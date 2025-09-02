from typing import Dict, Any, List, Optional, Tuple
import asyncio
import traceback
from ..toolbox import Tool, ToolDefinition, ToolParameter
from .datamodel import DBpediaEntity, DBpediaProperty, SearchResult, convert_api_entity_to_model, convert_api_property_to_model, convert_api_search_to_model
from .dbpedia_api import dbpedia_api
from .rate_limiter import rate_limited_request
from .utils import order_properties_by_degree

class GetEntityInfoTool(Tool):
    """Tool for getting comprehensive information about a DBpedia entity, including properties."""
    
    def __init__(self):
        super().__init__(
            name="get_entity_info",
            description="Get structured information about a DBpedia entity, including a human-readable label, a natural language description, and main properties (value-like characteristics of this entity) and relationships with neighboring nodes." \
            "USEFUL FOR: getting detailed info from an entity in DBpedia and its main properties, relationships, and context within the knowledge graph."
        )
        self.entity_limits = {}
        self.initial_limit = 20
        self.increment = 10
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_id",
                    type="string",
                    description="DBpedia entity ID (e.g., Douglas_Adams) or full URI"
                ),
                ToolParameter(
                    name="order_by_degree",
                    type="boolean",
                    description="Whether to order properties by degree",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="increase_limit",
                    type="boolean",
                    description="Whether to increase the number of results shown",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="include_properties",
                    type="boolean",
                    description="Whether to include detailed properties information",
                    required=False,
                    default=True
                ),
            ],
            return_type="dict",
            return_description="Comprehensive entity information including properties and relationships"
        )
    
    async def _fetch_property_details_parallel(self, prop_ids: List[str]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """Fetch property details in parallel for multiple property IDs."""
        print(f"🔄 Fetching property details for {len(prop_ids)} properties in parallel...")
        print(f"Property IDs: {prop_ids}")
        
        async def fetch_single_property(prop_id: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
            print(f"    ⏳ Starting fetch for property: {prop_id}")
            try:
                prop_data = await dbpedia_api.get_property(prop_id)
                print(f"    ✅ Successfully fetched property: {prop_id}")
                return prop_id, prop_data, None
            except Exception as e:
                error_msg = f"Error fetching property {prop_id}: {e}"
                print(f"    ⚠️  {error_msg}")
                return prop_id, {}, error_msg
        
        print("Creating property fetch tasks...")
        property_tasks = [fetch_single_property(prop_id) for prop_id in prop_ids]
        print(f"Created {len(property_tasks)} tasks. Gathering results...")
        property_results = await asyncio.gather(*property_tasks, return_exceptions=True)
        print("Gathered property results.")
        
        prop_details = {}
        errors = []
        for idx, result in enumerate(property_results):
            print(f"    Processing result {idx}: {result}")
            if isinstance(result, tuple) and len(result) == 3:
                prop_id, prop_data, error = result
                prop_details[prop_id] = prop_data
                if error:
                    errors.append(error)
            else:
                error_msg = f"Error fetching property details: {result}"
                print(f"    ⚠️  {error_msg}")
                errors.append(error_msg)
        
        print(f"✅ Property details fetched successfully. Total: {len(prop_details)}")
        if errors:
            print(f"⚠️  {len(errors)} errors occurred during property fetching")
        return prop_details, errors
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool to get entity information."""
        entity_id = kwargs.get("entity_id")
        order_by_degree = kwargs.get("order_by_degree", False)
        increase_limit = kwargs.get("increase_limit", False)
        include_properties = kwargs.get("include_properties", True)
        
        if not entity_id:
            raise ValueError("entity_id is required")
        
        # Initialize error tracking
        errors = []
        partial_data = False
        
        try:
            api_data = await dbpedia_api.get_entity(entity_id)
            entity = convert_api_entity_to_model(api_data)
        except Exception as e:
            error_msg = f"Failed to fetch entity {entity_id}: {e}"
            errors.append(error_msg)
            raise ValueError(error_msg)
        
        # Basic entity information
        result = {
            "id": entity.id,
            "uri": entity.uri,
            "label": entity.label,
            "description": entity.description,
            "aliases": entity.aliases,
            "statement_count": len(entity.statements),
            "link": entity.link,
            "errors": errors,
            "partial_data": partial_data
        }
        
        # Add properties information if requested
        if include_properties and entity.statements:
            # Fetch property details for all statements in parallel
            prop_ids = list(entity.statements.keys())
            prop_details, prop_errors = await self._fetch_property_details_parallel(prop_ids)
            errors.extend(prop_errors)
            if prop_errors:
                partial_data = True

            limit = self.entity_limits.get(entity_id, self.initial_limit)
            if increase_limit:
                limit += self.increment
            
            self.entity_limits[entity_id] = limit

            result.update({
                "statements": entity.statements,
                "property_details": prop_details,
                "limit": limit,
                "order_by_degree": order_by_degree,
                "errors": errors,
                "partial_data": partial_data
            })

            # Order properties if requested
            if order_by_degree:
                ordered_props = order_properties_by_degree(entity.statements, prop_details, limit)
                result["ordered_properties"] = ordered_props

        return result

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "Entity not found."
        
        # Basic entity information
        entity_label = result.get('label', result.get('id', 'N/A'))
        entity_id = result.get('id', 'N/A')
        description = result.get('description', 'N/A')
        statement_count = result.get('statement_count', 0)
        
        summary = f"Entity: {entity_label} ({entity_id})\n"
        summary += f"Description: {description[:100]}{'...' if len(description) > 100 else ''}\n"
        summary += f"Total statements: {statement_count}\n"
        
        # Add error information if present
        errors = result.get('errors', [])
        partial_data = result.get('partial_data', False)
        if errors:
            summary += f"⚠️ WARNING: {len(errors)} errors occurred during data retrieval.\n"
            if partial_data:
                summary += "Some data may be incomplete due to API failures.\n"
            # Include first few errors for context
            if len(errors) <= 3:
                summary += f"Errors: {'; '.join(errors)}\n"
            else:
                summary += f"First 3 errors: {'; '.join(errors[:3])}\n"
        
        # Add property information if present
        if 'statements' in result and result['statements']:
            limit = result.get('limit', 20)
            summary += f"\nKey properties (showing up to {limit}):\n"
            
            property_details = result.get('property_details', {})
            statements = result['statements']
            
            # Order properties if specified
            ordered_props = result.get('ordered_properties')
            if ordered_props:
                prop_ids = ordered_props[:limit]
            else:
                prop_ids = list(statements.keys())[:limit]
            
            for prop_id in prop_ids:
                if prop_id in statements:
                    prop_label = property_details.get(prop_id, {}).get('labels', {}).get('en', prop_id)
                    value_count = len(statements[prop_id])
                    summary += f"  - {prop_label}: {value_count} value(s)\n"
        
        return summary.strip()

class GetPropertyInfoTool(Tool):
    """Tool for getting information about a DBpedia property."""
    
    def __init__(self):
        super().__init__(
            name="get_property_info",
            description="Get information about a DBpedia property, including its label, description, and datatype." \
            "USEFUL FOR: understanding what a property represents and how it's used in the DBpedia ontology."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="property_id",
                    type="string",
                    description="DBpedia property ID (e.g., birthPlace) or full URI"
                )
            ],
            return_type="dict",
            return_description="Property information including label, description and datatype"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool to get property information."""
        property_id = kwargs.get("property_id")
        
        if not property_id:
            raise ValueError("property_id is required")
        
        try:
            api_data = await dbpedia_api.get_property(property_id)
            property_obj = convert_api_property_to_model(api_data)
            
            return {
                "id": property_obj.id,
                "uri": property_obj.uri,
                "label": property_obj.label,
                "description": property_obj.description,
                "aliases": property_obj.aliases,
                "datatype": property_obj.datatype,
                "link": property_obj.link
            }
        except Exception as e:
            error_msg = f"Failed to fetch property {property_id}: {e}"
            raise ValueError(error_msg)

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "Property not found."
        
        prop_id = result.get('id', 'N/A')
        prop_label = result.get('label', prop_id)
        description = result.get('description', 'N/A')
        datatype = result.get('datatype', 'unknown')
        
        summary = f"Property: {prop_label} ({prop_id})\n"
        summary += f"Description: {description[:100]}{'...' if len(description) > 100 else ''}\n"
        summary += f"Datatype: {datatype}\n"
        
        aliases = result.get('aliases', [])
        if aliases:
            summary += f"Aliases: {', '.join(aliases[:3])}\n"
        
        return summary.strip()

class SearchEntitiesTool(Tool):
    """Tool for searching DBpedia entities by text query."""
    
    def __init__(self):
        super().__init__(
            name="search_entities",
            description="Search for DBpedia entities using natural language queries." \
            "USEFUL FOR: finding entities when you know their name or description but not their exact ID." \
            "Returns a list of matching entities with their IDs and descriptions."
        )
        self.search_limits = {}
        self.initial_limit = 10
        self.increment = 5
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query (entity name or keywords)"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of results to return",
                    required=False,
                    default=10
                ),
                ToolParameter(
                    name="increase_limit",
                    type="boolean",
                    description="Whether to increase the number of results shown",
                    required=False,
                    default=False
                )
            ],
            return_type="dict",
            return_description="List of matching entities with their basic information"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool to search for entities."""
        query = kwargs.get("query")
        limit = kwargs.get("limit", 10)
        increase_limit = kwargs.get("increase_limit", False)
        
        if not query:
            raise ValueError("query is required")
        
        # Manage search limits
        query_hash = hash(query)
        display_limit = self.search_limits.get(query_hash, self.initial_limit)
        if increase_limit:
            display_limit += self.increment
        self.search_limits[query_hash] = display_limit
        
        # Use the higher of the requested limit or display limit
        actual_limit = max(limit, display_limit)
        
        try:
            api_results = await dbpedia_api.search_entities(query, actual_limit)
            search_results = convert_api_search_to_model(api_results)
            
            return {
                "query": query,
                "results": [
                    {
                        "id": result.id,
                        "uri": result.uri,
                        "label": result.label,
                        "description": result.description,
                        "categories": result.categories,
                        "classes": result.classes
                    } for result in search_results[:display_limit]
                ],
                "total_results": len(search_results),
                "displayed_results": min(len(search_results), display_limit),
                "limit": display_limit
            }
        except Exception as e:
            error_msg = f"Failed to search entities with query '{query}': {e}"
            raise ValueError(error_msg)

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the search result into a readable, concise string."""
        if not result:
            return "No search results found."
        
        query = result.get('query', 'N/A')
        total_results = result.get('total_results', 0)
        displayed_results = result.get('displayed_results', 0)
        
        summary = f"Search query: '{query}'\n"
        summary += f"Found {total_results} results, showing {displayed_results}:\n\n"
        
        results = result.get('results', [])
        for i, entity in enumerate(results[:5], 1):  # Show top 5 results
            label = entity.get('label', entity.get('id', 'N/A'))
            entity_id = entity.get('id', 'N/A')
            description = entity.get('description', 'No description')
            
            summary += f"{i}. {label} ({entity_id})\n"
            summary += f"   {description[:80]}{'...' if len(description) > 80 else ''}\n"
        
        if len(results) > 5:
            summary += f"... and {len(results) - 5} more results\n"
        
        return summary.strip()

class GetWikipediaPageTool(Tool):
    """Tool for fetching the Wikipedia page content corresponding to a DBpedia entity."""
    
    def __init__(self):
        super().__init__(
            name="get_wikipedia_page",
            description="Get the Wikipedia page content for a DBpedia entity. Since DBpedia is extracted from Wikipedia, " \
            "most entities have corresponding Wikipedia pages with detailed information." \
            "USEFUL FOR: getting detailed article content, accessing infobox data, retrieving comprehensive descriptions " \
            "that complement the structured data from DBpedia."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_id",
                    type="string",
                    description="DBpedia entity ID (e.g., Douglas_Adams) or full URI"
                ),
                ToolParameter(
                    name="sections",
                    type="array",
                    description="Specific sections to retrieve (e.g., ['introduction', 'biography']). If not specified, returns full content.",
                    required=False,
                    items={"type": "string"}
                ),
                ToolParameter(
                    name="max_length",
                    type="integer",
                    description="Maximum length of content to return (in characters)",
                    required=False,
                    default=5000
                ),
                ToolParameter(
                    name="include_infobox",
                    type="boolean",
                    description="Whether to include Wikipedia infobox data",
                    required=False,
                    default=True
                )
            ],
            return_type="dict",
            return_description="Wikipedia page content with metadata, sections, and optional infobox data"
        )
    
    async def _get_wikipedia_url_from_dbpedia(self, entity_id: str) -> str:
        """Get the Wikipedia URL for a DBpedia entity."""
        entity_uri = dbpedia_api._normalize_uri(entity_id)
        
        # Query DBpedia for the Wikipedia link
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        
        SELECT ?wikipediaUrl WHERE {{
            <{entity_uri}> foaf:isPrimaryTopicOf ?wikipediaUrl .
            FILTER(CONTAINS(STR(?wikipediaUrl), "en.wikipedia.org"))
        }}
        LIMIT 1
        """
        
        try:
            results = await dbpedia_api._execute_sparql_query(query)
            bindings = results.get('results', {}).get('bindings', [])
            
            if bindings:
                return bindings[0]['wikipediaUrl']['value']
            else:
                # Fallback: construct URL from entity name
                entity_name = entity_id.replace('_', ' ')
                return f"https://en.wikipedia.org/wiki/{entity_id}"
        except Exception as e:
            print(f"Warning: Could not fetch Wikipedia URL from DBpedia: {e}")
            # Fallback: construct URL from entity name
            return f"https://en.wikipedia.org/wiki/{entity_id}"
    
    async def _fetch_wikipedia_content(self, wikipedia_url: str, max_length: int = 5000) -> Dict[str, Any]:
        """Fetch Wikipedia page content using the Wikipedia API."""
        import aiohttp
        import re
        
        # Extract page title from URL
        page_title = wikipedia_url.split('/')[-1]
        
        # Wikipedia API endpoint
        api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + page_title
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'RoboData/1.0 (Python)',
                    'Accept': 'application/json'
                }
                
                async with session.get(api_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Get full page content for sections
                        full_content_url = f"https://en.wikipedia.org/w/api.php"
                        params = {
                            'action': 'query',
                            'format': 'json',
                            'titles': page_title,
                            'prop': 'extracts|pageprops',
                            'exintro': 0,
                            'explaintext': 1,
                            'exsectionformat': 'wiki'
                        }
                        
                        async with session.get(full_content_url, params=params, headers=headers) as full_response:
                            full_data = {}
                            if full_response.status == 200:
                                full_result = await full_response.json()
                                pages = full_result.get('query', {}).get('pages', {})
                                for page_id, page_data in pages.items():
                                    if page_id != '-1':  # Page exists
                                        full_data = page_data
                        
                        # Process the content
                        extract = full_data.get('extract', data.get('extract', ''))
                        if len(extract) > max_length:
                            extract = extract[:max_length] + "..."
                        
                        return {
                            'title': data.get('title', page_title),
                            'url': wikipedia_url,
                            'summary': data.get('extract', ''),
                            'full_content': extract,
                            'page_id': data.get('pageid'),
                            'last_modified': data.get('timestamp'),
                            'content_length': len(extract),
                            'thumbnail': data.get('thumbnail', {}).get('source') if data.get('thumbnail') else None,
                            'coordinates': data.get('coordinates'),
                            'description': data.get('description', ''),
                            'lang': data.get('lang', 'en')
                        }
                    else:
                        raise ValueError(f"Wikipedia API request failed with status {response.status}")
                        
        except Exception as e:
            raise ValueError(f"Failed to fetch Wikipedia content: {str(e)}")
    
    async def _extract_infobox_from_dbpedia(self, entity_id: str) -> Dict[str, Any]:
        """Extract infobox-like data from DBpedia properties."""
        entity_uri = dbpedia_api._normalize_uri(entity_id)
        
        # Query for common infobox properties
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbp: <http://dbpedia.org/property/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?property ?value ?valueLabel WHERE {{
            <{entity_uri}> ?property ?value .
            OPTIONAL {{ ?value rdfs:label ?valueLabel . FILTER(LANG(?valueLabel) = "en") }}
            FILTER(
                ?property = dbo:birthDate ||
                ?property = dbo:deathDate ||
                ?property = dbo:birthPlace ||
                ?property = dbo:occupation ||
                ?property = dbo:nationality ||
                ?property = dbo:almaMater ||
                ?property = dbo:spouse ||
                ?property = dbo:knownFor ||
                ?property = dbp:name ||
                ?property = dbp:born ||
                ?property = dbp:died
            )
        }}
        """
        
        try:
            results = await dbpedia_api._execute_sparql_query(query)
            infobox = {}
            
            for binding in results.get('results', {}).get('bindings', []):
                prop_uri = binding['property']['value']
                prop_name = dbpedia_api._extract_name_from_uri(prop_uri)
                value = binding['value']['value']
                display_value = binding.get('valueLabel', {}).get('value', value)
                
                if prop_name not in infobox:
                    infobox[prop_name] = []
                infobox[prop_name].append(display_value)
            
            return infobox
        except Exception as e:
            print(f"Warning: Could not extract infobox data: {e}")
            return {}
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool to get Wikipedia page content."""
        entity_id = kwargs.get("entity_id")
        sections = kwargs.get("sections", [])
        max_length = kwargs.get("max_length", 5000)
        include_infobox = kwargs.get("include_infobox", True)
        
        if not entity_id:
            raise ValueError("entity_id is required")
        
        errors = []
        
        try:
            # Get Wikipedia URL from DBpedia
            print(f"🔗 Finding Wikipedia page for entity: {entity_id}")
            wikipedia_url = await self._get_wikipedia_url_from_dbpedia(entity_id)
            print(f"📄 Wikipedia URL: {wikipedia_url}")
            
            # Fetch Wikipedia content
            print(f"📥 Fetching Wikipedia content...")
            wikipedia_data = await self._fetch_wikipedia_content(wikipedia_url, max_length)
            
            result = {
                "entity_id": entity_id,
                "wikipedia_url": wikipedia_url,
                "title": wikipedia_data['title'],
                "summary": wikipedia_data['summary'],
                "full_content": wikipedia_data['full_content'],
                "content_length": wikipedia_data['content_length'],
                "page_id": wikipedia_data.get('page_id'),
                "last_modified": wikipedia_data.get('last_modified'),
                "thumbnail": wikipedia_data.get('thumbnail'),
                "coordinates": wikipedia_data.get('coordinates'),
                "description": wikipedia_data.get('description', ''),
                "errors": errors
            }
            
            # Add infobox data if requested
            if include_infobox:
                print(f"📊 Extracting infobox data...")
                try:
                    infobox_data = await self._extract_infobox_from_dbpedia(entity_id)
                    result["infobox"] = infobox_data
                except Exception as e:
                    error_msg = f"Failed to extract infobox data: {e}"
                    errors.append(error_msg)
                    result["infobox"] = {}
            
            # Filter sections if specified
            if sections:
                # This is a simplified section filtering - could be enhanced
                filtered_content = ""
                content_lines = wikipedia_data['full_content'].split('\n')
                current_section = ""
                include_current = False
                
                for line in content_lines:
                    if line.strip().endswith('==') or line.strip().endswith('==='):
                        current_section = line.strip().replace('=', '').strip().lower()
                        include_current = any(section.lower() in current_section for section in sections)
                    
                    if include_current or not sections:  # Include if in requested section or no sections specified
                        filtered_content += line + '\n'
                
                result["filtered_content"] = filtered_content[:max_length]
            
            result["errors"] = errors
            print(f"✅ Wikipedia page fetched successfully")
            return result
            
        except Exception as e:
            error_msg = f"Failed to fetch Wikipedia page for {entity_id}: {e}"
            errors.append(error_msg)
            raise ValueError(error_msg)
    
    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the Wikipedia page result into a readable string."""
        if not result:
            return "No Wikipedia page found."
        
        entity_id = result.get('entity_id', 'N/A')
        title = result.get('title', 'N/A')
        wikipedia_url = result.get('wikipedia_url', 'N/A')
        content_length = result.get('content_length', 0)
        
        summary = f"Wikipedia Page: {title}\n"
        summary += f"Entity: {entity_id}\n"
        summary += f"URL: {wikipedia_url}\n"
        summary += f"Content length: {content_length} characters\n\n"
        
        # Add summary/description
        description = result.get('description', '')
        if description:
            summary += f"Description: {description}\n\n"
        
        # Add page summary
        page_summary = result.get('summary', '')
        if page_summary:
            summary += f"Summary:\n{page_summary[:300]}{'...' if len(page_summary) > 300 else ''}\n\n"
        
        # Add infobox data if available
        infobox = result.get('infobox', {})
        if infobox:
            summary += "Key Information:\n"
            for prop_name, values in list(infobox.items())[:5]:  # Show top 5 infobox items
                summary += f"  {prop_name}: {', '.join(values[:3])}\n"  # Show up to 3 values per property
        
        # Add thumbnail info
        thumbnail = result.get('thumbnail')
        if thumbnail:
            summary += f"\nThumbnail: {thumbnail}\n"
        
        # Add error information
        errors = result.get('errors', [])
        if errors:
            summary += f"\n⚠️ {len(errors)} warnings/errors occurred."
        
        return summary.strip()

# Simple direct test for GetEntityInfoTool
if __name__ == "__main__":
    import asyncio
    import logging
    logging.basicConfig(level=logging.INFO)
    print("Testing GetEntityInfoTool...")
    tool = GetEntityInfoTool()
    entity_id = "Douglas_Adams"  # Example entity
    async def test_entity_info():
        try:
            print(f"Running get_entity_info for entity: {entity_id}")
            result = await tool.execute(entity_id=entity_id, order_by_degree=True, include_properties=True)
            print("Result:")
            print(tool.format_result(result))
        except Exception as e:
            print(f"Error during test: {e}")
            import traceback
            traceback.print_exc()
    asyncio.run(test_entity_info())