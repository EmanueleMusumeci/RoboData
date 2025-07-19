from typing import Dict, Any, List, Optional, Tuple
import requests
import asyncio
import traceback
from ..toolbox import Tool, ToolDefinition, ToolParameter
from .datamodel import WikidataEntity, WikidataProperty, SearchResult, WikidataStatement, convert_api_entity_to_model, convert_api_property_to_model, convert_api_search_to_model
from .wikidata_api import wikidata_api
from .utils import order_properties_by_degree

class GetEntityInfoTool(Tool):
    """Tool for getting comprehensive information about a Wikidata entity, including properties."""
    
    def __init__(self):
        super().__init__(
            name="get_entity_info",
            description="Get structured information about a Wikidata entity, including a human-readable label, a natural language description, and main properties (value-like characteristics of this entity) and relationships with neighboring nodes." \
            "USEFUL FOR: getting a detailed info from an entity in Wikidata and its main properties, relationships, and context within the knowledge graph."
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
                    description="Wikidata entity ID (e.g., Q42)"
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
                )
            ],
            return_type="object",
            return_description="Detailed structured description of the input entity with a human-readable label, natural language description, and main properties (value-like characteristics of this entity) and relationships with neighboring nodes."
        )

    async def _fetch_property_details_parallel(self, property_ids: List[str]) -> Dict[str, Dict[str, str]]:
        """Fetch property names and descriptions in parallel for multiple property IDs."""
        
        async def fetch_single_prop(prop_id: str) -> Tuple[str, Dict[str, str]]:
            try:
                prop_api_data = await wikidata_api.get_property(prop_id)
                labels = prop_api_data.get('labels', {})
                descriptions = prop_api_data.get('descriptions', {})
                
                # Labels and descriptions are usually strings, not dicts
                prop_name = labels.get('en', prop_id)
                prop_desc = descriptions.get('en', 'No description available')
                
                return prop_id, {
                    'name': prop_name,
                    'description': prop_desc
                }
            except Exception as e:
                print(f"Error fetching property {prop_id}: {e}")
                return prop_id, {
                    'name': prop_id,
                    'description': 'No description available'
                }

        tasks = [fetch_single_prop(prop_id) for prop_id in property_ids]
        results = await asyncio.gather(*tasks)
        
        return {prop_id: prop_details for prop_id, prop_details in results}
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Get comprehensive information about a Wikidata entity."""
        entity_id = kwargs.get("entity_id")
        order_by_degree = kwargs.get("order_by_degree", False)
        increase_limit = kwargs.get("increase_limit", False)
        include_properties = kwargs.get("include_properties", True)
        
        if not entity_id:
            raise ValueError("entity_id is required")
            
        api_data = await wikidata_api.get_entity(entity_id)
        entity = convert_api_entity_to_model(api_data)
        
        # Basic entity information
        result = {
            "id": entity.id,
            "label": entity.label,
            "description": entity.description,
            "aliases": entity.aliases,
            "statement_count": len(entity.statements),
            "link": entity.link
        }
        
        # Add properties information if requested
        if include_properties and entity.statements:
            # Fetch property details for all statements in parallel
            prop_ids = list(entity.statements.keys())
            prop_details = await self._fetch_property_details_parallel(prop_ids)

            limit = self.entity_limits.get(entity_id, self.initial_limit)
            if increase_limit:
                limit += self.increment
            
            self.entity_limits[entity_id] = limit

            result.update({
                "statements": entity.statements,
                "property_details": prop_details,
                "limit": limit,
                "order_by_degree": order_by_degree
            })

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
        
        # Add properties information if available
        statements = result.get("statements")
        if statements:
            prop_details = result.get("property_details", {})
            limit = result.get("limit", 10)
            order_by_degree = result.get("order_by_degree", False)

            ordered_props = order_properties_by_degree(statements, enabled=order_by_degree)
            
            summary += f"\nTop {min(limit, len(ordered_props))} of {len(ordered_props)} properties (ordered by degree: {order_by_degree}):\n"
            for i, prop_id in enumerate(ordered_props[:limit]):
                prop_info = prop_details.get(prop_id, {'name': prop_id, 'description': 'No description available'})
                prop_name = prop_info.get('name', prop_id)
                prop_desc = prop_info.get('description', 'No description available')
                # Show first 100 chars of description
                desc_preview = prop_desc[:100] + ('...' if len(prop_desc) > 100 else '')
                summary += f"  {i+1}. {prop_name} ({prop_id}): {desc_preview}\n"
        
        return summary.strip()

class GetPropertyInfoTool(Tool):
    """Tool for getting information about a Wikidata property."""
    
    def __init__(self):
        super().__init__(
            name="get_property_info",
            description="Get information about a Wikidata property, including its label, description, datatype, and link to the property page." \
            "USEFUL FOR: understanding the nature of a property in Wikidata, its type, and how it can be used in queries."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="property_id",
                    type="string",
                    description="Wikidata property ID (e.g., P31)"
                )
            ],
            return_type="object",
            return_description="Structured information about the Wikidata property, including its ID, label, description, datatype, and link to the property page."
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Get information about a Wikidata property."""
        property_id = kwargs.get("property_id")
        if not property_id:
            raise ValueError("property_id is required")
            
        api_data = await wikidata_api.get_property(property_id)
        prop = convert_api_property_to_model(api_data)
        return {
            "id": prop.id,
            "label": prop.label,
            "description": prop.description,
            "datatype": prop.datatype,
            "link": prop.link
        }

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "Property not found."
        
        prop_label = result.get('label', 'N/A')
        prop_id = result.get('id', 'N/A')
        datatype = result.get('datatype', 'N/A')
        description = result.get('description', 'No description available')
        
        summary = f"Property: {prop_label} ({prop_id})\n"
        summary += f"Type: {datatype}\n"
        summary += f"Description: {description}"
        
        return summary

class SearchEntitiesTool(Tool):
    """Tool for searching Wikidata entities by text query with detailed information."""
    
    def __init__(self):
        super().__init__(
            name="search_entities",
            description="Search for Wikidata entities by text query with detailed information about the main results. Optionally, you can get the main properties for each entity." \
            "USEFUL FOR: finding entities in Wikidata based on a text query, retrieving their main properties. Useful to find a starting point for further exploration of the knowledge graph."
        )
        self.query_limits = {}
        self.initial_limit = 5
        self.increment = 5
        self.prop_limits = {}
        self.initial_prop_limit = 20
        self.prop_increment = 20
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Text query to search for"
                ),
                ToolParameter(
                    name="increase_limit",
                    type="boolean",
                    description=f"Whether to increase the number of results shown (increments by {self.increment})",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="include_properties",
                    type="boolean",
                    description="Whether to include main properties for each found entity",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="increase_prop_limit",
                    type="boolean",
                    description=f"Whether to increase the number of properties shown per entity (increments by {self.prop_increment})",
                    required=False,
                    default=False
                )
            ],
            return_type="array",
            return_description="List of search results with IDs, labels, descriptions, and optionally main properties"
        )

    async def _fetch_single_property(self, prop_id: str) -> Tuple[str, Dict[str, str]]:
        """Fetch a single property's details."""
        try:
            prop_api_data = await wikidata_api.get_property(prop_id)
            labels = prop_api_data.get('labels', {})
            descriptions = prop_api_data.get('descriptions', {})
            
            # Labels and descriptions are usually strings, not dicts
            prop_name = labels.get('en', prop_id)
            prop_desc = descriptions.get('en', 'No description available')
            
            return prop_id, {
                'name': prop_name,
                'description': prop_desc
            }
        except Exception as e:
            print(f"Error fetching property {prop_id}: {e}")
            return prop_id, {
                'name': prop_id,
                'description': 'No description available'
            }

    async def _fetch_entity_properties(self, entity_id: str, limit: int) -> Dict[str, Any]:
        """Fetch properties for a single entity."""
        try:
            api_data = await wikidata_api.get_entity(entity_id)
            entity = convert_api_entity_to_model(api_data)
            
            if not entity.statements:
                return {"statements": {}, "property_details": {}}
            
            # Get top properties by degree
            ordered_props = order_properties_by_degree(entity.statements, enabled=True)
            top_props = ordered_props[:limit]
            
            # Fetch details for top properties in parallel
            tasks = [self._fetch_single_property(prop_id) for prop_id in top_props]
            property_results = await asyncio.gather(*tasks)
            
            prop_details = {prop_id: prop_info for prop_id, prop_info in property_results}
            
            return {
                "statements": {prop_id: entity.statements[prop_id] for prop_id in top_props if prop_id in entity.statements},
                "property_details": prop_details
            }
        except Exception as e:
            print(f"Error fetching entity {entity_id}: {e}")
            return {"statements": {}, "property_details": {}}
    
    async def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """Search for Wikidata entities by text query."""
        query = kwargs.get("query")
        increase_limit = kwargs.get("increase_limit", False)
        include_properties = kwargs.get("include_properties", False)
        increase_prop_limit = kwargs.get("increase_prop_limit", False)
        # The API limit is not the same as the display limit
        limit = kwargs.get("limit", 50)
        
        if not query:
            raise ValueError("query is required")
            
        api_data = await wikidata_api.search_entities(query, limit)
        results = convert_api_search_to_model(api_data)

        display_limit = self.query_limits.get(query, self.initial_limit)
        if increase_limit:
            display_limit += self.increment
        self.query_limits[query] = display_limit
        
        prop_limit = self.prop_limits.get(query, self.initial_prop_limit)
        if increase_prop_limit:
            prop_limit += self.prop_increment
        self.prop_limits[query] = prop_limit
        
        # Get the results to process
        results_to_process = results[:display_limit]
        
        # Prepare basic entity data for all results
        enhanced_results = []
        for result in results_to_process:
            entity_data = {
                "id": result.id,
                "label": result.label,
                "description": result.description,
                "url": result.url,
                "limit": display_limit,
                "query": query,
                "include_properties": include_properties,
                "prop_limit": prop_limit
            }
            enhanced_results.append(entity_data)
        
        # Fetch properties in parallel if requested
        if include_properties and results_to_process:
            # Create tasks for parallel property fetching
            property_tasks = [
                self._fetch_entity_properties(result.id, prop_limit) 
                for result in results_to_process
            ]
            
            # Execute all property fetching tasks in parallel
            properties_results = await asyncio.gather(*property_tasks)
            
            # Update enhanced_results with property data
            for i, props_data in enumerate(properties_results):
                enhanced_results[i].update(props_data)
        
        return enhanced_results

    def format_result(self, result: List[Dict[str, Any]]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No entities found."
        
        limit = result[0].get("limit", 5)
        query = result[0].get("query")
        include_properties = result[0].get("include_properties", False)
        prop_limit = result[0].get("prop_limit", 5)

        summary = f"Found {len(result)} entities for '{query}'. Showing top {min(limit, len(result))}:\n\n"
        
        for i, entity in enumerate(result[:limit]):
            entity_id = entity.get('id', 'N/A')
            entity_label = entity.get('label', 'N/A')
            entity_desc = entity.get('description', 'No description available')
            
            summary += f"{i+1}. {entity_label} ({entity_id})\n"
            summary += f"   Description: {entity_desc}\n"
            
            # Add properties if included
            if include_properties:
                statements = entity.get("statements", {})
                prop_details = entity.get("property_details", {})
                
                if statements:
                    summary += f"   Top {min(prop_limit, len(statements))} properties:\n"
                    for j, prop_id in enumerate(list(statements.keys())[:prop_limit]):
                        prop_info = prop_details.get(prop_id, {'name': prop_id, 'description': 'No description available'})
                        prop_name = prop_info.get('name', prop_id)
                        prop_desc = prop_info.get('description', 'No description available')
                        # Show first 50 chars of description for search results
                        desc_preview = prop_desc[:50] + ('...' if len(prop_desc) > 50 else '')
                        summary += f"     {j+1}. {prop_name} ({prop_id}): {desc_preview}\n"
                else:
                    summary += f"   No properties found\n"
            
            summary += "\n"
            
        return summary.strip()

# Legacy sync wrapper functions for backward compatibility
async def get_entity_info_async(entity_id: str) -> WikidataEntity:
    """Get basic information about a Wikidata entity using the async API."""
    api_data = await wikidata_api.get_entity(entity_id)
    return convert_api_entity_to_model(api_data)

def get_entity_info(entity_id: str) -> WikidataEntity:
    """Get basic information about a Wikidata entity (sync wrapper)."""
    return asyncio.run(get_entity_info_async(entity_id))

async def get_property_info_async(property_id: str) -> WikidataProperty:
    """Get information about a Wikidata property using the async API."""
    api_data = await wikidata_api.get_property(property_id)
    return convert_api_property_to_model(api_data)

def get_property_info(property_id: str) -> WikidataProperty:
    """Get information about a Wikidata property (sync wrapper)."""
    return asyncio.run(get_property_info_async(property_id))

async def search_entities_async(query: str, limit: int = 10) -> List[SearchResult]:
    """Search for Wikidata entities by text query using the async API."""
    api_data = await wikidata_api.search_entities(query, limit)
    return convert_api_search_to_model(api_data)

def search_entities(query: str, limit: int = 10) -> List[SearchResult]:
    """Search for Wikidata entities by text query (sync wrapper)."""
    return asyncio.run(search_entities_async(query, limit))

if __name__ == "__main__":
    """Test cases for base tools with result formatting."""
    import asyncio
    import traceback
    
    async def main():
        """Run test cases for base tools and their format_result methods."""
        print("=== Testing Wikidata Base Tools and Result Formatting ===\n")
        
        # Test 1: GetEntityInfoTool
        print("1. Testing GetEntityInfoTool...")
        try:
            tool = GetEntityInfoTool()
            result = await tool.execute(entity_id='Q42')
            formatted_result = tool.format_result(result)
            print(f"   ✓ GetEntityInfoTool execution successful for 'Douglas Adams' (Q42).")
            print(f"   - Formatted result: {formatted_result}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            traceback.print_exc()
        
        print()
        
        # Test 2: GetPropertyInfoTool
        print("2. Testing GetPropertyInfoTool...")
        try:
            tool = GetPropertyInfoTool()
            result = await tool.execute(property_id='P31')
            formatted_result = tool.format_result(result)
            print(f"   ✓ GetPropertyInfoTool execution successful for 'instance of' (P31).")
            print(f"   - Formatted result: {formatted_result}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            traceback.print_exc()
        
        print()
        
        # Test 3: SearchEntitiesTool
        print("3. Testing SearchEntitiesTool...")
        try:
            tool = SearchEntitiesTool()
            result = await tool.execute(query='Douglas Adams', limit=5)
            formatted_result = tool.format_result(result)
            print(f"   ✓ SearchEntitiesTool execution successful for 'Douglas Adams'.")
            print(f"   - Formatted result: {formatted_result}")
            
            # Test with properties
            print("   - Testing with properties...")
            result_with_props = await tool.execute(query='Douglas Adams', include_properties=True, limit=3)
            formatted_result_with_props = tool.format_result(result_with_props)
            print(f"   - Formatted result with properties: {formatted_result_with_props}")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            traceback.print_exc()
            
        print()

        # Test 4: Handling not found cases
        print("4. Testing not found cases...")
        
        # Test GetEntityInfoTool with empty result
        entity_tool = GetEntityInfoTool()
        formatted_entity_not_found = entity_tool.format_result({})
        print(f"   - Formatted result for non-existent entity: {formatted_entity_not_found}")

        # Test GetPropertyInfoTool with empty result
        property_tool = GetPropertyInfoTool()
        formatted_property_not_found = property_tool.format_result({})
        print(f"   - Formatted result for non-existent property: {formatted_property_not_found}")

        # Test SearchEntitiesTool with empty result
        search_tool = SearchEntitiesTool()
        formatted_search_not_found = search_tool.format_result([])
        print(f"   - Formatted result for no search results: {formatted_search_not_found}")

        print("\n=== All tests completed ===")
    
    # Run the main function
    asyncio.run(main())
