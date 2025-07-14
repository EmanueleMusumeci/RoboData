from typing import Dict, Any, List, Optional, Tuple
import requests
import asyncio
import traceback
from ..toolbox import Tool, ToolDefinition, ToolParameter
from .datamodel import WikidataEntity, WikidataProperty, SearchResult, WikidataStatement, convert_api_entity_to_model, convert_api_property_to_model, convert_api_search_to_model
from .wikidata_api import wikidata_api
from .utils import order_properties_by_degree

class GetEntityInfoTool(Tool):
    """Tool for getting basic information about a Wikidata entity."""
    
    def __init__(self):
        super().__init__(
            name="get_entity_info",
            description="Get basic information about a Wikidata entity"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_id",
                    type="string",
                    description="Wikidata entity ID (e.g., Q42)"
                )
            ],
            return_type="object",
            return_description="WikidataEntity with labels, descriptions, aliases, and statements"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Get basic information about a Wikidata entity."""
        entity_id = kwargs.get("entity_id")
        if not entity_id:
            raise ValueError("entity_id is required")
            
        api_data = await wikidata_api.get_entity(entity_id)
        entity = convert_api_entity_to_model(api_data)
        

        return {
            "id": entity.id,
            "label": entity.label,
            "description": entity.description,
            "aliases": entity.aliases,
            "statement_count": len(entity.statements),
            "link": entity.link
        }

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "Entity not found."
        return f"Entity: {result.get('label', 'N/A')} ({result.get('id', 'N/A')}), Description: {result.get('description', 'N/A')[:100]}..., Stmts: {result.get('statement_count', 0)}"

class GetEntityPropertiesTool(Tool):
    """Tool for getting all properties of a Wikidata entity, ordered by importance."""
    
    def __init__(self):
        super().__init__(
            name="get_entity_properties",
            description="Get all properties of a Wikidata entity, ordered by importance."
        )
        self.entity_limits = {}
        self.initial_limit = 10
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
                )
            ],
            return_type="object",
            return_description="WikidataEntity with statements, and ordered properties"
        )

    async def _fetch_property_names_parallel(self, property_ids: List[str]) -> Dict[str, str]:
        """Fetch property names in parallel for multiple property IDs."""
        
        async def fetch_single_prop(prop_id: str) -> Tuple[str, str]:
            try:
                prop_api_data = await wikidata_api.get_property(prop_id)
                return prop_id, prop_api_data.get('labels', {}).get('en', {})
            except Exception as e:
                print(f"Error fetching property {prop_id}: {e}")
                return prop_id, prop_id

        tasks = [fetch_single_prop(prop_id) for prop_id in property_ids]
        results = await asyncio.gather(*tasks)
        
        return {prop_id: prop_name for prop_id, prop_name in results}
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Get all properties of a Wikidata entity."""
        entity_id = kwargs.get("entity_id")
        order_by_degree = kwargs.get("order_by_degree", False)
        increase_limit = kwargs.get("increase_limit", False)
        if not entity_id:
            raise ValueError("entity_id is required")
            
        api_data = await wikidata_api.get_entity(entity_id)
        entity = convert_api_entity_to_model(api_data)
        
        # Fetch property names for all statements in parallel
        prop_ids = list(entity.statements.keys())
        prop_names = await self._fetch_property_names_parallel(prop_ids)

        limit = self.entity_limits.get(entity_id, self.initial_limit)
        if increase_limit:
            limit += self.increment
        
        self.entity_limits[entity_id] = limit

        return {
            "id": entity.id,
            "label": entity.label,
            "statements": entity.statements,
            "property_names": prop_names,
            "limit": limit,
            "order_by_degree": order_by_degree
        }

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the result into a readable, concise string."""
        if not result or not result.get("statements"):
            return f"No properties found for entity {result.get('id', 'N/A')}."

        entity_label = result.get('label', result.get('id', 'N/A'))
        statements = result.get("statements", {})
        prop_names = result.get("property_names", {})
        limit = result.get("limit", 10)
        order_by_degree = result.get("order_by_degree", False)

        ordered_props = order_properties_by_degree(statements, enabled=order_by_degree)
        
        summary = f"Top {min(limit, len(ordered_props))} of {len(ordered_props)} properties for '{entity_label}' (ordered by degree: {order_by_degree}):\n"
        for i, prop_id in enumerate(ordered_props[:limit]):
            prop_name = prop_names.get(prop_id, prop_id)
            degree = len(statements.get(prop_id, []))
            summary += f"  {i+1}. {prop_name} ({prop_id}) - Degree: {degree}\n"
            
        return summary.strip()

class GetPropertyInfoTool(Tool):
    """Tool for getting information about a Wikidata property."""
    
    def __init__(self):
        super().__init__(
            name="get_property_info",
            description="Get information about a Wikidata property"
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
            return_description="WikidataProperty with label, description, and datatype"
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
        desc = result.get('description', '')
        desc_str = f", Description: {desc[:100]}..." if desc else ""
        return f"Property: {result.get('label', 'N/A')} ({result.get('id', 'N/A')}), Type: {result.get('datatype', 'N/A')}{desc_str}"

class SearchEntitiesTool(Tool):
    """Tool for searching Wikidata entities by text query."""
    
    def __init__(self):
        super().__init__(
            name="search_entities",
            description="Search for Wikidata entities by text query"
        )
        self.query_limits = {}
        self.initial_limit = 5
        self.increment = 5
    
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
                )
            ],
            return_type="array",
            return_description="List of search results with IDs, labels, and descriptions"
        )
    
    async def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """Search for Wikidata entities by text query."""
        query = kwargs.get("query")
        increase_limit = kwargs.get("increase_limit", False)
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
        
        return [
            {
                "id": result.id,
                "label": result.label,
                "description": result.description,
                "url": result.url,
                "limit": display_limit,
                "query": query
            }
            for result in results
        ]

    def format_result(self, result: List[Dict[str, Any]]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No entities found."
        
        limit = result[0].get("limit", 5)
        query = result[0].get("query")

        formatted_results = [
            f"{r.get('label', 'N/A')} ({r.get('id', 'N/A')})"
            for r in result[:limit]
        ]
        
        summary = f"Found {len(result)} entities for '{query}'. Top {min(limit, len(result))}: {'; '.join(formatted_results)}"
        if len(result) > limit:
            summary += "..."
            
        return summary

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

        print()

        # Test 5: GetEntityPropertiesTool
        print("5. Testing GetEntityPropertiesTool...")
        try:
            tool = GetEntityPropertiesTool()
            # Test without degree ordering
            result = await tool.execute(entity_id='Q42')
            formatted_result = tool.format_result(result)
            print(f"   ✓ GetEntityPropertiesTool execution successful for 'Douglas Adams' (Q42).")
            print(f"   - Formatted result (default ordering):\n{formatted_result}")

            # Test with degree ordering
            result_ordered = await tool.execute(entity_id='Q42', order_by_degree=True)
            formatted_result_ordered = tool.format_result(result_ordered)
            print(f"   - Formatted result (ordered by degree):\n{formatted_result_ordered}")

            # Test increasing limit
            print("   - Testing increase_limit...")
            result_increased = await tool.execute(entity_id='Q42', increase_limit=True)
            formatted_result_increased = tool.format_result(result_increased)
            print(f"   - Formatted result (increased limit):\n{formatted_result_increased}")

        except Exception as e:
            print(f"   ✗ Error: {e}")
            traceback.print_exc()


        print("\n=== All tests completed ===")
    
    # Run the main function
    asyncio.run(main())
