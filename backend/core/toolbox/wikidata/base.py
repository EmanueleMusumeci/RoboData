from typing import Dict, Any, List, Optional
import requests
import asyncio
import traceback
from ..toolbox import Tool, ToolDefinition, ToolParameter
from .datamodel import WikidataEntity, WikidataProperty, SearchResult, WikidataStatement, convert_api_entity_to_model, convert_api_property_to_model, convert_api_search_to_model
from .wikidata_api import wikidata_api

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

class SearchEntitiesTool(Tool):
    """Tool for searching Wikidata entities by text query."""
    
    def __init__(self):
        super().__init__(
            name="search_entities",
            description="Search for Wikidata entities by text query"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Text query to search for"
                )
            ],
            return_type="array",
            return_description="List of search results with IDs, labels, and descriptions"
        )
    
    async def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """Search for Wikidata entities by text query."""
        query = kwargs.get("query")
        limit = kwargs.get("limit", 10)

        # Force at least 5 results
        
        if not query:
            raise ValueError("query is required")
            
        api_data = await wikidata_api.search_entities(query, limit)
        print(api_data)
        results = convert_api_search_to_model(api_data)
        return [
            {
                "id": result.id,
                "label": result.label,
                "description": result.description,
                "url": result.url
            }
            for result in results
        ]

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
    """Test cases without testing infrastructure."""
    import asyncio
    
    async def main():
        """Run test cases for base functions."""
        print("=== Testing Wikidata Base Functions ===\n")
        
        # Test 1: Get entity Douglas Adams (Q42)
        print("1. Getting entity data for Douglas Adams (Q42)...")
        try:
            result = await get_entity_info_async('Q42')
            print(f"   ID: {result.id}")
            print(f"   Label: {result.label}")
            print(f"   Description: {result.description[:50]}..." if result.description else "   No description")
            print(f"   Aliases: {len(result.aliases)} found")
            print(f"   Statements: {len(result.statements)} properties")
            
            # Check entity references
            entity_refs = sum(1 for statements in result.statements.values() 
                            for stmt in statements if stmt.is_entity_ref)
            print(f"   Entity references: {entity_refs}")
        except Exception as e:
            print(f"   Error: {e}")
            traceback.print_exc()
        
        print()
        
        # Test 2: Get property info for P31 (instance of)
        print("2. Getting property info for P31 (instance of)...")
        try:
            result = await get_property_info_async('P31')
            print(f"   Property ID: {result.id}")
            print(f"   Label: {result.label}")
            print(f"   Datatype: {result.datatype}")
        except Exception as e:
            print(f"   Error: {e}")
            traceback.print_exc()
        
        print()
        
        # Test 3: Search entities
        print("3. Searching for 'Douglas Adams'...")
        try:
            result = await search_entities_async('Douglas Adams', limit=3)
            print(f"   Found {len(result)} results")
            if result:
                first = result[0]
                print(f"   First result: {first.id} - {first.label}")
        except Exception as e:
            print(f"   Error: {e}")
            traceback.print_exc()
        
        print()
        
        # Test 4: Test sync wrappers
        print("4. Testing sync wrapper functions...")
        try:
            entity = get_entity_info('Q42')
            print(f"   ✓ Sync entity: {entity.label}")
            
            prop = get_property_info('P31')
            print(f"   ✓ Sync property: {prop.label}")
            
            results = search_entities('Douglas Adams', limit=2)
            print(f"   ✓ Sync search: {len(results)} results")
        except Exception as e:
            print(f"   Error: {e}")
            traceback.print_exc()
        
        print()
        
        # Test 5: Test conversion consistency
        print("5. Testing conversion consistency...")
        try:
            from .wikidata_api import wikidata_api
            from .datamodel import convert_api_entity_to_model
            
            api_data = await wikidata_api.get_entity("Q42")
            converted_entity = convert_api_entity_to_model(api_data)
            base_entity = await get_entity_info_async("Q42")
            
            assert converted_entity.id == base_entity.id
            assert converted_entity.label == base_entity.label
            print("   ✓ Conversion consistency verified")
        except Exception as e:
            print(f"   Error: {e}")
            traceback.print_exc()
        
        print("\n=== All tests completed ===")
    
    # Run the main function
    asyncio.run(main())
