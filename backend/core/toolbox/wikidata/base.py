from typing import Dict, Any, List, Optional
import requests
import asyncio
from .datamodel import WikidataEntity, WikidataProperty, SearchResult, WikidataStatement, convert_api_entity_to_model, convert_api_property_to_model, convert_api_search_to_model
from .wikidata_api import wikidata_api

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
        
        print("\n=== All tests completed ===")
    
    # Run the main function
    asyncio.run(main())
