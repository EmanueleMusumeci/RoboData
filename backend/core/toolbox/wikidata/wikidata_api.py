from wikidata.client import Client
from typing import Dict, Any, List, Optional
import asyncio
import aiohttp
import pprint

#TODO: Improve modeling of entity and property data structures, for now they're just dicts (with filtered values)

class WikidataRestAPI:
    """Wrapper for Wikidata API using wikidata library."""
    
    def __init__(self):
        self.client = Client()
    
    def _process_claims(self, raw_claims: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Process raw claims into a structured format with datavalue ids as keys."""
        processed_claims = {}
        
        for prop, claims in raw_claims.items():
            processed_claims[prop] = {}
            for claim in claims:
                datavalue = claim["mainsnak"].get("datavalue")
                
                # Determine the key for this claim
                if datavalue:
                    if datavalue.get("type") == "wikibase-entityid":
                        # Entity reference - use entity ID as key
                        key = datavalue["value"]["id"]
                    else:
                        # Native type - use the value itself as key
                        value = datavalue["value"]
                        key = str(value) if not isinstance(value, str) else value
                else:
                    # No datavalue - use a placeholder
                    key = "no_value"
                
                # Store the claim data with the key
                processed_claims[prop][key] = {
                    "property_id": claim["mainsnak"]["property"],
                    "datavalue": datavalue,
                    "datatype": claim["mainsnak"]["datatype"],
                    "qualifiers": {
                        qualifier_id: qualifier_data
                        for qualifier_id, qualifier_data in claim.get("qualifiers", {}).items()
                    }
                }
        
        return processed_claims
        
    async def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """Get entity data."""
        try:
            loop = asyncio.get_event_loop()
            entity = await loop.run_in_executor(None, self.client.get, entity_id, True)
            
            labels = {lang: label["value"] for lang, label in entity.data["labels"].items()}
            descriptions = {lang: desc["value"] for lang, desc in entity.data["descriptions"].items()}
            aliases = {lang: [alias["value"] for alias in aliases] for lang, aliases in entity.data["aliases"].items()}
            
            statements = self._process_claims(entity.data.get('claims', {}))
            
            return {
                'id': entity.id,
                'labels': labels,
                'descriptions': descriptions,
                'aliases': aliases,
                'statements': statements,
                'pageid': entity.data["pageid"],
                'sitelinks': entity.data["sitelinks"],
                'link': f"https://www.wikidata.org/wiki/{entity.id}"
            }
        except Exception as e:
            raise ValueError(f"Failed to get entity {entity_id}: {str(e)}")
    
    async def get_entity_statements(self, entity_id: str, property_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statements for an entity, optionally filtered by property."""
        loop = asyncio.get_event_loop()
        entity = await loop.run_in_executor(None, self.client.get, entity_id, True)
        
        raw_claims = entity.data.get('claims', {})
        if property_id and property_id in raw_claims:
            raw_claims = {property_id: raw_claims[property_id]}
        
        return self._process_claims(raw_claims)
    
    async def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities using the REST API."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://www.wikidata.org/w/api.php"
                params = {
                    'action': 'wbsearchentities',
                    'search': query,
                    'language': 'en',
                    'format': 'json',
                    'limit': limit
                }
                
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return data.get('search', [])
        except Exception as e:
            raise ValueError(f"Failed to search entities: {str(e)}")
    
    async def get_property(self, property_id: str) -> Dict[str, Any]:
        """Get property information."""
        try:
            loop = asyncio.get_event_loop()
            prop = await loop.run_in_executor(None, self.client.get, property_id, True)
            
            labels = {lang: label["value"] for lang, label in prop.data.get("labels", {}).items()}
            descriptions = {lang: desc["value"] for lang, desc in prop.data.get("descriptions", {}).items()}
            aliases = {lang: [alias["value"] for alias in aliases] for lang, aliases in prop.data.get("aliases", {}).items()}
            
            return {
            'id': prop.id,
            'labels': labels,
            'descriptions': descriptions,
            'aliases': aliases,
            'datatype': prop.data.get('datatype', 'unknown'),
            'pageid': prop.data.get("pageid"),
            'sitelinks': prop.data.get("sitelinks", {}),
            'link': f"https://www.wikidata.org/wiki/{prop.id}"
            }
        except Exception as e:
            raise ValueError(f"Failed to get property {property_id}: {str(e)}")

# Global instance
wikidata_api = WikidataRestAPI()

if __name__ == "__main__":
    async def main():
        """Run test cases without testing infrastructure."""
        api = WikidataRestAPI()
        
        print("=== Testing WikidataRestAPI ===\n")
        
        # Test 1: Get entity Douglas Adams (Q42)
        print("1. Getting entity data for Douglas Adams (Q42)...")
        try:
            result = await api.get_entity('Q42')
            print(f"   ID: {result['id']}")
            print(f"   English label: {result['labels'].get('en', 'N/A')}")
            print(f"   Number of statements: {len(result['statements'])}")
            pprint.pp(result["labels"])
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 2: Get entity statements for instance of (P31)
        print("2. Getting statements for Douglas Adams P31 (instance of)...")
        try:
            result = await api.get_entity_statements('Q42', 'P31')
            print(f"   Found P31 statements: {'P31' in result}")
            if 'P31' in result:
                print(f"   Number of P31 values: {len(result['P31'])}")
                pprint.pp(result['P31'])
                # Check if human (Q5) is in the values
                if 'Q5' in result['P31']:
                    print("   ✓ Douglas Adams is confirmed as human (Q5)")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 3: Get all statements
        print("3. Getting all statements for Douglas Adams...")
        try:
            result = await api.get_entity_statements('Q42')
            print(f"   Total properties: {len(result)}")
            common_props = ['P31', 'P106', 'P569']
            for prop in common_props:
                if prop in result:
                    print(f"   ✓ Has {prop}: {len(result[prop])} values")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 4: Search entities
        print("4. Searching for 'Douglas Adams'...")
        try:
            result = await api.search_entities('Douglas Adams', limit=5)
            print(f"   Found {len(result)} results")
            if result:
                first = result[0]
                print(f"   First result: {first['id']} - {first['label']}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 5: Get property info
        print("5. Getting property info for P31 (instance of)...")
        try:
            result = await api.get_property('P31')
            print(f"   Property ID: {result['id']}")
            print(f"   English label: {result['labels'].get('en', "N/A")}")
            print(f"   Data type: {result['datatype']}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 6: Test empty search
        print("6. Testing with rare query...")
        try:
            result = await api.search_entities('xyzabcnonexistentquery123', limit=1)
            print(f"   Results for rare query: {len(result)}")
        except Exception as e:
            print(f"   Error: {e}")
        

        # Test 7: Test getting non existing entity
        print("7. Testing with non existing entity...")
        try:
            result = await api.get_entity('Q99999999')
            print(f"   Results for non existing entity: {len(result)}")
            pprint.pp(result)
        except Exception as e:
            #Print type of exception
            print(f"   Error: {e.__class__.__name__} - {str(e)}")
            #Print the error message
            print(f"   Error: {e}")


        print("8. Getting property info for non-existing property P9999999...")
        try:
            result = await api.get_property('P9999999')
            print(f"   Property ID: {result['id']}")
            print(f"   English label: {result['labels'].get('en', 'N/A')}")
            print(f"   Data type: {result['datatype']}")
        except Exception as e:
            print(f"   Error: {e}")

        print("\n=== All tests completed ===")
    
    # Run the main function
    asyncio.run(main())
