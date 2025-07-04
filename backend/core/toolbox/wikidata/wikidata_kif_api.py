from kif_lib import Store
from kif_lib.model import Item, Property, Statement, Text, IRI, Entity
from kif_lib.vocabulary import wd
import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class WikidataKIFAPI:
    """Wikidata API wrapper using IBM's KIF (Knowledge Integration Framework)."""
    
    def __init__(self):
        """Initialize the KIF store with Wikidata backend."""
        self.store = Store('wdqs')
    
    def _entity_to_dict(self, entity: Union[Item, Property]) -> Dict[str, Any]:
        """Convert KIF Entity to dictionary format."""
        result = {
            'id': str(entity).replace('wd:', ''),
            'iri': str(entity),
            'labels': {},
            'descriptions': {},
            'aliases': {},
            'link': f"https://www.wikidata.org/wiki/{str(entity).replace('wd:', '')}"
        }
        
        # Get labels, descriptions, and aliases
        try:
            # Query for labels
            for stmt in self.store.filter(subject=entity, property=wd.rdfs.label):
                if isinstance(stmt.object, Text):
                    lang = stmt.object.language or 'en'
                    result['labels'][lang] = str(stmt.object)
            
            # Query for descriptions  
            for stmt in self.store.filter(subject=entity, property=wd.schema.description):
                if isinstance(stmt.object, Text):
                    lang = stmt.object.language or 'en'
                    result['descriptions'][lang] = str(stmt.object)
                    
            # Query for aliases
            for stmt in self.store.filter(subject=entity, property=wd.skos.altLabel):
                if isinstance(stmt.object, Text):
                    lang = stmt.object.language or 'en'
                    if lang not in result['aliases']:
                        result['aliases'][lang] = []
                    result['aliases'][lang].append(str(stmt.object))
                    
        except Exception as e:
            logger.warning(f"Error fetching labels/descriptions for {entity}: {e}")
        
        return result
    
    def _statement_to_dict(self, statement: Statement) -> Dict[str, Any]:
        """Convert KIF Statement to dictionary format."""
        result = {
            'property_id': str(statement.property).replace('wd:', ''),
            'property_iri': str(statement.property),
            'subject_id': str(statement.subject).replace('wd:', ''),
            'datatype': None,
            'value': None,
            'value_type': type(statement.object).__name__,
            'qualifiers': []
        }
        
        # Handle different object types
        if isinstance(statement.object, Item):
            result['value'] = str(statement.object).replace('wd:', '')
            result['datatype'] = 'wikibase-item'
        elif isinstance(statement.object, Text):
            result['value'] = str(statement.object)
            result['datatype'] = 'string'
            if statement.object.language:
                result['language'] = statement.object.language
        elif isinstance(statement.object, IRI):
            result['value'] = str(statement.object)
            result['datatype'] = 'url'
        else:
            result['value'] = str(statement.object)
            result['datatype'] = 'literal'
        
        return result
    
    async def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """Get entity data using KIF."""
        try:
            loop = asyncio.get_event_loop()
            
            def _get_entity():
                # Create entity from ID
                if entity_id.startswith('P'):
                    entity = Property(f"wd:{entity_id}")
                else:
                    entity = Item(f"wd:{entity_id}")
                
                # Convert to dict
                result = self._entity_to_dict(entity)
                
                # Get statements
                statements = {}
                for stmt in self.store.filter(subject=entity):
                    prop_id = str(stmt.property).replace('wd:', '')
                    if prop_id not in statements:
                        statements[prop_id] = {}
                    
                    # Use object as key
                    if isinstance(stmt.object, Item):
                        key = str(stmt.object).replace('wd:', '')
                    else:
                        key = str(stmt.object)
                    
                    statements[prop_id][key] = self._statement_to_dict(stmt)
                
                result['statements'] = statements
                return result
            
            return await loop.run_in_executor(None, _get_entity)
            
        except Exception as e:
            raise ValueError(f"Failed to get entity {entity_id}: {str(e)}")
    
    async def get_entity_statements(self, entity_id: str, property_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statements for an entity, optionally filtered by property."""
        try:
            loop = asyncio.get_event_loop()
            
            def _get_statements():
                # Create entity from ID
                if entity_id.startswith('P'):
                    entity = Property(f"wd:{entity_id}")
                else:
                    entity = Item(f"wd:{entity_id}")
                
                statements = {}
                
                # Filter by property if specified
                if property_id:
                    prop = Property(f"wd:{property_id}")
                    for stmt in self.store.filter(subject=entity, property=prop):
                        if property_id not in statements:
                            statements[property_id] = {}
                        
                        # Use object as key
                        if isinstance(stmt.object, Item):
                            key = str(stmt.object).replace('wd:', '')
                        else:
                            key = str(stmt.object)
                        
                        statements[property_id][key] = self._statement_to_dict(stmt)
                else:
                    # Get all statements
                    for stmt in self.store.filter(subject=entity):
                        prop_id = str(stmt.property).replace('wd:', '')
                        if prop_id not in statements:
                            statements[prop_id] = {}
                        
                        if isinstance(stmt.object, Item):
                            key = str(stmt.object).replace('wd:', '')
                        else:
                            key = str(stmt.object)
                        
                        statements[prop_id][key] = self._statement_to_dict(stmt)
                
                return statements
            
            return await loop.run_in_executor(None, _get_statements)
            
        except Exception as e:
            raise ValueError(f"Failed to get statements for {entity_id}: {str(e)}")
    
    async def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities using Wikidata REST API."""
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
    
    async def get_property(self, property_id: str) -> Dict[str, Any]:
        """Get property information using KIF."""
        try:
            loop = asyncio.get_event_loop()
            
            def _get_property():
                prop = Property(f"wd:{property_id}")
                result = self._entity_to_dict(prop)
                
                # Get property-specific data
                try:
                    # Get datatype from property statements
                    for stmt in self.store.filter(subject=prop, property=wd.wikibase.propertyType):
                        if isinstance(stmt.object, Item):
                            result['datatype'] = str(stmt.object).replace('wd:', '')
                            break
                    else:
                        result['datatype'] = 'unknown'
                except Exception as e:
                    logger.warning(f"Could not determine datatype for {property_id}: {e}")
                    result['datatype'] = 'unknown'
                
                return result
            
            return await loop.run_in_executor(None, _get_property)
            
        except Exception as e:
            raise ValueError(f"Failed to get property {property_id}: {str(e)}")
    
    async def query_sparql(self, sparql_query: str) -> List[Dict[str, Any]]:
        """Execute SPARQL query using Wikidata endpoint."""
        async with aiohttp.ClientSession() as session:
            url = "https://query.wikidata.org/sparql"
            headers = {
                'Accept': 'application/sparql-results+json',
                'User-Agent': 'RoboData/1.0 (https://github.com/yourusername/robodata)'
            }
            
            async with session.get(url, params={'query': sparql_query}, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for binding in data.get('results', {}).get('bindings', []):
                        row = {}
                        for var, value in binding.items():
                            if value['type'] == 'uri' and 'wikidata.org' in value['value']:
                                # Extract entity ID from Wikidata URI
                                entity_id = value['value'].split('/')[-1]
                                row[var] = {
                                    'type': 'entity',
                                    'id': entity_id,
                                    'iri': value['value']
                                }
                            else:
                                row[var] = {
                                    'type': value['type'],
                                    'value': value['value']
                                }
                                if 'xml:lang' in value:
                                    row[var]['language'] = value['xml:lang']
                        results.append(row)
                    
                    return results
                else:
                    raise ValueError(f"SPARQL query failed with status {response.status}")

# Global instance
wikidata_kif_api = WikidataKIFAPI()

if __name__ == "__main__":
    async def main():
        """Test the KIF-based API."""
        api = WikidataKIFAPI()
        
        print("=== Testing WikidataKIFAPI ===\n")
        
        # Test 1: Get entity Douglas Adams (Q42)
        print("1. Getting entity data for Douglas Adams (Q42)...")
        try:
            result = await api.get_entity('Q42')
            print(f"   ID: {result['id']}")
            print(f"   English label: {result['labels'].get('en', 'N/A')}")
            print(f"   Number of statements: {len(result['statements'])}")
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
                if 'Q5' in result['P31']:
                    print("   ✓ Douglas Adams is confirmed as human (Q5)")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 3: Get property info
        print("3. Getting property info for P31 (instance of)...")
        try:
            result = await api.get_property('P31')
            print(f"   Property ID: {result['id']}")
            print(f"   English label: {result['labels'].get('en', 'N/A')}")
            print(f"   Data type: {result['datatype']}")
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
        
        # Test 5: SPARQL query
        print("5. Testing SPARQL query...")
        try:
            sparql = """
            SELECT ?item ?itemLabel WHERE {
                ?item wdt:P31 wd:Q5 .
                ?item rdfs:label "Douglas Adams"@en .
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
            }
            LIMIT 1
            """
            result = await api.query_sparql(sparql)
            print(f"   SPARQL results: {len(result)}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\n=== KIF API tests completed ===")
    
    # Run the main function
    asyncio.run(main())
