from typing import Dict, Any, List, Optional
import asyncio
import aiohttp
import urllib.parse
import time
from .rate_limiter import rate_limited_request

class DBpediaRestAPI:
    """Wrapper for DBpedia API using REST endpoints and SPARQL."""
    
    def __init__(self):
        self.base_url = "https://dbpedia.org"
        self.sparql_endpoint = "https://dbpedia.org/sparql"
        # Using the original/old lookup API (the new one is different)
        self.lookup_endpoint = "https://lookup.dbpedia.org/api/search"
        
    def _normalize_uri(self, uri: str) -> str:
        """Normalize a DBpedia URI."""
        if uri.startswith("http://dbpedia.org/resource/"):
            return uri
        elif uri.startswith("dbr:"):
            return uri.replace("dbr:", "http://dbpedia.org/resource/")
        else:
            # Assume it's a simple name and create the full URI
            return f"http://dbpedia.org/resource/{uri}"
    
    def _extract_name_from_uri(self, uri: str) -> str:
        """Extract the simple name from a DBpedia URI."""
        if uri.startswith("http://dbpedia.org/resource/"):
            return uri.replace("http://dbpedia.org/resource/", "")
        elif uri.startswith("http://dbpedia.org/property/"):
            return uri.replace("http://dbpedia.org/property/", "")
        elif uri.startswith("http://dbpedia.org/ontology/"):
            return uri.replace("http://dbpedia.org/ontology/", "")
        return uri
    
    async def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """Get entity data from DBpedia."""
        try:
            # Normalize the entity ID to a full URI
            entity_uri = self._normalize_uri(entity_id)
            
            # Query DBpedia for entity information
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX dbp: <http://dbpedia.org/property/>
            PREFIX foaf: <http://www.foaf.org.1/name>
            
            SELECT DISTINCT ?property ?value ?valueLabel WHERE {{
                <{entity_uri}> ?property ?value .
                OPTIONAL {{ ?value rdfs:label ?valueLabel . FILTER(LANG(?valueLabel) = "en") }}
            }}
            LIMIT 1000
            """
            
            results = await self._execute_sparql_query(query)
            
            # Process the results
            entity_data = {
                'id': entity_id,
                'uri': entity_uri,
                'labels': {},
                'descriptions': {},
                'aliases': {},
                'statements': {},
                'link': entity_uri
            }
            
            # Get basic info with a separate query
            basic_info_query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX dbo: <http://dbpedia.org/ontology/>
            
            SELECT ?label ?comment WHERE {{
                <{entity_uri}> rdfs:label ?label .
                OPTIONAL {{ <{entity_uri}> rdfs:comment ?comment . FILTER(LANG(?comment) = "en") }}
                FILTER(LANG(?label) = "en")
            }}
            LIMIT 1
            """
            
            basic_results = await self._execute_sparql_query(basic_info_query)
            
            if basic_results.get('results', {}).get('bindings'):
                binding = basic_results['results']['bindings'][0]
                if 'label' in binding:
                    entity_data['labels']['en'] = binding['label']['value']
                if 'comment' in binding:
                    entity_data['descriptions']['en'] = binding['comment']['value']
            
            # Process statements
            if results.get('results', {}).get('bindings'):
                for binding in results['results']['bindings']:
                    property_uri = binding['property']['value']
                    value = binding['value']['value']
                    
                    # Skip some meta properties for cleaner output
                    if property_uri in ['http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
                                       'http://www.w3.org/2000/01/rdf-schema#label',
                                       'http://www.w3.org/2000/01/rdf-schema#comment']:
                        continue
                    
                    prop_name = self._extract_name_from_uri(property_uri)
                    
                    if prop_name not in entity_data['statements']:
                        entity_data['statements'][prop_name] = {}
                    
                    # Use value label if available, otherwise use the value itself
                    display_value = binding.get('valueLabel', {}).get('value', value)
                    
                    entity_data['statements'][prop_name][value] = {
                        "property_id": prop_name,
                        "datavalue": {"value": value, "type": "string"},
                        "datatype": "string",
                        "display_value": display_value,
                        "qualifiers": {}
                    }
            
            return entity_data
            
        except Exception as e:
            raise ValueError(f"Failed to get entity {entity_id}: {str(e)}")
    
    async def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities using the DBpedia Lookup API with rate limiting."""
        async with rate_limited_request():
            try:
                async with aiohttp.ClientSession() as session:
                    search_url = "https://lookup.dbpedia.org/api/search/KeywordSearch"
                    params = {
                        'QueryString': query,
                        'MaxHits': limit,
                    }
                    headers = {
                        'Accept': 'application/json',
                        'User-Agent': 'RoboData/1.0 (Python)'
                    }
                    print(f"[DBpedia Lookup] Requesting: {search_url} params={params}")
                    async with session.get(search_url, params=params, headers=headers) as response:
                        print(f"[DBpedia Lookup] Response status: {response.status}")
                        print(f"[DBpedia Lookup] Content-Type: {response.headers.get('content-type', '')}")
                        text_data = await response.text()
                        print(f"[DBpedia Lookup] Response body (first 500 chars): {text_data[:500]}")
                        if response.status == 200:
                            content_type = response.headers.get('content-type', '')
                            if 'application/json' in content_type:
                                try:
                                    data = await response.json()
                                except Exception as e:
                                    print(f"[DBpedia Lookup] JSON decode error: {e}")
                                    data = None
                            else:
                                import xml.etree.ElementTree as ET
                                try:
                                    root = ET.fromstring(text_data)
                                    results = []
                                    # DBpedia Lookup API XML uses capital 'Result' for elements
                                    for result in root.findall('.//Result'):
                                        uri_elem = result.find('URI')
                                        label_elem = result.find('Label')
                                        description_elem = result.find('Description')
                                        categories_elem = result.find('Categories')
                                        classes_elem = result.find('Classes')
                                        uri_text = uri_elem.text if uri_elem is not None and uri_elem.text is not None else ''
                                        label_text = label_elem.text if label_elem is not None and label_elem.text is not None else ''
                                        desc_text = description_elem.text if description_elem is not None and description_elem.text is not None else ''
                                        categories = []
                                        if categories_elem is not None and categories_elem.text and categories_elem.text != 'None':
                                            categories = [cat.strip() for cat in categories_elem.text.split(',') if cat.strip()]
                                        classes = []
                                        if classes_elem is not None and classes_elem.text and classes_elem.text != 'None':
                                            classes = [cls.strip() for cls in classes_elem.text.split(',') if cls.strip()]
                                        results.append({
                                            'id': self._extract_name_from_uri(uri_text),
                                            'uri': uri_text,
                                            'label': label_text,
                                            'description': desc_text,
                                            'categories': categories,
                                            'classes': classes
                                        })
                                    return results
                                except Exception as e:
                                    print(f"[DBpedia Lookup] XML parse error: {e}")
                                    return []
                            if data and 'results' in data:
                                results = []
                                for result in data['results']:
                                    results.append({
                                        'id': self._extract_name_from_uri(result.get('uri', '')),
                                        'uri': result.get('uri', ''),
                                        'label': result.get('label', ''),
                                        'description': result.get('description', ''),
                                        'categories': result.get('categories', []),
                                        'classes': result.get('classes', [])
                                    })
                                return results
                            else:
                                return []
                        else:
                            print(f"[DBpedia Lookup] Non-200 response, trying SPARQL fallback...")
                            return await self._search_entities_sparql_fallback(query, limit)
            except Exception as e:
                print(f"[DBpedia Lookup] Exception: {e}, trying SPARQL fallback...")
                return await self._search_entities_sparql_fallback(query, limit)
    
    async def _search_entities_sparql_fallback(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback search using SPARQL when the Lookup API fails."""
        try:
            # Clean the query for SPARQL
            query_clean = query.replace('"', '').replace("'", "")
            
            sparql_query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX dbo: <http://dbpedia.org/ontology/>
            
            SELECT DISTINCT ?uri ?label ?comment WHERE {{
                ?uri rdfs:label ?label .
                OPTIONAL {{ ?uri rdfs:comment ?comment . FILTER(LANG(?comment) = "en") }}
                FILTER(LANG(?label) = "en" && (
                    CONTAINS(LCASE(?label), LCASE("{query_clean}")) ||
                    REGEX(?label, "^{query_clean}", "i")
                ))
            }}
            ORDER BY STRLEN(?label)
            LIMIT {limit}
            """
            
            results = await self._execute_sparql_query(sparql_query)
            entities = []
            
            for binding in results.get('results', {}).get('bindings', []):
                uri = binding.get('uri', {}).get('value', '')
                label = binding.get('label', {}).get('value', '')
                description = binding.get('comment', {}).get('value', '')
                
                entities.append({
                    'id': self._extract_name_from_uri(uri),
                    'uri': uri,
                    'label': label,
                    'description': description,
                    'categories': [],
                    'classes': []
                })
            
            return entities
        except Exception as e:
            print(f"SPARQL fallback search also failed: {e}")
            return []
    
    async def get_property(self, property_id: str) -> Dict[str, Any]:
        """Get property information from DBpedia."""
        try:
            # Properties in DBpedia can be in ontology or property namespace
            if not property_id.startswith("http://"):
                property_uri = f"http://dbpedia.org/ontology/{property_id}"
                alt_property_uri = f"http://dbpedia.org/property/{property_id}"
            else:
                property_uri = property_id
                alt_property_uri = None

            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?label ?comment ?type WHERE {{
                {{
                    <{property_uri}> rdfs:label ?label .
                    OPTIONAL {{ <{property_uri}> rdfs:comment ?comment . FILTER(LANG(?comment) = "en") }}
                    OPTIONAL {{ <{property_uri}> rdf:type ?type }}
                    FILTER(LANG(?label) = "en")
                }}
                {f'UNION {{ <{alt_property_uri}> rdfs:label ?label . OPTIONAL {{ <{alt_property_uri}> rdfs:comment ?comment . FILTER(LANG(?comment) = "en") }} OPTIONAL {{ <{alt_property_uri}> rdf:type ?type }} FILTER(LANG(?label) = "en") }}' if alt_property_uri else ''}
            }}
            LIMIT 1
            """

            # Do NOT use rate_limited_request here; only use it in top-level API calls
            results = await self._execute_sparql_query(query)

            prop_data = {
                'id': property_id,
                'labels': {},
                'descriptions': {},
                'aliases': {},
                'datatype': 'unknown',
                'link': property_uri
            }

            if results.get('results', {}).get('bindings'):
                binding = results['results']['bindings'][0]
                if 'label' in binding:
                    prop_data['labels']['en'] = binding['label']['value']
                if 'comment' in binding:
                    prop_data['descriptions']['en'] = binding['comment']['value']
                if 'type' in binding:
                    prop_data['datatype'] = self._extract_name_from_uri(binding['type']['value'])

            return prop_data

        except Exception as e:
            raise ValueError(f"Failed to get property {property_id}: {str(e)}")
    
    async def _execute_sparql_query(self, query: str, timeout: int = 60) -> Dict[str, Any]:
        """Execute a SPARQL query against DBpedia with rate limiting."""
        try:
            timeout_config = aiohttp.ClientTimeout(total=timeout)
            # Only rate-limit the actual HTTP request
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                headers = {
                    'Accept': 'application/sparql-results+json',
                    'User-Agent': 'RoboData/1.0 (Python)'
                }
                params = {
                    'query': query,
                    'format': 'json'
                }
                async with rate_limited_request():
                    async with session.get(self.sparql_endpoint, params=params, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result
                        else:
                            raise ValueError(f"SPARQL query failed with status {response.status}")
        except Exception as e:
            raise ValueError(f"SPARQL query error: {str(e)}")

# Global instance
dbpedia_api = DBpediaRestAPI()

if __name__ == "__main__":
    async def main():
        """Run test cases for DBpedia API."""
        api = DBpediaRestAPI()
        
        print("=== Testing DBpediaRestAPI ===\n")
        
        # Test 1: Get entity Douglas Adams
        print("1. Getting entity data for Douglas Adams...")
        try:
            result = await api.get_entity('Douglas_Adams')
            print(f"   ID: {result['id']}")
            print(f"   English label: {result['labels'].get('en', 'N/A')}")
            print(f"   Number of statements: {len(result['statements'])}")
            print(f"   Description: {result['descriptions'].get('en', 'N/A')[:100]}...")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 2: Search entities
        print("2. Searching for 'Einstein'...")
        try:
            results = await api.search_entities('Einstein', limit=5)
            print(f"   Found {len(results)} results:")
            for result in results[:3]:
                print(f"     - {result['label']} ({result['id']})")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 3: Get property information
        print("3. Getting property information for 'birthPlace'...")
        try:
            result = await api.get_property('birthPlace')
            print(f"   ID: {result['id']}")
            print(f"   English label: {result['labels'].get('en', 'N/A')}")
            print(f"   Description: {result['descriptions'].get('en', 'N/A')}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Test 4: Test SPARQL query
        print("4. Testing SPARQL query for writers...")
        try:
            query = """
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?writer ?name WHERE {
                ?writer a dbo:Writer ;
                        rdfs:label ?name .
                FILTER(LANG(?name) = "en")
            } LIMIT 5
            """
            result = await api._execute_sparql_query(query)
            bindings = result.get('results', {}).get('bindings', [])
            print(f"   Found {len(bindings)} writers:")
            for binding in bindings[:3]:
                name = binding.get('name', {}).get('value', 'N/A')
                print(f"     - {name}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Run the test
    asyncio.run(main())
