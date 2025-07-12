from typing import Dict, Any, List, Optional
import aiohttp
import asyncio
from ..toolbox import Tool, ToolDefinition, ToolParameter
import requests

class SPARQLQueryTool(Tool):
    """Tool for executing SPARQL queries on Wikidata."""
    
    def __init__(self):
        super().__init__(
            name="sparql_query",
            description="Execute SPARQL queries on Wikidata"
        )
        self.endpoint = "https://query.wikidata.org/sparql"
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="SPARQL query string"
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Query timeout in seconds",
                    required=False,
                    default=30
                )
            ],
            return_type="dict",
            return_description="Query results in JSON format"
        )
    
    async def execute(self, query: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute a SPARQL query."""
        headers = {
            'Accept': 'application/sparql-results+json',
            'User-Agent': 'RoboData/1.0 (Python)'
        }
        
        try:
            response = requests.get(
            self.endpoint,
            params={'query': query},
            headers=headers,
            timeout=timeout
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise ValueError(f"SPARQL query failed: {response.status_code}")
        except requests.RequestException as e:
            raise ValueError(f"SPARQL query error: {e}")

class SubclassQueryTool(Tool):
    """Tool for querying subclasses of an entity."""
    
    def __init__(self):
        super().__init__(
            name="query_subclasses",
            description="Query all subclasses of a given Wikidata entity"
        )
        self.sparql_tool = SPARQLQueryTool()
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_id",
                    type="string",
                    description="Wikidata entity ID (e.g., Q35120)"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=100
                )
            ],
            return_type="list",
            return_description="List of subclass entities with IDs and labels"
        )
    
    async def execute(self, entity_id: str, limit: int = 100) -> List[Dict[str, str]]:
        """Get subclasses of an entity."""
        query = f"""
        SELECT DISTINCT ?subclass ?subclassLabel WHERE {{
            ?subclass wdt:P279 wd:{entity_id} .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT {limit}
        """
        
        result = await self.sparql_tool.execute(query)
        subclasses = []
        
        for binding in result.get('results', {}).get('bindings', []):
            subclass_id = binding['subclass']['value'].split('/')[-1]
            subclass_label = binding.get('subclassLabel', {}).get('value', subclass_id)
            subclasses.append({
                'id': subclass_id,
                'label': subclass_label
            })
        
        return subclasses

class SuperclassQueryTool(Tool):
    """Tool for querying superclasses of an entity."""
    
    def __init__(self):
        super().__init__(
            name="query_superclasses",
            description="Query all superclasses of a given Wikidata entity"
        )
        self.sparql_tool = SPARQLQueryTool()
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_id",
                    type="string",
                    description="Wikidata entity ID (e.g., Q35120)"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=100
                )
            ],
            return_type="list",
            return_description="List of superclass entities with IDs and labels"
        )
    
    async def execute(self, entity_id: str, limit: int = 100) -> List[Dict[str, str]]:
        """Get superclasses of an entity."""
        query = f"""
        SELECT DISTINCT ?superclass ?superclassLabel WHERE {{
            wd:{entity_id} wdt:P279 ?superclass .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT {limit}
        """
        
        result = await self.sparql_tool.execute(query)
        superclasses = []
        
        for binding in result.get('results', {}).get('bindings', []):
            superclass_id = binding['superclass']['value'].split('/')[-1]
            superclass_label = binding.get('superclassLabel', {}).get('value', superclass_id)
            superclasses.append({
                'id': superclass_id,
                'label': superclass_label
            })
        
        return superclasses

class InstanceQueryTool(Tool):
    """Tool for querying instances of a class."""
    
    def __init__(self):
        super().__init__(
            name="query_instances",
            description="Query instances of a given Wikidata class"
        )
        self.sparql_tool = SPARQLQueryTool()
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="class_id",
                    type="string",
                    description="Wikidata class ID (e.g., Q35120)"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=100
                )
            ],
            return_type="list",
            return_description="List of instance entities with IDs and labels"
        )
    
    async def execute(self, class_id: str, limit: int = 100) -> List[Dict[str, str]]:
        """Get instances of a class."""
        query = f"""
        SELECT DISTINCT ?instance ?instanceLabel WHERE {{
            ?instance wdt:P31 wd:{class_id} .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT {limit}
        """
        
        result = await self.sparql_tool.execute(query)
        instances = []
        
        for binding in result.get('results', {}).get('bindings', []):
            instance_id = binding['instance']['value'].split('/')[-1]
            instance_label = binding.get('instanceLabel', {}).get('value', instance_id)
            instances.append({
                'id': instance_id,
                'label': instance_label
            })
        
        return instances

class InstanceOfQueryTool(Tool):
    """Tool for querying what classes an entity is an instance of."""
    
    def __init__(self):
        super().__init__(
            name="query_instance_of",
            description="Query what classes a given Wikidata entity is an instance of"
        )
        self.sparql_tool = SPARQLQueryTool()
    
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
                    name="limit",
                    type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=100
                )
            ],
            return_type="list",
            return_description="List of class entities that this entity is an instance of"
        )
    
    async def execute(self, entity_id: str, limit: int = 100) -> List[Dict[str, str]]:
        """Get what classes an entity is an instance of."""
        query = f"""
        SELECT DISTINCT ?class ?classLabel WHERE {{
            wd:{entity_id} wdt:P31 ?class .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT {limit}
        """
        
        result = await self.sparql_tool.execute(query)
        classes = []
        
        for binding in result.get('results', {}).get('bindings', []):
            class_id = binding['class']['value'].split('/')[-1]
            class_label = binding.get('classLabel', {}).get('value', class_id)
            classes.append({
                'id': class_id,
                'label': class_label
            })
        
        return classes


if __name__ == "__main__":
    import asyncio
    import sys
    import os
    
    # Add the parent directories to the path for absolute imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    # Import with absolute paths when running as main
    try:
        from backend.core.toolbox.toolbox import Tool, ToolDefinition, ToolParameter
    except ImportError:
        print("Error: Could not import required modules. Make sure you're running from the correct directory.")
        print("Try running: python -m backend.core.toolbox.wikidata.queries")
        sys.exit(1)
    
    async def test_sparql_tools():
        """Fast test of SPARQL query tools without testing framework."""
        print("Testing Wikidata SPARQL Query Tools")
        print("=" * 40)
        
        # Test 1: Basic SPARQL Query
        print("\n1. Testing SPARQLQueryTool")
        try:
            sparql_tool = SPARQLQueryTool()
            print(f"Tool definition: {sparql_tool.get_definition().name}")
            
            # Simple query for Douglas Adams
            query = """
            SELECT ?item ?itemLabel WHERE {
                ?item wdt:P31 wd:Q5 .
                ?item rdfs:label "Douglas Adams"@en .
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
            }
            """
            
            print(f"Query:\n{query}")
            result = await sparql_tool.execute(query)
            print(f"✓ SPARQL query successful")
            
            bindings = result.get('results', {}).get('bindings', [])
            print(f"  - Found {len(bindings)} total results, showing first 15:")
            
            for i, binding in enumerate(bindings[:15]):
                item_id = binding.get('item', {}).get('value', '').split('/')[-1]
                item_label = binding.get('itemLabel', {}).get('value', 'No label')
                print(f"  - {i+1}. {item_label} ({item_id})")
        
        except Exception as e:
            print(f"✗ SPARQL query failed: {e}")
        
        # Test 2: Subclass Query
        print("\n2. Testing SubclassQueryTool")
        try:
            subclass_tool = SubclassQueryTool()
            print(f"Tool definition: {subclass_tool.get_definition().name}")
            
            # Get subclasses of "human" (Q5) - fetch all, show 15
            result = await subclass_tool.execute("Q5", limit=1000)
            print(f"✓ Subclass query successful")
            print(f"  - Found {len(result)} total subclasses of 'human', showing first 15:")
            
            for i, subclass in enumerate(result[:15]):
                print(f"  - {i+1}. {subclass['label']} ({subclass['id']})")
        
        except Exception as e:
            print(f"✗ Subclass query failed: {e}")
        
        # Test 3: Superclass Query
        print("\n3. Testing SuperclassQueryTool")
        try:
            superclass_tool = SuperclassQueryTool()
            print(f"Tool definition: {superclass_tool.get_definition().name}")
            
            # Get superclasses of "writer" (Q36180) - fetch all, show 15
            result = await superclass_tool.execute("Q36180", limit=1000)
            print(f"✓ Superclass query successful")
            print(f"  - Found {len(result)} total superclasses of 'writer', showing first 15:")
            
            for i, superclass in enumerate(result[:15]):
                print(f"  - {i+1}. {superclass['label']} ({superclass['id']})")
        
        except Exception as e:
            print(f"✗ Superclass query failed: {e}")
        
        # Test 4: Instance Query
        print("\n4. Testing InstanceQueryTool")
        try:
            instance_tool = InstanceQueryTool()
            print(f"Tool definition: {instance_tool.get_definition().name}")
            
            # Get instances of "programming language" (Q9143) - fetch all, show 15
            result = await instance_tool.execute("Q9143", limit=1000)
            print(f"✓ Instance query successful")
            print(f"  - Found {len(result)} total instances of 'programming language', showing first 15:")
            
            for i, instance in enumerate(result[:15]):
                print(f"  - {i+1}. {instance['label']} ({instance['id']})")
        
        except Exception as e:
            print(f"✗ Instance query failed: {e}")
        
        # Test 5: Instance Of Query
        print("\n5. Testing InstanceOfQueryTool")
        try:
            instance_of_tool = InstanceOfQueryTool()
            print(f"Tool definition: {instance_of_tool.get_definition().name}")
            
            # Get what Douglas Adams (Q42) is an instance of - fetch all, show 15
            result = await instance_of_tool.execute("Q42", limit=1000)
            print(f"✓ Instance of query successful")
            print(f"  - Found {len(result)} total classes that Douglas Adams is an instance of, showing first 15:")
            
            for i, class_item in enumerate(result[:15]):
                print(f"  - {i+1}. {class_item['label']} ({class_item['id']})")
        
        except Exception as e:
            print(f"✗ Instance of query failed: {e}")
        
        # Test 6: Simple SPARQL Query - Countries
        print("\n6. Testing Simple SPARQL Query - Countries")
        try:
            sparql_tool = SPARQLQueryTool()
            
            # Simple query for countries - no LIMIT to get all
            simple_query = """
            SELECT ?country ?countryLabel WHERE {
                ?country wdt:P31 wd:Q6256 .
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
            }
            """
            
            print(f"Query:\n{simple_query}")
            result = await sparql_tool.execute(simple_query)
            print(f"✓ Simple SPARQL query successful")
            
            bindings = result.get('results', {}).get('bindings', [])
            print(f"  - Found {len(bindings)} total countries, showing first 15:")
            
            for i, binding in enumerate(bindings[:15]):
                name = binding.get('countryLabel', {}).get('value', 'Unknown')
                country_id = binding.get('country', {}).get('value', '').split('/')[-1]
                print(f"  - {i+1}. {name} ({country_id})")
        
        except Exception as e:
            print(f"✗ Simple SPARQL query failed: {e}")
        
        print("\n" + "=" * 40)
        print("SPARQL Tools testing completed!")
        print("=" * 40)
    
    # Run the async tests
    asyncio.run(test_sparql_tools())