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
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute a SPARQL query."""
        query = kwargs.get("query")
        timeout = kwargs.get("timeout", 30)

        if not query:
            raise ValueError("SPARQL query string is required")

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

    def format_result(self, result: Optional[Dict[str, Any]], query: Optional[str] = None) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "SPARQL query returned no result."
        
        summary = ""
        if query:
            summary += f"Query:\n```sparql\n{query}\n```\n"

        bindings = result.get('results', {}).get('bindings', [])
        count = len(bindings)
        
        if count == 0:
            summary += "SPARQL query returned 0 results."
            return summary
            
        # Extract headers from the first result if available
        headers = list(bindings[0].keys()) if bindings else []
        
        # Format a few results for preview
        preview_items = []
        for binding in bindings[:5]:  # Preview first 5 results
            item_parts = []
            for header in headers:
                value_data = binding.get(header, {})
                value = value_data.get('value', 'N/A')
                value_label = value_data.get('label', None) # Check for a label

                # Shorten long URLs and prefer labels
                if value.startswith("http://www.wikidata.org/entity/"):
                    entity_id = value.split('/')[-1]
                    display_value = f"'{value_label}' ({entity_id})" if value_label else entity_id
                else:
                    display_value = value

                item_parts.append(f"{header}: {display_value}")
            preview_items.append(f"{{{', '.join(item_parts)}}}")
            
        preview_str = "; ".join(preview_items)
        
        summary += f"SPARQL query returned {count} results. Headers: {', '.join(headers)}. Preview: {preview_str}"
        if count > 5:
            summary += "..."
            
        return summary

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
    
    async def execute(self, **kwargs) -> List[Dict[str, str]]:
        """Get subclasses of an entity."""
        entity_id = kwargs.get("entity_id")
        limit = kwargs.get("limit", 100)

        if not entity_id:
            raise ValueError("entity_id is required")

        query = f"""
        SELECT DISTINCT ?subclass ?subclassLabel WHERE {{
            ?subclass wdt:P279 wd:{entity_id} .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT {limit}
        """
        
        result = await self.sparql_tool.execute(query=query)
        
        # Pass the query to the formatter
        result['query'] = query

        subclasses = []
        
        bindings = result.get('results', {}).get('bindings', [])
        for binding in bindings:
            subclass_id = binding.get('subclass', {}).get('value', '').split('/')[-1]
            subclass_label = binding.get('subclassLabel', {}).get('value', subclass_id)
            subclasses.append({'id': subclass_id, 'label': subclass_label})
            
        return subclasses
    
    def format_result(self, result: List[Dict[str, str]]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No subclasses found."
            
        count = len(result)
        
        # Format a few results for preview
        preview_items = [f"'{item['label']}' ({item['id']})" for item in result[:10]]
        preview_str = "; ".join(preview_items)
        
        summary = f"Found {count} subclasses. Preview: {preview_str}"
        if count > 10:
            summary += "..."
            
        return summary

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
                    description="Wikidata entity ID (e.g., Q146)"
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
    
    async def execute(self, **kwargs) -> List[Dict[str, str]]:
        """Get superclasses of an entity."""
        entity_id = kwargs.get("entity_id")
        limit = kwargs.get("limit", 100)

        if not entity_id:
            raise ValueError("entity_id is required")

        query = f"""
        SELECT DISTINCT ?superclass ?superclassLabel WHERE {{
            wd:{entity_id} wdt:P279 ?superclass .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT {limit}
        """
        
        result = await self.sparql_tool.execute(query=query)
        
        # Pass the query to the formatter
        result['query'] = query

        superclasses = []
        
        bindings = result.get('results', {}).get('bindings', [])
        for binding in bindings:
            superclass_id = binding.get('superclass', {}).get('value', '').split('/')[-1]
            superclass_label = binding.get('superclassLabel', {}).get('value', superclass_id)
            superclasses.append({'id': superclass_id, 'label': superclass_label})
            
        return superclasses
    
    def format_result(self, result: List[Dict[str, str]]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No superclasses found."
            
        count = len(result)
        
        # Format a few results for preview
        preview_items = [f"'{item['label']}' ({item['id']})" for item in result[:10]]
        preview_str = "; ".join(preview_items)
        
        summary = f"Found {count} superclasses. Preview: {preview_str}"
        if count > 10:
            summary += "..."
            
        return summary

class GetInstancesQueryTool(Tool):
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
    
    async def execute(self, **kwargs) -> List[Dict[str, str]]:
        """Get instances of a class."""
        class_id = kwargs.get("class_id")
        limit = kwargs.get("limit", 100)

        if not class_id:
            raise ValueError("class_id is required")

        query = f"""
        SELECT DISTINCT ?instance ?instanceLabel WHERE {{
            ?instance wdt:P31 wd:{class_id} .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT {limit}
        """
        
        result = await self.sparql_tool.execute(query=query)
        
        # Pass the query to the formatter
        result['query'] = query
        
        instances = []
        
        bindings = result.get('results', {}).get('bindings', [])
        for binding in bindings:
            instance_id = binding.get('instance', {}).get('value', '').split('/')[-1]
            instance_label = binding.get('instanceLabel', {}).get('value', instance_id)
            instances.append({'id': instance_id, 'label': instance_label})
            
        return instances

    def format_result(self, result: List[Dict[str, str]]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No instances found."
            
        count = len(result)
        
        # Format a few results for preview
        preview_items = [f"'{item['label']}' ({item['id']})" for item in result[:10]]
        preview_str = "; ".join(preview_items)
        
        summary = f"Found {count} instances. Preview: {preview_str}"
        if count > 10:
            summary += "..."
            
        return summary

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
            return_description="List of classes the entity is an instance of, with IDs and labels"
        )
    
    async def execute(self, **kwargs) -> List[Dict[str, str]]:
        """Get classes an entity is an instance of."""
        entity_id = kwargs.get("entity_id")
        limit = kwargs.get("limit", 100)

        if not entity_id:
            raise ValueError("entity_id is required")

        query = f"""
        SELECT DISTINCT ?class ?classLabel WHERE {{
            wd:{entity_id} wdt:P31 ?class .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT {limit}
        """
        
        result = await self.sparql_tool.execute(query=query)
        
        # Pass the query to the formatter
        result['query'] = query
        
        classes = []
        
        bindings = result.get('results', {}).get('bindings', [])
        for binding in bindings:
            class_id = binding.get('class', {}).get('value', '').split('/')[-1]
            class_label = binding.get('classLabel', {}).get('value', class_id)
            classes.append({'id': class_id, 'label': class_label})
            
        return classes

    def format_result(self, result: List[Dict[str, str]]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No 'instance of' relationships found."
            
        count = len(result)
        
        # Format a few results for preview
        preview_items = [f"'{item['label']}' ({item['id']})" for item in result[:10]]
        preview_str = "; ".join(preview_items)
        
        summary = f"Found {count} 'instance of' relationships. Preview: {preview_str}"
        if count > 10:
            summary += "..."
            
        return summary

class PropertyQueryTool(Tool):
    """Tool for querying entities with a specific property value."""
    
    def __init__(self):
        super().__init__(
            name="query_property_values",
            description="Query entities that have a specific property value"
        )
        self.sparql_tool = SPARQLQueryTool()
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="property_id",
                    type="string",
                    description="Wikidata property ID (e.g., P31)"
                ),
                ToolParameter(
                    name="value_id",
                    type="string",
                    description="Wikidata value ID to filter by (e.g., Q146)"
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
            return_description="List of entity IDs and labels that have the specified property value"
        )
    
    async def execute(self, **kwargs) -> List[Dict[str, str]]:
        """Get entities with a specific property value."""
        property_id = kwargs.get("property_id")
        value_id = kwargs.get("value_id")
        limit = kwargs.get("limit", 100)

        if not property_id or not value_id:
            raise ValueError("property_id and value_id are required")

        query = f"""
        SELECT DISTINCT ?entity ?entityLabel WHERE {{
            ?entity wdt:{property_id} wd:{value_id} .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT {limit}
        """
        
        result = await self.sparql_tool.execute(query=query)
        
        # Pass the query to the formatter
        result['query'] = query
        
        entities = []
        
        bindings = result.get('results', {}).get('bindings', [])
        for binding in bindings:
            entity_id = binding.get('entity', {}).get('value', '').split('/')[-1]
            entity_label = binding.get('entityLabel', {}).get('value', entity_id)
            entities.append({'id': entity_id, 'label': entity_label})
            
        return entities

    def format_result(self, result: List[Dict[str, str]]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No entities found with this property."
            
        count = len(result)
        
        # Format a few results for preview
        preview_items = [f"'{item['label']}' ({item['id']})" for item in result[:10]]
        preview_str = "; ".join(preview_items)
        
        summary = f"Found {count} entities. Preview: {preview_str}"
        if count > 10:
            summary += "..."
            
        return summary

if __name__ == "__main__":
    import sys
    import os
    import asyncio
    import traceback

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
        """Fast test of SPARQL query tools with result formatting."""
        print("Testing Wikidata SPARQL Query Tools")
        print("=" * 40)
        
        sparql_tool = SPARQLQueryTool()
        subclass_tool = SubclassQueryTool()
        instance_tool = GetInstancesQueryTool()
        property_tool = PropertyQueryTool()
        superclass_tool = SuperclassQueryTool()
        instance_of_tool = InstanceOfQueryTool()

        # Test 1: Basic SPARQL Query
        print("\n1. Testing SPARQLQueryTool")
        try:
            # Test with a valid query
            query = '''
            SELECT ?item ?itemLabel WHERE {
                ?item wdt:P31 wd:Q146.
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            } LIMIT 10
            '''
            result = await sparql_tool.execute(query=query)
            print(f"  ✓ SPARQL query successful.")
            
            # Test result formatting
            formatted_result = sparql_tool.format_result(result, query=query)
            print(f"  - Formatted result (Success):\n{formatted_result}")

            # Test formatting for an empty/failed result
            formatted_empty_result = sparql_tool.format_result(None)
            print(f"  - Formatted result (Empty): {formatted_empty_result}")

        except Exception as e:
            print(f"  ✗ SPARQL query failed: {e}")
            traceback.print_exc()
        
        # Test 2: Subclass Query
        print("\n2. Testing SubclassQueryTool")
        try:
            # Test with a valid entity
            result = await subclass_tool.execute(entity_id="Q35120", limit=5)
            print(f"  ✓ Subclass query successful for 'entity'.")
            
            # Test result formatting
            formatted_result = subclass_tool.format_result(result)
            print(f"  - Formatted result (Success): {formatted_result}")

            # Test formatting for an empty/failed result
            formatted_empty_result = subclass_tool.format_result([])
            print(f"  - Formatted result (Empty): {formatted_empty_result}")

        except Exception as e:
            print(f"  ✗ Subclass query failed: {e}")
            traceback.print_exc()
        
        # Test 3: Superclass Query
        print("\n3. Testing SuperclassQueryTool")
        try:
            # Test with a valid entity
            result = await superclass_tool.execute(entity_id="Q146", limit=5) # house cat
            print(f"  ✓ Superclass query successful for 'house cat' (Q146).")
            
            # Test result formatting
            formatted_result = superclass_tool.format_result(result)
            print(f"  - Formatted result (Success): {formatted_result}")

            # Test formatting for an empty/failed result
            formatted_empty_result = superclass_tool.format_result([])
            print(f"  - Formatted result (Empty): {formatted_empty_result}")

        except Exception as e:
            print(f"  ✗ Superclass query failed: {e}")
            traceback.print_exc()

        # Test 4: Instance Query
        print("\n4. Testing GetInstancesQueryTool")
        try:
            # Test with a valid class
            result = await instance_tool.execute(class_id="Q515", limit=5)
            print(f"  ✓ Instance query successful for 'city'.")
            
            # Test result formatting
            formatted_result = instance_tool.format_result(result)
            print(f"  - Formatted result (Success): {formatted_result}")

            # Test formatting for an empty/failed result
            formatted_empty_result = instance_tool.format_result([])
            print(f"  - Formatted result (Empty): {formatted_empty_result}")

        except Exception as e:
            print(f"  ✗ Instance query failed: {e}")
            traceback.print_exc()
        
        # Test 5: Instance Of Query
        print("\n5. Testing InstanceOfQueryTool")
        try:
            # Test with a valid entity
            result = await instance_of_tool.execute(entity_id="Q42", limit=5) # Douglas Adams
            print(f"  ✓ InstanceOf query successful for 'Douglas Adams' (Q42).")
            
            # Test result formatting
            formatted_result = instance_of_tool.format_result(result)
            print(f"  - Formatted result (Success): {formatted_result}")

            # Test formatting for an empty/failed result
            formatted_empty_result = instance_of_tool.format_result([])
            print(f"  - Formatted result (Empty): {formatted_empty_result}")

        except Exception as e:
            print(f"  ✗ InstanceOf query failed: {e}")
            traceback.print_exc()

        # Test 6: Property Query
        print("\n6. Testing PropertyQueryTool")
        try:
            # Test with a valid property
            result = await property_tool.execute(property_id="P31", value_id="Q146", limit=5)
            print(f"  ✓ Property query successful for 'instance of cat'.")
            
            # Test result formatting
            formatted_result = property_tool.format_result(result)
            print(f"  - Formatted result (Success): {formatted_result}")

            # Test formatting for an empty/failed result
            formatted_empty_result = property_tool.format_result([])
            print(f"  - Formatted result (Empty): {formatted_empty_result}")

        except Exception as e:
            print(f"  ✗ Property query failed: {e}")
            traceback.print_exc()
        
        print("\n" + "=" * 40)
        print("All tests completed.")

    
    # Run the async tests
    asyncio.run(test_sparql_tools())