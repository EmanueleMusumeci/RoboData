from typing import Dict, Any, List, Optional
import aiohttp
import asyncio
import traceback
import sys
import os
from ..toolbox import Tool, ToolDefinition, ToolParameter
import requests

class SPARQLQueryTool(Tool):
    """Tool for executing SPARQL queries on Wikidata."""
    
    def __init__(self):
        super().__init__(
            name="sparql_query",
            description="Execute SPARQL queries on Wikidata, where SPARQL is a query language for databases, able to retrieve and manipulate data stored in Resource Description Framework (RDF) format." \
            "This tool should be preferred if we wish to retrieve sets of data with a common property. It is generally faster than the other tools, as it can retrieve multiple data in a single query." \
            "Be careful when using the LIMIT operator: the higher the number of results, the better for you." \
            "USEFUL FOR: retrieving structured data from Wikidata, such as entities, properties, and relationships." \
            "Retrieving data with complex relationships, such as subclasses, superclasses, instances, and properties of entities or even property chains." \
            "Retrievin data with specific conditions, such as entities or properties with a certain property value or instances of a class." \
        )
        self.endpoint = "https://query.wikidata.org/sparql"
        self.query_limits = {}
        self.initial_limit = 50
        self.increment = 25
    
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
                    default=60
                ),
                ToolParameter(
                    name="increase_limit",
                    type="boolean",
                    description=f"Whether to increase the number of results shown (increments by {self.increment})",
                    required=False,
                    default=False
                )
            ],
            return_type="dict",
            return_description="Query results in JSON format with enhanced display formatting"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute a SPARQL query. Any warning/error/notification from the endpoint is included in the result."""
        import traceback
        query = kwargs.get("query")
        timeout = kwargs.get("timeout", 60)
        increase_limit = kwargs.get("increase_limit", False)

        if not query:
            return {"success": False, "error": "SPARQL query string is required"}

        # Manage display limits
        query_hash = hash(query)
        display_limit = self.query_limits.get(query_hash, self.initial_limit)
        if increase_limit:
            display_limit += self.increment
        self.query_limits[query_hash] = display_limit

        headers = {
            'Accept': 'application/sparql-results+json',
            'User-Agent': 'RoboData/1.0 (Python)'
        }
        result: Dict[str, Any] = {}
        try:
            import requests
            response = requests.get(
                self.endpoint,
                params={'query': query},
                headers=headers,
                timeout=timeout
            )
            if response.status_code == 200:
                data = response.json()
                # Add metadata for better formatting
                data['_metadata'] = {
                    'query': query,
                    'display_limit': display_limit,
                    'total_results': len(data.get('results', {}).get('bindings', []))
                }
                result["success"] = True
                result["data"] = data
            else:
                result["success"] = False
                result["error"] = f"SPARQL query failed: {response.status_code}"
                try:
                    result["endpoint_message"] = response.json()
                except Exception:
                    result["endpoint_message"] = response.text
        except Exception as e:
            result["success"] = False
            
            # Provide specific handling for timeout errors with optimization suggestions
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                result["error"] = f"SPARQL query timed out after {timeout} seconds: {e}"
                result["optimization_suggestions"] = self._get_optimization_suggestions(query)
            else:
                result["error"] = f"SPARQL query error: {e}"
            
            result["traceback"] = traceback.format_exc()
        return result

    def format_result(self, result: Optional[Dict[str, Any]], query: Optional[str] = None) -> str:
        """Format the result into a readable, extensive string. Any warning/error/notification from the endpoint is included in the output."""
        import json
        if not result:
            return "SPARQL query returned no result."
        
        # Extract query text for both success and error cases
        query_text = query
        if result.get("data") and result["data"].get('_metadata'):
            query_text = result["data"]["_metadata"].get('query') or query
        
        if not result.get("success", True):
            out = f"SPARQL query failed: {result.get('error', 'Unknown error.')}"
            if result.get("traceback"):
                out += f"\nTraceback:\n{result['traceback']}"
            if result.get("endpoint_message"):
                out += f"\nEndpoint message: {json.dumps(result['endpoint_message'], indent=2)}"
            
            # Add optimization suggestions for timeout errors
            if result.get("optimization_suggestions"):
                out += f"\n\nOPTIMIZATION SUGGESTIONS:\n"
                for i, suggestion in enumerate(result["optimization_suggestions"], 1):
                    out += f"{i}. {suggestion}\n"
                
                # Add alternative queries if it's a timeout error
                if "timeout" in result.get("error", "").lower():
                    alternatives = self._suggest_alternative_queries(query_text or "")
                    if alternatives:
                        out += f"\nALTERNATIVE QUERIES:\n"
                        for alt in alternatives:
                            out += f"{alt}\n"
            
            return out
        data = result.get("data", result)
        # Extract metadata
        metadata = data.get('_metadata', {})
        query_text = metadata.get('query') or query
        display_limit = metadata.get('display_limit', self.initial_limit)
        total_results = metadata.get('total_results', 0)
        bindings = data.get('results', {}).get('bindings', [])
        count = len(bindings)
        if count == 0:
            summary = "SPARQL query returned 0 results.\n"
            if query_text:
                summary += f"\nQuery:\n```sparql\n{query_text}\n```"
            return summary
        # Start building the summary
        summary = f"SPARQL query returned {count} results. Showing top {min(display_limit, count)}:\n"
        # Add query if available
        if query_text:
            summary += f"\nQuery: {query_text}\n"
        # Extract headers from the first result if available
        headers = list(bindings[0].keys()) if bindings else []
        if headers:
            summary += f"Columns: {', '.join(headers)}\n\n"
        # Display results in detail, similar to SearchEntitiesTool
        results_to_show = bindings[:display_limit]
        for i, binding in enumerate(results_to_show):
            summary += f"{i+1}. "
            column_parts = []
            for header in headers:
                value_data = binding.get(header, {})
                raw_value = value_data.get('value', 'N/A')
                value_type = value_data.get('type', 'literal')
                if value_type == 'uri' and raw_value.startswith("http://www.wikidata.org/entity/"):
                    entity_id = raw_value.split('/')[-1]
                    formatted_value = entity_id
                elif value_type == 'uri':
                    formatted_value = raw_value
                else:
                    formatted_value = raw_value
                column_parts.append(f"{header}: {formatted_value}")
            summary += " | ".join(column_parts) + "\n"
            if i < len(results_to_show) - 1:
                summary += "\n"
        if count > display_limit:
            summary += f"\n... and {count - display_limit} more results"
        return summary.strip()

    def _get_optimization_suggestions(self, query: str) -> List[str]:
        """Generate optimization suggestions for complex SPARQL queries."""
        suggestions = []
        
        if not query:
            return suggestions
        
        query_lower = query.lower()
        
        # Check for multiple joins that might cause timeouts
        if query_lower.count('?') > 10:
            suggestions.append("Consider reducing the number of variables in your query")
        
        # Check for missing LIMIT clause
        if 'limit' not in query_lower:
            suggestions.append("Add a LIMIT clause to restrict the number of results (e.g., LIMIT 100)")
        
        # Check for complex property chains
        if query_lower.count('wdt:') > 5:
            suggestions.append("Consider breaking complex property chains into separate queries")
        
        # Check for multiple OPTIONAL clauses
        optional_count = query_lower.count('optional')
        if optional_count > 3:
            suggestions.append("Too many OPTIONAL clauses can slow down queries - consider making some required")
        
        # Check for complex filtering
        if 'filter' in query_lower and len(query) > 500:
            suggestions.append("Complex FILTER conditions can be expensive - try to simplify or move filtering to WHERE clauses")
        
        # General timeout suggestions
        suggestions.extend([
            "Try increasing the timeout parameter if the query is necessary",
            "Consider using more specific entity IDs instead of broad classes",
            "Break complex queries into smaller, simpler queries",
            "Use DISTINCT sparingly as it can be expensive on large result sets"
        ])
        
        return suggestions[:5]  # Return top 5 suggestions

    def _suggest_alternative_queries(self, original_query: str) -> List[str]:
        """Suggest alternative, simpler queries for complex cases."""
        alternatives = []
        
        if not original_query:
            return alternatives
        
        query_lower = original_query.lower()
        
        # General suggestions for complex queries
        if 'limit' not in query_lower:
            alternatives.append("""
# Add a LIMIT clause to your query
# Example: Add 'LIMIT 100' at the end of your query
""")
        
        if query_lower.count('wdt:') > 5:
            alternatives.append("""
# Break complex queries into smaller parts
# Example: Query one relationship at a time, then combine results
""")
        
        return alternatives

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