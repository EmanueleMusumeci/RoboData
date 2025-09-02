from typing import Dict, Any, List, Optional
import aiohttp
import asyncio
import traceback
from ..toolbox import Tool, ToolDefinition, ToolParameter
from .dbpedia_api import dbpedia_api

class SPARQLQueryTool(Tool):
    """Tool for executing SPARQL queries on DBpedia."""
    
    def __init__(self):
        super().__init__(
            name="sparql_query",
            description="Execute SPARQL queries on DBpedia, where SPARQL is a query language for databases, able to retrieve and manipulate data stored in Resource Description Framework (RDF) format." \
            "This tool should be preferred if we wish to retrieve sets of data with a common property. It is generally faster than the other tools, as it can retrieve multiple data in a single query." \
            "Be careful when using the LIMIT operator: the higher the number of results, the better for you." \
            "USEFUL FOR: retrieving structured data from DBpedia, such as entities, properties, and relationships." \
            "Retrieving data with complex relationships, such as subclasses, superclasses, instances, and properties of entities or even property chains." \
            "Retrieving data with specific conditions, such as entities or properties with a certain property value or instances of a class."
        )
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
                    description="SPARQL query string (use DBpedia prefixes: dbo: for ontology, dbr: for resources, dbp: for properties)"
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

        try:
            data = await dbpedia_api._execute_sparql_query(query, timeout)
            # Add metadata for better formatting
            data['_metadata'] = {
                'query': query,
                'display_limit': display_limit,
                'total_results': len(data.get('results', {}).get('bindings', []))
            }
            return {"success": True, "data": data}
        except Exception as e:
            error_msg = f"SPARQL query failed: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return {"success": False, "error": error_msg, "query": query}

    def format_result(self, result: Optional[Dict[str, Any]], query: Optional[str] = None) -> str:
        """Format the SPARQL result into a readable, extensive string."""
        import json
        if not result:
            return "SPARQL query returned no result."
        
        # Extract query text for both success and error cases
        query_text = query
        if result.get("data") and result["data"].get('_metadata'):
            query_text = result["data"]["_metadata"].get('query') or query
        
        if not result.get("success", True):
            out = f"SPARQL query failed: {result.get('error', 'Unknown error.')}"
            if query_text:
                out += f"\nQuery: {query_text}"
            return out
        
        data = result.get("data", result)
        if not data or 'results' not in data:
            return "No SPARQL results found."
        
        bindings = data['results'].get('bindings', [])
        metadata = data.get('_metadata', {})
        total_results = metadata.get('total_results', len(bindings))
        display_limit = metadata.get('display_limit', 50)
        
        summary = f"SPARQL Query Results ({total_results} total):\n"
        if query_text:
            summary += f"Query: {query_text[:100]}{'...' if len(query_text or '') > 100 else ''}\n\n"
        
        # Show results
        for i, binding in enumerate(bindings[:display_limit], 1):
            summary += f"{i}. "
            for var, value_info in binding.items():
                value = value_info.get('value', '')
                summary += f"{var}: {value} | "
            summary = summary.rstrip(" | ") + "\n"
        
        if total_results > display_limit:
            summary += f"... and {total_results - display_limit} more results\n"
        
        return summary.strip()

class SubclassQueryTool(Tool):
    """Tool for finding subclasses of a given DBpedia class."""
    
    def __init__(self):
        super().__init__(
            name="get_subclasses",
            description="Get direct subclasses of a DBpedia class using rdfs:subClassOf relationships." \
            "USEFUL FOR: exploring class hierarchies and finding more specific categories within a broader class."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="class_uri",
                    type="string",
                    description="DBpedia class URI (e.g., dbo:Person) or simple name"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of subclasses to return",
                    required=False,
                    default=20
                )
            ],
            return_type="dict",
            return_description="List of subclasses with their labels and URIs"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the subclass query."""
        class_uri = kwargs.get("class_uri")
        limit = kwargs.get("limit", 20)
        
        if not class_uri:
            raise ValueError("class_uri is required")
        
        # Normalize the class URI
        if not class_uri.startswith("http://"):
            if ":" not in class_uri:
                class_uri = f"dbo:{class_uri}"
        
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        
        SELECT DISTINCT ?subclass ?label WHERE {{
            ?subclass rdfs:subClassOf <{class_uri}> .
            OPTIONAL {{ ?subclass rdfs:label ?label . FILTER(LANG(?label) = "en") }}
        }}
        LIMIT {limit}
        """
        
        try:
            data = await dbpedia_api._execute_sparql_query(query)
            
            subclasses = []
            for binding in data.get('results', {}).get('bindings', []):
                subclass_uri = binding.get('subclass', {}).get('value', '')
                label = binding.get('label', {}).get('value', subclass_uri.split('/')[-1])
                
                subclasses.append({
                    'uri': subclass_uri,
                    'label': label,
                    'id': subclass_uri.split('/')[-1] if '/' in subclass_uri else subclass_uri
                })
            
            return {
                "class_uri": class_uri,
                "subclasses": subclasses,
                "count": len(subclasses),
                "query": query
            }
        except Exception as e:
            error_msg = f"Failed to get subclasses for {class_uri}: {e}"
            raise ValueError(error_msg)

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the subclass result into a readable string."""
        if not result:
            return "No subclasses found."
        
        class_uri = result.get('class_uri', 'N/A')
        subclasses = result.get('subclasses', [])
        count = result.get('count', 0)
        
        summary = f"Subclasses of {class_uri}:\n"
        summary += f"Found {count} subclasses:\n\n"
        
        for i, subclass in enumerate(subclasses[:10], 1):  # Show top 10
            label = subclass.get('label', subclass.get('id', 'N/A'))
            class_id = subclass.get('id', 'N/A')
            summary += f"{i}. {label} ({class_id})\n"
        
        if len(subclasses) > 10:
            summary += f"... and {len(subclasses) - 10} more subclasses\n"
        
        return summary.strip()

class SuperclassQueryTool(Tool):
    """Tool for finding superclasses of a given DBpedia class."""
    
    def __init__(self):
        super().__init__(
            name="get_superclasses",
            description="Get direct superclasses of a DBpedia class using rdfs:subClassOf relationships." \
            "USEFUL FOR: exploring class hierarchies and finding broader categories that contain a specific class."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="class_uri",
                    type="string",
                    description="DBpedia class URI (e.g., dbo:Person) or simple name"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of superclasses to return",
                    required=False,
                    default=10
                )
            ],
            return_type="dict",
            return_description="List of superclasses with their labels and URIs"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the superclass query."""
        class_uri = kwargs.get("class_uri")
        limit = kwargs.get("limit", 10)
        
        if not class_uri:
            raise ValueError("class_uri is required")
        
        # Normalize the class URI
        if not class_uri.startswith("http://"):
            if ":" not in class_uri:
                class_uri = f"dbo:{class_uri}"
        
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        
        SELECT DISTINCT ?superclass ?label WHERE {{
            <{class_uri}> rdfs:subClassOf ?superclass .
            OPTIONAL {{ ?superclass rdfs:label ?label . FILTER(LANG(?label) = "en") }}
        }}
        LIMIT {limit}
        """
        
        try:
            data = await dbpedia_api._execute_sparql_query(query)
            
            superclasses = []
            for binding in data.get('results', {}).get('bindings', []):
                superclass_uri = binding.get('superclass', {}).get('value', '')
                label = binding.get('label', {}).get('value', superclass_uri.split('/')[-1])
                
                superclasses.append({
                    'uri': superclass_uri,
                    'label': label,
                    'id': superclass_uri.split('/')[-1] if '/' in superclass_uri else superclass_uri
                })
            
            return {
                "class_uri": class_uri,
                "superclasses": superclasses,
                "count": len(superclasses),
                "query": query
            }
        except Exception as e:
            error_msg = f"Failed to get superclasses for {class_uri}: {e}"
            raise ValueError(error_msg)

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the superclass result into a readable string."""
        if not result:
            return "No superclasses found."
        
        class_uri = result.get('class_uri', 'N/A')
        superclasses = result.get('superclasses', [])
        count = result.get('count', 0)
        
        summary = f"Superclasses of {class_uri}:\n"
        summary += f"Found {count} superclasses:\n\n"
        
        for i, superclass in enumerate(superclasses, 1):
            label = superclass.get('label', superclass.get('id', 'N/A'))
            class_id = superclass.get('id', 'N/A')
            summary += f"{i}. {label} ({class_id})\n"
        
        return summary.strip()

class GetInstancesQueryTool(Tool):
    """Tool for finding instances of a given DBpedia class."""
    
    def __init__(self):
        super().__init__(
            name="get_instances",
            description="Get instances (individuals) of a DBpedia class using rdf:type relationships." \
            "USEFUL FOR: finding specific entities that belong to a particular class or category."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="class_uri",
                    type="string",
                    description="DBpedia class URI (e.g., dbo:Person) or simple name"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of instances to return",
                    required=False,
                    default=50
                )
            ],
            return_type="dict",
            return_description="List of instances with their labels and URIs"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the instances query."""
        class_uri = kwargs.get("class_uri")
        limit = kwargs.get("limit", 50)
        
        if not class_uri:
            raise ValueError("class_uri is required")
        
        # Normalize the class URI
        if not class_uri.startswith("http://"):
            if ":" not in class_uri:
                class_uri = f"dbo:{class_uri}"
        
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        
        SELECT DISTINCT ?instance ?label WHERE {{
            ?instance rdf:type <{class_uri}> .
            OPTIONAL {{ ?instance rdfs:label ?label . FILTER(LANG(?label) = "en") }}
        }}
        LIMIT {limit}
        """
        
        try:
            data = await dbpedia_api._execute_sparql_query(query)
            
            instances = []
            for binding in data.get('results', {}).get('bindings', []):
                instance_uri = binding.get('instance', {}).get('value', '')
                label = binding.get('label', {}).get('value', instance_uri.split('/')[-1])
                
                instances.append({
                    'uri': instance_uri,
                    'label': label,
                    'id': instance_uri.split('/')[-1] if '/' in instance_uri else instance_uri
                })
            
            return {
                "class_uri": class_uri,
                "instances": instances,
                "count": len(instances),
                "query": query
            }
        except Exception as e:
            error_msg = f"Failed to get instances for {class_uri}: {e}"
            raise ValueError(error_msg)

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the instances result into a readable string."""
        if not result:
            return "No instances found."
        
        class_uri = result.get('class_uri', 'N/A')
        instances = result.get('instances', [])
        count = result.get('count', 0)
        
        summary = f"Instances of {class_uri}:\n"
        summary += f"Found {count} instances:\n\n"
        
        for i, instance in enumerate(instances[:15], 1):  # Show top 15
            label = instance.get('label', instance.get('id', 'N/A'))
            instance_id = instance.get('id', 'N/A')
            summary += f"{i}. {label} ({instance_id})\n"
        
        if len(instances) > 15:
            summary += f"... and {len(instances) - 15} more instances\n"
        
        return summary.strip()

class InstanceOfQueryTool(Tool):
    """Tool for finding what classes an entity is an instance of."""
    
    def __init__(self):
        super().__init__(
            name="get_instance_of",
            description="Get the classes that an entity is an instance of using rdf:type relationships." \
            "USEFUL FOR: understanding what categories or types an entity belongs to."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_uri",
                    type="string",
                    description="DBpedia entity URI (e.g., dbr:Douglas_Adams) or simple name"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of classes to return",
                    required=False,
                    default=20
                )
            ],
            return_type="dict",
            return_description="List of classes that the entity is an instance of"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the instance-of query."""
        entity_uri = kwargs.get("entity_uri")
        limit = kwargs.get("limit", 20)
        
        if not entity_uri:
            raise ValueError("entity_uri is required")
        
        # Normalize the entity URI
        if not entity_uri.startswith("http://"):
            if ":" not in entity_uri:
                entity_uri = f"dbr:{entity_uri}"
        
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        
        SELECT DISTINCT ?class ?label WHERE {{
            <{entity_uri}> rdf:type ?class .
            OPTIONAL {{ ?class rdfs:label ?label . FILTER(LANG(?label) = "en") }}
        }}
        LIMIT {limit}
        """
        
        try:
            data = await dbpedia_api._execute_sparql_query(query)
            
            classes = []
            for binding in data.get('results', {}).get('bindings', []):
                class_uri = binding.get('class', {}).get('value', '')
                label = binding.get('label', {}).get('value', class_uri.split('/')[-1])
                
                classes.append({
                    'uri': class_uri,
                    'label': label,
                    'id': class_uri.split('/')[-1] if '/' in class_uri else class_uri
                })
            
            return {
                "entity_uri": entity_uri,
                "classes": classes,
                "count": len(classes),
                "query": query
            }
        except Exception as e:
            error_msg = f"Failed to get classes for {entity_uri}: {e}"
            raise ValueError(error_msg)

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the instance-of result into a readable string."""
        if not result:
            return "No classes found."
        
        entity_uri = result.get('entity_uri', 'N/A')
        classes = result.get('classes', [])
        count = result.get('count', 0)
        
        summary = f"Classes that {entity_uri} is an instance of:\n"
        summary += f"Found {count} classes:\n\n"
        
        for i, class_info in enumerate(classes, 1):
            label = class_info.get('label', class_info.get('id', 'N/A'))
            class_id = class_info.get('id', 'N/A')
            summary += f"{i}. {label} ({class_id})\n"
        
        return summary.strip()
