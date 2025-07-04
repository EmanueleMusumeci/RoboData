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