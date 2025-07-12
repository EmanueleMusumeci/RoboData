from typing import Dict, Any, List, Optional
from ..toolbox import Tool, ToolDefinition, ToolParameter
from ...knowledge_base.graph import get_knowledge_graph
import json

class AddNodeTool(Tool):
    """Tool for adding nodes to the graph database."""
    
    def __init__(self):
        super().__init__(
            name="add_node",
            description="Add a new node to the graph database"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_id",
                    type="string",
                    description="Unique ID for the entity"
                ),
                ToolParameter(
                    name="entity_type",
                    type="string",
                    description="Type of the entity (e.g. 'WikidataEntity')"
                ),
                ToolParameter(
                    name="properties",
                    type="object",
                    description="Entity properties as JSON object"
                )
            ],
            return_type="string",
            return_description="ID of the created entity"
        )
    
    async def execute(self, **kwargs) -> str:
        """Add an entity to the knowledge graph."""
        entity_id = kwargs.get("entity_id")
        entity_type = kwargs.get("entity_type")
        properties = kwargs.get("properties", {})
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        return await graph.add_entity(entity_id, entity_type, properties)

class AddEdgeTool(Tool):
    """Tool for adding edges to the graph database."""
    
    def __init__(self):
        super().__init__(
            name="add_edge",
            description="Add a new edge/relationship between two entities"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="from_node_id",
                    type="string",
                    description="ID of the source entity"
                ),
                ToolParameter(
                    name="to_node_id",
                    type="string",
                    description="ID of the target entity"
                ),
                ToolParameter(
                    name="relationship_type",
                    type="string",
                    description="Type of relationship"
                ),
                ToolParameter(
                    name="properties",
                    type="object",
                    description="Edge properties as JSON object",
                    required=False,
                    default={}
                )
            ],
            return_type="string",
            return_description="ID of the created edge"
        )
    
    async def execute(self, **kwargs) -> str:
        """Add a relationship to the knowledge graph."""
        from_node_id = kwargs.get("from_node_id")
        to_node_id = kwargs.get("to_node_id")
        relationship_type = kwargs.get("relationship_type")
        properties = kwargs.get("properties", {})
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        return await graph.add_relationship(from_node_id, to_node_id, relationship_type, properties)

class GetNodeTool(Tool):
    """Tool for retrieving nodes from the graph database."""
    
    def __init__(self):
        super().__init__(
            name="get_node",
            description="Get a node from the graph database by ID"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="node_id",
                    type="string",
                    description="ID of the node to retrieve"
                )
            ],
            return_type="object",
            return_description="Node data including properties and labels"
        )
    
    async def execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Get an entity from the knowledge graph."""
        node_id = kwargs.get("node_id")
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        return await graph.get_entity(node_id)

class GetEdgeTool(Tool):
    """Tool for retrieving edges from the graph database."""
    
    def __init__(self):
        super().__init__(
            name="get_edge",
            description="Get an edge from the graph database by ID"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="edge_id",
                    type="string",
                    description="ID of the edge to retrieve"
                )
            ],
            return_type="object",
            return_description="Edge data including properties and connected nodes"
        )
    
    async def execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Get a relationship from the knowledge graph."""
        edge_id = kwargs.get("edge_id")
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        return await graph.get_relationship(edge_id)

class RemoveNodeTool(Tool):
    """Tool for removing nodes from the graph database."""
    
    def __init__(self):
        super().__init__(
            name="remove_node",
            description="Remove a node from the graph database"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="node_id",
                    type="string",
                    description="ID of the node to remove"
                )
            ],
            return_type="boolean",
            return_description="True if node was successfully removed"
        )
    
    async def execute(self, **kwargs) -> bool:
        """Remove an entity from the knowledge graph."""
        node_id = kwargs.get("node_id")
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        return await graph.remove_entity(node_id)

class RemoveEdgeTool(Tool):
    """Tool for removing edges from the graph database."""
    
    def __init__(self):
        super().__init__(
            name="remove_edge",
            description="Remove an edge from the graph database"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="edge_id",
                    type="string",
                    description="ID of the edge to remove"
                )
            ],
            return_type="boolean",
            return_description="True if edge was successfully removed"
        )
    
    async def execute(self, **kwargs) -> bool:
        """Remove a relationship from the knowledge graph."""
        edge_id = kwargs.get("edge_id")
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        return await graph.remove_relationship(edge_id)

class QueryGraphTool(Tool):
    """Tool for executing custom queries on the graph database."""
    
    def __init__(self):
        super().__init__(
            name="query_graph",
            description="Execute a custom Cypher query on the graph database"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Cypher query to execute"
                ),
                ToolParameter(
                    name="parameters",
                    type="object",
                    description="Query parameters as JSON object",
                    required=False,
                    default={}
                )
            ],
            return_type="array",
            return_description="Query results as array of objects"
        )
    
    async def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """Execute a custom query on the knowledge graph."""
        query = kwargs.get("query")
        parameters = kwargs.get("parameters", {})
        
        if not query:
            return []
        
        try:
            graph = get_knowledge_graph()
            if not await graph.is_connected():
                await graph.connect()
            
            results = await graph.execute_custom_query(query, parameters)
            return results if results else []
        except Exception as e:
            # Log error but return empty results rather than failing
            print(f"Query execution failed: {e}")
            return []

class FindNodesTool(Tool):
    """Tool for finding nodes in the graph database."""
    
    def __init__(self):
        super().__init__(
            name="find_nodes",
            description="Find nodes in the graph database by type and properties"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_type",
                    type="string",
                    description="Entity type to filter by",
                    required=False
                ),
                ToolParameter(
                    name="properties",
                    type="object",
                    description="Properties to match as JSON object",
                    required=False,
                    default={}
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of results to return",
                    required=False
                )
            ],
            return_type="array",
            return_description="Array of matching nodes"
        )
    
    async def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """Find entities in the knowledge graph."""
        entity_type = kwargs.get("entity_type")
        properties = kwargs.get("properties", {})
        limit = kwargs.get("limit")
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        return await graph.find_entities(entity_type, properties, limit)

class GetNeighborsTool(Tool):
    """Tool for getting neighboring nodes in the graph database."""
    
    def __init__(self):
        super().__init__(
            name="get_neighbors",
            description="Get neighboring nodes of a given node"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="node_id",
                    type="string",
                    description="ID of the node to get neighbors for"
                ),
                ToolParameter(
                    name="relationship_type",
                    type="string",
                    description="Filter by relationship type",
                    required=False
                ),
                ToolParameter(
                    name="direction",
                    type="string",
                    description="Direction of relationships ('incoming', 'outgoing', 'both')",
                    required=False,
                    default="both"
                )
            ],
            return_type="array",
            return_description="Array of neighboring nodes with relationship info"
        )
    
    async def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """Get neighbors of an entity in the knowledge graph."""
        node_id = kwargs.get("node_id")
        relationship_type = kwargs.get("relationship_type")
        direction = kwargs.get("direction", "both")
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        return await graph.get_neighbors(node_id, relationship_type, direction)

class GetSubgraphTool(Tool):
    """Tool for getting a subgraph around specified nodes."""
    
    def __init__(self):
        super().__init__(
            name="get_subgraph",
            description="Get a subgraph containing specified nodes and their connections"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="node_ids",
                    type="array",
                    description="Array of node IDs to include in the subgraph",
                    items={"type": "string"}
                ),
                ToolParameter(
                    name="max_depth",
                    type="integer",
                    description="Maximum depth of connections to include",
                    required=False,
                    default=1
                )
            ],
            return_type="object",
            return_description="Subgraph containing nodes and relationships"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Get a subgraph from the knowledge graph."""
        node_ids = kwargs.get("node_ids", [])
        max_depth = kwargs.get("max_depth", 1)
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        return await graph.get_subgraph(node_ids, max_depth)

class GetGraphStatsTool(Tool):
    """Tool for getting graph database statistics."""
    
    def __init__(self):
        super().__init__(
            name="get_graph_stats",
            description="Get statistics about the graph database"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[],
            return_type="object",
            return_description="Graph statistics including node count, relationship count, etc."
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Get statistics from the knowledge graph."""
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        return await graph.get_graph_statistics()

class CypherQueryTool(Tool):
    """Tool for executing custom Cypher queries on the graph database with enhanced error handling."""
    
    def __init__(self):
        super().__init__(
            name="cypher_query",
            description="Execute a custom Cypher query on the graph database with enhanced error handling and result formatting"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="cypher_query",
                    type="string",
                    description="Cypher query to execute (e.g., 'MATCH (n:Person) RETURN n LIMIT 10')"
                ),
                ToolParameter(
                    name="parameters",
                    type="object",
                    description="Query parameters as JSON object",
                    required=False,
                    default={}
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Query timeout in seconds",
                    required=False,
                    default=30
                )
            ],
            return_type="object",
            return_description="Query results with metadata including execution time and record count"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute a Cypher query with enhanced error handling and result formatting."""
        import time
        
        cypher_query = kwargs.get("cypher_query")
        parameters = kwargs.get("parameters", {})
        timeout = kwargs.get("timeout", 30)
        
        if not cypher_query:
            return {
                "success": False,
                "error": "cypher_query is required",
                "results": [],
                "metadata": {
                    "record_count": 0,
                    "execution_time_seconds": 0,
                    "query": "",
                    "parameters": {}
                }
            }
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        start_time = time.time()
        
        try:
            # Execute the query
            results = await graph.execute_custom_query(cypher_query, parameters)
            
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "results": results,
                "metadata": {
                    "record_count": len(results),
                    "execution_time_seconds": round(execution_time, 3),
                    "query": cypher_query,
                    "parameters": parameters
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "metadata": {
                    "record_count": 0,
                    "execution_time_seconds": round(execution_time, 3),
                    "query": cypher_query,
                    "parameters": parameters
                }
            }

# Export all tools for registration
GRAPH_TOOLS = [
    AddNodeTool,
    AddEdgeTool,
    GetNodeTool,
    GetEdgeTool,
    RemoveNodeTool,
    RemoveEdgeTool,
    QueryGraphTool,
    FindNodesTool,
    GetNeighborsTool,
    GetSubgraphTool,
    GetGraphStatsTool,
    CypherQueryTool
]
