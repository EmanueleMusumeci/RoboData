from typing import Dict, Any, List, Optional
from ..toolbox import Tool, ToolDefinition, ToolParameter
from ...knowledge_base.graph import get_graph_db
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
                    name="label",
                    type="string",
                    description="Node label/type"
                ),
                ToolParameter(
                    name="properties",
                    type="object",
                    description="Node properties as JSON object"
                )
            ],
            return_type="string",
            return_description="ID of the created node"
        )
    
    async def execute(self, label: str, properties: Dict[str, Any]) -> str:
        """Add a node to the graph."""
        db = get_graph_db()
        if not db.driver:
            await db.connect()
        
        return await db.add_node(label, properties)

class AddEdgeTool(Tool):
    """Tool for adding edges to the graph database."""
    
    def __init__(self):
        super().__init__(
            name="add_edge",
            description="Add a new edge/relationship between two nodes"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="from_node_id",
                    type="string",
                    description="ID of the source node"
                ),
                ToolParameter(
                    name="to_node_id",
                    type="string",
                    description="ID of the target node"
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
    
    async def execute(self, from_node_id: str, to_node_id: str, relationship_type: str, properties: Dict[str, Any] = None) -> str:
        """Add an edge to the graph."""
        db = get_graph_db()
        if not db.driver:
            await db.connect()
        
        return await db.add_edge(from_node_id, to_node_id, relationship_type, properties or {})

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
    
    async def execute(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node from the graph."""
        db = get_graph_db()
        if not db.driver:
            await db.connect()
        
        return await db.get_node(node_id)

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
    
    async def execute(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get an edge from the graph."""
        db = get_graph_db()
        if not db.driver:
            await db.connect()
        
        return await db.get_edge(edge_id)

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
            return_description="True if node was removed, False otherwise"
        )
    
    async def execute(self, node_id: str) -> bool:
        """Remove a node from the graph."""
        db = get_graph_db()
        if not db.driver:
            await db.connect()
        
        return await db.remove_node(node_id)

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
            return_description="True if edge was removed, False otherwise"
        )
    
    async def execute(self, edge_id: str) -> bool:
        """Remove an edge from the graph."""
        db = get_graph_db()
        if not db.driver:
            await db.connect()
        
        return await db.remove_edge(edge_id)

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
    
    async def execute(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a custom query on the graph."""
        db = get_graph_db()
        if not db.driver:
            await db.connect()
        
        return await db.query(query, parameters or {})

class FindNodesTool(Tool):
    """Tool for finding nodes in the graph database."""
    
    def __init__(self):
        super().__init__(
            name="find_nodes",
            description="Find nodes in the graph database by label and/or properties"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="label",
                    type="string",
                    description="Node label to search for",
                    required=False
                ),
                ToolParameter(
                    name="properties",
                    type="object",
                    description="Properties to match as JSON object",
                    required=False,
                    default={}
                )
            ],
            return_type="array",
            return_description="List of matching nodes"
        )
    
    async def execute(self, label: str = None, properties: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find nodes in the graph."""
        db = get_graph_db()
        if not db.driver:
            await db.connect()
        
        return await db.find_nodes(label, properties or {})

class GetNeighborsTool(Tool):
    """Tool for getting neighboring nodes."""
    
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
                    description="Type of relationship to follow",
                    required=False
                ),
                ToolParameter(
                    name="direction",
                    type="string",
                    description="Direction of relationships (incoming, outgoing, both)",
                    required=False,
                    default="both"
                )
            ],
            return_type="array",
            return_description="List of neighboring nodes with relationship info"
        )
    
    async def execute(self, node_id: str, relationship_type: str = None, direction: str = "both") -> List[Dict[str, Any]]:
        """Get neighboring nodes."""
        db = get_graph_db()
        if not db.driver:
            await db.connect()
        
        return await db.get_node_neighbors(node_id, relationship_type, direction)
