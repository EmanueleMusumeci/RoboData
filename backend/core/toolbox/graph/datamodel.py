from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel

class GraphNode(BaseModel):
    """Standardized representation of a graph node."""
    id: str
    labels: List[str] = []
    properties: Dict[str, Any] = {}

class GraphEdge(BaseModel):
    """Standardized representation of a graph edge/relationship."""
    id: str
    type: str
    from_id: str
    to_id: str
    properties: Dict[str, Any] = {}

class GraphSearchResult(BaseModel):
    """Result from searching/finding nodes in the graph."""
    nodes: List[GraphNode] = []
    total_count: int = 0
    query_info: Dict[str, Any] = {}

class GraphOperationResult(BaseModel):
    """Generic result for graph operations (add, remove, etc.)."""
    success: bool
    operation: str
    entity_id: Optional[str] = None
    message: str = ""
    error: Optional[str] = None

def convert_node_to_model(node_data: Dict[str, Any]) -> GraphNode:
    """Convert raw node data to GraphNode model."""
    return GraphNode(
        id=node_data.get('id', ''),
        labels=node_data.get('labels', []),
        properties={k: v for k, v in node_data.items() if k not in ['id', 'labels']}
    )

def convert_edge_to_model(edge_data: Dict[str, Any]) -> GraphEdge:
    """Convert raw edge data to GraphEdge model."""
    return GraphEdge(
        id=edge_data.get('id', ''),
        type=edge_data.get('type', ''),
        from_id=edge_data.get('from_id', ''),
        to_id=edge_data.get('to_id', ''),
        properties={k: v for k, v in edge_data.items() if k not in ['id', 'type', 'from_id', 'to_id']}
    )
