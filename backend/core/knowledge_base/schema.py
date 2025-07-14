"""
High-level abstractions for the Knowledge Base, including Node, Edge, and Graph objects.
"""

from typing import Dict, Any, List, Optional, Set
import networkx as nx
from rdflib import Graph as RDFGraph, URIRef, Literal
from rdflib.namespace import RDF, RDFS

class Node:
    """Represents a node in the knowledge graph."""
    
    def __init__(self, node_id: str, node_type: str, label: str = "", description: str = "", properties: Optional[Dict[str, Any]] = None):
        self.id = node_id
        self.type = node_type
        self.label = label
        self.description = description
        self.properties = properties.copy() if properties is not None and properties else {}
        
        # Ensure core properties are set
        self.properties.update({
            'id': self.id,
            'type': self.type,
            'label': self.label,
            'description': self.description
        })

    def __repr__(self) -> str:
        return f"Node(id={self.id}, type={self.type}, label='{self.label}')"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type,
            'label': self.label,
            'description': self.description,
            'properties': self.properties
        }

class Edge:
    """Represents an edge in the knowledge graph."""
    
    def __init__(self, source_id: str, target_id: str, relationship_type: str, label: str = "", properties: Optional[Dict[str, Any]] = None):
        self.source_id = source_id
        self.target_id = target_id
        self.type = relationship_type
        self.label = label or relationship_type
        self.properties = properties or {}
        
        # Ensure core properties are set
        self.properties.update({
            'source_id': self.source_id,
            'target_id': self.target_id,
            'type': self.type,
            'label': self.label
        })

    def __repr__(self) -> str:
        return f"Edge(source={self.source_id}, target={self.target_id}, type='{self.type}')"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'type': self.type,
            'label': self.label,
            'properties': self.properties
        }

class Graph:
    """Represents a knowledge graph using networkx."""
    
    def __init__(self):
        self._graph = nx.MultiDiGraph()

    def add_node(self, node: Node):
        """Adds a node to the graph."""
        self._graph.add_node(node.id, **node.to_dict())

    def add_edge(self, edge: Edge):
        """Adds an edge to the graph."""
        self._graph.add_edge(edge.source_id, edge.target_id, key=edge.type, **edge.to_dict())

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieves a node by its ID."""
        if self._graph.has_node(node_id):
            node_data = self._graph.nodes[node_id]
            return Node(
                node_id=node_data['id'],
                node_type=node_data['type'],
                label=node_data.get('label', ''),
                description=node_data.get('description', ''),
                properties=node_data.get('properties', {})
            )
        return None

    @property
    def nodes(self) -> List[Node]:
        """Returns a list of all nodes in the graph."""
        node_list = [self.get_node(node_id) for node_id in self._graph.nodes]
        return [node for node in node_list if node is not None]

    @property
    def edges(self) -> List[Edge]:
        """Returns a list of all edges in the graph."""
        edge_list = []
        for u, v, key, data in self._graph.edges(data=True, keys=True):
            edge_list.append(Edge(
                source_id=u,
                target_id=v,
                relationship_type=key,
                label=data.get('label', ''),
                properties=data.get('properties', {})
            ))
        return edge_list

    def to_json(self) -> Dict[str, Any]:
        """Exports the graph to a JSON-serializable dictionary."""
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges]
        }

    def to_rdf(self, base_uri: str = "http://robodata.io/graph/") -> RDFGraph:
        """Exports the graph to an RDF graph."""
        rdf_graph = RDFGraph()
        for node in self.nodes:
            node_uri = URIRef(f"{base_uri}{node.id}")
            rdf_graph.add((node_uri, RDF.type, URIRef(f"{base_uri}{node.type}")))
            rdf_graph.add((node_uri, RDFS.label, Literal(node.label)))
            if node.description:
                rdf_graph.add((node_uri, RDFS.comment, Literal(node.description)))
        
        for edge in self.edges:
            source_uri = URIRef(f"{base_uri}{edge.source_id}")
            target_uri = URIRef(f"{base_uri}{edge.target_id}")
            edge_uri = URIRef(f"{base_uri}{edge.type}")
            rdf_graph.add((source_uri, edge_uri, target_uri))
            
        return rdf_graph

    def get_subgraph(self, node_ids: List[str]) -> 'Graph':
        """Returns a subgraph containing the specified nodes and their connections."""
        nx_subgraph = self._graph.subgraph(node_ids)
        subgraph = Graph()
        subgraph._graph = nx_subgraph
        return subgraph
