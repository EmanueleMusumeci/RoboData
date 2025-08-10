"""
High-level Graph Database Abstraction for RoboData

This module provides a high-level interface for working with knowledge graphs,
using the Neo4j interface for low-level database operations.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from backend.settings import settings_manager
from .interfaces.neo4j_interface import Neo4jInterface, DatabaseInterface
from .schema import Node, Edge, Graph

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    High-level knowledge graph interface that provides semantic operations
    for managing entities and relationships in a graph database.
    """
    
    def __init__(self, db_interface: Optional[DatabaseInterface] = None, retain_existing_data: bool = False):
        """
        Initialize the knowledge graph.
        
        Args:
            db_interface: Database interface to use. If None, creates Neo4j interface from settings.
            retain_existing_data: If False, clears the database on connection. Default: False.
        """
        if db_interface is None:
            # Create Neo4j interface from settings
            neo4j_settings = settings_manager.get_settings().neo4j
            self.db = Neo4jInterface(
                uri=neo4j_settings.uri,
                username=neo4j_settings.username,
                password=neo4j_settings.password,
                database=neo4j_settings.database
            )
        else:
            self.db = db_interface
        
        self.retain_existing_data = retain_existing_data
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to the graph database."""
        await self.db.connect()
        self._connected = True
        logger.info("Knowledge graph connected")
        
        # Clear database if not retaining existing data
        if not self.retain_existing_data:
            try:
                stats = await self.get_graph_statistics()
                node_count = stats.get('node_count', 0)
                rel_count = stats.get('relationship_count', 0)
                
                if node_count > 0 or rel_count > 0:
                    logger.info(f"Clearing existing graph data: {node_count} nodes, {rel_count} relationships")
                    await self.clear_graph()
                    logger.info("Graph database cleared")
                else:
                    logger.info("Graph database was already empty")
            except Exception as e:
                logger.warning(f"Could not clear graph database: {e}")
                # Continue anyway - this is not a fatal error
    
    async def disconnect(self) -> None:
        """Disconnect from the graph database."""
        await self.db.close()
        self._connected = False
        logger.info("Knowledge graph disconnected")
    
    async def is_connected(self) -> bool:
        """Check if the graph database is connected."""
        return self._connected and await self.db.is_connected()
    
    async def close(self) -> None:
        """Alias for disconnect."""
        await self.disconnect()

    # === Entity Management ===
    
    async def add_entity(self, node: Node) -> str:
        """
        Add an entity (node) to the knowledge graph.
        
        Args:
            node: The Node object to add.
            
        Returns:
            str: The entity ID
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        labels = ["Entity", node.type]
        properties = node.properties
        properties['id'] = node.id
        properties['label'] = node.label
        properties['description'] = node.description
        
        node_id = await self.db.create_node(labels, properties)
        logger.info(f"Added entity {node.id} of type {node.type}")
        return node_id or node.id
    
    async def get_entity(self, entity_id: str) -> Optional[Node]:
        """
        Get an entity by its ID.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            Node object or None if not found
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        return await self.db.get_node_by_id(entity_id)
    
    async def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update an entity's properties.
        
        Args:
            entity_id: The entity ID
            properties: Properties to update
            
        Returns:
            bool: True if successful
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        success = await self.db.update_node(entity_id, properties)
        if success:
            logger.info(f"Updated entity {entity_id}")
        return success
    
    async def remove_entity(self, entity_id: str) -> bool:
        """
        Remove an entity from the knowledge graph.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            bool: True if successful
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        success = await self.db.delete_node(entity_id)
        if success:
            logger.info(f"Removed entity {entity_id}")
        return success
    
    async def find_entities(self, entity_type: Optional[str] = None, 
                          properties: Optional[Dict[str, Any]] = None,
                          limit: Optional[int] = None) -> List[Node]:
        """
        Find entities by type and/or properties.
        
        Args:
            entity_type: Entity type to filter by
            properties: Properties to match
            limit: Maximum number of results
            
        Returns:
            List of matching Node objects
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        labels = ["Entity"]
        if entity_type:
            labels.append(entity_type)
        
        return await self.db.find_nodes(labels, properties, limit)
    
    # === Relationship Management ===
    
    async def add_relationship(self, edge: Edge) -> str:
        """
        Add a relationship between two entities.
        
        Args:
            edge: The Edge object to add.
            
        Returns:
            str: The relationship ID
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        rel_id = await self.db.create_relationship(
            edge.source_id, 
            edge.target_id, 
            edge.type, 
            edge.properties
        )
        logger.info(f"Added relationship {edge.type} from {edge.source_id} to {edge.target_id}")
        return rel_id or f"{edge.source_id}_{edge.type}_{edge.target_id}"
    
    async def get_relationship(self, relationship_id: str) -> Optional[Edge]:
        """Get a relationship by its ID."""
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        return await self.db.get_relationship_by_id(relationship_id)
    
    async def remove_relationship(self, relationship_id: str) -> bool:
        """Remove a relationship from the knowledge graph."""
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        success = await self.db.delete_relationship(relationship_id)
        if success:
            logger.info(f"Removed relationship {relationship_id}")
        return success
    
    async def get_entity_relationships(self, entity_id: str, direction: str = "both",
                                     relationship_type: Optional[str] = None) -> List[Edge]:
        """
        Get all relationships for an entity.
        
        Args:
            entity_id: The entity ID
            direction: "incoming", "outgoing", or "both"
            relationship_type: Filter by relationship type
            
        Returns:
            List of Edge objects
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        return await self.db.get_node_relationships(entity_id, direction, relationship_type)
    
    async def find_relationships(self, relationship_type: Optional[str] = None, 
                                 properties: Optional[Dict[str, Any]] = None,
                                 limit: Optional[int] = None) -> List[Edge]:
        """
        Find relationships by type and/or properties.
        
        Args:
            relationship_type: Relationship type to filter by
            properties: Properties to match
            limit: Maximum number of results
            
        Returns:
            List of matching Edge objects
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        return await self.db.find_relationships(relationship_type, properties, limit)

    # === Graph Operations ===
    
    async def get_neighbors(self, entity_id: str, 
                          relationship_type: Optional[str] = None,
                          limit: Optional[int] = None) -> List[Node]:
        """
        Get neighboring entities of a given entity.
        
        Args:
            entity_id: The entity ID
            relationship_type: Filter by relationship type
            limit: Maximum number of neighbors to return
            
        Returns:
            List of neighboring Node objects
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")

        rel_pattern = f":`{relationship_type}`" if relationship_type else ""
        
        match_clause = f"MATCH (n {{id: $entity_id}})-[r{rel_pattern}]-(neighbor)"

        return_clause = "RETURN DISTINCT neighbor"
        
        query_parts = [match_clause, return_clause]

        if limit:
            limit_clause = f"LIMIT {limit}"
            query_parts.append(limit_clause)

        query = " ".join(filter(None, query_parts))
        
        params = {'entity_id': entity_id}
        
        result = await self.db.execute_query(query, params)
        
        neighbors = []
        if result:
            for record in result:
                node_data = record.get('neighbor')
                if node_data:

                    # Assuming Node can be constructed from a dict of its properties
                    neighbors.append(Node(
                        node_id=node_data.get('id'),
                        node_type=next((label for label in node_data.get('labels', []) if label != 'Entity'), 'Entity'),
                        label=node_data.get('label', ''),
                        description=node_data.get('properties', {}).get('description', ''),
                        properties=node_data.get('properties', {})
                    ))
        return neighbors
    
    async def get_whole_graph(self) -> Graph:
        """
        Get the entire knowledge graph as a Graph object.
        
        Returns:
            Graph object containing all nodes and relationships
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        return await self.db.get_all_nodes_and_relationships()
    
    async def to_triples(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get local graph data as a list of triples, showing nodes as "ID (label)".
        
        Returns:
            A list of dictionaries, where each dictionary represents a triple, or None if the graph is empty.
        """
        graph_data = await self.get_whole_graph()
        
        if not graph_data:
            return None
        
        triples = []
        
        # Process nodes
        nodes = graph_data.nodes
        if not nodes:
            return None  # Empty graph
            
        for node in nodes:
            if not node:
                continue
                
            node_id = node.id
            node_label = node.label or node.type
            subject_repr = f"{node_id} ({node_label})" if node_id and node_label else node_id or node_label
            if not node_id:
                continue
                
            labels = [node.type]
            for label in labels:
                if label:
                    triples.append({
                        'subject': subject_repr,
                        'predicate': 'rdf:type',
                        'object': label
                    })
            
            for key, value in node.properties.items():
                if key not in ['id', 'labels', 'type', 'label', 'description'] and value is not None:
                    triples.append({
                        'subject': subject_repr,
                        'predicate': key,
                        'object': value
                    })
        
        # Process relationships
        relationships = graph_data.edges
        for rel in relationships:
            if not rel:
                continue
                
            source_id = rel.source_id
            target_id = rel.target_id
            rel_type = rel.type

            source_label = next((n.label for n in nodes if n.id == source_id), "")
            target_label = next((n.label for n in nodes if n.id == target_id), "")
            source_repr = f"{source_id} ({source_label})" if source_id and source_label else source_id
            target_repr = f"{target_id} ({target_label})" if target_id and target_label else target_id

            if source_id and target_id and rel_type:
                triples.append({
                    'subject': source_repr,
                    'predicate': rel_type,
                    'object': target_repr
                })
                
                for key, value in rel.properties.items():
                    if key not in ['source_id', 'target_id', 'type', 'label', 'id'] and value is not None:
                        rel_id = f"{source_id}_{rel_type}_{target_id}"
                        rel_label = rel.label or rel_type
                        rel_repr = f"{rel_id} ({rel_label})" if rel_id and rel_label else rel_id
                        triples.append({
                            'subject': rel_repr,
                            'predicate': key,
                            'object': value
                        })
        
        return triples if triples else None

    async def to_readable_format(self) -> Optional[str]:
        """
        Get local graph data in a readable format for prompts.
        
        Returns:
            A formatted string with Entities, Relations, and Structure sections, or None if the graph is empty.
        """
        graph_data = await self.get_whole_graph()
        
        if not graph_data:
            return None
        
        nodes = graph_data.nodes
        edges = graph_data.edges
        
        if not nodes:
            return None  # Empty graph
        
        output_lines = []
        
        # === ENTITIES SECTION ===
        output_lines.append("Entities:")
        for node in nodes:
            if not node or not node.id:
                continue
            
            node_id = node.id
            node_label = node.label or node.type or ""
            node_desc = node.description or ""
            
            if node_desc:
                entity_line = f"- {node_id} ({node_label}): {node_desc}"
            else:
                entity_line = f"- {node_id} ({node_label})"
            
            output_lines.append(entity_line)
        
        # === RELATIONS SECTION ===
        output_lines.append("\nRelations:")
        relation_types = {}
        
        # Collect distinct relationship types and their descriptions
        for edge in edges:
            if not edge or not edge.type:
                continue
                
            rel_type = edge.type
            rel_label = edge.label or edge.type
            rel_desc = edge.description or ""
            
            # Use the first description found for each relation type
            if rel_type not in relation_types:
                relation_types[rel_type] = {
                    'label': rel_label,
                    'description': rel_desc
                }
            elif not relation_types[rel_type]['description'] and rel_desc:
                # Update with description if we didn't have one before
                relation_types[rel_type]['description'] = rel_desc
        
        # Output distinct relation types
        for rel_type, rel_info in relation_types.items():
            rel_label = rel_info['label']
            rel_desc = rel_info['description']
            
            if rel_desc:
                relation_line = f"- {rel_type} ({rel_label}):\n   DESCRIPTION: {rel_desc}"
            else:
                relation_line = f"- {rel_type} ({rel_label})"
            
            output_lines.append(relation_line)
        
        # === STRUCTURE SECTION ===
        output_lines.append("\nStructure:")
        
        for node in nodes:
            if not node or not node.id:
                continue
            
            node_id = node.id
            node_label = node.label or node.type or ""
            node_desc = node.description or ""
            
            # Add entity header
            if node_desc:
                entity_header = f"- {node_id} ({node_label}):\n   DESCRIPTION: {node_desc}"
            else:
                entity_header = f"- {node_id} ({node_label})"
            
            output_lines.append(entity_header)
            
            # Add RDF triples in proper format - only relationships and properties
            entity_triples = []
            
            # Add other properties (excluding standard ones that are part of the entity definition)
            #for key, value in node.properties.items():
            #    if key not in ['id', 'type', 'label', 'description'] and value is not None:
            #        # For properties, we use the key as property_id and assume key is also the label
            #        property_label = key  # This could be enhanced to map property IDs to labels
            #        if isinstance(value, str):
            #            entity_triples.append(f'  <{node_id}> <{key} ({property_label})> "{value}" .')
            #        else:
            #            entity_triples.append(f'  <{node_id}> <{key} ({property_label})> {value} .')
            
            # Add outgoing relationships
            for edge in edges:
                if edge.source_id == node_id:
                    target_label = next((n.label for n in nodes if n.id == edge.target_id), "")
                    rel_label = edge.label or edge.type
                    entity_triples.append(f"  <{node_id}> <{edge.type} ({rel_label})> <{edge.target_id} ({target_label})> .")
            
            # Add incoming relationships as comments for context
            incoming_rels = [edge for edge in edges if edge.target_id == node_id]
            if incoming_rels:
                entity_triples.append("   RELATIONS:")
                for edge in incoming_rels:
                    source_label = next((n.label for n in nodes if n.id == edge.source_id), "")
                    rel_label = edge.label or edge.type
                    entity_triples.append(f"   # <{edge.source_id} ({source_label})> <{edge.type} ({rel_label})> <{node_id}> .")
            
            if entity_triples:
                output_lines.extend(entity_triples)
            else:
                output_lines.append("   # No additional triples")
        
        return "\n".join(output_lines) if output_lines else None

    async def get_subgraph(self, entity_ids: List[str], max_depth: int = 1) -> Graph:
        """
        Get a subgraph containing specified entities and their connections.
        
        Args:
            entity_ids: List of entity IDs to include
            max_depth: Maximum depth of connections to include
            
        Returns:
            Graph object containing nodes and relationships
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        return await self.db.get_subgraph(entity_ids, max_depth)
    
    async def find_path(self, from_entity_id: str, to_entity_id: str, 
                       max_length: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        Find a path between two entities.
        
        Args:
            from_entity_id: Source entity ID
            to_entity_id: Target entity ID
            max_length: Maximum path length
            
        Returns:
            List representing the path or None if no path found
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        query = f"""
        MATCH path = shortestPath((a {{id: $from_id}})-[*1..{max_length}]-(b {{id: $to_id}}))
        RETURN nodes(path) as nodes, relationships(path) as relationships
        """
        
        result = await self.db.execute_query(query, {
            'from_id': from_entity_id,
            'to_id': to_entity_id
        })
        
        if result:
            path_data = result[0]
            path = []
            nodes_data = path_data.get('nodes', [])
            relationships_data = path_data.get('relationships', [])
            
            for i, node_data in enumerate(nodes_data):
                path_item = {'node': dict(node_data)}
                if i < len(relationships_data):
                    path_item['relationship'] = dict(relationships_data[i])
                path.append(path_item)
            
            logger.info(f"Found path from {from_entity_id} to {to_entity_id} with {len(nodes_data)} nodes")
            return path
        
        return None
    
    # === Advanced Operations ===
    
    async def execute_custom_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a custom Cypher query with notifications.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Dict containing query results, notifications, and metadata
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        # Use the new method that includes notifications
        if hasattr(self.db, 'execute_query_with_notifications'):
            return await self.db.execute_query_with_notifications(query, parameters)
        else:
            # Fallback for databases that don't support notifications
            records = await self.db.execute_query(query, parameters)
            return {
                'records': records,
                'notifications': [],
                'query': query
            }
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        stats = await self.db.get_database_stats()
        logger.info(f"Graph statistics: {stats}")
        return stats
    
    async def clear_graph(self) -> bool:
        """
        Clear all data from the knowledge graph. Use with extreme caution!
        
        Returns:
            bool: True if successful
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        logger.warning("Clearing entire knowledge graph!")
        return await self.db.clear_database()
    
    # === Export and Serialization Methods ===
    
    async def export_to_json(self) -> Dict[str, Any]:
        """
        Export the entire knowledge graph to JSON format.
        
        Returns:
            Dictionary containing nodes and edges in JSON-serializable format
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        graph_data = await self.get_whole_graph()
        if not graph_data:
            return {"nodes": [], "edges": []}
        
        return graph_data.to_json()
    
    async def export_to_file(self, file_path: Union[str, Path], format: str = "json") -> None:
        """
        Export the knowledge graph to a file.
        
        Args:
            file_path: Path to save the file
            format: Export format ('json', 'cypher', 'graphml')
        """
        import json
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            data = await self.export_to_json()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "cypher":
            await self._export_to_cypher(file_path)
        
        elif format.lower() == "graphml":
            await self._export_to_graphml(file_path)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Knowledge graph exported to {file_path} in {format} format")
    
    async def _export_to_cypher(self, file_path: Path) -> None:
        """Export graph as Cypher CREATE statements."""
        
        graph_data = await self.get_whole_graph()
        if not graph_data:
            with open(file_path, 'w') as f:
                f.write("// Empty graph\n")
            return
        
        cypher_statements = []
        
        # Generate CREATE statements for nodes
        for node in graph_data.nodes:
            props = []
            for key, value in node.properties.items():
                if isinstance(value, str):
                    props.append(f'{key}: "{value}"')
                else:
                    props.append(f'{key}: {value}')
            
            if props:
                props_str = ", ".join(props)
                cypher_statements.append(
                    f'CREATE (:{node.type} {{id: "{node.id}", label: "{node.label}", '
                    f'description: "{node.description or ""}", {props_str}}})'
                )
            else:
                cypher_statements.append(
                    f'CREATE (:{node.type} {{id: "{node.id}", label: "{node.label}", '
                    f'description: "{node.description or ""}"}})'
                )
        
        # Generate CREATE statements for relationships
        for edge in graph_data.edges:
            props = []
            for key, value in edge.properties.items():
                if isinstance(value, str):
                    props.append(f'{key}: "{value}"')
                else:
                    props.append(f'{key}: {value}')
            
            props_str = ", " + ", ".join(props) if props else ""
            cypher_statements.append(
                f'MATCH (source {{id: "{edge.source_id}"}}), (target {{id: "{edge.target_id}"}}) '
                f'CREATE (source)-[:{edge.type}{props_str}]->(target)'
            )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("// Neo4j Cypher export\n")
            f.write("// Generated by RoboData Knowledge Graph\n\n")
            for statement in cypher_statements:
                f.write(statement + ";\n")
    
    async def _export_to_graphml(self, file_path: Path) -> None:
        """Export graph as GraphML format."""
        
        graph_data = await self.get_whole_graph()
        if not graph_data:
            # Create empty GraphML
            with open(file_path, 'w') as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
                f.write('  <graph id="RoboDataGraph" edgedefault="directed">\n')
                f.write('  </graph>\n')
                f.write('</graphml>\n')
            return
        
        # Convert to NetworkX and export as GraphML
        import networkx as nx
        
        # Create NetworkX graph
        nx_graph = nx.DiGraph()
        
        # Add nodes
        for node in graph_data.nodes:
            # Prepare node attributes, avoiding duplicate description
            node_attrs = {
                "label": node.label,
                "node_type": node.type,
                **node.properties
            }
            # Only add description if it's not already in properties
            if "description" not in node_attrs:
                node_attrs["description"] = node.description or ""
            
            nx_graph.add_node(node.id, **node_attrs)
        
        # Add edges
        for edge in graph_data.edges:
            nx_graph.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.type,
                **edge.properties
            )
        
        # Write GraphML
        nx.write_graphml(nx_graph, file_path)
    
    async def create_backup(self, backup_path: Union[str, Path]) -> None:
        """
        Create a complete backup of the knowledge graph.
        
        Args:
            backup_path: Directory path to store backup files
        """
        from datetime import datetime
        
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export in multiple formats
        await self.export_to_file(backup_path / f"graph_backup_{timestamp}.json", "json")
        await self.export_to_file(backup_path / f"graph_backup_{timestamp}.cypher", "cypher")
        
        # Also export statistics
        stats = await self.get_graph_statistics()
        stats_file = backup_path / f"graph_stats_{timestamp}.json"
        
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Knowledge graph backup created in {backup_path}")

    # === Utility Methods ===
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# Global instance for easy access
_global_graph: Optional[KnowledgeGraph] = None


def get_knowledge_graph(retain_existing_data: bool = False) -> KnowledgeGraph:
    """Get or create the global knowledge graph instance."""
    global _global_graph
    if _global_graph is None:
        _global_graph = KnowledgeGraph(retain_existing_data=retain_existing_data)
    return _global_graph


async def create_knowledge_graph(db_interface: Optional[DatabaseInterface] = None, retain_existing_data: bool = False) -> KnowledgeGraph:
    """
    Create and connect a new knowledge graph instance.
    
    Args:
        db_interface: Optional database interface to use
        retain_existing_data: If False, clears the database on connection. Default: False.
        
    Returns:
        Connected KnowledgeGraph instance
    """
    graph = KnowledgeGraph(db_interface, retain_existing_data=retain_existing_data)
    await graph.connect()
    return graph


# Compatibility aliases for existing code
GraphDatabase = DatabaseInterface  # Abstract base class
Neo4jGraph = Neo4jInterface  # Specific implementation


if __name__ == "__main__":
    async def test_knowledge_graph():
        """Test the knowledge graph functionality."""
        print("=== Testing Knowledge Graph ===\n")
        
        try:
            # Test with context manager
            async with KnowledgeGraph() as graph:
                # Test 1: Add entities
                print("1. Adding entities...")
                person_node = Node("person_alice", "Person", label="Alice", properties={
                    "name": "Alice", "age": 30, "occupation": "Engineer"
                })
                company_node = Node("company_techcorp", "Company", label="TechCorp", properties={
                    "name": "TechCorp", "industry": "Technology", "founded": 2010
                })
                person_id = await graph.add_entity(person_node)
                company_id = await graph.add_entity(company_node)
                print(f"   Added person: {person_id}")
                print(f"   Added company: {company_id}")
                
                # Test 2: Add relationship
                print("\n2. Adding relationship...")
                work_edge = Edge(person_id, company_id, "WORKS_FOR", label="Works For", properties={
                    "since": "2020", "role": "Senior Engineer"
                })
                rel_id = await graph.add_relationship(work_edge)
                print(f"   Added relationship: {rel_id}")
                
                # Test 3: Get entity
                print("\n3. Getting entity...")
                person = await graph.get_entity(person_id)
                if person:
                    print(f"   Retrieved: {person.label}, age: {person.properties.get('age')}")
                
                # Test 4: Find entities
                print("\n4. Finding entities...")
                people = await graph.find_entities("Person")
                print(f"   Found {len(people)} people")
                
                # Test 5: Get neighbors
                print("\n5. Getting neighbors...")
                neighbors = await graph.get_neighbors(person_id)
                print(f"   Found {len(neighbors)} neighbors")
                if neighbors:
                    print(f"   First neighbor: {neighbors[0].label}")
                
                # Test 6: Get statistics
                print("\n6. Getting graph statistics...")
                stats = await graph.get_graph_statistics()
                print(f"   Nodes: {stats.get('node_count', 'N/A')}")
                print(f"   Relationships: {stats.get('relationship_count', 'N/A')}")
                
                # Test 7: Custom query
                print("\n7. Custom query...")
                result = await graph.execute_custom_query(
                    "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name as person, c.name as company"
                )
                records = result.get('records', []) if isinstance(result, dict) else result
                print(f"   Query results: {len(records)}")
                if records:
                    print(f"   First result: {records[0]}")
                    
                # Show notifications if any
                notifications = result.get('notifications', []) if isinstance(result, dict) else []
                if notifications:
                    print(f"   Database notifications: {len(notifications)}")
                    for notification in notifications:
                        print(f"     - {notification.get('severity', 'UNKNOWN')}: {notification.get('title', 'No title')}")
                
                # Test 8: Cleanup
                print("\n8. Cleaning up...")
                await graph.remove_entity(person_id)
                await graph.remove_entity(company_id)
                print("   Cleaned up test data")
            
            print("\n=== Knowledge Graph tests completed ===")
            
        except Exception as e:
            print(f"Error during testing: {e}")
            print("Make sure Neo4j is running and accessible")
    
    asyncio.run(test_knowledge_graph())
