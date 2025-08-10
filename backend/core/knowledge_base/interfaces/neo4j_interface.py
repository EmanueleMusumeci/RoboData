"""
Neo4j Interface for RoboData Knowledge Base

This module provides a high-level interface to Neo4j graph database operations,
abstracting away the low-level database interactions and providing a clean API
for the graph.py module.
"""

from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING, cast
import logging
from abc import ABC, abstractmethod

from ..schema import Node, Edge, Graph

if TYPE_CHECKING:
    from neo4j import AsyncDriver, Query
    from typing_extensions import LiteralString

try:
    from neo4j import GraphDatabase, Driver, AsyncGraphDatabase, Query
    from typing_extensions import LiteralString
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    # Define dummy classes for type hinting if neo4j is not installed
    class AsyncGraphDatabase: pass
    class Query: pass
    class Driver: pass
    class LiteralString: pass

logger = logging.getLogger(__name__)


class DatabaseInterface(ABC):
    """Abstract interface for database operations."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the database."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if database is connected."""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        pass
    
    async def execute_query_with_notifications(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a query and return results with notifications. Default implementation for backwards compatibility."""
        records = await self.execute_query(query, parameters)
        return {
            'records': records,
            'notifications': [],
            'query': query
        }
    
    # Node operations
    @abstractmethod
    async def create_node(self, labels: List[str], properties: Dict[str, Any]) -> Optional[str]:
        """Create a node with given labels and properties."""
        pass
    
    @abstractmethod
    async def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID."""
        pass
    
    @abstractmethod
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        pass
    
    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its relationships."""
        pass
    
    @abstractmethod
    async def find_nodes(self, labels: Optional[List[str]] = None, properties: Optional[Dict[str, Any]] = None, 
                        limit: Optional[int] = None) -> List[Node]:
        """Find nodes by labels and/or properties."""
        pass
    
    # Relationship operations
    @abstractmethod
    async def create_relationship(self, from_node_id: str, to_node_id: str, 
                                relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a relationship between two nodes."""
        pass
    
    @abstractmethod
    async def get_relationship_by_id(self, rel_id: str) -> Optional[Edge]:
        """Get a relationship by its ID."""
        pass
    
    @abstractmethod
    async def delete_relationship(self, rel_id: str) -> bool:
        """Delete a relationship by its ID."""
        pass
    
    @abstractmethod
    async def get_node_relationships(self, node_id: str, direction: str = "both", 
                                   relationship_type: Optional[str] = None) -> List[Edge]:
        """Get relationships for a node."""
        pass
    
    @abstractmethod
    async def find_relationships(self, relationship_type: Optional[str] = None, 
                                 properties: Optional[Dict[str, Any]] = None, 
                                 limit: Optional[int] = None) -> List[Edge]:
        """Find relationships by type and/or properties."""
        pass
    
    # Graph operations
    @abstractmethod
    async def get_subgraph(self, node_ids: List[str], max_depth: int = 1) -> Graph:
        """Get a subgraph containing specified nodes and their connections."""
        pass
    
    @abstractmethod
    async def get_all_nodes_and_relationships(self) -> Graph:
        """Get all nodes and relationships in the database."""
        pass
    
    @abstractmethod
    async def clear_database(self) -> bool:
        """Clear all nodes and relationships from the database."""
        pass
    
    @abstractmethod
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass


class Neo4jInterface(DatabaseInterface):
    """Neo4j-specific database interface implementation."""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j interface.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            username: Database username
            password: Database password
            database: Database name (default: "neo4j")
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
        
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional["AsyncDriver"] = None
        self._connected = False
        
    async def connect(self) -> None:
        """Connect to Neo4j database."""
        try:
            if NEO4J_AVAILABLE:
                self.driver = AsyncGraphDatabase.driver(
                    self.uri, 
                    auth=(self.username, self.password)
                )
                # Test connection
                await self.driver.verify_connectivity()
            self._connected = True
            logger.info(f"Connected to Neo4j database at {self.uri}")
        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Could not connect to Neo4j: {e}")
    
    async def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            self._connected = False
            logger.info("Closed Neo4j connection")
    
    async def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected and self.driver is not None
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        if not await self.is_connected():
            raise RuntimeError("Database not connected")
        
        if parameters is None:
            parameters = {}
        
        if self.driver is None:
            raise RuntimeError("Driver not initialized")
            
        async with self.driver.session(database=self.database) as session:
            try:
                literal_query = cast(LiteralString, query)
                result = await session.run(literal_query, parameters)
                records = []
                async for record in result:
                    records.append(dict(record))        

                # Get summary and check for notifications
                try:
                    summary = await result.consume()
                    if summary and hasattr(summary, 'notifications') and summary.notifications:
                        warning_count = len([n for n in summary.notifications if getattr(n, 'severity', None) == 'WARNING'])
                        if warning_count > 0:
                            logger.debug(f"Query returned {warning_count} warnings (likely missing labels/properties/relationships)")
                except Exception as e:
                    logger.warning(f"Could not get query summary: {e}")
                
                return records
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                logger.error(f"Query: {query}")
                logger.error(f"Parameters: {parameters}")
                raise

    async def execute_query_with_notifications(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a Cypher query and return results with notifications for tool usage."""
        if not await self.is_connected():
            raise RuntimeError("Database not connected")
        
        if parameters is None:
            parameters = {}
        
        if self.driver is None:
            raise RuntimeError("Driver not initialized")
            
        async with self.driver.session(database=self.database) as session:
            try:
                literal_query = cast(LiteralString, query)
                result = await session.run(literal_query, parameters)
                records = []
                async for record in result:
                    records.append(dict(record))
                

                # Get summary and check for notifications
                notifications = []
                try:
                    summary = await result.consume()
                    if summary and hasattr(summary, 'notifications') and summary.notifications:
                        for notification in summary.notifications:
                            # Extract notification data - notifications are dict-like objects
                            notification_dict = {
                                'severity': notification.get('severity', 'UNKNOWN'),
                                'code': notification.get('code', 'UNKNOWN'),
                                'category': notification.get('category', 'UNKNOWN'),
                                'title': notification.get('title', 'UNKNOWN'),
                                'description': notification.get('description', 'UNKNOWN'),
                                'position': notification.get('position', {})
                            }
                            notifications.append(notification_dict)
                        
                        warning_count = len([n for n in notifications if n.get('severity') == 'WARNING'])
                        if warning_count > 0:
                            logger.debug(f"Query returned {warning_count} warnings (likely missing labels/properties/relationships)")
                except Exception as e:
                    logger.warning(f"Could not get query summary: {e}")
                
                return {
                    'records': records,
                    'notifications': notifications,
                    'query': query
                }
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                logger.error(f"Query: {query}")
                logger.error(f"Parameters: {parameters}")
                raise
    
    # === Node Operations ===
    
    async def create_node(self, labels: List[str], properties: Dict[str, Any]) -> Optional[str]:
        """
        Create a node with given labels and properties.
        
        Args:
            labels: List of node labels
            properties: Node properties
            
        Returns:
            Optional[str]: Generated node ID or None if failed
        """
        # Generate unique ID if not provided
        if 'id' not in properties:
            import uuid
            properties['id'] = str(uuid.uuid4())
        
        labels_str = ':'.join(labels)
        query = f"""
        CREATE (n:{labels_str} $properties)
        RETURN n.id as node_id
        """
        
        result = await self.execute_query(query, {'properties': properties})
        return result[0]['node_id'] if result else None
    
    async def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID."""
        query = """
        MATCH (n {id: $node_id})
        RETURN n, labels(n) as labels
        """
        
        result = await self.execute_query(query, {'node_id': node_id})
        if result:
            node_data = dict(result[0]['n'])
            labels = result[0].get('labels') or []
            node_id_val = node_data.get('id', node_id)
            if not node_id_val:
                return None
            return Node(
                node_id=node_id_val,
                node_type=next((l for l in labels if l != 'Entity'), 'Entity'),
                label=node_data.get('label', ''),
                description=node_data.get('description', ''),
                properties=node_data
            )
        return None
    
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        query = """
        MATCH (n {id: $node_id})
        SET n += $properties
        RETURN count(n) as updated_count
        """
        
        result = await self.execute_query(query, {
            'node_id': node_id,
            'properties': properties
        })
        return result[0]['updated_count'] > 0 if result else False
    
    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its relationships."""
        query = """
        MATCH (n {id: $node_id})
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """
        
        result = await self.execute_query(query, {'node_id': node_id})
        return result[0]['deleted_count'] > 0 if result else False
    
    async def find_nodes(self, labels: Optional[List[str]] = None, properties: Optional[Dict[str, Any]] = None, 
                        limit: Optional[int] = None) -> List[Node]:
        """Find nodes by labels and/or properties."""
        # Build query dynamically
        if labels:
            labels_str = ':'.join(labels)
            query = f"MATCH (n:{labels_str})"
        else:
            query = "MATCH (n)"
        
        where_conditions = []
        parameters = {}
        
        if properties:
            for key, value in properties.items():
                param_name = f"prop_{key}"
                where_conditions.append(f"n.{key} = ${param_name}")
                parameters[param_name] = value
        
        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)
        
        query += " RETURN n, labels(n) as labels"
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = await self.execute_query(query, parameters)
        nodes = []
        for record in result:
            node_data = dict(record['n'])
            labels = record.get('labels') or []
            node_id_val = node_data.get('id')
            if not node_id_val:
                continue
            nodes.append(Node(
                node_id=node_id_val,
                node_type=next((l for l in labels if l != 'Entity'), 'Entity'),
                label=node_data.get('label', ''),
                description=node_data.get('description', ''),
                properties=node_data
            ))
        return nodes
    
    # === Relationship Operations ===
    
    async def create_relationship(self, from_node_id: str, to_node_id: str, 
                                relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a relationship between two nodes."""
        if properties is None:
            properties = {}
        
        # Generate unique ID for the relationship
        import uuid
        rel_id = str(uuid.uuid4())
        properties['id'] = rel_id
        
        query = f"""
        MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
        CREATE (a)-[r:{relationship_type} $properties]->(b)
        RETURN r.id as rel_id
        """
        
        result = await self.execute_query(query, {
            'from_id': from_node_id,
            'to_id': to_node_id,
            'properties': properties
        })
        return result[0]['rel_id'] if result else None
    
    async def get_relationship_by_id(self, rel_id: str) -> Optional[Edge]:
        """Get a relationship by its ID."""
        query = """
        MATCH (a)-[r {id: $rel_id}]->(b)
        RETURN r, type(r) as type, a.id as from_id, b.id as to_id
        """
        
        result = await self.execute_query(query, {'rel_id': rel_id})
        if result:
            rel_data = dict(result[0]['r'])
            return Edge(
                source_id=result[0]['from_id'],
                target_id=result[0]['to_id'],
                relationship_type=result[0]['type'],
                label=rel_data.get('label', result[0]['type']),
                properties=rel_data
            )
        return None
    
    async def delete_relationship(self, rel_id: str) -> bool:
        """Delete a relationship by its ID."""
        query = """
        MATCH ()-[r {id: $rel_id}]->()
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        result = await self.execute_query(query, {'rel_id': rel_id})
        return result[0]['deleted_count'] > 0 if result else False
    
    async def get_node_relationships(self, node_id: str, direction: str = "both", 
                                   relationship_type: Optional[str] = None) -> List[Edge]:
        """Get relationships for a node."""
        # Build query based on direction
        if direction == "incoming":
            if relationship_type:
                query = f"MATCH (neighbor)-[r:{relationship_type}]->(n {{id: $node_id}})"
            else:
                query = "MATCH (neighbor)-[r]->(n {id: $node_id})"
        elif direction == "outgoing":
            if relationship_type:
                query = f"MATCH (n {{id: $node_id}})-[r:{relationship_type}]->(neighbor)"
            else:
                query = "MATCH (n {id: $node_id})-[r]->(neighbor)"
        else:  # both
            if relationship_type:
                query = f"MATCH (n {{id: $node_id}})-[r:{relationship_type}]-(neighbor)"
            else:
                query = "MATCH (n {id: $node_id})-[r]-(neighbor)"
        
        query += " RETURN r, type(r) as rel_type, startNode(r).id as from_id, endNode(r).id as to_id"
        
        result = await self.execute_query(query, {'node_id': node_id})
        relationships = []
        for record in result:
            rel_data = dict(record['r'])
            relationships.append(Edge(
                source_id=record['from_id'],
                target_id=record['to_id'],
                relationship_type=record['rel_type'],
                label=rel_data.get('label', record['rel_type']),
                properties=rel_data
            ))
        return relationships
    
    async def find_relationships(self, relationship_type: Optional[str] = None, 
                                 properties: Optional[Dict[str, Any]] = None, 
                                 limit: Optional[int] = None) -> List[Edge]:
        """Find relationships by type and/or properties."""
        # Build query dynamically
        rel_type_str = f":`{relationship_type}`" if relationship_type else ""
        query = f"MATCH (a)-[r{rel_type_str}]->(b)"
        
        where_conditions = []
        parameters = {}
        
        if properties:
            for key, value in properties.items():
                param_name = f"prop_{key}"
                where_conditions.append(f"r.{key} = ${param_name}")
                parameters[param_name] = value
        
        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)
        
        query += " RETURN r, type(r) as rel_type, startNode(r).id as from_id, endNode(r).id as to_id"
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = await self.execute_query(query, parameters)
        relationships = []
        for record in result:
            rel_data = dict(record['r'])
            relationships.append(Edge(
                source_id=record['from_id'],
                target_id=record['to_id'],
                relationship_type=record['rel_type'],
                label=rel_data.get('label', record['rel_type']),
                properties=rel_data
            ))
        return relationships

    async def get_neighbors(self, node_id: str, relationship_type: Optional[str] = None, limit: int = 20) -> List['Node']:
        """Get neighbors of a node."""
        query = f"""
        MATCH (n)-[r]-(neighbor)
        WHERE n.id = $node_id
        { "AND type(r) = $relationship_type" if relationship_type else "" }
        RETURN neighbor, r as relationship
        ORDER BY r.weight DESC, r.rank DESC
        LIMIT $limit
        """
        parameters = {"node_id": node_id, "limit": limit}
        if relationship_type:
            parameters["relationship_type"] = relationship_type
            
        results = await self.execute_query(query, parameters)
        
        neighbors = []
        for record in results:
            node_data = record['neighbor']
            relationship_data = record['relationship']
            
            # Add relationship properties to the node properties for context
            props = dict(node_data.items())
            props['relationship_properties'] = dict(relationship_data.items())

            neighbors.append(Node(
                node_id=node_data.get('id'),
                node_type=list(node_data.labels)[0] if node_data.labels else 'Unknown',
                label=node_data.get('label', ''),
                description=node_data.get('description', ''),
                properties=props
            ))
        return neighbors

    # === Graph Operations ===
    
    async def get_all_nodes_and_relationships(self) -> Graph:
        """Get all nodes and relationships in the database."""
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN COLLECT(DISTINCT {data: n, labels: labels(n)}) AS nodes, COLLECT(DISTINCT {data: r, type: type(r), from: startNode(r).id, to: endNode(r).id}) AS relationships
        """
        result = await self.execute_query(query)
        
        graph = Graph()
        
        if result and result[0]:
            nodes_raw = result[0].get('nodes') or []
            for node_entry in nodes_raw:
                if node_entry and node_entry.get('data'):
                    node_data = dict(node_entry['data'])
                    labels = node_entry.get('labels') or []
                    node_id_val = node_data.get('id')
                    if not node_id_val:
                        continue
                    graph.add_node(Node(
                        node_id=node_id_val,
                        node_type=next((l for l in labels if l != 'Entity'), 'Entity'),
                        label=node_data.get('label', ''),
                        description=node_data.get('description', ''),
                        properties=node_data
                    ))

            relationships_raw = result[0].get('relationships') or []
            for rel_entry in relationships_raw:
                if rel_entry and rel_entry.get('data'):
                    rel_data = dict(rel_entry['data'])
                    graph.add_edge(Edge(
                        source_id=rel_entry['from'],
                        target_id=rel_entry['to'],
                        relationship_type=rel_entry['type'],
                        label=rel_data.get('label', rel_entry['type']),
                        properties=rel_data
                    ))
        return graph

    async def get_subgraph(self, node_ids: List[str], max_depth: int = 1) -> Graph:
        """Get a subgraph containing specified nodes and their connections."""
        if not node_ids:
            return Graph()
        
        # Get nodes and their relationships up to max_depth
        # Use double curly braces to escape Cypher map syntax for Python .format()
        query = """
        MATCH path = (start)-[*0..{max_depth}]-(connected)
        WHERE start.id IN $node_ids
        WITH start, connected, relationships(path) as rels, labels(start) as start_labels, labels(connected) as connected_labels
        UNWIND rels as r
        RETURN DISTINCT 
            {{data: start, labels: start_labels}} as start_node,
            {{data: connected, labels: connected_labels}} as connected_node,
            {{data: r, type: type(r), from: startNode(r).id, to: endNode(r).id}} as relationship
        """.format(max_depth=max_depth)
        
        result = await self.execute_query(query, {'node_ids': node_ids})
        
        graph = Graph()
        
        for record in result:
            # Add start node
            start_node_entry = record['start_node']
            if start_node_entry and start_node_entry.get('data'):
                start_node_data = dict(start_node_entry['data'])
                start_labels = start_node_entry.get('labels') or []
                node_id_val = start_node_data.get('id')
                if not node_id_val:
                    continue
                graph.add_node(Node(
                    node_id=node_id_val,
                    node_type=next((l for l in start_labels if l != 'Entity'), 'Entity'),
                    label=start_node_data.get('label', ''),
                    description=start_node_data.get('description', ''),
                    properties=start_node_data
                ))

            # Add connected node
            connected_node_entry = record['connected_node']
            if connected_node_entry and connected_node_entry.get('data'):
                connected_node_data = dict(connected_node_entry['data'])
                connected_labels = connected_node_entry.get('labels') or []
                node_id_val = connected_node_data.get('id')
                if not node_id_val:
                    continue
                graph.add_node(Node(
                    node_id=node_id_val,
                    node_type=next((l for l in connected_labels if l != 'Entity'), 'Entity'),
                    label=connected_node_data.get('label', ''),
                    description=connected_node_data.get('description', ''),
                    properties=connected_node_data
                ))

            # Add relationship
            rel_entry = record['relationship']
            if rel_entry and rel_entry.get('data'):
                rel_data = dict(rel_entry['data'])
                graph.add_edge(Edge(
                    source_id=rel_entry['from'],
                    target_id=rel_entry['to'],
                    relationship_type=rel_entry['type'],
                    label=rel_data.get('label', rel_entry['type']),
                    properties=rel_data
                ))
        
        return graph
    
    async def clear_database(self) -> bool:
        """Clear all nodes and relationships from the database. Use with caution!"""
        query = "MATCH (n) DETACH DELETE n RETURN count(n) as deleted_count"
        result = await self.execute_query(query)
        deleted_count = result[0]['deleted_count'] if result else 0
        logger.info(f"Cleared database: deleted {deleted_count} nodes")
        return deleted_count > 0
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        queries = {
            'node_count': "MATCH (n) RETURN count(n) as count",
            'relationship_count': "MATCH ()-[r]->() RETURN count(r) as count",
            'labels': "CALL db.labels() YIELD label RETURN collect(label) as labels",
            'relationship_types': "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
        }
        
        stats = {}
        for stat_name, query in queries.items():
            try:
                result = await self.execute_query(query)
                if stat_name in ['node_count', 'relationship_count']:
                    stats[stat_name] = result[0]['count'] if result else 0
                elif stat_name == 'labels':
                    stats[stat_name] = result[0]['labels'] if result else []
                elif stat_name == 'relationship_types':
                    stats[stat_name] = result[0]['types'] if result else []
            except Exception as e:
                logger.warning(f"Could not get {stat_name}: {e}")
                stats[stat_name] = None
        
        return stats
