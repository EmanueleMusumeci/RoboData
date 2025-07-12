"""
Neo4j Interface for RoboData Knowledge Base

This module provides a high-level interface to Neo4j graph database operations,
abstracting away the low-level database interactions and providing a clean API
for the graph.py module.
"""

from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
import logging
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from neo4j import AsyncDriver

try:
    from neo4j import GraphDatabase, Driver, AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

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
    
    # Node operations
    @abstractmethod
    async def create_node(self, labels: List[str], properties: Dict[str, Any]) -> Optional[str]:
        """Create a node with given labels and properties."""
        pass
    
    @abstractmethod
    async def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
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
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find nodes by labels and/or properties."""
        pass
    
    # Relationship operations
    @abstractmethod
    async def create_relationship(self, from_node_id: str, to_node_id: str, 
                                relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a relationship between two nodes."""
        pass
    
    @abstractmethod
    async def get_relationship_by_id(self, rel_id: str) -> Optional[Dict[str, Any]]:
        """Get a relationship by its ID."""
        pass
    
    @abstractmethod
    async def delete_relationship(self, rel_id: str) -> bool:
        """Delete a relationship by its ID."""
        pass
    
    @abstractmethod
    async def get_node_relationships(self, node_id: str, direction: str = "both", 
                                   relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get relationships for a node."""
        pass
    
    # Graph operations
    @abstractmethod
    async def get_subgraph(self, node_ids: List[str], max_depth: int = 1) -> Dict[str, Any]:
        """Get a subgraph containing specified nodes and their connections."""
        pass
    
    @abstractmethod
    async def get_all_nodes_and_relationships(self) -> Dict[str, Any]:
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
                result = await session.run(query, parameters)
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
                except Exception:
                    # Ignore summary processing errors
                    pass
                
                return records
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
    
    async def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by its ID."""
        query = """
        MATCH (n {id: $node_id})
        RETURN n, labels(n) as labels
        """
        
        result = await self.execute_query(query, {'node_id': node_id})
        if result:
            node = dict(result[0]['n'])
            node['labels'] = result[0]['labels']
            return node
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
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
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
            node = dict(record['n'])
            node['labels'] = record['labels']
            nodes.append(node)
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
    
    async def get_relationship_by_id(self, rel_id: str) -> Optional[Dict[str, Any]]:
        """Get a relationship by its ID."""
        query = """
        MATCH (a)-[r {id: $rel_id}]->(b)
        RETURN r, type(r) as type, a.id as from_id, b.id as to_id
        """
        
        result = await self.execute_query(query, {'rel_id': rel_id})
        if result:
            relationship = dict(result[0]['r'])
            relationship['type'] = result[0]['type']
            relationship['from_id'] = result[0]['from_id']
            relationship['to_id'] = result[0]['to_id']
            return relationship
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
                                   relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
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
        
        query += " RETURN r, type(r) as rel_type, neighbor, labels(neighbor) as neighbor_labels"
        
        result = await self.execute_query(query, {'node_id': node_id})
        relationships = []
        for record in result:
            rel = dict(record['r'])
            rel['type'] = record['rel_type']
            
            neighbor = dict(record['neighbor'])
            neighbor['labels'] = record['neighbor_labels']
            
            relationships.append({
                'relationship': rel,
                'neighbor': neighbor
            })
        return relationships
    
    # === Graph Operations ===
    
    async def get_all_nodes_and_relationships(self) -> Dict[str, Any]:
        """Get all nodes and relationships in the database."""
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN COLLECT(DISTINCT n) AS nodes, COLLECT(DISTINCT r) AS relationships
        """
        result = await self.execute_query(query)
        if result and result[0]:
            # Handle case where nodes/relationships might be null or empty
            nodes_raw = result[0].get('nodes') or []
            relationships_raw = result[0].get('relationships') or []
            
            # Filter out null values and convert to dict
            nodes = [dict(node) for node in nodes_raw if node is not None]
            relationships = [dict(rel) for rel in relationships_raw if rel is not None]
        else:
            nodes = []
            relationships = []
        return {
            'nodes': nodes,
            'relationships': relationships
        }

    async def get_subgraph(self, node_ids: List[str], max_depth: int = 1) -> Dict[str, Any]:
        """Get a subgraph containing specified nodes and their connections."""
        if not node_ids:
            return {'nodes': [], 'relationships': []}
        
        # Get nodes and their relationships up to max_depth
        query = """
        MATCH path = (start)-[*0..{}]-(connected)
        WHERE start.id IN $node_ids
        WITH DISTINCT start, connected, relationships(path) as rels
        RETURN DISTINCT start, connected, rels, labels(start) as start_labels, labels(connected) as connected_labels
        """.format(max_depth)
        
        result = await self.execute_query(query, {'node_ids': node_ids})
        
        nodes = {}
        relationships = []
        
        for record in result:
            # Add start node
            start_node = dict(record['start'])
            start_node['labels'] = record['start_labels']
            nodes[start_node['id']] = start_node
            
            # Add connected node
            connected_node = dict(record['connected'])
            connected_node['labels'] = record['connected_labels']
            nodes[connected_node['id']] = connected_node
            
            # Add relationships
            for rel in record['rels']:
                rel_dict = dict(rel)
                relationships.append(rel_dict)
        
        return {
            'nodes': list(nodes.values()),
            'relationships': relationships
        }
    
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
