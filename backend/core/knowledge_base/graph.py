"""
High-level Graph Database Abstraction for RoboData

This module provides a high-level interface for working with knowledge graphs,
using the Neo4j interface for low-level database operations.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
from abc import ABC, abstractmethod

from ...settings import settings_manager
from .interfaces.neo4j_interface import Neo4jInterface, DatabaseInterface

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
    
    # === Entity Management ===
    
    async def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]) -> str:
        """
        Add an entity (node) to the knowledge graph.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity (e.g., "Person", "WikidataEntity")
            properties: Additional properties for the entity
            
        Returns:
            str: The entity ID
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        # Ensure entity has an ID
        properties['id'] = entity_id
        properties['entity_type'] = entity_type
        
        labels = ["Entity", entity_type]
        node_id = await self.db.create_node(labels, properties)
        logger.info(f"Added entity {entity_id} of type {entity_type}")
        return node_id or entity_id
    
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by its ID.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            Dict containing entity data or None if not found
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
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find entities by type and/or properties.
        
        Args:
            entity_type: Entity type to filter by
            properties: Properties to match
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        labels = ["Entity"]
        if entity_type:
            labels.append(entity_type)
        
        return await self.db.find_nodes(labels, properties, limit)
    
    # === Relationship Management ===
    
    async def add_relationship(self, from_entity_id: str, to_entity_id: str, 
                             relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a relationship between two entities.
        
        Args:
            from_entity_id: Source entity ID
            to_entity_id: Target entity ID
            relationship_type: Type of relationship
            properties: Additional properties for the relationship
            
        Returns:
            str: The relationship ID
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        if properties is None:
            properties = {}
        
        rel_id = await self.db.create_relationship(from_entity_id, to_entity_id, relationship_type, properties)
        logger.info(f"Added relationship {relationship_type} from {from_entity_id} to {to_entity_id}")
        return rel_id or f"{from_entity_id}_{relationship_type}_{to_entity_id}"
    
    async def get_relationship(self, relationship_id: str) -> Optional[Dict[str, Any]]:
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
                                     relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity.
        
        Args:
            entity_id: The entity ID
            direction: "incoming", "outgoing", or "both"
            relationship_type: Filter by relationship type
            
        Returns:
            List of relationships with neighbor information
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        return await self.db.get_node_relationships(entity_id, direction, relationship_type)
    
    # === Graph Operations ===
    
    async def get_neighbors(self, entity_id: str, relationship_type: Optional[str] = None,
                          direction: str = "both") -> List[Dict[str, Any]]:
        """
        Get neighboring entities of a given entity.
        
        Args:
            entity_id: The entity ID
            relationship_type: Filter by relationship type
            direction: "incoming", "outgoing", or "both"
            
        Returns:
            List of neighboring entities
        """
        relationships = await self.get_entity_relationships(entity_id, direction, relationship_type)
        neighbors = []
        for rel_data in relationships:
            neighbor = rel_data.get('neighbor')
            if neighbor:
                neighbor['relationship_info'] = rel_data.get('relationship')
                neighbors.append(neighbor)
        return neighbors
    
    async def get_whole_graph(self) -> Dict[str, Any]:
        """
        Get the entire knowledge graph as nodes and relationships.
        
        Returns:
            Dict containing all nodes and relationships
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        return await self.db.get_all_nodes_and_relationships()

    async def get_subgraph(self, entity_ids: List[str], max_depth: int = 1) -> Dict[str, Any]:
        """
        Get a subgraph containing specified entities and their connections.
        
        Args:
            entity_ids: List of entity IDs to include
            max_depth: Maximum depth of connections to include
            
        Returns:
            Dict containing nodes and relationships
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
            nodes = path_data['nodes']
            relationships = path_data['relationships']
            
            for i, node in enumerate(nodes):
                path_item = {'node': dict(node)}
                if i < len(relationships):
                    path_item['relationship'] = dict(relationships[i])
                path.append(path_item)
            
            logger.info(f"Found path from {from_entity_id} to {to_entity_id} with {len(nodes)} nodes")
            return path
        
        return None
    
    # === Advanced Operations ===
    
    async def execute_custom_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query results
        """
        if not await self.is_connected():
            raise RuntimeError("Graph database not connected")
        
        return await self.db.execute_query(query, parameters)
    
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
    
    # === Context Manager Support ===
    
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
                person_id = await graph.add_entity("person_alice", "Person", {
                    "name": "Alice",
                    "age": 30,
                    "occupation": "Engineer"
                })
                company_id = await graph.add_entity("company_techcorp", "Company", {
                    "name": "TechCorp",
                    "industry": "Technology",
                    "founded": 2010
                })
                print(f"   Added person: {person_id}")
                print(f"   Added company: {company_id}")
                
                # Test 2: Add relationship
                print("\n2. Adding relationship...")
                rel_id = await graph.add_relationship(person_id, company_id, "WORKS_FOR", {
                    "since": "2020",
                    "role": "Senior Engineer"
                })
                print(f"   Added relationship: {rel_id}")
                
                # Test 3: Get entity
                print("\n3. Getting entity...")
                person = await graph.get_entity(person_id)
                if person:
                    print(f"   Retrieved: {person['name']}, age: {person['age']}")
                
                # Test 4: Find entities
                print("\n4. Finding entities...")
                people = await graph.find_entities("Person")
                print(f"   Found {len(people)} people")
                
                # Test 5: Get neighbors
                print("\n5. Getting neighbors...")
                neighbors = await graph.get_neighbors(person_id)
                print(f"   Found {len(neighbors)} neighbors")
                if neighbors:
                    print(f"   First neighbor: {neighbors[0]['name']}")
                
                # Test 6: Get statistics
                print("\n6. Getting graph statistics...")
                stats = await graph.get_graph_statistics()
                print(f"   Nodes: {stats.get('node_count', 'N/A')}")
                print(f"   Relationships: {stats.get('relationship_count', 'N/A')}")
                
                # Test 7: Custom query
                print("\n7. Custom query...")
                results = await graph.execute_custom_query(
                    "MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name as person, c.name as company"
                )
                print(f"   Query results: {len(results)}")
                if results:
                    print(f"   First result: {results[0]}")
                
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
