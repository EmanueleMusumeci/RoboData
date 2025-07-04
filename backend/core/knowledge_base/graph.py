from typing import Dict, Any, List, Optional, Union
import asyncio
from abc import ABC, abstractmethod
import logging

try:
    from neo4j import GraphDatabase as Neo4jDriver, AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

logger = logging.getLogger(__name__)

class GraphDatabase(ABC):
    """Abstract base class for graph database operations."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the database."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    async def add_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Add a node to the graph."""
        pass
    
    @abstractmethod
    async def add_edge(self, from_node_id: str, to_node_id: str, relationship_type: str, properties: Dict[str, Any] = None) -> str:
        """Add an edge between two nodes."""
        pass
    
    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        pass
    
    @abstractmethod
    async def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get an edge by ID."""
        pass
    
    @abstractmethod
    async def remove_node(self, node_id: str) -> bool:
        """Remove a node from the graph."""
        pass
    
    @abstractmethod
    async def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the graph."""
        pass
    
    @abstractmethod
    async def query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a custom query."""
        pass
    
    @abstractmethod
    async def find_nodes(self, label: str = None, properties: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find nodes by label and/or properties."""
        pass

class Neo4jGraph(GraphDatabase):
    """Neo4j implementation of GraphDatabase."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", username: str = "neo4j", password: str = "password"):
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
        
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
    async def connect(self) -> None:
        """Connect to Neo4j database."""
        try:
            self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            await self.driver.verify_connectivity()
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Closed Neo4j connection")
    
    async def add_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Add a node to the graph."""
        if not self.driver:
            raise RuntimeError("Database not connected")
        
        # Generate a unique ID if not provided
        if 'id' not in properties:
            import uuid
            properties['id'] = str(uuid.uuid4())
        
        query = f"""
        CREATE (n:{label} $properties)
        RETURN n.id as node_id
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, properties=properties)
            record = await result.single()
            return record['node_id'] if record else None
    
    async def add_edge(self, from_node_id: str, to_node_id: str, relationship_type: str, properties: Dict[str, Any] = None) -> str:
        """Add an edge between two nodes."""
        if not self.driver:
            raise RuntimeError("Database not connected")
        
        if properties is None:
            properties = {}
        
        # Generate a unique ID for the relationship
        import uuid
        rel_id = str(uuid.uuid4())
        properties['id'] = rel_id
        
        query = f"""
        MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
        CREATE (a)-[r:{relationship_type} $properties]->(b)
        RETURN r.id as rel_id
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, from_id=from_node_id, to_id=to_node_id, properties=properties)
            record = await result.single()
            return record['rel_id'] if record else None
    
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        if not self.driver:
            raise RuntimeError("Database not connected")
        
        query = """
        MATCH (n {id: $node_id})
        RETURN n, labels(n) as labels
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, node_id=node_id)
            record = await result.single()
            
            if record:
                node = dict(record['n'])
                node['labels'] = record['labels']
                return node
            return None
    
    async def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get an edge by ID."""
        if not self.driver:
            raise RuntimeError("Database not connected")
        
        query = """
        MATCH (a)-[r {id: $edge_id}]->(b)
        RETURN r, type(r) as type, a.id as from_id, b.id as to_id
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, edge_id=edge_id)
            record = await result.single()
            
            if record:
                edge = dict(record['r'])
                edge['type'] = record['type']
                edge['from_id'] = record['from_id']
                edge['to_id'] = record['to_id']
                return edge
            return None
    
    async def remove_node(self, node_id: str) -> bool:
        """Remove a node from the graph."""
        if not self.driver:
            raise RuntimeError("Database not connected")
        
        query = """
        MATCH (n {id: $node_id})
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, node_id=node_id)
            record = await result.single()
            return record['deleted_count'] > 0 if record else False
    
    async def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the graph."""
        if not self.driver:
            raise RuntimeError("Database not connected")
        
        query = """
        MATCH ()-[r {id: $edge_id}]->()
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, edge_id=edge_id)
            record = await result.single()
            return record['deleted_count'] > 0 if record else False
    
    async def query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query."""
        if not self.driver:
            raise RuntimeError("Database not connected")
        
        if parameters is None:
            parameters = {}
        
        async with self.driver.session() as session:
            result = await session.run(query, parameters)
            records = []
            async for record in result:
                records.append(dict(record))
            return records
    
    async def find_nodes(self, label: str = None, properties: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find nodes by label and/or properties."""
        if not self.driver:
            raise RuntimeError("Database not connected")
        
        # Build query dynamically
        if label:
            query = f"MATCH (n:{label})"
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
        
        async with self.driver.session() as session:
            result = await session.run(query, parameters)
            nodes = []
            async for record in result:
                node = dict(record['n'])
                node['labels'] = record['labels']
                nodes.append(node)
            return nodes
    
    async def get_node_neighbors(self, node_id: str, relationship_type: str = None, direction: str = "both") -> List[Dict[str, Any]]:
        """Get neighboring nodes of a given node."""
        if not self.driver:
            raise RuntimeError("Database not connected")
        
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
        
        query += " RETURN neighbor, r, type(r) as rel_type, labels(neighbor) as labels"
        
        async with self.driver.session() as session:
            result = await session.run(query, node_id=node_id)
            neighbors = []
            async for record in result:
                neighbor = dict(record['neighbor'])
                neighbor['labels'] = record['labels']
                neighbor['relationship'] = {
                    'type': record['rel_type'],
                    'properties': dict(record['r'])
                }
                neighbors.append(neighbor)
            return neighbors

# Global instance for easy access
graph_db = None

def get_graph_db() -> Neo4jGraph:
    """Get or create the global graph database instance."""
    global graph_db
    if graph_db is None:
        graph_db = Neo4jGraph()
    return graph_db

if __name__ == "__main__":
    async def test_neo4j():
        """Test Neo4j functionality."""
        print("=== Testing Neo4j Graph Database ===\n")
        
        try:
            # Initialize database
            db = Neo4jGraph()
            await db.connect()
            
            # Test 1: Add nodes
            print("1. Adding nodes...")
            person_id = await db.add_node("Person", {"name": "Alice", "age": 30})
            company_id = await db.add_node("Company", {"name": "TechCorp", "industry": "Technology"})
            print(f"   Added Person: {person_id}")
            print(f"   Added Company: {company_id}")
            
            # Test 2: Add edge
            print("\n2. Adding relationship...")
            rel_id = await db.add_edge(person_id, company_id, "WORKS_FOR", {"since": "2020", "role": "Engineer"})
            print(f"   Added relationship: {rel_id}")
            
            # Test 3: Get node
            print("\n3. Getting node...")
            node = await db.get_node(person_id)
            if node:
                print(f"   Retrieved node: {node['name']}, age: {node['age']}")
            
            # Test 4: Find nodes
            print("\n4. Finding nodes...")
            people = await db.find_nodes("Person")
            print(f"   Found {len(people)} people")
            
            # Test 5: Get neighbors
            print("\n5. Getting neighbors...")
            neighbors = await db.get_node_neighbors(person_id)
            print(f"   Found {len(neighbors)} neighbors")
            if neighbors:
                print(f"   First neighbor: {neighbors[0]['name']}")
            
            # Test 6: Custom query
            print("\n6. Custom query...")
            results = await db.query("MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name, c.name")
            print(f"   Query results: {len(results)}")
            
            # Cleanup
            print("\n7. Cleaning up...")
            await db.remove_node(person_id)
            await db.remove_node(company_id)
            print("   Cleaned up test data")
            
            await db.close()
            print("\n=== Neo4j tests completed ===")
            
        except Exception as e:
            print(f"Error during testing: {e}")
            print("Make sure Neo4j is running and accessible")
    
    asyncio.run(test_neo4j())
