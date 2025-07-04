from typing import Dict, Any, List, Optional, Set
from ..toolbox import Tool, ToolDefinition, ToolParameter
from .base import get_entity_info_async, get_entity_info, WikidataEntity
from .datamodel import WikidataStatement


class EntityExplorationTool(Tool):
    """Tool for exploring entity neighbors and relationships."""
    
    def __init__(self):
        super().__init__(
            name="explore_entity",
            description="Explore an entity's direct relationships and properties"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_id",
                    type="string",
                    description="Wikidata entity ID to explore"
                ),
                ToolParameter(
                    name="include_properties",
                    type="array",
                    description="Specific properties to include (empty for all)",
                    required=False,
                    default=[]
                ),
                ToolParameter(
                    name="max_values_per_property",
                    type="integer",
                    description="Maximum values per property",
                    required=False,
                    default=10
                )
            ],
            return_type="dict",
            return_description="Entity information with neighbors and relationships"
        )
    
    async def execute(self, entity_id: str, include_properties: List[str] = None, max_values_per_property: int = 10) -> Dict[str, Any]:
        """Explore an entity's relationships."""
        # Get basic entity info using base.py async function
        entity = await get_entity_info_async(entity_id)
        
        neighbors = {}
        relationships = {}
        
        for prop_id, statements in entity.statements.items():
            if include_properties and prop_id not in include_properties:
                continue
                
            # Limit statements per property
            limited_statements = statements[:max_values_per_property]
            relationships[prop_id] = [stmt.value for stmt in limited_statements]
            
            # Collect entity neighbors from entity references
            for stmt in limited_statements:
                if stmt.is_entity_ref and stmt.entity_type == "item":
                    try:
                        neighbor = await get_entity_info_async(stmt.value)
                        neighbors[stmt.value] = {
                            'id': neighbor.id,
                            'label': neighbor.label,
                            'description': neighbor.description
                        }
                    except:
                        neighbors[stmt.value] = {'id': stmt.value, 'label': stmt.value, 'description': ''}
        
        return {
            'entity': entity.dict(),
            'neighbors': neighbors,
            'relationships': relationships,
            'total_properties': len(entity.statements),
            'neighbor_count': len(neighbors)
        }

'''
class PathFindingTool(Tool):
    """Tool for finding paths between entities."""
    
    def __init__(self):
        super().__init__(
            name="find_path",
            description="Find a path between two Wikidata entities"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="start_entity",
                    type="string",
                    description="Starting entity ID"
                ),
                ToolParameter(
                    name="end_entity",
                    type="string",
                    description="Target entity ID"
                ),
                ToolParameter(
                    name="max_depth",
                    type="integer",
                    description="Maximum search depth",
                    required=False,
                    default=3
                ),
                ToolParameter(
                    name="properties",
                    type="array",
                    description="Properties to follow for path finding",
                    required=False,
                    default=["P279", "P31"]
                )
            ],
            return_type="list",
            return_description="Path between entities as list of entity IDs"
        )
    
    async def execute(self, start_entity: str, end_entity: str, max_depth: int = 3, properties: List[str] = None) -> List[str]:
        """Find path between two entities using BFS."""
        if properties is None:
            properties = ["P279", "P31"]  # subclass of, instance of
        
        if start_entity == end_entity:
            return [start_entity]
        
        visited: Set[str] = set()
        queue = [(start_entity, [start_entity])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
                
            if current in visited:
                continue
                
            visited.add(current)
            
            try:
                entity = await get_entity_info_async(current)
                
                for prop in properties:
                    if prop in entity.statements:
                        for stmt in entity.statements[prop]:
                            if stmt.is_entity_ref and stmt.entity_type == "item":
                                neighbor = stmt.value
                                if neighbor == end_entity:
                                    return path + [neighbor]
                                
                                if neighbor not in visited:
                                    queue.append((neighbor, path + [neighbor]))
            except:
                continue
        
        return []  # No path found
'''

class LocalGraphTool(Tool):
    """Tool for building local graph around an entity."""
    
    def __init__(self):
        super().__init__(
            name="build_local_graph",
            description="Build a local graph around an entity up to specified depth"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="center_entity",
                    type="string",
                    description="Central entity ID"
                ),
                ToolParameter(
                    name="depth",
                    type="integer",
                    description="Graph depth (1-3 recommended)",
                    required=False,
                    default=2
                ),
                ToolParameter(
                    name="properties",
                    type="array",
                    description="Properties to follow",
                    required=False,
                    default=["P279", "P31"]
                ),
                ToolParameter(
                    name="max_nodes",
                    type="integer",
                    description="Maximum nodes in graph",
                    required=False,
                    default=100
                )
            ],
            return_type="dict",
            return_description="Graph structure with nodes and edges"
        )
    
    async def execute(self, center_entity: str, depth: int = 2, properties: List[str] = None, max_nodes: int = 100) -> Dict[str, Any]:
        """Build local graph around entity."""
        if properties is None:
            properties = ["P279", "P31"]
        
        nodes = {}
        edges = []
        to_explore = [(center_entity, 0)]
        explored = set()
        
        while to_explore and len(nodes) < max_nodes:
            entity_id, current_depth = to_explore.pop(0)
            
            if entity_id in explored or current_depth >= depth:
                continue
                
            explored.add(entity_id)
            
            try:
                entity = await get_entity_info_async(entity_id)
                nodes[entity_id] = {
                    'id': entity.id,
                    'label': entity.label,
                    'description': entity.description,
                    'depth': current_depth
                }
                
                # Add neighbors from entity statements
                for prop in properties:
                    if prop in entity.statements:
                        for stmt in entity.statements[prop]:
                            if stmt.is_entity_ref and stmt.entity_type == "item":
                                neighbor = stmt.value
                                edges.append({
                                    'source': entity_id,
                                    'target': neighbor,
                                    'property': prop
                                })
                                
                                if neighbor not in explored and current_depth + 1 < depth:
                                    to_explore.append((neighbor, current_depth + 1))
                                    
            except Exception as e:
                print(f"Error exploring {entity_id}: {e}")
                continue
        
        return {
            'nodes': nodes,
            'edges': edges,
            'center': center_entity,
            'depth': depth,
            'total_nodes': len(nodes),
            'total_edges': len(edges)
        }