from typing import Dict, Any, List, Optional, Set, Tuple
from abc import abstractmethod
from ..toolbox import Tool, ToolDefinition, ToolParameter
from .wikidata_api import wikidata_api
from .datamodel import (
    WikidataStatement, WikidataEntity, NeighborExplorationResult, LocalGraphResult,
    convert_api_entity_to_model
)
import asyncio
import traceback


class ExplorationTool(Tool):
    """Base class for Wikidata exploration tools with common functionality."""
    
    def __init__(self, name: str, description: str):
        super().__init__(name=name, description=description)
    
    @abstractmethod
    def visualize_results(self, result: Any, test_identifier: str, test_number: int) -> Tuple[str, str]:
        """
        Create visualizations for the exploration results.
        
        Args:
            result: The result object from the tool execution
            test_identifier: Unique identifier for this test
            test_number: Sequential test number
            
        Returns:
            Tuple of (static_path, dynamic_path)
        """
        pass
    
    async def _fetch_property_name(self, prop_id: str) -> tuple[str, str]:
        """Fetch property name for a given property ID."""
        try:
            prop_api_data = await wikidata_api.get_property(prop_id)
            prop_name = prop_api_data.get('labels', {}).get('en', prop_id)
            return prop_id, prop_name
        except Exception as e:
            print(f"Error fetching property {prop_id}: {e}")
            traceback.print_exc()
            return prop_id, prop_id
    
    async def _fetch_property_names_parallel(self, property_ids: List[str]) -> Dict[str, str]:
        """Fetch property names in parallel for multiple property IDs."""
        print(f"🔄 Fetching property names for {len(property_ids)} properties in parallel...")
        
        property_tasks = [self._fetch_property_name(prop_id) for prop_id in property_ids]
        property_results = await asyncio.gather(*property_tasks, return_exceptions=True)
        
        prop_names = {}
        for result in property_results:
            if isinstance(result, tuple):
                prop_id, prop_name = result
                prop_names[prop_id] = prop_name
            else:
                print(f"    ⚠️  Error fetching property name: {result}")
        
        print(f"✅ Property names fetched successfully")
        return prop_names
    
    async def _fetch_neighbor_entity(self, entity_ref: str) -> tuple[str, Dict[str, Any]]:
        """Fetch a single neighbor entity."""
        try:
            neighbor_api_data = await wikidata_api.get_entity(entity_ref)
            neighbor = convert_api_entity_to_model(neighbor_api_data)
            return entity_ref, {
                'id': neighbor.id,
                'label': neighbor.label,
                'description': neighbor.description
            }
        except Exception as e:
            print(f"      ❌ Failed to load neighbor {entity_ref}: {e}")
            traceback.print_exc()
            return entity_ref, {'id': entity_ref, 'label': entity_ref, 'description': ''}
    
    async def _fetch_neighbors_parallel(self, entity_refs: Set[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple neighbor entities in parallel."""
        if not entity_refs:
            return {}
        
        print(f"🔄 Fetching {len(entity_refs)} neighbor entities in parallel...")
        
        neighbor_tasks = [self._fetch_neighbor_entity(entity_ref) for entity_ref in entity_refs]
        neighbor_results = await asyncio.gather(*neighbor_tasks, return_exceptions=True)
        
        neighbors = {}
        for result in neighbor_results:
            if isinstance(result, tuple):
                entity_ref, neighbor_data = result
                neighbors[entity_ref] = neighbor_data
                if neighbor_data['label'] != entity_ref:  # Successfully loaded
                    print(f"      ✅ Neighbor loaded: '{neighbor_data['label']}'")
            else:
                print(f"    ⚠️  Error fetching neighbor: {result}")
        
        return neighbors
    
    async def _process_entity_batch(self, entity_batch: List[tuple[str, int]]) -> List[tuple[str, int, Any]]:
        """Process a batch of entities in parallel."""
        async def process_single_entity(entity_id: str, current_depth: int) -> tuple[str, int, Any]:
            try:
                api_data = await wikidata_api.get_entity(entity_id)
                entity = convert_api_entity_to_model(api_data)
                return entity_id, current_depth, entity
            except Exception as e:
                print(f"    ❌ Error exploring {entity_id}: {e}")
                traceback.print_exc()
                return entity_id, current_depth, None
        
        batch_tasks = [process_single_entity(entity_id, current_depth) for entity_id, current_depth in entity_batch]
        return await asyncio.gather(*batch_tasks, return_exceptions=True)


class NeighborsExplorationTool(ExplorationTool):
    """Tool for exploring entity neighbors and relationships."""
    
    def __init__(self):
        super().__init__(
            name="explore_entity_neighbors",
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
                    default=[],
                    items={"type": "string"}
                ),
                ToolParameter(
                    name="max_values_per_property",
                    type="integer",
                    description="Maximum values per property",
                    required=False,
                    default=10
                )
            ],
            return_type="NeighborExplorationResult",
            return_description="Entity information with neighbors and relationships"
        )
    
    async def execute(self, entity_id: str, include_properties: Optional[List[str]] = None, max_values_per_property: int = -1) -> NeighborExplorationResult:
        """Explore an entity's relationships."""
        # Get basic entity info using wikidata_api
        api_data = await wikidata_api.get_entity(entity_id)
        entity = convert_api_entity_to_model(api_data)
        
        neighbors = {}
        relationships = {}
        
        print(f"🔍 Exploring entity: {entity.id} - '{entity.label}'")
        print(f"📊 Found {len(entity.statements.items())} properties to analyze")
        
        # Get all property IDs that we'll need to process
        relevant_props = []
        for prop_id in entity.statements.keys():
            if not include_properties or prop_id in include_properties:
                relevant_props.append(prop_id)
        
        # Fetch property names in parallel
        prop_names = await self._fetch_property_names_parallel(relevant_props)
        
        # Process each property and collect entity references for parallel fetching
        entity_refs_to_fetch = set()
        
        for prop_id, statements in entity.statements.items():
            if include_properties and prop_id not in include_properties:
                continue
            
            prop_name = prop_names.get(prop_id, prop_id)
            print(f"  📋 Property {prop_id} ({prop_name}): {len(statements)} statements")
                
            # Limit statements per property
            if max_values_per_property < 0:
                limited_statements = statements
            else:
                limited_statements = statements[:max_values_per_property]
            
            if len(limited_statements) < len(statements):
                print(f"    📏 Limited to {len(limited_statements)} statements (max: {max_values_per_property})")
            
            relationships[prop_id] = [stmt.value for stmt in limited_statements]
            
            # Collect entity references for parallel fetching
            for stmt in limited_statements:
                if stmt.is_entity_ref and stmt.entity_type == "item":
                    entity_refs_to_fetch.add(stmt.value)
                    print(f"    🔗 Found entity reference: {stmt.value}")
                else:
                    print(f"    📝 Statement value: {stmt.value} (type: {stmt.entity_type})")
        
        # Fetch neighbor entities in parallel and convert to WikidataEntity objects
        neighbor_data = await self._fetch_neighbors_parallel(entity_refs_to_fetch)
        for entity_id, neighbor_info in neighbor_data.items():
            if neighbor_info['label'] != entity_id:  # Successfully loaded
                neighbors[entity_id] = WikidataEntity(
                    id=neighbor_info['id'],
                    label=neighbor_info['label'],
                    description=neighbor_info['description'],
                    aliases=[],
                    statements={},
                    link=f"https://www.wikidata.org/entity/{neighbor_info['id']}"
                )
        
        print(f"🎯 Exploration complete: {len(neighbors)} neighbors found")
        return NeighborExplorationResult(
            entity=entity,
            neighbors=neighbors,
            relationships=relationships,
            property_names=prop_names,
            total_properties=len(entity.statements),
            neighbor_count=len(neighbors)
        )
    
    def visualize_results(self, result: NeighborExplorationResult, test_identifier: str, test_number: int) -> Tuple[str, str]:
        """Create visualizations for neighbor exploration results."""
        # Set matplotlib backend to avoid threading issues
        import matplotlib
        matplotlib.use('Agg')
        
        from backend.core.toolbox.graph.visualization import GraphVisualizer
        import os
        
        # Use home directory for visualizations
        output_dir = os.path.expanduser("~/graph_visualizations")
        visualizer = GraphVisualizer(output_dir=output_dir)
        
        # Convert to visualization format
        entity_label = result.entity.label
        entity_id = result.entity.id
        title = f"Test {test_number}: Neighbors of {entity_label} ({entity_id})"
        
        # Create nodes dictionary for visualization
        vis_nodes = {
            result.entity.id: {
                'id': result.entity.id,
                'label': result.entity.label,
                'description': result.entity.description or 'No description available',
                'depth': 0
            }
        }
        for neighbor_id, neighbor_entity in result.neighbors.items():
            vis_nodes[neighbor_id] = {
                'id': neighbor_entity.id,
                'label': neighbor_entity.label,
                'description': neighbor_entity.description or 'No description available',
                'depth': 1
            }
        
        # Create edges list for visualization
        vis_edges = []
        for prop_id, values in result.relationships.items():
            prop_name = result.property_names.get(prop_id, prop_id)
            for target in values:
                if target in result.neighbors:
                    vis_edges.append({
                        'source': result.entity.id,
                        'target': target,
                        'property': prop_id,
                        'property_name': prop_name
                    })
        
        # Create static visualization first (more reliable)
        static_path = None
        dynamic_path = None
        
        try:
            static_path = visualizer.create_static_visualization(
                vis_nodes, vis_edges, title,
                filename=f"test_{test_number}_{test_identifier}_exploration_static.png"
            )
        except Exception as e:
            print(f"⚠️ Static visualization failed: {e}")
            traceback.print_exc()
            static_path = "Failed to create static visualization"
        
        # Create dynamic visualization with error handling
        try:
            dynamic_path = visualizer.create_dynamic_visualization(
                vis_nodes, vis_edges, title,
                filename=f"test_{test_number}_{test_identifier}_exploration_interactive.html"
            )
        except Exception as e:
            print(f"⚠️ Dynamic visualization failed: {e}")
            traceback.print_exc()
            dynamic_path = "Failed to create dynamic visualization"
        
        return static_path, dynamic_path


class LocalGraphTool(ExplorationTool):
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
                    default=["P279", "P31"],
                    items={"type": "string"}
                ),
                ToolParameter(
                    name="max_nodes",
                    type="integer",
                    description="Maximum nodes in graph",
                    required=False,
                    default=100
                )
            ],
            return_type="LocalGraphResult",
            return_description="Graph structure with nodes and edges"
        )
    
    async def execute(self, center_entity: str, depth: int = 2, properties: Optional[List[str]] = None, max_nodes: int = 100) -> LocalGraphResult:
        """Build local graph around entity."""
        if properties is None:
            properties = ["P279", "P31"]
        
        print(f"🌐 Building local graph around {center_entity}")
        print(f"📏 Parameters: depth={depth}, max_nodes={max_nodes}")
        print(f"🔍 Following properties: {properties}")
        
        # Fetch property names in parallel
        prop_names = await self._fetch_property_names_parallel(properties)
        print(f"📋 Property details: {', '.join([f'{p} ({prop_names[p]})' for p in properties])}")
        
        nodes = {}
        edges = []
        to_explore = [(center_entity, 0)]
        explored = set()
        
        while to_explore and len(nodes) < max_nodes:
            # Collect entities to process at this iteration for parallel fetching
            current_batch = []
            batch_size = min(10, len(to_explore))  # Process up to 10 entities in parallel
            
            for _ in range(batch_size):
                if not to_explore:
                    break
                entity_id, current_depth = to_explore.pop(0)
                if entity_id not in explored and current_depth < depth:
                    current_batch.append((entity_id, current_depth))
            
            if not current_batch:
                break
            
            # Process batch of entities in parallel
            batch_results = await self._process_entity_batch(current_batch)
            
            # Process batch results
            for result in batch_results:
                if isinstance(result, tuple):
                    entity_id, current_depth, entity = result
                    if entity is None or entity_id in explored:
                        continue
                        
                    explored.add(entity_id)
                    print(f"  🔍 Exploring {entity_id} at depth {current_depth}")
                    
                    # Store as WikidataEntity with depth information
                    entity_with_depth = WikidataEntity(
                        id=entity.id,
                        label=entity.label,
                        description=entity.description,
                        aliases=entity.aliases,
                        statements=entity.statements,
                        link=entity.link
                    )
                    # Add depth as a custom attribute (not part of the model but accessible)
                    entity_with_depth.__dict__['depth'] = current_depth
                    
                    nodes[entity_id] = entity_with_depth
                    print(f"    ✅ Added node: '{entity.label}'")
                    
                    # Add neighbors from entity statements
                    neighbors_found = 0
                    for prop in properties:
                        if prop in entity.statements:
                            for stmt in entity.statements[prop]:
                                if stmt.is_entity_ref and stmt.entity_type == "item":
                                    neighbor = stmt.value
                                    edges.append({
                                        'source': entity_id,
                                        'target': neighbor,
                                        'property': prop,
                                        'property_name': prop_names.get(prop, prop)
                                    })
                                    neighbors_found += 1
                                    
                                    if neighbor not in explored and current_depth + 1 < depth:
                                        to_explore.append((neighbor, current_depth + 1))
                                        print(f"      🔗 Added to explore: {neighbor} via {prop} ({prop_names[prop]})")
                    
                    print(f"    📊 Found {neighbors_found} neighbors via target properties")
                else:
                    print(f"    ⚠️  Error in batch processing: {result}")
        
        print(f"🎯 Graph building complete: {len(nodes)} nodes, {len(edges)} edges")
        return LocalGraphResult(
            nodes=nodes,
            edges=edges,
            center=center_entity,
            depth=depth,
            properties=properties,
            property_names=prop_names,
            total_nodes=len(nodes),
            total_edges=len(edges)
        )
    
    def visualize_results(self, result: LocalGraphResult, test_identifier: str, test_number: int) -> Tuple[str, str]:
        """Create visualizations for local graph results."""
        # Set matplotlib backend to avoid threading issues
        import matplotlib
        matplotlib.use('Agg')
        
        from backend.core.toolbox.graph.visualization import GraphVisualizer
        import os
        
        # Use home directory for visualizations
        output_dir = os.path.expanduser("~/graph_visualizations")
        visualizer = GraphVisualizer(output_dir=output_dir)
        
        # Convert to visualization format
        center_id = result.center
        depth = result.depth
        node_count = result.total_nodes
        title = f"Test {test_number}: Local Graph {center_id} (Depth {depth}, {node_count} nodes)"
        
        # Create nodes dictionary for visualization
        vis_nodes = {}
        for entity_id, entity in result.nodes.items():
            vis_nodes[entity_id] = {
                'id': entity.id,
                'label': entity.label,
                'description': entity.description or 'No description available',
                'depth': getattr(entity, 'depth', 0)
            }
        
        # Create static visualization first (more reliable)
        static_path = None
        dynamic_path = None
        
        try:
            static_path = visualizer.create_static_visualization(
                vis_nodes, result.edges, title,
                filename=f"test_{test_number}_{test_identifier}_graph_static.png"
            )
        except Exception as e:
            print(f"⚠️ Static visualization failed: {e}")
            traceback.print_exc()
            static_path = "Failed to create static visualization"
        
        # Create dynamic visualization with error handling
        try:
            dynamic_path = visualizer.create_dynamic_visualization(
                vis_nodes, result.edges, title,
                filename=f"test_{test_number}_{test_identifier}_graph_interactive.html"
            )
        except Exception as e:
            print(f"⚠️ Dynamic visualization failed: {e}")
            traceback.print_exc()
            dynamic_path = "Failed to create dynamic visualization"
        
        return static_path, dynamic_path


if __name__ == "__main__":
    import asyncio
    import sys
    import os
    
    # Add the parent directories to the path for absolute imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    # Import with absolute paths when running as main
    try:
        from backend.core.toolbox.toolbox import Tool, ToolDefinition, ToolParameter
        from backend.core.toolbox.wikidata.wikidata_api import wikidata_api
        from backend.core.toolbox.wikidata.datamodel import WikidataStatement, convert_api_entity_to_model
        from backend.core.toolbox.graph.visualization import GraphVisualizer
    except ImportError:
        print("Error: Could not import required modules. Make sure you're running from the correct directory.")
        print("Try running: python -m backend.core.toolbox.wikidata.exploration")
        sys.exit(1)
    
    async def test_tools():
        """Test all exploration tools and create visualizations."""
        print("Testing Wikidata Exploration Tools")
        print("=" * 40)
        
        test_counter = 1
        
        # Test NeighborsExplorationTool
        print(f"\n{test_counter}. Testing NeighborsExplorationTool")
        exploration_tool = NeighborsExplorationTool()
        print(f"Tool definition: {exploration_tool.get_definition().name}")
        
        try:
            # Test with Albert Einstein (Q937)
            result = await exploration_tool.execute("Q937", max_values_per_property=5)
            print(f"✓ Entity exploration successful: {result.entity.label}")
            print(f"  - Properties: {result.total_properties}")
            print(f"  - Neighbors: {result.neighbor_count}")
            
            # Create visualization immediately
            print(f"📊 Creating visualization for test {test_counter}...")
            static_path, dynamic_path = exploration_tool.visualize_results(
                result, 'neighbors_basic', test_counter
            )
            if "Failed" not in static_path:
                print(f"✓ Created static visualization: {static_path}")
            if "Failed" not in dynamic_path:
                print(f"✓ Created dynamic visualization: {dynamic_path}")
                
        except Exception as e:
            print(f"✗ Entity exploration failed: {e}")
            traceback.print_exc()
        
        test_counter += 1
        
        # Test LocalGraphTool - Basic Test
        print(f"\n{test_counter}. Testing LocalGraphTool - Basic Test")
        graph_tool = LocalGraphTool()
        print(f"Tool definition: {graph_tool.get_definition().name}")
        
        try:
            # Test with default parameters
            result = await graph_tool.execute("Q937", depth=2, max_nodes=20)
            print(f"✓ Local graph building successful")
            print(f"  - Center: {result.center}")
            print(f"  - Nodes: {result.total_nodes}")
            print(f"  - Edges: {result.total_edges}")
            
            # Create visualization immediately
            print(f"📊 Creating visualization for test {test_counter}...")
            static_path, dynamic_path = graph_tool.visualize_results(
                result, 'basic_depth2', test_counter
            )
            if "Failed" not in static_path:
                print(f"✓ Created static visualization: {static_path}")
            if "Failed" not in dynamic_path:
                print(f"✓ Created dynamic visualization: {dynamic_path}")
                
        except Exception as e:
            print(f"✗ Local graph building failed: {e}")
            traceback.print_exc()
        
        test_counter += 1
        
        # Test LocalGraphTool with all properties at depth 2
        print(f"\n{test_counter}. Testing LocalGraphTool - All Properties (Depth 2)")
        try:
            # Get all properties from the entity first
            api_data = await wikidata_api.get_entity("Q937")
            entity = convert_api_entity_to_model(api_data)
            all_props = list(entity.statements.keys())[:10]  # Limit to first 10 for testing
            
            result = await graph_tool.execute("Q937", depth=2, properties=all_props, max_nodes=30)
            print(f"✓ Local graph (all properties) successful")
            print(f"  - Used properties: {len(all_props)}")
            print(f"  - Nodes: {result.total_nodes}")
            print(f"  - Edges: {result.total_edges}")
            
            # Create visualization immediately
            print(f"📊 Creating visualization for test {test_counter}...")
            static_path, dynamic_path = graph_tool.visualize_results(
                result, 'all_props_depth2', test_counter
            )
            if "Failed" not in static_path:
                print(f"✓ Created static visualization: {static_path}")
            if "Failed" not in dynamic_path:
                print(f"✓ Created dynamic visualization: {dynamic_path}")
                
        except Exception as e:
            print(f"✗ Local graph (all properties) failed: {e}")
            traceback.print_exc()
        
        test_counter += 1
        
        # Test LocalGraphTool with default properties at depth 3
        print(f"\n{test_counter}. Testing LocalGraphTool - Default Properties (Depth 3)")
        try:
            result = await graph_tool.execute("Q937", depth=3, max_nodes=50)
            print(f"✓ Local graph (depth 3) successful")
            print(f"  - Depth: {result.depth}")
            print(f"  - Nodes: {result.total_nodes}")
            print(f"  - Edges: {result.total_edges}")
            
            # Create visualization immediately
            print(f"📊 Creating visualization for test {test_counter}...")
            static_path, dynamic_path = graph_tool.visualize_results(
                result, 'default_depth3', test_counter
            )
            if "Failed" not in static_path:
                print(f"✓ Created static visualization: {static_path}")
            if "Failed" not in dynamic_path:
                print(f"✓ Created dynamic visualization: {dynamic_path}")
                
        except Exception as e:
            print(f"✗ Local graph (depth 3) failed: {e}")
            traceback.print_exc()
        
        print("\n" + "=" * 40)
        print("Testing and Visualization completed!")
        print(f"📁 Visualization files saved to: ~/graph_visualizations")
        print("=" * 40)
    
    # Run the async tests
    asyncio.run(test_tools())