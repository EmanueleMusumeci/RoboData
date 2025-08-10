from typing import Dict, Any, List, Optional, Set, Tuple
from abc import abstractmethod
import asyncio
import os
import traceback
import matplotlib
from ..toolbox import Tool, ToolDefinition, ToolParameter
from .wikidata_api import wikidata_api
from .datamodel import (
    WikidataStatement, WikidataEntity, NeighborExplorationResult, LocalGraphResult,
    convert_api_entity_to_model
)
from ...knowledge_base.schema import Graph as VisGraph, Node as VisNode, Edge as VisEdge, LiteralNode
from .utils import order_properties_by_degree, order_local_graph_edges


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
    
    async def _fetch_property_name(self, prop_id: str) -> tuple[str, str, Optional[str]]:
        """Fetch property name for a given property ID."""
        try:
            prop_api_data = await wikidata_api.get_property(prop_id)
            prop_name = prop_api_data.get('labels', {}).get('en', prop_id)
            return prop_id, prop_name, None
        except Exception as e:
            error_msg = f"Error fetching property {prop_id}: {e}"
            print(error_msg)
            traceback.print_exc()
            return prop_id, prop_id, error_msg
    
    async def _fetch_property_names_parallel(self, property_ids: List[str]) -> tuple[Dict[str, str], List[str]]:
        """Fetch property names in parallel for multiple property IDs."""
        print(f"🔄 Fetching property names for {len(property_ids)} properties in parallel...")
        
        property_tasks = [self._fetch_property_name(prop_id) for prop_id in property_ids]
        property_results = await asyncio.gather(*property_tasks, return_exceptions=True)
        
        prop_names = {}
        errors = []
        for result in property_results:
            if isinstance(result, tuple) and len(result) == 3:
                prop_id, prop_name, error = result
                prop_names[prop_id] = prop_name
                if error:
                    errors.append(error)
            else:
                error_msg = f"Error fetching property name: {result}"
                print(f"    ⚠️  {error_msg}")
                errors.append(error_msg)
        
        print(f"✅ Property names fetched successfully")
        if errors:
            print(f"⚠️  {len(errors)} errors occurred during property name fetching")
        return prop_names, errors
    
    async def _fetch_neighbor_entity(self, entity_ref: str) -> tuple[str, Dict[str, Any], Optional[str]]:
        """Fetch a single neighbor entity."""
        try:
            neighbor_api_data = await wikidata_api.get_entity(entity_ref)
            return entity_ref, neighbor_api_data, None
        except Exception as e:
            error_msg = f"Failed to load neighbor {entity_ref}: {e}"
            print(f"      ❌ {error_msg}")
            traceback.print_exc()
            return entity_ref, {'id': entity_ref, 'label': entity_ref, 'description': ''}, error_msg
    
    async def _fetch_neighbors_parallel(self, entity_refs: Set[str]) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
        """Fetch multiple neighbor entities in parallel."""
        if not entity_refs:
            return {}, []
        
        print(f"🔄 Fetching {len(entity_refs)} neighbor entities in parallel...")
        
        neighbor_tasks = [self._fetch_neighbor_entity(entity_ref) for entity_ref in entity_refs]
        neighbor_results = await asyncio.gather(*neighbor_tasks, return_exceptions=True)
        
        neighbors = {}
        errors = []
        for result in neighbor_results:
            if isinstance(result, tuple) and len(result) == 3:
                entity_ref, neighbor_data, error = result
                neighbors[entity_ref] = neighbor_data
                if error:
                    errors.append(error)
                # Check if we have a valid entity with a label
                if isinstance(neighbor_data, dict) and 'labels' in neighbor_data:
                    labels = neighbor_data.get('labels', {})
                    label = labels.get('en', list(labels.values())[0] if labels else entity_ref)
                    print(f"      ✅ Neighbor loaded: '{label}' ({entity_ref})")
                else:
                    print(f"      ⚠️  Neighbor data incomplete for {entity_ref}")
            else:
                error_msg = f"Error fetching neighbor: {result}"
                print(f"    ⚠️  {error_msg}")
                errors.append(error_msg)
        
        if errors:
            print(f"⚠️  {len(errors)} errors occurred during neighbor fetching")
        return neighbors, errors
    
    async def _process_entity_batch(self, entity_batch: List[tuple[str, int]]) -> List[Any]:
        """Process a batch of entities in parallel."""
        async def process_single_entity(entity_id: str, current_depth: int) -> tuple[str, int, Any]:
            try:
                api_data = await wikidata_api.get_entity(entity_id)
                entity = convert_api_entity_to_model(api_data)
                return (entity_id, current_depth, entity)
            except Exception as e:
                error_msg = f"Error processing entity {entity_id}: {e}"
                print(f"⚠️ {error_msg}")
                return (entity_id, current_depth, None)
        
        batch_tasks = [process_single_entity(entity_id, current_depth) for entity_id, current_depth in entity_batch]
        return await asyncio.gather(*batch_tasks, return_exceptions=True)


class NeighborsExplorationTool(ExplorationTool):
    """Tool for exploring entity neighbors and relationships."""
    
    def __init__(self):
        super().__init__(
            name="explore_entity_neighbors",
            description="Explore an entity's direct relationships with neighbors or properties/values of this entity that can not represented as other nodes in the graph. " \
            "USEFUL FOR: understanding the context and connections of an entity within the knowledge graph. Explores the entity's neighbors and properties, returning a structured result."
        )
        self.entity_limits = {}
        self.initial_limit = 20
        self.increment = 10

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
                ),
                ToolParameter(
                    name="order_by_degree",
                    type="boolean",
                    description="Whether to order results by property degree",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="increase_limit",
                    type="boolean",
                    description=f"Whether to increase the number of results shown (increments by {self.increment})",
                    required=False,
                    default=False
                )
                ],
            return_type="NeighborExplorationResult",
            return_description="Detailed structured description of the entity's neighboring entities in the knowledge graph, including relationships and value properties."
        )
    
    async def execute(self, **kwargs) -> NeighborExplorationResult:
        """Explore an entity's relationships."""
        entity_id = kwargs.get("entity_id")
        include_properties = kwargs.get("include_properties")
        max_values_per_property = kwargs.get("max_values_per_property", -1)
        order_by_degree = kwargs.get("order_by_degree", False)
        increase_limit = kwargs.get("increase_limit", False)

        if not entity_id:
            raise ValueError("entity_id is required")

        # Initialize error tracking
        all_errors = []
        partial_data = False

        # Get progressive limit
        limit = self.entity_limits.get(entity_id, self.initial_limit)
        if increase_limit:
            limit += self.increment
        self.entity_limits[entity_id] = limit

        # Get basic entity info using wikidata_api
        try:
            api_data = await wikidata_api.get_entity(entity_id)
            entity = convert_api_entity_to_model(api_data)
        except Exception as e:
            error_msg = f"Failed to fetch main entity {entity_id}: {e}"
            all_errors.append(error_msg)
            raise ValueError(error_msg)
        
        neighbors = {}
        relationships = {}
        
        print(f"🔍 Exploring entity: {entity.id} - '{entity.label}'")
        print(f"📊 Found {len(entity.statements.items())} properties to analyze")
        
        # Get all property IDs that we'll need to process
        all_prop_ids = list(entity.statements.keys())
        if include_properties:
            relevant_props = [p for p in all_prop_ids if p in include_properties]
        else:
            relevant_props = all_prop_ids
        
        # Fetch property names in parallel
        prop_names, prop_errors = await self._fetch_property_names_parallel(relevant_props)
        all_errors.extend(prop_errors)
        if prop_errors:
            partial_data = True
        
        # Process each property and collect entity references for parallel fetching
        entity_refs_to_fetch = set()
        
        for prop_id, statements in entity.statements.items():
            if prop_id not in relevant_props:
                continue
            
            prop_name = prop_names.get(prop_id, prop_id)
            print(f"  📋 Property {prop_id} ({prop_name}): {len(statements)} statements")

            # Limit statements per property
            if max_values_per_property < 0:
                statements_to_process = statements
            else:
                statements_to_process = statements[:max_values_per_property]

            if len(statements_to_process) < len(statements):
                print(f"    📏 Limited to {len(statements_to_process)} statements (max: {max_values_per_property})")
            
            relationships[prop_id] = []
            
            for statement in statements_to_process:
                relationships[prop_id].append(statement)
                if statement.is_entity_ref:
                    entity_refs_to_fetch.add(statement.value)
                    if statement.entity_type == "item":
                        print(f"    🔗 Found entity reference: {statement.value}")
                else:
                    print(f"    📝 Statement value: {statement.value} (type: {statement.datatype})")


        # Fetch neighbor entities in parallel and convert to WikidataEntity objects
        neighbor_data, neighbor_errors = await self._fetch_neighbors_parallel(entity_refs_to_fetch)
        all_errors.extend(neighbor_errors)
        if neighbor_errors:
            partial_data = True
            
        for entity_id_val, neighbor_info in neighbor_data.items():
            if neighbor_info:
                neighbors[entity_id_val] = convert_api_entity_to_model(neighbor_info)
        
        print(f"🎯 Exploration complete: {len(neighbors)} neighbors found")
        if all_errors:
            print(f"⚠️  Total errors encountered: {len(all_errors)}")
            
        return NeighborExplorationResult(
            entity=entity,
            neighbors=neighbors,
            relationships=relationships,
            property_names=prop_names,
            total_properties=len(entity.statements),
            neighbor_count=len(neighbors),
            limit=limit,
            order_by_degree=order_by_degree,
            errors=all_errors,
            partial_data=partial_data
        )
    
    def format_result(self, result: Optional[NeighborExplorationResult]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No exploration result."
        
        entity = result.entity
        prop_count = len(result.property_names)
        neighbor_count = result.neighbor_count
        
        summary = f"Explored '{entity.label}' ({entity.id}). Found {prop_count} properties and {neighbor_count} neighbors."
        
        # Add error information if present
        if result.errors:
            summary += f" ⚠️ WARNING: {len(result.errors)} errors occurred during exploration."
            if result.partial_data:
                summary += " Some data may be incomplete due to API failures."
            # Include first few errors for context
            if len(result.errors) <= 3:
                summary += f" Errors: {'; '.join(result.errors)}"
            else:
                summary += f" First 3 errors: {'; '.join(result.errors[:3])}..."
        
        # Order properties and get top examples based on progressive limit
        order_by_degree = result.order_by_degree
        ordered_props = order_properties_by_degree(result.entity.statements, enabled=order_by_degree)
        limit = result.limit
        
        examples = []
        for prop_id in ordered_props:
            if len(examples) >= limit:
                break
            
            prop_name = result.property_names.get(prop_id, prop_id)
            for stmt in result.entity.statements[prop_id]:
                if stmt.is_entity_ref and stmt.value in result.neighbors:
                    neighbor_label = result.neighbors[stmt.value].label
                    examples.append(
                        f"'{entity.label}' -> '{prop_name} ({prop_id})' -> '{neighbor_label} ({stmt.value})'"
                    )
                    if len(examples) >= limit:
                        break
        
        if examples:
            summary += f" Top {len(examples)} examples (ordered by degree: {order_by_degree}): {'; '.join(examples)}"
            
        return summary

    def visualize_results(self, result: NeighborExplorationResult, test_identifier: str, test_number: int) -> Tuple[str, str]:
        """Create visualizations for neighbor exploration results."""
        # Set matplotlib backend to avoid threading issues
        matplotlib.use('Agg')
        
        from backend.core.toolbox.graph.visualization import GraphVisualizer
        
        # Use home directory for visualizations
        output_dir = os.path.expanduser("~/graph_visualizations")
        visualizer = GraphVisualizer(output_dir=output_dir)
        
        # Convert to visualization format
        entity_label = result.entity.label
        entity_id = result.entity.id
        title = f"Test {test_number}: Neighbors of {entity_label} ({entity_id})"
        
        vis_graph = VisGraph()

        # Create nodes dictionary for visualization
        vis_graph.add_node(VisNode(
            node_id=result.entity.id,
            node_type='center',
            label=result.entity.label,
            description=result.entity.description or 'No description available',
            properties={'depth': 0}
        ))

        for neighbor_id, neighbor_entity in result.neighbors.items():
            vis_graph.add_node(VisNode(
                node_id=neighbor_id,
                node_type='neighbor',
                label=neighbor_entity.label,
                description=neighbor_entity.description or 'No description available',
                properties={'depth': 1}
            ))
        
        # Create edges list for visualization
        for prop_id, statements in result.relationships.items():
            prop_name = result.property_names.get(prop_id, prop_id)
            for statement in statements:
                if statement.is_entity_ref:
                    if statement.value in result.neighbors:
                        vis_graph.add_edge(VisEdge(
                            source_id=result.entity.id,
                            target_id=statement.value,
                            relationship_type=prop_id,
                            label=prop_name
                        ))
                else:
                    # Handle literal nodes
                    literal_id = f"{prop_id}_{statement.value}"
                    vis_graph.add_node(LiteralNode(
                        node_id=literal_id,
                        label=str(statement.value),
                        datatype=statement.datatype
                    ))
                    vis_graph.add_edge(VisEdge(
                        source_id=result.entity.id,
                        target_id=literal_id,
                        relationship_type=prop_id,
                        label=prop_name
                    ))

        # Create static visualization first (more reliable)
        static_path = None
        dynamic_path = None
        
        try:
            static_path = visualizer.create_static_visualization(
                graph=vis_graph, title=title,
                filename=f"test_{test_number}_{test_identifier}_neighbors_static.png"
            )
        except Exception as e:
            print(f"⚠️ Static visualization failed: {e}")
            traceback.print_exc()
            static_path = "Failed to create static visualization"
        
        # Create dynamic visualization with error handling
        try:
            dynamic_path = visualizer.create_dynamic_visualization(
                graph=vis_graph, title=title,
                filename=f"test_{test_number}_{test_identifier}_neighbors_interactive.html"
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
            description="Get the local graph around an entity up to specified depth, performing a depth-first search (DFS). " \
            "USEFUL FOR: understanding the local context and relationships of an entity within the knowledge graph with a focus on depth and breadth of connections."
        )
        self.entity_limits = {}
        self.initial_limit = 10
        self.increment = 10

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
                ),
                ToolParameter(
                    name="order_by_degree",
                    type="boolean",
                    description="Whether to order the connections by degree",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="increase_limit",
                    type="boolean",
                    description=f"Whether to increase the number of results shown (increments by {self.increment})",
                    required=False,
                    default=False
                )
            ],
            return_type="LocalGraphResult",
            return_description="Detailed structured description of the local graph around the entity, including nodes, edges, properties followed, and total counts, up to a specified depth around the center entity."
        )
    
    async def execute(self, **kwargs) -> LocalGraphResult:
        """Build local graph around entity."""
        center_entity = kwargs.get("center_entity")
        depth = kwargs.get("depth", 2)
        properties = kwargs.get("properties")
        max_nodes = kwargs.get("max_nodes", 100)
        order_by_degree = kwargs.get("order_by_degree", False)
        increase_limit = kwargs.get("increase_limit", False)

        if not center_entity:
            raise ValueError("center_entity is required")

        # Initialize error tracking
        all_errors = []
        partial_data = False

        limit = self.entity_limits.get(center_entity, self.initial_limit)
        if increase_limit:
            limit += self.increment
        self.entity_limits[center_entity] = limit

        if properties is None:
            properties = ["P279", "P31"]
        
        print(f"🌐 Building local graph around {center_entity}")
        print(f"📏 Parameters: depth={depth}, max_nodes={max_nodes}")
        print(f"🔍 Following properties: {properties}")
        
        # Fetch property names in parallel
        prop_names, prop_errors = await self._fetch_property_names_parallel(properties)
        all_errors.extend(prop_errors)
        if prop_errors:
            partial_data = True
            
        print(f"📋 Property details: {', '.join([f'{p} ({prop_names.get(p, p)})' for p in properties])}")
        
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
                    if entity is None:
                        error_msg = f"Failed to fetch entity {entity_id} at depth {current_depth}"
                        all_errors.append(error_msg)
                        partial_data = True
                        continue
                    if entity_id in explored:
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
                                if stmt.is_entity_ref and stmt.value:
                                    neighbor = stmt.value
                                    edges.append({
                                        'from': entity_id,
                                        'to': neighbor,
                                        'property': prop,
                                        'property_name': prop_names.get(prop, prop)
                                    })
                                    neighbors_found += 1
                                    
                                    if neighbor not in explored and current_depth + 1 < depth:
                                        to_explore.append((neighbor, current_depth + 1))
                                        print(f"      🔗 Added to explore: {neighbor} via {prop} ({prop_names.get(prop, prop)})")
                    
                    print(f"    📊 Found {neighbors_found} neighbors via target properties")
                else:
                    error_msg = f"Error in batch processing: {result}"
                    all_errors.append(error_msg)
                    partial_data = True
                    print(f"    ⚠️  {error_msg}")
        
        print(f"🎯 Graph building complete: {len(nodes)} nodes, {len(edges)} edges")
        
        # Fetch leaf nodes that were not fully explored but are part of edges
        leaf_node_ids = set(edge['to'] for edge in edges) - set(nodes.keys())
        if leaf_node_ids:
            print(f"🍃 Fetching details for {len(leaf_node_ids)} leaf nodes...")
            # The depth here is just for the call, it won't be used for further exploration
            leaf_node_results = await self._process_entity_batch([(node_id, depth) for node_id in leaf_node_ids])
            for result in leaf_node_results:
                if isinstance(result, tuple):
                    entity_id, _, entity = result
                    if entity:
                        # Add depth as a custom attribute
                        entity.__dict__['depth'] = depth
                        nodes[entity_id] = entity
                        print(f"    🌿 Added leaf node: '{entity.label}' ({entity_id})")
                    else:
                        error_msg = f"Failed to fetch leaf node {entity_id}"
                        all_errors.append(error_msg)
                        partial_data = True
                else:
                    # This handles exceptions returned by asyncio.gather
                    error_msg = f"Error fetching leaf node details: {result}"
                    all_errors.append(error_msg)
                    partial_data = True
                    print(f"    ⚠️  {error_msg}")

        if all_errors:
            print(f"⚠️  Total errors encountered: {len(all_errors)}")

        return LocalGraphResult(
            nodes=nodes,
            edges=edges,
            center=center_entity,
            depth=depth,
            properties=properties,
            property_names=prop_names,
            total_nodes=len(nodes),
            total_edges=len(edges),
            limit=limit,
            order_by_degree=order_by_degree,
            errors=all_errors,
            partial_data=partial_data
        )
    
    def format_result(self, result: Optional[LocalGraphResult]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No local graph built."
        
        center_node = result.nodes.get(result.center)
        center_label = center_node.label if center_node else result.center
            
        summary = (f"Built local graph around '{center_label}' ({result.center}) "
                   f"with depth {result.depth}. Found {result.total_nodes} nodes and {result.total_edges} edges.")

        # Add error information if present
        if result.errors:
            summary += f" ⚠️ WARNING: {len(result.errors)} errors occurred during graph building."
            if result.partial_data:
                summary += " Some data may be incomplete due to API failures."
            # Include first few errors for context
            if len(result.errors) <= 3:
                summary += f" Errors: {'; '.join(result.errors)}"
            else:
                summary += f" First 3 errors: {'; '.join(result.errors[:3])}..."

        order_by_degree = result.order_by_degree
        ordered_edges = order_local_graph_edges(result, enabled=order_by_degree)
        limit = result.limit
        
        if ordered_edges:
            summary += f"\nTop {min(limit, len(ordered_edges))} of {len(ordered_edges)} connections (ordered by degree: {order_by_degree}):"
            for i, edge in enumerate(ordered_edges[:limit]):
                source_node = result.nodes.get(edge['from'])
                target_node = result.nodes.get(edge['to'])
                
                source_label = source_node.label if source_node else edge['from']
                target_label = target_node.label if target_node else edge['to']
                
                prop_name = edge['property_name']
                prop_id = edge['property']
                
                source_id = edge['from']
                target_id = edge['to']

                summary += f"\n  {i+1}. '{source_label} ({source_id})' -> '{prop_name} ({prop_id})' -> '{target_label} ({target_id})'"
                
        return summary

    def visualize_results(self, result: LocalGraphResult, test_identifier: str, test_number: int) -> Tuple[str, str]:
        """Create visualizations for local graph results."""
        # Set matplotlib backend to avoid threading issues
        matplotlib.use('Agg')
        
        from backend.core.toolbox.graph.visualization import GraphVisualizer
        
        # Use home directory for visualizations
        output_dir = os.path.expanduser("~/graph_visualizations")
        visualizer = GraphVisualizer(output_dir=output_dir)
        
        # Convert to visualization format
        center_id = result.center
        depth = result.depth
        node_count = result.total_nodes
        title = f"Test {test_number}: Local Graph {center_id} (Depth {depth}, {node_count} nodes)"
        
        vis_graph = VisGraph()
        # Create nodes dictionary for visualization
        for entity_id, entity in result.nodes.items():
            vis_graph.add_node(VisNode(
                node_id=entity.id,
                node_type='entity',
                label=entity.label,
                description=entity.description or 'No description available',
                properties={'depth': getattr(entity, 'depth', 0)}
            ))
        
        for edge in result.edges:
            vis_graph.add_edge(VisEdge(
                source_id=edge['from'],
                target_id=edge['to'],
                relationship_type=edge['property_name'],
                label=edge['property_name']
            ))

        # Create static visualization first (more reliable)
        static_path = None
        dynamic_path = None
        
        try:
            static_path = visualizer.create_static_visualization(
                graph=vis_graph, title=title,
                filename=f"test_{test_number}_{test_identifier}_graph_static.png"
            )
        except Exception as e:
            print(f"⚠️ Static visualization failed: {e}")
            traceback.print_exc()
            static_path = "Failed to create static visualization"
        
        # Create dynamic visualization with error handling
        try:
            dynamic_path = visualizer.create_dynamic_visualization(
                graph=vis_graph, title=title,
                filename=f"test_{test_number}_{test_identifier}_graph_interactive.html"
            )
        except Exception as e:
            print(f"⚠️ Dynamic visualization failed: {e}")
            traceback.print_exc()
            dynamic_path = "Failed to create dynamic visualization"
        
        return static_path, dynamic_path


if __name__ == "__main__":
    import sys
    import os
    import asyncio
    import traceback
    import matplotlib
        
    # Add the parent directories to the path for absolute imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    # Import with absolute paths when running as main
    try:
        from backend.core.toolbox.toolbox import Tool, ToolDefinition, ToolParameter
        from backend.core.toolbox.wikidata.wikidata_api import wikidata_api
        from backend.core.toolbox.wikidata.datamodel import WikidataStatement, convert_api_entity_to_model
    except ImportError:
        print("Error: Could not import required modules. Make sure you're running from the correct directory.")
        print("Try running: python -m backend.core.toolbox.wikidata.exploration")
        sys.exit(1)
    
    async def test_tools():
        """Test all exploration tools and their result formatting."""
        print("Testing Wikidata Exploration Tools and Result Formatting")
        print("=" * 40)
        
        test_counter = 1
        
        # Test 1: NeighborsExplorationTool
        print(f"\n{test_counter}. Testing NeighborsExplorationTool")
        exploration_tool = NeighborsExplorationTool()
        
        try:
            # Test with a valid entity
            result = await exploration_tool.execute(entity_id="Q937")
            print(f"  ✓ Entity exploration successful for '{result.entity.label}'.")
            
            # Test result formatting
            formatted_result = exploration_tool.format_result(result)
            print(f"  - Formatted result (Success, default ordering): {formatted_result}")

            # Test with degree ordering
            result_ordered = await exploration_tool.execute(entity_id="Q937", order_by_degree=True)
            formatted_result_ordered = exploration_tool.format_result(result_ordered)
            print(f"  - Formatted result (Success, degree ordering): {formatted_result_ordered}")

            # Test again to see if limit increases
            print("  - Testing again to check for increased limit...")
            result_increased = await exploration_tool.execute(entity_id="Q937", increase_limit=True)
            formatted_result_increased = exploration_tool.format_result(result_increased)
            print(f"  - Formatted result (increased limit): {formatted_result_increased}")

            # Test formatting for an empty/failed result
            formatted_empty_result = exploration_tool.format_result(None)
            print(f"  - Formatted result (Empty): {formatted_empty_result}")

        except Exception as e:
            print(f"  ✗ Entity exploration failed: {e}")
            traceback.print_exc()
        
        test_counter += 1
        
        # Test 2: LocalGraphTool
        print(f"\n{test_counter}. Testing LocalGraphTool")
        graph_tool = LocalGraphTool()
        
        try:
            # Test with default parameters
            result = await graph_tool.execute(center_entity="Q5", depth=1, max_nodes=15)
            print(f"  ✓ Local graph building successful.")
            
            # Test result formatting
            formatted_result = graph_tool.format_result(result)
            print(f"  - Formatted result (Success, default ordering):\n{formatted_result}")

            # Test with degree ordering
            result_ordered = await graph_tool.execute(center_entity="Q5", depth=1, max_nodes=15, order_by_degree=True)
            formatted_result_ordered = graph_tool.format_result(result_ordered)
            print(f"  - Formatted result (Success, degree ordering):\n{formatted_result_ordered}")

            # Test again to see if limit increases
            print("  - Testing again to check for increased limit...")
            result_increased = await graph_tool.execute(center_entity="Q5", depth=1, max_nodes=15, increase_limit=True)
            formatted_result_increased = graph_tool.format_result(result_increased)
            print(f"  - Formatted result (increased limit):\n{formatted_result_increased}")

            # Test formatting for an empty/failed result
            formatted_empty_result = graph_tool.format_result(None)
            print(f"  - Formatted result (Empty): {formatted_empty_result}")
            
        except Exception as e:
            print(f"  ✗ Local graph building failed: {e}")
            traceback.print_exc()
        
        print("\n" + "=" * 40)
        print("All formatting tests completed.")

    
    # Run the async tests
    asyncio.run(test_tools())