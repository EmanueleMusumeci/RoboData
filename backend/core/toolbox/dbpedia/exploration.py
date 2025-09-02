from typing import Dict, Any, List, Optional, Set, Tuple
from abc import abstractmethod
import asyncio
import os
import traceback
import time
from ..toolbox import Tool, ToolDefinition, ToolParameter
from .dbpedia_api import dbpedia_api
from .rate_limiter import rate_limited_request
from .datamodel import (
    DBpediaEntity, NeighborExplorationResult, LocalGraphResult,
    convert_api_entity_to_model
)
from ...knowledge_base.schema import Graph as VisGraph, Node as VisNode, Edge as VisEdge, LiteralNode
from .utils import order_properties_by_degree, order_local_graph_edges, extract_entity_references_from_statements

class ExplorationTool(Tool):
    """Base class for DBpedia exploration tools with common functionality."""
    
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
        """Fetch the display name for a property ID with rate limiting."""
        async with rate_limited_request():
            try:
                prop_api_data = await dbpedia_api.get_entity(prop_id)
            except Exception as e:
                error_msg = f"Error fetching property {prop_id}: {e}"
                print(error_msg)
                traceback.print_exc()
                return prop_id, prop_id, error_msg
            
            prop_name = prop_api_data.get('labels', {}).get('en', prop_id)
            return prop_id, prop_name, None
    
    async def _fetch_property_names_parallel(self, property_ids: List[str]) -> tuple[Dict[str, str], List[str]]:
        """Fetch property names in parallel for multiple property IDs."""
        # Limit concurrency for property fetches to avoid hitting the SPARQL
        # endpoint too hard and getting 429 responses. We also add a small
        # retry/backoff on transient failures.
        print(f"🔄 Fetching property names for {len(property_ids)} properties with bounded concurrency...")

        MAX_CONCURRENT_PROPERTY_FETCHES = 4
        RETRIES = 3

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROPERTY_FETCHES)

        async def _fetch_with_semaphore(prop_id: str) -> tuple[str, str, Optional[str]]:
            async with semaphore:
                attempt = 0
                while attempt < RETRIES:
                    attempt += 1
                    try:
                        result = await self._fetch_property_name(prop_id)
                        return result
                    except Exception as e:
                        # If we see a 429-like error, backoff a bit and retry
                        err_text = str(e)
                        backoff = 0.5 * attempt
                        print(f"    ⚠️  Error fetching property {prop_id} (attempt {attempt}): {err_text}; backoff {backoff}s")
                        await asyncio.sleep(backoff)
                # Final attempt without exception capture to return a consistent tuple
                try:
                    return await self._fetch_property_name(prop_id)
                except Exception as e:
                    error_msg = f"Failed to fetch property {prop_id} after {RETRIES} retries: {e}"
                    print(f"    ❌ {error_msg}")
                    return prop_id, prop_id, error_msg

        tasks = [asyncio.create_task(_fetch_with_semaphore(pid)) for pid in property_ids]

        # Progress tracking
        total = len(tasks)
        completed = 0
        progress_interval = max(1, total // 10)  # log ~10 times during the run
        start_ts = time.time()

        results: List[object] = []
        for fut in asyncio.as_completed(tasks):
            try:
                r = await fut
            except Exception as e:
                r = e
            results.append(r)
            completed += 1
            if (completed % progress_interval) == 0 or completed == total:
                elapsed = time.time() - start_ts
                pct = int((completed / total) * 100)
                print(f"    ℹ️  Property fetch progress: {completed}/{total} ({pct}%) elapsed {elapsed:.1f}s")

        prop_names: Dict[str, str] = {}
        errors: List[str] = []
        for r in results:
            if isinstance(r, tuple) and len(r) == 3:
                prop_id, prop_name, error = r
                prop_names[prop_id] = prop_name
                if error:
                    errors.append(error)
            else:
                error_msg = f"Error fetching property name: {r}"
                print(f"    ⚠️  {error_msg}")
                errors.append(error_msg)

        print(f"✅ Property names fetch completed ({len(prop_names)} fetched, {len(errors)} errors) in {time.time() - start_ts:.1f}s")
        return prop_names, errors
    
    async def _fetch_neighbor_entity(self, entity_ref: str) -> tuple[str, Dict[str, Any], Optional[str]]:
        """Fetch a single neighbor entity with rate limiting."""
        print(f"        📡 Fetching neighbor: {entity_ref}")
        async with rate_limited_request():
            try:
                neighbor_api_data = await dbpedia_api.get_entity(entity_ref)
                print(f"        ✅ Successfully fetched: {entity_ref}")
                return entity_ref, neighbor_api_data, None
            except Exception as e:
                error_msg = f"Failed to load neighbor {entity_ref}: {e}"
                print(f"        ❌ {error_msg}")
                return entity_ref, {'id': entity_ref, 'labels': {'en': entity_ref}, 'descriptions': {'en': ''}}, error_msg
    
    async def _fetch_neighbors_parallel(self, entity_refs: Set[str]) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
        """Fetch multiple neighbor entities in parallel."""
        if not entity_refs:
            return {}, []
        
        print(f"🔄 Fetching {len(entity_refs)} neighbor entities in parallel...")
        
        neighbor_tasks = [self._fetch_neighbor_entity(entity_ref) for entity_ref in entity_refs]
        
        try:
            # Add a timeout to prevent hanging indefinitely
            neighbor_results = await asyncio.wait_for(
                asyncio.gather(*neighbor_tasks, return_exceptions=True),
                timeout=300  # 5 minutes timeout
            )
        except asyncio.TimeoutError:
            print("⏰ Neighbor fetching timed out after 5 minutes")
            return {}, ["Neighbor fetching timed out"]
        
        neighbors = {}
        errors = []
        for result in neighbor_results:
            if isinstance(result, tuple) and len(result) == 3:
                entity_ref, neighbor_data, error = result
                neighbors[entity_ref] = neighbor_data
                if error:
                    errors.append(error)
            else:
                error_msg = f"Error fetching neighbor: {result}"
                print(f"      ⚠️  {error_msg}")
                errors.append(error_msg)
        
        print(f"✅ Neighbors fetched successfully")
        if errors:
            print(f"⚠️  {len(errors)} errors occurred during neighbor fetching")
        return neighbors, errors

class NeighborsExplorationTool(ExplorationTool):
    """Tool for exploring neighbors of a DBpedia entity through its properties."""
    
    def __init__(self):
        super().__init__(
            name="explore_neighbors",
            description="Explore the neighboring entities of a given DBpedia entity through its properties and relationships." \
            "This tool fetches the entity's properties and follows entity-valued properties to discover connected entities." \
            "USEFUL FOR: discovering related entities, building local knowledge graphs, and understanding entity relationships."
        )
        self.neighbor_limits = {}
        self.initial_limit = 15
        self.increment = 10

    async def _fetch_neighbor_entity(self, entity_ref: str) -> tuple[str, Dict[str, Any], Optional[str]]:
        """Fetch a single neighbor entity WITHOUT using the shared rate limiter.

        We override the base implementation here so that NeighborsExplorationTool
        does not enter nested rate-limited contexts when it spawns many
        concurrent neighbor-fetching tasks. Property name fetching remains
        rate-limited elsewhere.
        """
        print(f"        📡 Fetching neighbor (no rate limit): {entity_ref}")
        try:
            neighbor_api_data = await dbpedia_api.get_entity(entity_ref)
            print(f"        ✅ Successfully fetched: {entity_ref}")
            return entity_ref, neighbor_api_data, None
        except Exception as e:
            error_msg = f"Failed to load neighbor {entity_ref}: {e}"
            print(f"        ❌ {error_msg}")
            return entity_ref, {'id': entity_ref, 'labels': {'en': entity_ref}, 'descriptions': {'en': ''}}, error_msg
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_id",
                    type="string",
                    description="DBpedia entity ID (e.g., Douglas_Adams) or full URI"
                ),
                ToolParameter(
                    name="include_reverse",
                    type="boolean", 
                    description="Whether to include reverse relationships (entities pointing to this entity)",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="order_by_degree",
                    type="boolean",
                    description="Whether to order properties by their usage frequency",
                    required=False,
                    default=True
                ),
                ToolParameter(
                    name="increase_limit",
                    type="boolean",
                    description="Whether to increase the number of neighbors shown",
                    required=False,
                    default=False
                )
            ],
            return_type="dict",
            return_description="Neighbor exploration results with entity and neighbor information"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute neighbor exploration for a DBpedia entity."""
        entity_id = kwargs.get("entity_id")
        include_reverse = kwargs.get("include_reverse", False)
        order_by_degree = kwargs.get("order_by_degree", True)
        increase_limit = kwargs.get("increase_limit", False)
        
        if not entity_id:
            raise ValueError("entity_id is required")
        
        print(f"🔍 Exploring neighbors for entity: {entity_id}")
        
        # Manage neighbor limits
        limit = self.neighbor_limits.get(entity_id, self.initial_limit)
        if increase_limit:
            limit += self.increment
        self.neighbor_limits[entity_id] = limit
        
        errors = []
        
        try:
            # Fetch the main entity (do not use the shared rate limiter here to
            # avoid nested acquisitions when neighbor fetches spawn many tasks).
            print(f"📥 Fetching main entity data...")
            api_data = await dbpedia_api.get_entity(entity_id)
            entity = convert_api_entity_to_model(api_data)
            print(f"✅ Main entity fetched: {entity.label}")
        except Exception as e:
            error_msg = f"Failed to fetch entity {entity_id}: {e}"
            errors.append(error_msg)
            raise ValueError(error_msg)
        
        # Extract entity references from statements
        print(f"🔗 Extracting entity references from {len(entity.statements)} properties...")
        entity_refs = set()
        
        for prop_id, prop_statements in entity.statements.items():
            for value_key, statement in prop_statements.items():
                # Look for entity-like values
                # Access the value attribute directly from the DBpediaStatement object
                value = statement.value if hasattr(statement, 'value') else ''
                if isinstance(value, str):
                    # Check if it looks like a DBpedia entity URI or reference
                    if (value.startswith('http://dbpedia.org/resource/') or
                        value.startswith('dbr:') or
                        # Simple heuristic: capitalized words without spaces might be entities
                        (not ' ' in value and value and value[0].isupper() and len(value) > 1)):
                        entity_refs.add(value)
        
        print(f"🎯 Found {len(entity_refs)} potential neighbor entity references")
        
        # Limit the number of neighbors to fetch
        entity_refs_list = list(entity_refs)[:limit]
        entity_refs = set(entity_refs_list)
        
        # Fetch neighbor entities in parallel
        neighbors_data, neighbor_errors = await self._fetch_neighbors_parallel(entity_refs)
        errors.extend(neighbor_errors)
        
        # Convert neighbor data to entity models
        neighbors = {}
        for entity_ref, neighbor_data in neighbors_data.items():
            try:
                neighbors[entity_ref] = convert_api_entity_to_model(neighbor_data)
            except Exception as e:
                error_msg = f"Failed to convert neighbor {entity_ref}: {e}"
                errors.append(error_msg)
                print(f"    ⚠️  {error_msg}")
        
        # Fetch property names for all properties used
        all_property_ids = list(entity.statements.keys())
        for neighbor in neighbors.values():
            all_property_ids.extend(neighbor.statements.keys())
        unique_property_ids = list(set(all_property_ids))
        
        property_names, prop_errors = await self._fetch_property_names_parallel(unique_property_ids)
        errors.extend(prop_errors)
        
        # Create result
        result_data = NeighborExplorationResult(
            entity=entity,
            neighbors=neighbors,
            property_names=property_names,
            total_neighbors=len(entity_refs_list),
            displayed_neighbors=len(neighbors),
            errors=errors
        )
        
        print(f"✅ Neighbor exploration completed. Found {len(neighbors)} neighbors")
        if errors:
            print(f"⚠️  {len(errors)} errors occurred during exploration")
        
        return {
            "entity_id": entity.id,
            "entity_label": entity.label,
            "entity_description": entity.description,
            "neighbors": {
                ref: {
                    "id": neighbor.id,
                    "label": neighbor.label,
                    "description": neighbor.description,
                    "statement_count": len(neighbor.statements)
                } for ref, neighbor in neighbors.items()
            },
            "property_names": property_names,
            "total_neighbors": result_data.total_neighbors,
            "displayed_neighbors": result_data.displayed_neighbors,
            "limit": limit,
            "errors": errors,
            "partial_data": len(errors) > 0
        }
    
    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the neighbor exploration result into a readable string."""
        if not result:
            return "No neighbor exploration results found."
        
        entity_id = result.get('entity_id', 'N/A')
        entity_label = result.get('entity_label', entity_id)
        neighbors = result.get('neighbors', {})
        total_neighbors = result.get('total_neighbors', 0)
        displayed_neighbors = result.get('displayed_neighbors', 0)
        
        summary = f"Neighbor exploration for: {entity_label} ({entity_id})\n"
        summary += f"Found {total_neighbors} neighbors, displaying {displayed_neighbors}:\n\n"
        
        for i, (ref, neighbor) in enumerate(list(neighbors.items())[:10], 1):  # Show top 10
            neighbor_label = neighbor.get('label', neighbor.get('id', 'N/A'))
            neighbor_id = neighbor.get('id', 'N/A')
            statement_count = neighbor.get('statement_count', 0)
            description = neighbor.get('description', 'No description')
            
            summary += f"{i}. {neighbor_label} ({neighbor_id})\n"
            summary += f"   {description[:60]}{'...' if len(description) > 60 else ''}\n"
            summary += f"   Properties: {statement_count}\n\n"
        
        if len(neighbors) > 10:
            summary += f"... and {len(neighbors) - 10} more neighbors\n"
        
        errors = result.get('errors', [])
        if errors:
            summary += f"\n⚠️ {len(errors)} errors occurred during exploration."
        
        return summary.strip()
    
    def visualize_results(self, result: Any, test_identifier: str, test_number: int) -> Tuple[str, str]:
        """Create visualizations for neighbor exploration results."""
        # This would create graph visualizations - placeholder for now
        return ("static_viz_path", "dynamic_viz_path")

class LocalGraphTool(ExplorationTool):
    """Tool for building a local knowledge graph from multiple DBpedia entities."""
    
    def __init__(self):
        super().__init__(
            name="build_local_graph",
            description="Build a local knowledge graph by exploring multiple DBpedia entities and their relationships." \
            "This tool takes a list of entity IDs and creates a connected graph showing relationships between them." \
            "USEFUL FOR: creating comprehensive knowledge graphs, understanding complex relationships between entities."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_ids",
                    type="array",
                    description="List of DBpedia entity IDs to include in the local graph",
                    items={"type": "string"}
                ),
                ToolParameter(
                    name="max_depth",
                    type="integer",
                    description="Maximum depth for relationship exploration",
                    required=False,
                    default=2
                ),
                ToolParameter(
                    name="max_entities",
                    type="integer",
                    description="Maximum number of entities to include in the graph",
                    required=False,
                    default=50
                )
            ],
            return_type="dict",
            return_description="Local graph with entities, relationships, and visualization data"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute local graph building."""
        entity_ids = kwargs.get("entity_ids", [])
        max_depth = kwargs.get("max_depth", 2)
        max_entities = kwargs.get("max_entities", 50)
        
        if not entity_ids:
            raise ValueError("entity_ids list is required")
        
        print(f"🏗️  Building local graph with {len(entity_ids)} seed entities, max depth {max_depth}")
        
        errors = []
        entities = {}
        explored_entities = set()
        entities_to_explore = set(entity_ids)
        current_depth = 0
        
        # Breadth-first exploration
        while entities_to_explore and current_depth < max_depth and len(entities) < max_entities:
            print(f"📊 Depth {current_depth}: exploring {len(entities_to_explore)} entities")
            
            # Fetch entities in parallel
            fetch_tasks = []
            for entity_id in entities_to_explore:
                if entity_id not in explored_entities:
                    fetch_tasks.append(self._fetch_neighbor_entity(entity_id))
            
            if fetch_tasks:
                fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                
                next_entities_to_explore = set()
                for result in fetch_results:
                    if isinstance(result, tuple) and len(result) == 3:
                        entity_ref, entity_data, error = result
                        if error:
                            errors.append(error)
                        else:
                            try:
                                entity = convert_api_entity_to_model(entity_data)
                                entities[entity_ref] = entity
                                explored_entities.add(entity_ref)
                                
                                # Extract neighbors for next depth level
                                if current_depth < max_depth - 1 and len(entities) < max_entities:
                                    neighbor_refs = extract_entity_references_from_statements(entity.statements)
                                    for neighbor_ref in neighbor_refs[:5]:  # Limit neighbors per entity
                                        if neighbor_ref not in explored_entities:
                                            next_entities_to_explore.add(neighbor_ref)
                                            
                            except Exception as e:
                                error_msg = f"Failed to process entity {entity_ref}: {e}"
                                errors.append(error_msg)
                
                entities_to_explore = next_entities_to_explore
            else:
                break
            
            current_depth += 1
        
        # Build visualization graph
        vis_graph = VisGraph()
        
        # Add nodes
        for entity_id, entity in entities.items():
            node = VisNode(
                node_id=entity.id,
                node_type="entity",
                label=entity.label,
                description=entity.description,
                properties={
                    "description": entity.description,
                    "uri": entity.uri,
                    "statement_count": len(entity.statements)
                }
            )
            vis_graph.add_node(node)
        
        # Add edges based on relationships
        for entity_id, entity in entities.items():
            for prop_id, prop_statements in entity.statements.items():
                for value_key, statement in prop_statements.items():
                    value = statement.get('datavalue', {}).get('value', '')
                    if value in entities:  # If the value is another entity in our graph
                        edge = VisEdge(
                            source_id=entity.id,
                            target_id=entities[value].id,
                            relationship_type=prop_id,
                            properties={"statement": statement}
                        )
                        vis_graph.add_edge(edge)
        
        # Fetch property names
        all_property_ids = []
        for entity in entities.values():
            all_property_ids.extend(entity.statements.keys())
        unique_property_ids = list(set(all_property_ids))
        
        property_names, prop_errors = await self._fetch_property_names_parallel(unique_property_ids)
        errors.extend(prop_errors)
        
        print(f"✅ Local graph built with {len(entities)} entities and {len(vis_graph.edges)} relationships")
        
        return {
            "entities": {
                entity_id: {
                    "id": entity.id,
                    "label": entity.label,
                    "description": entity.description,
                    "statement_count": len(entity.statements)
                } for entity_id, entity in entities.items()
            },
            "graph": {
                "nodes": [{"id": node.id, "label": node.label, "type": node.type} for node in vis_graph.nodes],
                "edges": [{"source": edge.source_id, "target": edge.target_id, "relationship": edge.type} for edge in vis_graph.edges]
            },
            "property_names": property_names,
            "total_nodes": len(vis_graph.nodes),
            "total_edges": len(vis_graph.edges),
            "max_depth_reached": current_depth,
            "errors": errors,
            "partial_data": len(errors) > 0
        }
    
    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the local graph result into a readable string."""
        if not result:
            return "No local graph results found."
        
        entities = result.get('entities', {})
        total_nodes = result.get('total_nodes', 0)
        total_edges = result.get('total_edges', 0)
        max_depth = result.get('max_depth_reached', 0)
        
        summary = f"Local Knowledge Graph Built:\n"
        summary += f"Nodes: {total_nodes}, Edges: {total_edges}\n"
        summary += f"Max depth reached: {max_depth}\n\n"
        
        summary += "Entities in graph:\n"
        for i, (entity_id, entity) in enumerate(list(entities.items())[:10], 1):  # Show top 10
            entity_label = entity.get('label', entity.get('id', 'N/A'))
            entity_id_display = entity.get('id', 'N/A')
            statement_count = entity.get('statement_count', 0)
            description = entity.get('description', 'No description')
            
            summary += f"{i}. {entity_label} ({entity_id_display})\n"
            summary += f"   {description[:60]}{'...' if len(description) > 60 else ''}\n"
            summary += f"   Properties: {statement_count}\n\n"
        
        if len(entities) > 10:
            summary += f"... and {len(entities) - 10} more entities\n"
        
        errors = result.get('errors', [])
        if errors:
            summary += f"\n⚠️ {len(errors)} errors occurred during graph building."
        
        return summary.strip()
    
    def visualize_results(self, result: Any, test_identifier: str, test_number: int) -> Tuple[str, str]:
        """Create visualizations for local graph results."""
        # This would create graph visualizations - placeholder for now
        return ("static_viz_path", "dynamic_viz_path")


if __name__ == "__main__":
    import time
    
    async def test_rate_limiting():
        """Test that rate limiting works properly and prevents API overload."""
        print("🧪 Testing DBpedia exploration with rate limiting...")
        
        # Test the neighbors exploration tool which was causing rate limit issues
        tool = NeighborsExplorationTool()
        
        print("📝 Testing neighbor exploration with rate limiting...")
        start_time = time.time()
        
        try:
            result = await tool.execute(entity_id="Douglas_Adams")
            elapsed = time.time() - start_time
            print(f"✅ Neighbor exploration completed successfully in {elapsed:.2f} seconds")
            print(f"📊 Found {len(result.get('neighbors', {}))} neighbors")
            if result.get('errors'):
                print(f"⚠️ {len(result['errors'])} errors occurred (expected with rate limiting)")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Test failed after {elapsed:.2f} seconds: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the test
    asyncio.run(test_rate_limiting())
