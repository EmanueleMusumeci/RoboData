from typing import Dict, Any, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from pathlib import Path
import asyncio
import json

try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    class Network: pass  # Dummy class for linter

from ...knowledge_base.graph import get_knowledge_graph
from ...knowledge_base.schema import Graph, Node, Edge


class GraphVisualizer:
    """Comprehensive graph visualization toolkit for Wikidata exploration results."""
    
    def __init__(self, output_dir: str = "/tmp/graph_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def create_static_visualization(
        self, 
        graph: Graph, 
        title: str = "Wikidata Graph",
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12),
        node_size_multiplier: float = 1.0,
        font_size: int = 8
    ) -> str:
        """
        Create a static graph visualization with labeled nodes and edges.
        
        Args:
            nodes: Dictionary of node data with IDs as keys
            edges: List of edge dictionaries with source, target, property
            title: Graph title
            filename: Output filename (auto-generated if None)
            figsize: Figure size tuple
            node_size_multiplier: Multiplier for node sizes
            font_size: Base font size
            
        Returns:
            Path to the generated image file
        """
        # Use the networkx graph from our Graph object
        G = graph._graph
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Generate layout
        if len(list(G.nodes())) > 50:
            layout = nx.spring_layout(G, k=3, iterations=50)
        else:
            layout = nx.spring_layout(G, k=2, iterations=100)
        
        # Draw edges first (so they appear behind nodes)
        edge_colors = ['#666666' for _ in G.edges()]
        nx.draw_networkx_edges(
            G, layout, 
            edge_color=edge_colors, # type: ignore
            alpha=0.6,
            width=1.5,
            ax=ax
        )
        
        # Calculate node sizes based on degree
        degrees = dict(G.degree()) # type: ignore
        max_degree = max(degrees.values()) if degrees else 1
        node_sizes = [
            300 + (degrees.get(node, 1) / max_degree) * 1000 * node_size_multiplier 
            for node in G.nodes()
        ]
        
        # Color nodes by depth if available
        node_colors = []
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            depth = node_data.get('properties', {}).get('depth', 0)
            if depth == 0:
                node_colors.append('#FF6B6B')  # Center node - red
            elif depth == 1:
                node_colors.append('#4ECDC4')  # First level - teal
            elif depth == 2:
                node_colors.append('#45B7D1')  # Second level - blue
            else:
                node_colors.append('#96CEB4')  # Other levels - green
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, layout,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax
        )
        
        # Add node labels (entity labels above nodes)
        node_label_pos = {node: (x, y + 0.15) for node, (x, y) in layout.items()}
        node_labels = {
            node_id: G.nodes[node_id].get('label', node_id)[:30] + ('...' if len(G.nodes[node_id].get('label', '')) > 30 else '')
            for node_id in G.nodes()
        }
        
        nx.draw_networkx_labels(
            G, node_label_pos,
            labels=node_labels,
            font_size=font_size,
            font_weight='bold',
            ax=ax
        )
        
        # Add node IDs at center of nodes
        nx.draw_networkx_labels(
            G, layout,
            labels={node_id: node_id for node_id in G.nodes()},
            font_size=max(6, font_size - 2),
            font_color='white',
            ax=ax
        )
        
        # Add edge labels (property names and IDs)
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            prop_id = data.get('type', '')
            prop_name = data.get('label', prop_id)
            if prop_name and prop_name != prop_id:
                label = f"{prop_name}\n({prop_id})"
            else:
                label = prop_id
            edge_labels[(u, v)] = label

        for (u, v), label in edge_labels.items():
            pos_u = layout[u]
            pos_v = layout[v]
            pos = ((pos_u[0] + pos_v[0]) / 2, (pos_u[1] + pos_v[1]) / 2)
            ax.text(
                pos[0], pos[1], label,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=max(6, font_size - 3),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                style='italic'
            )

        # Add legend
        legend_elements = [
            patches.Patch(color='#FF6B6B', label='Center Entity'),
            patches.Patch(color='#4ECDC4', label='Depth 1'),
            patches.Patch(color='#45B7D1', label='Depth 2'),
            patches.Patch(color='#96CEB4', label='Depth 3+')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Format axes
        ax.set_axis_off()
        plt.tight_layout()
        
        # Save the figure
        if filename is None:
            filename = f"graph_{title.lower().replace(' ', '_')}.png"
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Static visualization saved to: {filepath}")
        return str(filepath)
    
    def create_dynamic_visualization(
        self,
        graph: Graph,
        title: str = "Interactive Wikidata Graph",
        filename: Optional[str] = None,
        height: str = "800px",
        width: str = "100%"
    ) -> str:
        """
        Create an interactive, draggable graph visualization using pyvis.
        
        Args:
            graph: Graph object to visualize
            title: Graph title
            filename: Output HTML filename
            height: HTML canvas height
            width: HTML canvas width
            
        Returns:
            Path to the generated HTML file
        """
        if not PYVIS_AVAILABLE:
            raise ImportError("pyvis not available. Install with: pip install pyvis")
        
        net = Network(
            height=height,
            width=width,
            bgcolor="#ffffff",
            font_color="black", # type: ignore
            directed=True
        )
        
        # Configure physics for elastic layout
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.1
            }
          },
          "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
          }
        }
        """)
        
        # Add nodes
        for node in graph.nodes:
            if node is None: continue
            label = node.label or node.id
            depth = node.properties.get('depth', 0)
            
            # Color based on depth
            if depth == 0:
                color = "#FF6B6B"  # Center node - red
                size = 30
            elif depth == 1:
                color = "#4ECDC4"  # First level - teal
                size = 25
            elif depth == 2:
                color = "#45B7D1"  # Second level - blue
                size = 20
            else:
                color = "#96CEB4"  # Other levels - green
                size = 15
            
            # Create hover info
            title_text = f"""
            <b>{label}</b><br>
            ID: {node.id}<br>
            Depth: {depth}<br>
            Description: {(node.description or 'No description')[:100]}...
            """
            
            net.add_node(
                node.id,
                label=f"{label}\n{node.id}",
                title=title_text,
                color=color,
                size=size,
                font={"size": 12, "color": "black"}
            )
        
        # Add edges
        for edge in graph.edges:
            if edge is None: continue
            prop_id = edge.type
            prop_name = edge.label or prop_id
            
            if prop_name and prop_name != prop_id:
                edge_label = f"{prop_name} ({prop_id})"
            else:
                edge_label = prop_id
            
            net.add_edge(
                edge.source_id,
                edge.target_id,
                label=edge_label,
                color="#666666",
                width=2,
                font={"size": 10, "color": "black"}
            )
        
        # Generate filename
        if filename is None:
            filename = f"interactive_graph_{title.lower().replace(' ', '_')}.html"
        
        filepath = self.output_dir / filename
        
        # Save the network
        net.show(str(filepath), notebook=False)
        
        print(f"🌐 Interactive visualization saved to: {filepath}")
        return str(filepath)
    
    async def store_in_neo4j(
        self,
        graph: Graph,
        clear_existing: bool = False
    ) -> bool:
        """
        Store graph data in Neo4j for advanced visualization and querying.
        
        Args:
            graph: Graph object to store
            clear_existing: Whether to clear existing data first
            
        Returns:
            Success status
        """
        try:
            db = get_knowledge_graph()
            if not await db.is_connected():
                await db.connect()
            
            # Clear existing data if requested
            if clear_existing:
                await db.clear_graph()
                print("🗑️  Cleared existing Neo4j data")
            
            # Add nodes
            node_count = 0
            for node in graph.nodes:
                if node:
                    await db.add_entity(node)
                    node_count += 1
            
            # Add edges
            edge_count = 0
            for edge in graph.edges:
                if edge:
                    await db.add_relationship(edge)
                    edge_count += 1
            
            print(f"💾 Stored {node_count} nodes and {edge_count} edges in Neo4j")
            return True
            
        except Exception as e:
            print(f"❌ Failed to store in Neo4j: {e}")
            return False
    
    def visualize_exploration_results(
        self,
        exploration_result: Dict[str, Any],
        title_prefix: str = "Entity Exploration"
    ) -> Tuple[str, str]:
        """
        Visualize results from NeighborsExplorationTool.
        
        Args:
            exploration_result: Result from exploration tool
            title_prefix: Prefix for visualization titles
            
        Returns:
            Tuple of (static_path, dynamic_path)
        """
        # Extract entity info
        entity = exploration_result['entity']
        neighbors = exploration_result['neighbors']
        
        graph = Graph()
        
        # Add center node
        center_node = Node(
            node_id=entity['id'],
            node_type='WikidataEntity',
            label=entity['label'],
            description=entity['description'],
            properties={'depth': 0}
        )
        graph.add_node(center_node)
        
        # Add neighbor nodes
        for neighbor_id, neighbor_data in neighbors.items():
            neighbor_node = Node(
                node_id=neighbor_id,
                node_type='WikidataEntity',
                label=neighbor_data['label'],
                description=neighbor_data['description'],
                properties={'depth': 1}
            )
            graph.add_node(neighbor_node)
        
        # Build edges from relationships
        relationships = exploration_result['relationships']
        
        for prop_id, values in relationships.items():
            for value in values:
                if graph.get_node(value):
                    edge = Edge(
                        source_id=entity['id'],
                        target_id=value,
                        relationship_type=prop_id,
                        label=exploration_result.get('property_names', {}).get(prop_id, prop_id)
                    )
                    graph.add_edge(edge)
        
        # Create visualizations
        title = f"{title_prefix}: {entity['label']} ({entity['id']})"
        
        static_path = self.create_static_visualization(
            graph, title,
            filename=f"exploration_{entity['id']}.png"
        )
        
        dynamic_path = self.create_dynamic_visualization(
            graph, title,
            filename=f"exploration_{entity['id']}.html"
        )
        
        return static_path, dynamic_path
    
    def visualize_local_graph(
        self,
        graph_result: Dict[str, Any],
        title_prefix: str = "Local Graph"
    ) -> Tuple[str, str]:
        """
        Visualize results from LocalGraphTool.
        
        Args:
            graph_result: Result from local graph tool
            title_prefix: Prefix for visualization titles
            
        Returns:
            Tuple of (static_path, dynamic_path)
        """
        nodes_data = graph_result['nodes']
        edges_data = graph_result['edges']
        center = graph_result['center']
        depth = graph_result['depth']
        
        graph = Graph()
        for node_id, node_info in nodes_data.items():
            graph.add_node(Node(
                node_id=node_id,
                node_type='WikidataEntity',
                label=node_info.get('label', ''),
                description=node_info.get('description', ''),
                properties=node_info
            ))
            
        for edge_info in edges_data:
            graph.add_edge(Edge(
                source_id=edge_info['source'],
                target_id=edge_info['target'],
                relationship_type=edge_info.get('property', ''),
                label=edge_info.get('property_name', '')
            ))

        title = f"{title_prefix}: {center} (Depth {depth})"
        
        static_path = self.create_static_visualization(
            graph, title,
            filename=f"local_graph_{center}_d{depth}.png"
        )
        
        dynamic_path = self.create_dynamic_visualization(
            graph, title,
            filename=f"local_graph_{center}_d{depth}.html"
        )
        
        return static_path, dynamic_path


# Global visualizer instance
visualizer = GraphVisualizer()

if __name__ == "__main__":
    async def test_visualization():
        """Test visualization functionality with sample data."""
        print("=== Testing Graph Visualization ===\n")
        
        # Create sample graph data
        graph = Graph()
        graph.add_node(Node(
            "Q42", "WikidataEntity", "Douglas Adams", "English writer and humorist",
            properties={'depth': 0}
        ))
        graph.add_node(Node(
            "Q5", "WikidataEntity", "human", "common name of Homo sapiens",
            properties={'depth': 1}
        ))
        graph.add_node(Node(
            "Q6581097", "WikidataEntity", "male", "male sex or gender",
            properties={'depth': 1}
        ))
        graph.add_node(Node(
            "Q36180", "WikidataEntity", "writer", "person who writes books or other texts",
            properties={'depth': 1}
        ))
        
        graph.add_edge(Edge("Q42", "Q5", "P31", "instance of"))
        graph.add_edge(Edge("Q42", "Q6581097", "P21", "sex or gender"))
        graph.add_edge(Edge("Q42", "Q36180", "P106", "occupation"))

        viz = GraphVisualizer()
        
        # Test static visualization
        print("1. Testing static visualization...")
        try:
            static_path = viz.create_static_visualization(
                graph, 
                "Sample Wikidata Graph"
            )
            print(f"✓ Static visualization created successfully")
        except Exception as e:
            print(f"✗ Static visualization failed: {e}")
        
        # Test dynamic visualization
        print("\n2. Testing dynamic visualization...")
        try:
            if PYVIS_AVAILABLE:
                dynamic_path = viz.create_dynamic_visualization(
                    graph,
                    "Sample Interactive Graph"
                )
                print(f"✓ Dynamic visualization created successfully")
            else:
                print("⚠️  Pyvis not available, skipping dynamic visualization")
        except Exception as e:
            print(f"✗ Dynamic visualization failed: {e}")
        
        # Test Neo4j storage
        print("\n3. Testing Neo4j storage...")
        try:
            success = await viz.store_in_neo4j(
                graph,
                clear_existing=True
            )
            if success:
                print(f"✓ Neo4j storage successful")
            else:
                print(f"✗ Neo4j storage failed")
        except Exception as e:
            print(f"⚠️  Neo4j storage error (database may not be running): {e}")
        
        print("\n=== Visualization tests completed ===")
    
    asyncio.run(test_visualization())
