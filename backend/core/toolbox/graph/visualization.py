from typing import Dict, Any, List, Optional, Tuple, Set
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from pathlib import Path
import asyncio
import json
import re
import math
import uuid
from adjustText import adjust_text

try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from pyvis.network import Network as PyvisNetwork  # type: ignore
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    class PyvisNetwork: pass  # Dummy class for linter

from backend.core.knowledge_base.graph import get_knowledge_graph
from backend.core.knowledge_base.schema import Graph, Node, Edge


def _format_label(label: str, item_id: str, max_word_len: int = 20, max_id_len: int = 11, words_per_line: int = 3) -> str:
    """Formats a label for display by wrapping, truncating long words, and truncating the ID."""
    # Truncate long words
    words = label.split(' ')
    truncated_words = []
    for word in words:
        if len(word) > max_word_len:
            truncated_words.append(word[:max_word_len - 3] + '...')
        else:
            truncated_words.append(word)
    
    # Wrap text
    wrapped_label = '\n'.join([' '.join(truncated_words[i:i + words_per_line]) for i in range(0, len(truncated_words), words_per_line)])

    # Truncate ID
    truncated_id = (item_id[:max_id_len - 3] + '...') if len(item_id) > max_id_len else item_id
    
    return f"{wrapped_label}\n({truncated_id})"


class GraphVisualizer:
    """Comprehensive graph visualization toolkit for Wikidata exploration results."""
    
    def __init__(self, output_dir: str = "/tmp/graph_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def extract_support_sets_from_answer(self, final_answer: str) -> List[List[str]]:
        """
        Extract support sets from the final answer text.
        
        Args:
            final_answer: The final answer containing support sets
            
        Returns:
            List of support sets, where each support set is a list of entity IDs
        """
        support_sets = []
        if not final_answer:
            return support_sets
        
        # Handle both old format "SUPPORT SET 001:" and new format "SUPPORT SET PS_001:"
        # Pattern to match support set headers
        support_header_pattern = r'SUPPORT SET (?:PS_)?(\d+):'
        
        # Split the text into lines for processing
        lines = final_answer.split('\n')
        
        current_support_set = []
        inside_support_set = False
        
        for line in lines:
            line = line.strip()
            
            # Check if this line is a support set header
            header_match = re.match(support_header_pattern, line)
            if header_match:
                # If we were already processing a support set, save it
                if current_support_set:
                    support_sets.append(current_support_set)
                
                # Start a new support set
                current_support_set = []
                inside_support_set = True
                
                # Check if the support set is on the same line (old format)
                remainder = line[header_match.end():].strip()
                if remainder:
                    # Old format: SUPPORT SET 001: (Q42, P22, Q14623675)
                    triple_match = re.search(r'\(([^)]+)\)', remainder)
                    if triple_match:
                        entities = self._extract_entities_from_triple(triple_match.group(1))
                        current_support_set.extend(entities)
                
                continue
            
            # If we're inside a support set and this line contains a triple
            if inside_support_set and line.startswith('(') and line.endswith(')'):
                # New format: each triple on its own line
                triple_content = line[1:-1]  # Remove parentheses
                entities = self._extract_entities_from_triple(triple_content)
                current_support_set.extend(entities)
                continue
            
            # If we encounter a line that doesn't belong to the support set, stop processing
            if inside_support_set and line and not line.startswith('('):
                # Check if it's another section (like "SENTENCE" or "FINAL ANSWER")
                if any(keyword in line.upper() for keyword in ['SENTENCE', 'FINAL ANSWER', 'SUPPORT SET']):
                    inside_support_set = False
        
        # Don't forget the last support set
        if current_support_set:
            support_sets.append(current_support_set)
        
        return support_sets
    
    def _extract_entities_from_triple(self, triple_content: str) -> List[str]:
        """
        Extract entity IDs from a triple string like "Q42, P22, Q14623675".
        
        Args:
            triple_content: Content of the triple without parentheses
            
        Returns:
            List of entity IDs found in the triple
        """
        entities = []
        parts = triple_content.split(',')
        for part in parts:
            part = part.strip()
            # Look for ID in parentheses or plain ID
            id_match = re.search(r'\(([QP]\d+)\)', part)
            if id_match:
                entities.append(id_match.group(1))
            elif re.match(r'^[QP]\d+$', part):
                entities.append(part)
        
        return entities
        
        return support_sets

    def create_static_visualization(
        self,
        graph: Graph,
        title: Optional[str] = None,
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 16),
        node_size_multiplier: float = 1.0,
        font_size: int = 12,
        final_answer: Optional[str] = None
    ) -> str:
        """
        Create a static graph visualization with labeled nodes and edges, optimized layout, 
        and support set highlighting.
        
        Args:
            graph: Graph object to visualize
            title: Optional title for the graph
            filename: Output filename (auto-generated if None)
            figsize: Figure size tuple
            node_size_multiplier: Multiplier for node sizes
            font_size: Base font size
            final_answer: Final answer text containing support sets to highlight
            
        Returns:
            Path to the generated image file
        """
        # Use the networkx graph from our Graph object
        G = graph._graph
        
        # Extract support sets from final answer if provided
        support_sets = []
        if final_answer:
            support_sets = self.extract_support_sets_from_answer(final_answer)
        
        # Create support set color mapping
        support_set_colors = [
            '#FF1744',  # Red
            '#FF9100',  # Orange  
            '#00E676',  # Green
            '#00BCD4',  # Cyan
            '#9C27B0',  # Purple
            '#FF5722',  # Deep Orange
            '#4CAF50',  # Light Green
            '#2196F3',  # Blue
            '#E91E63',  # Pink
            '#795548'   # Brown
        ]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate an elastic layout with more spacing to ensure edge length and component separation
        layout = nx.spring_layout(G, k=4.0, iterations=500, seed=42)

        # Calculate degrees for node sizing
        degrees = dict(G.degree())  # type: ignore
        degree_values = [int(deg) for deg in degrees.values()] if degrees else [1]
        max_degree = max(degree_values) if degree_values else 1
        min_degree = min(degree_values) if degree_values else 1
        
        # Separate nodes into three size categories based on degree
        node_sizes = []
        for node in G.nodes():
            degree = int(degrees.get(node, 1))
            if max_degree == min_degree:
                size = 1200
            else:
                # Normalize degree to 0-1 range
                normalized_degree = (degree - min_degree) / (max_degree - min_degree)
                
                # Three size categories: small, medium, large
                if normalized_degree < 0.33:
                    size = 1000
                elif normalized_degree < 0.67:
                    size = 1800
                else:
                    size = 3000
            
            node_sizes.append(size * node_size_multiplier)

        # Determine node and edge colors based on support sets
        node_colors = {}
        edge_colors = {}
        edge_widths = {}

        default_node_color = '#cccccc'  # Grey
        default_edge_color = '#cccccc'

        for node_id in G.nodes():
            node_colors[node_id] = default_node_color

        for u, v in G.edges():
            edge_colors[(u, v)] = default_edge_color
            edge_widths[(u, v)] = 1.5

        for i, support_set in enumerate(support_sets):
            color = support_set_colors[i % len(support_set_colors)]
            # A support set is (source, relation, target)
            if len(support_set) >= 3:
                source, _, target = support_set[0], support_set[1], support_set[2]
                if source in G:
                    node_colors[source] = color
                if target in G:
                    node_colors[target] = color
                if G.has_edge(source, target):
                    edge_colors[(source, target)] = color
                    edge_widths[(source, target)] = 3.0
                # Also handle reverse edge if graph is not directed
                if not G.is_directed() and G.has_edge(target, source):
                    edge_colors[(target, source)] = color
                    edge_widths[(target, source)] = 3.0

        # Draw edges
        edge_color_list = [edge_colors.get(edge, default_edge_color) for edge in G.edges()]
        edge_width_list = [edge_widths.get(edge, 1.5) for edge in G.edges()]

        nx.draw_networkx_edges(
            G, layout, 
            edge_color=edge_color_list,
            width=edge_width_list,
            alpha=0.8,
            ax=ax,
            arrows=True,
            arrowsize=30,
            arrowstyle='-|>'
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            G, layout,
            node_color=[node_colors.get(node, default_node_color) for node in G.nodes()],
            node_size=node_sizes,
            alpha=0.95,
            ax=ax
        )
        
        # Add node labels with overlap avoidance
        node_labels = {
            node_id: _format_label(G.nodes[node_id].get('label', node_id), node_id)
            for node_id in G.nodes()
        }
        
        texts = []
        for node_id, text in node_labels.items():
            x, y = layout[node_id]
            texts.append(ax.text(x, y, text, fontsize=font_size, fontweight='bold', ha='center', va='center'))

        # Use adjust_text to prevent label overlap, with more visible connectors
        adjust_text(
            texts, 
            arrowprops=dict(arrowstyle='-', color='black', lw=1.0),
            force_text=(0.5, 0.5),
            force_points=(0.2, 0.2)
        )

        # Add edge labels
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            prop_id = data.get('type', '')
            prop_name = data.get('label', prop_id)
            edge_labels[(u, v)] = _format_label(prop_name, prop_id, words_per_line=2)

        edge_label_objects = nx.draw_networkx_edge_labels(
            G, layout,
            edge_labels=edge_labels,
            font_size=max(10, font_size - 2),
            font_weight='bold',
            font_color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7)
        )

        # Rerun adjust_text to also avoid edge labels
        adjust_text(
            texts, 
            add_objects=list(edge_label_objects.values()), 
            arrowprops=dict(arrowstyle='-', color='black', lw=1.0),
            force_text=(0.5, 0.5),
            force_points=(0.2, 0.2)
        )

        # Create legend
        legend_elements = []
        if support_sets:
            legend_elements.append(patches.Patch(color='white', label='Sentences:'))
            for i, _ in enumerate(support_sets):
                color = support_set_colors[i % len(support_set_colors)]
                legend_elements.append(
                    patches.Patch(color=color, label=f'{i+1}')
                )
        
        ax.legend(handles=legend_elements, loc='upper right', prop={'weight':'bold', 'size':'x-large'})
        
        # Add title if provided
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Format axes and ensure all nodes fit
        plt.tight_layout()
        ax.margins(0.15) 
        
        # Save the figure
        if filename is None:
            # Create a more generic filename
            filename = f"graph_visualization_{uuid.uuid4().hex[:8]}.png"
        
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
        
        net = PyvisNetwork(  # type: ignore
            height=height,  # type: ignore
            width=width,  # type: ignore
            bgcolor="#ffffff",  # type: ignore
            font_color="black",  # type: ignore
            directed=True  # type: ignore
        )
        
        # Configure physics for elastic layout
        net.set_options("""  # type: ignore[attr-defined]
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
            
            net.add_node(  # type: ignore
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
            
            net.add_edge(  # type: ignore
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
        net.show(str(filepath), notebook=False)  # type: ignore
        
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
            graph,
            title=title,
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
            graph,
            title=title,
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
                title="Sample Wikidata Graph"
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
