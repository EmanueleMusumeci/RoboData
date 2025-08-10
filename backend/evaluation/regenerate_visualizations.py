#!/usr/bin/env python3
"""
Regenerates the static graph visualization for all completed experiments.

This script iterates through all subdirectories in the 'experiments' folder,
loads the knowledge graph and final answer for each, and then generates
an updated static visualization image, saving it within the experiment's folder.
"""

import sys
import json
from pathlib import Path
import traceback

# Add backend to path to resolve local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.toolbox.graph.visualization import GraphVisualizer
from backend.core.knowledge_base.schema import Graph, Node, Edge

def regenerate_all_visualizations():
    """
    Iterates through all experiment folders and regenerates the static graph visualization.
    """
    experiments_root = Path(__file__).resolve().parents[2] / 'experiments'
    if not experiments_root.exists():
        print(f"❌ Experiments directory not found at: {experiments_root}")
        return

    print(f"🔍 Starting regeneration process in: {experiments_root}")
    
    visualizer = GraphVisualizer() # Re-use the same visualizer instance

    for experiment_dir in experiments_root.iterdir():
        if not experiment_dir.is_dir():
            continue

        print(f"\nProcessing experiment: {experiment_dir.name}")

        kg_file = experiment_dir / 'knowledge_graph.json'
        final_answer_file = experiment_dir / 'final_answer.txt'

        if not kg_file.exists():
            print(f"  -> ⚠️  Skipping: knowledge_graph.json not found.")
            continue
        
        if not final_answer_file.exists():
            print(f"  -> ⚠️  Skipping: final_answer.txt not found.")
            continue

        try:
            # Load data
            with open(kg_file) as f:
                kg_data = json.load(f)
            
            with open(final_answer_file) as f:
                final_answer = f.read()
            
            print(f"  -> 📊 Loaded {len(kg_data.get('nodes', []))} nodes and {len(kg_data.get('edges', []))} edges.")

            # Create Graph object
            graph = Graph()
            for node_data in kg_data.get('nodes', []):
                graph.add_node(Node(
                    node_id=node_data['id'],
                    node_type=node_data.get('type', 'Unknown'),
                    label=node_data.get('label', ''),
                    description=node_data.get('description', ''),
                    properties=node_data.get('properties', {})
                ))
            
            for edge_data in kg_data.get('edges', []):
                graph.add_edge(Edge(
                    source_id=edge_data['source_id'],
                    target_id=edge_data['target_id'],
                    relationship_type=edge_data['type'],
                    label=edge_data.get('label', ''),
                    description=edge_data.get('description', ''),
                    properties=edge_data.get('properties', {})
                ))

            # Set the output directory for the visualizer to the current experiment folder
            visualizer.output_dir = experiment_dir

            # Generate new visualization
            viz_path = visualizer.create_static_visualization(
                graph,
                filename='knowledge_graph_visualization.png',
                final_answer=final_answer
            )
            
            print(f"  -> ✅ Successfully generated visualization: {viz_path}")

        except Exception as e:
            print(f"  -> ❌ Error processing {experiment_dir.name}: {e}")
            traceback.print_exc()

    print("\n🎉 All visualizations regenerated.")

if __name__ == "__main__":
    regenerate_all_visualizations()
