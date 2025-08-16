# Knowledge Graph Visualization

This document explains the new knowledge graph visualization features added to the RoboData system.

## Overview

The system now automatically generates visualizations of the knowledge graph at the end of each turn, allowing you to see how the graph evolves during the exploration process.

## Features

### 1. Turn-by-Turn Visualizations
- Automatically generates a PNG image at the end of each turn
- Images are named with format: `turn_XXX_<substate>.png`
- Shows the current state of the knowledge graph with turn and substate information

### 2. Final Graph Visualization
- Creates a final visualization with support sets highlighted
- Support sets are extracted from the final answer and color-coded
- Named: `final_graph_with_answer.png`

### 3. Evolution Animation
- Automatically creates an animated GIF showing the complete evolution
- Named: `graph_evolution.gif`
- Each frame shows one turn of the process

### 4. HTML Index
- Creates an `index.html` file with all visualizations organized
- Includes the query, experiment details, and embedded images
- Provides a complete overview of the exploration process

## Output Structure

```
/tmp/graph_visualizations/<experiment_id>/
├── turn_001_eval_local_data.png
├── turn_002_remote_graph_exploration.png
├── turn_003_local_graph_update.png
├── turn_004_eval_local_data.png
├── turn_005_produce_answer.png
├── final_graph_with_answer.png
├── graph_evolution.gif
└── index.html
```

## Result Information

When a query completes, the result now includes a `visualizations` section:

```python
{
    "answer": "...",
    "turns_taken": 5,
    "visualizations": {
        "output_dir": "/tmp/graph_visualizations/experiment_123",
        "animation_path": "/tmp/graph_visualizations/experiment_123/graph_evolution.gif",
        "final_image": "/tmp/graph_visualizations/experiment_123/final_graph_with_answer.png",
        "index_html": "/tmp/graph_visualizations/experiment_123/index.html"
    }
}
```

## Usage

### Command Line
The visualizations are generated automatically when running queries:

```bash
python main.py -q "Who was Albert Einstein?"
```

### Programmatic Access
```python
result = await orchestrator.process_user_query("Who was Albert Einstein?")
vis_info = result['visualizations']
print(f"View results at: {vis_info['index_html']}")
```

## Features

### Support Set Highlighting
The final visualization highlights entities and relationships that were used as evidence in the final answer. Each support set (proof set) is color-coded to match the sentences in the answer.

### Node Sizing
Nodes are sized based on their connectivity (degree) in the graph:
- Small nodes: Low connectivity
- Medium nodes: Medium connectivity 
- Large nodes: High connectivity

### Edge Labels
Relationship types are displayed as edge labels, truncated for readability.

### Layout Optimization
- Uses a spring layout algorithm for natural positioning
- Prevents label overlap using automatic text adjustment
- Includes arrows to show relationship direction

## Dependencies

The visualization system requires:
- matplotlib (for static graphs)
- networkx (for graph layout)
- adjustText (for label positioning)
- PIL/Pillow (for animation generation, optional)

## Configuration

Visualization settings can be customized in the GraphVisualizer class:
- Output directory
- Image size and resolution
- Node size multipliers
- Font sizes
- Color schemes

## Troubleshooting

### No Images Generated
- Check that the knowledge graph contains nodes and edges
- Verify write permissions to `/tmp/graph_visualizations/`
- Check logs for visualization errors (they don't break execution)

### Animation Not Created
- Ensure PIL/Pillow is installed: `pip install Pillow`
- Check that multiple turn images exist

### Large File Sizes
- Reduce `figsize` parameter in visualization calls
- Lower DPI setting (currently 300)
- Use smaller `node_size_multiplier` values

## Examples

### Viewing Results
After running a query, open the HTML index:
```bash
# Run a query
python main.py -q "What is the capital of France?"

# View results (example path)
open /tmp/graph_visualizations/experiment_20250809_123456/index.html
```

### Creating Custom Visualizations
```python
from backend.core.toolbox.graph.visualization import GraphVisualizer

visualizer = GraphVisualizer(output_dir="/my/custom/path")
image_path = visualizer.create_static_visualization(
    graph=my_graph,
    title="My Custom Graph",
    filename="custom_graph.png",
    figsize=(20, 16)
)
```
