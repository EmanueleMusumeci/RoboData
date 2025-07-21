import asyncio
import sys
import argparse
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from settings import settings_manager
from core.agents.gemini import GeminiAgent
from core.agents.openai import OpenAIAgent
from core.agents.wattool_slm import WatToolSLMAgent
from core.knowledge_base.graph import get_knowledge_graph
from core.orchestrator.multi_stage.multi_stage_orchestrator import MultiStageOrchestrator
from core.orchestrator.multi_stage.toolboxes import (
    create_local_exploration_toolbox,
    create_remote_exploration_toolbox,
    create_graph_update_toolbox,
    create_evaluation_toolbox
)
from core.logging import log_debug

def get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
    return {
        "experiment_id": "",
        "orchestrator": {
            "type": "multi_stage",
            "context_length": 8000,
            "model": "gpt-4o",
            "memory": {
                "use_summary_memory": True,
                "max_memory_slots": 50
            },
            "max_turns": 20,
            "toolboxes": {
                "local_exploration": [],
                "remote_exploration": [],
                "graph_update": [],
                "evaluation": []
            }
        },
        "log_level": "DEBUG",
        "memory": "",
        "dataset": {
            "path": "",
            "type": "auto",  # auto-detect, json, jsonl, lcquad
            "load_on_start": False
        },
        "query": "",
        "output": {
            "save_results": True,
            "export_formats": ["json", "cypher"],
            "create_visualizations": True,
            "results_directory": "experiments"
        }
    }


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    # Set defaults
    defaults = get_default_config()
    
    # Merge with defaults
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict) and isinstance(config[key], dict):
            for subkey, subvalue in value.items():
                if subkey not in config[key]:
                    config[key][subkey] = subvalue
    
    return config


def create_agent(model_name: str, toolbox=None):
    """Create the appropriate agent based on model name."""
    if model_name == "local":
        print("Using local WatTool SLM Agent")
        return WatToolSLMAgent(toolbox=toolbox)
    elif model_name.startswith("gpt"):
        print(f"Using OpenAI Agent with model: {model_name}")
        return OpenAIAgent(model=model_name)
    elif model_name.startswith("gemini"):
        print(f"Using Gemini Agent with model: {model_name}")
        # Gemini agent needs a toolbox, create empty one if none provided
        if toolbox is None:
            from core.toolbox.toolbox import Toolbox
            toolbox = Toolbox()
        return GeminiAgent(toolbox=toolbox)
    else:
        # Default to OpenAI
        print(f"Using OpenAI Agent with model: {model_name}")
        return OpenAIAgent(model=model_name)


async def run_multi_stage_orchestrator(config: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Run the multi-stage orchestrator with the given configuration and query."""
    
    # Generate experiment ID if not provided
    experiment_id = config.get("experiment_id") or f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create toolboxes
    local_exploration_toolbox = create_local_exploration_toolbox()
    remote_exploration_toolbox = create_remote_exploration_toolbox()
    graph_update_toolbox = create_graph_update_toolbox()
    evaluation_toolbox = create_evaluation_toolbox()
    
    # Create agent
    orchestrator_config = config["orchestrator"]
    model_name = orchestrator_config.get("model", "gpt-4o")
    
    # Create agent based on model type
    if model_name == "local":
        agent = create_agent(model_name, remote_exploration_toolbox)
    elif model_name.startswith("gemini"):
        agent = create_agent(model_name, remote_exploration_toolbox)
    else:
        agent = create_agent(model_name)
    
    # Get knowledge graph instance
    knowledge_graph = get_knowledge_graph()
    
    # Create orchestrator
    orchestrator = MultiStageOrchestrator(
        agent, 
        knowledge_graph,
        context_length=orchestrator_config.get("context_length", 8000),
        local_exploration_toolbox=local_exploration_toolbox,
        remote_exploration_toolbox=remote_exploration_toolbox,
        graph_update_toolbox=graph_update_toolbox,
        evaluation_toolbox=evaluation_toolbox,
        use_summary_memory=orchestrator_config["memory"].get("use_summary_memory", True),
        memory_max_slots=orchestrator_config["memory"].get("max_memory_slots", 50),
        max_turns=orchestrator_config.get("max_turns", 20),
        experiment_id=experiment_id
    )
    
    # Process the query
    print(f"🤖 Processing query: {query}")
    result = await orchestrator.process_user_query(query)
    
    # Log results
    log_debug(f"Answer: {result['answer']}")
    log_debug(f"Attempts: {result['attempts']}")
    log_debug(f"Turns taken: {result['turns_taken']} / {result['max_turns'] if result['max_turns'] > 0 else 'unlimited'}")
    
    # Save results to files after orchestrator execution (if enabled)
    if config.get("output", {}).get("save_results", True):
        await save_orchestrator_results(result, experiment_id, query, knowledge_graph)
    else:
        print("ℹ️  Skipping result saving (disabled)")
    
    return result


async def save_orchestrator_results(result: Dict[str, Any], experiment_id: str, query: str, knowledge_graph) -> None:
    """
    Save orchestrator results to files including answer, graph export, and visualizations.
    
    Args:
        result: The orchestrator result dictionary
        experiment_id: The experiment ID for naming files
        query: The original query
        knowledge_graph: The knowledge graph instance
    """
    from pathlib import Path
    from datetime import datetime
    import json
    
    # Create results directory
    results_dir = Path("experiments") / experiment_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving results to {results_dir}")
    
    try:
        # 1. Save final answer to text file
        answer_file = results_dir / "final_answer.txt"
        with open(answer_file, 'w', encoding='utf-8') as f:
            f.write(f"Query: {query}\n")
            f.write(f"Experiment ID: {experiment_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Turns taken: {result['turns_taken']}\n")
            f.write(f"Max turns: {result['max_turns']}\n")
            f.write("="*60 + "\n")
            f.write(f"ANSWER:\n{result['answer']}\n")
            f.write("="*60 + "\n")
            f.write(f"Remote explorations: {result['attempts']['remote_explorations']}\n")
            f.write(f"Local explorations: {result['attempts']['local_explorations']}\n")
            if result['attempts']['failures']:
                f.write(f"Failures: {len(result['attempts']['failures'])}\n")
        
        print(f"✅ Final answer saved to {answer_file}")
        
        # 2. Save complete result data as JSON
        complete_result_file = results_dir / "complete_result.json"
        with open(complete_result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "result": result
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Complete result saved to {complete_result_file}")
        
        # 3. Export knowledge graph in multiple formats
        if await knowledge_graph.is_connected():
            try:
                # JSON export
                graph_json_file = results_dir / "knowledge_graph.json"
                await knowledge_graph.export_to_file(graph_json_file, "json")
                print(f"✅ Knowledge graph JSON saved to {graph_json_file}")
                
                # Cypher export
                graph_cypher_file = results_dir / "knowledge_graph.cypher"
                await knowledge_graph.export_to_file(graph_cypher_file, "cypher")
                print(f"✅ Knowledge graph Cypher saved to {graph_cypher_file}")
                
                # GraphML export (if networkx is available)
                try:
                    graph_graphml_file = results_dir / "knowledge_graph.graphml"
                    await knowledge_graph.export_to_file(graph_graphml_file, "graphml")
                    print(f"✅ Knowledge graph GraphML saved to {graph_graphml_file}")
                except Exception as e:
                    log_debug(f"Could not export GraphML: {e}")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not export knowledge graph: {e}")
                log_debug(f"Knowledge graph export error: {e}")
        else:
            print("⚠️  Warning: Knowledge graph not connected, skipping export")
        
        # 4. Create graph visualizations
        try:
            from core.toolbox.graph.visualization import visualizer
            
            # Get current graph data
            graph_data = await knowledge_graph.get_whole_graph()
            
            if graph_data and (graph_data.nodes or graph_data.edges):
                # Create static visualization (PNG)
                static_viz_path = visualizer.create_static_visualization(
                    graph_data,
                    title=f"Knowledge Graph: {experiment_id}",
                    filename=f"{experiment_id}_graph.png"
                )
                
                # Move to results directory
                import shutil
                static_target = results_dir / "knowledge_graph_static.png"
                shutil.move(static_viz_path, static_target)
                print(f"✅ Static graph visualization saved to {static_target}")
                
                # Create interactive visualization (HTML)
                try:
                    dynamic_viz_path = visualizer.create_dynamic_visualization(
                        graph_data,
                        title=f"Interactive Knowledge Graph: {experiment_id}",
                        filename=f"{experiment_id}_graph.html"
                    )
                    
                    # Move to results directory
                    dynamic_target = results_dir / "knowledge_graph_interactive.html"
                    shutil.move(dynamic_viz_path, dynamic_target)
                    print(f"✅ Interactive graph visualization saved to {dynamic_target}")
                    
                except Exception as e:
                    print(f"⚠️  Warning: Could not create interactive visualization: {e}")
                    log_debug(f"Interactive visualization error: {e}")
                
            else:
                print("ℹ️  No graph data to visualize (empty graph)")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not create visualizations: {e}")
            log_debug(f"Visualization error: {e}")
        
        print(f"🎉 All results saved successfully to {results_dir}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        log_debug(f"Error in save_orchestrator_results: {e}")
        import traceback
        traceback.print_exc()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RoboData - AI-powered knowledge exploration system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to experiment configuration YAML file"
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Natural language query to process"
    )
    
    parser.add_argument(
        "-o", "--orchestrator",
        type=str,
        default="multi_stage",
        choices=["multi_stage"],
        help="Orchestrator type to use (default: multi_stage)"
    )
    
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        help="Path to dataset file (JSON, JSONL, or LC-QUAD format)"
    )
    
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["auto", "json", "jsonl", "lcquad"],
        default="auto",
        help="Dataset type (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving results to files"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # If no arguments provided, show help
    if not any([args.config, args.query, args.dataset]):
        print("🤖 RoboData - AI-powered knowledge exploration system")
        print("\nUsage examples:")
        print("  python main.py -q 'What is climate change?'")
        print("  python main.py -c experiment.yaml")
        print("  python main.py -c experiment.yaml -q 'Who was Albert Einstein?'")
        print("  python main.py -d dataset.json --dataset-type lcquad")
        print("\nUse -h for detailed help.")
        return
    
    try:
        # Load configuration
        config = {}
        if args.config:
            config = load_experiment_config(args.config)
            print(f"✅ Loaded configuration from: {args.config}")
        else:
            config = get_default_config()
            print("✅ No config file provided, using default configuration.")
        
        # Handle dataset loading
        dataset_loader = None
        if args.dataset or config.get("dataset", {}).get("path"):
            dataset_path = args.dataset or config["dataset"]["path"]
            dataset_type = args.dataset_type if args.dataset_type != "auto" else config["dataset"].get("type", "auto")
            
            try:
                from core.datasets.utils import load_dataset, validate_dataset
                
                print(f"📂 Loading dataset from: {dataset_path}")
                dataset_loader = load_dataset(dataset_path, dataset_type if dataset_type != "auto" else None)
                
                # Validate the dataset
                validation_report = validate_dataset(dataset_loader, sample_size=3)
                if validation_report['is_valid']:
                    print(f"✅ Dataset loaded successfully: {validation_report['total_items']} items")
                    print(f"   Format: {validation_report['metadata'].get('format', 'unknown')}")
                    if validation_report['metadata'].get('dataset_type'):
                        print(f"   Type: {validation_report['metadata']['dataset_type']}")
                else:
                    print(f"⚠️  Dataset validation warnings: {len(validation_report['errors'])} errors, {len(validation_report['warnings'])} warnings")
                    for error in validation_report['errors'][:3]:  # Show first 3 errors
                        print(f"   Error: {error}")
                
            except Exception as e:
                print(f"❌ Error loading dataset: {e}")
                print("   Continuing without dataset...")
                dataset_loader = None
        
        # Determine query
        query = args.query or config.get("query", "")
        
        # If we have a dataset but no specific query, we could run evaluation mode
        if dataset_loader and not query:
            print("📊 Dataset loaded but no specific query provided.")
            print("   Use -q to specify a query, or implement batch evaluation mode")
            # For now, just show dataset info and exit
            metadata = dataset_loader.get_metadata()
            print(f"   Dataset info: {metadata}")
            return
        
        if not query:
            print("❌ No query specified. Use -q option or specify 'query' in config file.")
            return
        
        # Override save setting from command line
        if args.no_save:
            config["output"]["save_results"] = False
        
        # Run orchestrator
        if args.orchestrator == "multi_stage":
            result = await run_multi_stage_orchestrator(config, query)
        else:
            print(f"❌ Unsupported orchestrator: {args.orchestrator}")
            return
        
        # Display final results
        print("\n" + "="*60)
        print("🎯 FINAL RESULT")
        print("="*60)
        print(f"Query: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Turns taken: {result['turns_taken']}")
        print(f"Remote explorations: {result['attempts']['remote_explorations']}")
        print(f"Local explorations: {result['attempts']['local_explorations']}")
        if result['attempts']['failures']:
            print(f"Failures: {len(result['attempts']['failures'])}")
        if dataset_loader:
            print(f"Dataset: {dataset_loader.get_metadata().get('total_items', 'unknown')} items")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

