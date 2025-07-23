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
from utils import (
    create_multi_stage_orchestrator,
    create_hil_orchestrator,
    save_orchestrator_results,
    validate_environment
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
            "enable_question_decomposition": False,
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


async def run_multi_stage_orchestrator(config: Dict[str, Any], query: str, enable_question_decomposition: bool = False) -> Dict[str, Any]:
    """Run the multi-stage orchestrator with the given configuration and query."""
    
    # Generate experiment ID if not provided
    experiment_id = config.get("experiment_id") or f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create orchestrator
    orchestrator = await create_multi_stage_orchestrator(config, experiment_id, enable_question_decomposition)
    
    # Process the query
    print(f"🤖 Processing query: {query}")
    result = await orchestrator.process_user_query(query)
    
    # Log results
    log_debug(f"Answer: {result['answer']}")
    log_debug(f"Attempts: {result['attempts']}")
    log_debug(f"Turns taken: {result['turns_taken']} / {result['max_turns'] if result['max_turns'] > 0 else 'unlimited'}")
    
    # Save results to files after orchestrator execution (if enabled)
    if config.get("output", {}).get("save_results", True):
        await save_orchestrator_results(result, experiment_id, query, orchestrator.knowledge_graph)
    else:
        print("ℹ️  Skipping result saving (disabled)")
    
    return result


async def run_interactive_mode(config: Dict[str, Any], use_experimental_gui: bool = False, enable_question_decomposition: bool = False) -> None:
    """Run the HIL orchestrator in interactive mode."""
    
    # Validate environment
    if not validate_environment():
        return
    
    # Generate experiment ID
    experiment_id = config.get("experiment_id") or f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("Creating orchestrator for simple interactive mode...")
    
    try:
        # Create regular MultiStage orchestrator for simple mode
        orchestrator = await create_multi_stage_orchestrator(config, experiment_id, enable_question_decomposition)
        
        # Import and run simple interactive interface
        from gui.simple_interactive import run_simple_interactive_session
        await run_simple_interactive_session(orchestrator, config)
        
        print("Interactive session completed successfully")
        
    except Exception as e:
        print(f"Error in interactive mode: {e}")
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
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--experimental-gui",
        action="store_true",
        help="Use experimental curses-based GUI interface (requires --interactive)"
    )
    
    parser.add_argument(
        "--enable-question-decomposition",
        action="store_true",
        help="Enable question decomposition (breaks complex queries into sub-questions)"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # If no arguments provided, show help
    if not any([args.config, args.query, args.dataset, args.interactive]):
        print("🤖 RoboData - AI-powered knowledge exploration system")
        print("\nUsage examples:")
        print("  python main.py -q 'What is climate change?'")
        print("  python main.py -q 'What is climate change?' --enable-question-decomposition")
        print("  python main.py -c experiment.yaml")
        print("  python main.py -c experiment.yaml -q 'Who was Albert Einstein?'")
        print("  python main.py -d dataset.json --dataset-type lcquad")
        print("  python main.py --interactive")
        print("  python main.py --interactive --experimental-gui")
        print("\nOptions:")
        print("  --enable-question-decomposition   Break complex queries into sub-questions")
        print("  --interactive                     Simple interactive mode (default)")
        print("                                   Shows normal orchestrator feed with input prompts")
        print("  --interactive --experimental-gui  Experimental curses-based GUI")
        print("                                   Split-screen interface (may not work in all terminals)")
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
        
        # Override save setting from command line
        if args.no_save:
            config["output"]["save_results"] = False
        
        # Check if interactive mode is requested
        if args.interactive:
            # Use command-line argument, or fall back to config setting
            enable_decomposition = args.enable_question_decomposition or config.get("orchestrator", {}).get("enable_question_decomposition", False)
            await run_interactive_mode(config, args.experimental_gui, enable_decomposition)
            return
        
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
        
        # Run orchestrator
        if args.orchestrator == "multi_stage":
            # Use command-line argument, or fall back to config setting
            enable_decomposition = args.enable_question_decomposition or config.get("orchestrator", {}).get("enable_question_decomposition", False)
            result = await run_multi_stage_orchestrator(config, query, enable_decomposition)
        else:
            print(f"❌ Unsupported orchestrator: {args.orchestrator}")
            return
        
        # Display final results
        print("\n" + "="*60)
        print("🎯 FINAL RESULT")
        print("="*60)
        print(f"Query: {query}")
        enable_decomposition = args.enable_question_decomposition or config.get("orchestrator", {}).get("enable_question_decomposition", False)
        print(f"Question decomposition: {'Enabled' if enable_decomposition else 'Disabled'}")
        if result.get('sub_questions'):
            print(f"Sub-questions generated: {len(result.get('sub_questions', []))}")
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

