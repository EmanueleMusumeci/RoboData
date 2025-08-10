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
    validate_environment,
    generate_experiment_id
)
from core.logging import log_debug


def get_default_config() -> Dict[str, Any]:
    """Get default configuration values from default_config.yaml."""
    # Path to default config file (relative to project root)
    default_config_path = backend_dir.parent / "default_config.yaml"
    
    if not default_config_path.exists():
        raise FileNotFoundError(f"Default configuration file not found: {default_config_path}")
    
    with open(default_config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    return config


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


async def run_multi_stage_orchestrator(config: Dict[str, Any], query: str, enable_question_decomposition: bool = False, enable_metacognition: bool = False) -> Dict[str, Any]:
    """Run the multi-stage orchestrator with the given configuration and query."""
    
    # Generate experiment ID using query
    experiment_id = config.get("experiment_id") or generate_experiment_id(query)
    
    # Create orchestrator
    orchestrator = await create_multi_stage_orchestrator(config, experiment_id, enable_question_decomposition, enable_metacognition, query)
    
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


async def run_multiple_queries(config: Dict[str, Any], queries: list, enable_question_decomposition: bool = False, enable_metacognition: bool = False) -> Dict[str, Any]:
    """Run multiple queries sequentially using the multi-stage orchestrator."""
    
    if not queries:
        raise ValueError("No queries provided")
    
    results = []
    total_turns = 0
    total_remote_explorations = 0
    total_local_explorations = 0
    all_failures = []
    
    print(f"🔄 Processing {len(queries)} queries...")
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"📊 QUERY {i}/{len(queries)}")
        print(f"{'='*60}")
        
        try:
            result = await run_multi_stage_orchestrator(config, query, enable_question_decomposition, enable_metacognition)
            
            # Aggregate statistics
            total_turns += result['turns_taken']
            total_remote_explorations += result['attempts']['remote_explorations']
            total_local_explorations += result['attempts']['local_explorations']
            all_failures.extend(result['attempts']['failures'])
            
            results.append({
                'query': query,
                'answer': result['answer'],
                'turns_taken': result['turns_taken'],
                'attempts': result['attempts'],
                'sub_questions': result.get('sub_questions', [])
            })
            
            print(f"✅ Query {i} completed successfully")
            
        except Exception as e:
            print(f"❌ Query {i} failed: {e}")
            results.append({
                'query': query,
                'answer': f"Error: {str(e)}",
                'turns_taken': 0,
                'attempts': {'remote_explorations': 0, 'local_explorations': 0, 'failures': [str(e)]},
                'sub_questions': []
            })
            all_failures.append(str(e))
    
    # Compile aggregate results
    aggregate_result = {
        'queries': queries,
        'results': results,
        'total_queries': len(queries),
        'successful_queries': len([r for r in results if not r['answer'].startswith('Error:')]),
        'failed_queries': len([r for r in results if r['answer'].startswith('Error:')]),
        'total_turns': total_turns,
        'avg_turns_per_query': total_turns / len(queries) if queries else 0,
        'attempts': {
            'remote_explorations': total_remote_explorations,
            'local_explorations': total_local_explorations,
            'failures': all_failures
        }
    }
    
    return aggregate_result


async def run_interactive_mode(config: Dict[str, Any], use_experimental_gui: bool = False, enable_question_decomposition: bool = False, enable_metacognition: bool = False) -> None:
    """Run the HIL orchestrator in interactive mode."""
    
    # Validate environment
    if not validate_environment():
        return
    
    # Generate experiment ID
    experiment_id = config.get("experiment_id") or f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("Creating orchestrator for simple interactive mode...")
    
    try:
        # Create regular MultiStage orchestrator for simple mode
        orchestrator = await create_multi_stage_orchestrator(config, experiment_id, enable_question_decomposition, enable_metacognition)
        
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
    
    parser.add_argument(
        "--enable-metacognition",
        action="store_true",
        help="Enable metacognitive strategic assessment and observation"
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
        print("  python main.py -c experiment_with_multiple_queries.yaml")
        print("  python main.py -d dataset.json --dataset-type lcquad")
        print("  python main.py --interactive")
        print("  python main.py --interactive --experimental-gui")
        print("\nOptions:")
        print("  --enable-question-decomposition   Break complex queries into sub-questions")
        print("  --enable-metacognition           Enable strategic assessment and meta-observation")
        print("  --interactive                     Simple interactive mode (default)")
        print("                                   Shows normal orchestrator feed with input prompts")
        print("  --interactive --experimental-gui  Experimental curses-based GUI")
        print("                                   Split-screen interface (may not work in all terminals)")
        print("\nConfiguration:")
        print("  - Use 'queries' in config for one or more queries (processed sequentially)")
        print("  - Command line -q creates a single-query list and takes precedence")
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
            enable_metacognition = args.enable_metacognition or config.get("orchestrator", {}).get("enable_metacognition", False)
            await run_interactive_mode(config, args.experimental_gui, enable_decomposition, enable_metacognition)
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
        
        # Determine queries list
        queries = []
        
        if args.query:
            # Command line query takes precedence and creates a single-item list
            queries = [args.query]
            print(f"✅ Using command line query: {args.query}")
        else:
            # Use queries from config file
            queries = config.get("queries", [])
            if queries:
                print(f"✅ Using {len(queries)} queries from configuration file")
        
        if not queries:
            print("❌ No queries specified. Use -q option or provide 'queries' list in config file.")
            return
        
        # If we have a dataset but no queries, we could run evaluation mode
        if dataset_loader and not queries:
            print("📊 Dataset loaded but no queries provided.")
            print("   Use -q to specify a query, or add 'queries' list to config file, or implement batch evaluation mode")
            # For now, just show dataset info and exit
            metadata = dataset_loader.get_metadata()
            print(f"   Dataset info: {metadata}")
            return
        
        # Run orchestrator
        if args.orchestrator == "multi_stage":
            # Use command-line argument, or fall back to config setting
            enable_decomposition = args.enable_question_decomposition or config.get("orchestrator", {}).get("enable_question_decomposition", False)
            enable_metacognition = args.enable_metacognition or config.get("orchestrator", {}).get("enable_metacognition", False)
            
            if len(queries) == 1:
                # Process single query (from command line or single item in config)
                result = await run_multi_stage_orchestrator(config, queries[0], enable_decomposition, enable_metacognition)
                # Keep original result for visualization info
                original_result = result.copy()
                # Convert single result to multiple queries format for consistent handling
                result = {
                    'queries': queries,
                    'results': [{
                        'query': queries[0],
                        'answer': result['answer'],
                        'turns_taken': result['turns_taken'],
                        'attempts': result['attempts'],
                        'sub_questions': result.get('sub_questions', [])
                    }],
                    'total_queries': 1,
                    'successful_queries': 1 if not result['answer'].startswith('Error:') else 0,
                    'failed_queries': 1 if result['answer'].startswith('Error:') else 0,
                    'total_turns': result['turns_taken'],
                    'avg_turns_per_query': result['turns_taken'],
                    'attempts': result['attempts'],
                    'original_result': original_result  # Keep for visualization info
                }
            else:
                # Process multiple queries
                result = await run_multiple_queries(config, queries, enable_decomposition, enable_metacognition)
        else:
            print(f"❌ Unsupported orchestrator: {args.orchestrator}")
            return
        
        # Display final results
        print("\n" + "="*60)
        print("🎯 FINAL RESULT")
        print("="*60)
        
        enable_decomposition = args.enable_question_decomposition or config.get("orchestrator", {}).get("enable_question_decomposition", False)
        enable_metacognition = args.enable_metacognition or config.get("orchestrator", {}).get("enable_metacognition", False)
        
        # All results now follow the multiple queries format
        print(f"Total queries processed: {result['total_queries']}")
        print(f"Successful queries: {result['successful_queries']}")
        print(f"Failed queries: {result['failed_queries']}")
        print(f"Question decomposition: {'Enabled' if enable_decomposition else 'Disabled'}")
        print(f"Metacognition: {'Enabled' if enable_metacognition else 'Disabled'}")
        print(f"Total turns taken: {result['total_turns']}")
        print(f"Average turns per query: {result['avg_turns_per_query']:.1f}")
        print(f"Total remote explorations: {result['attempts']['remote_explorations']}")
        print(f"Total local explorations: {result['attempts']['local_explorations']}")
        if result['attempts']['failures']:
            print(f"Total failures: {len(result['attempts']['failures'])}")
        
        # Show individual results
        if result['total_queries'] == 1:
            print(f"\n📋 RESULT:")
            query_result = result['results'][0]
            print(f"Query: {query_result['query']}")
            print(f"Answer: {query_result['answer']}")
            print(f"Turns: {query_result['turns_taken']}")
            if query_result.get('sub_questions'):
                print(f"Sub-questions: {len(query_result['sub_questions'])}")
            
            # Show visualization information if available
            if 'original_result' in result and 'visualizations' in result['original_result']:
                vis_info = result['original_result']['visualizations']
                print(f"\n🎨 VISUALIZATIONS:")
                print(f"Output directory: {vis_info['output_dir']}")
                if vis_info.get('index_html'):
                    print(f"📄 Index HTML: {vis_info['index_html']}")
                if vis_info.get('animation_path'):
                    print(f"🎬 Animation: {vis_info['animation_path']}")
                if vis_info.get('final_image'):
                    print(f"🖼️  Final graph: {vis_info['final_image']}")
        else:
            print(f"\n📋 INDIVIDUAL RESULTS:")
            for i, query_result in enumerate(result['results'], 1):
                print(f"\n--- Query {i} ---")
                print(f"Query: {query_result['query']}")
                print(f"Answer: {query_result['answer']}")
                print(f"Turns: {query_result['turns_taken']}")
                if query_result.get('sub_questions'):
                    print(f"Sub-questions: {len(query_result['sub_questions'])}")
        
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

