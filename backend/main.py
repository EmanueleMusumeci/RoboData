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
        "dataset": "",
        "query": ""
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
    print(f"� Processing query: {query}")
    result = await orchestrator.process_user_query(query)
    
    # Log results
    log_debug(f"Answer: {result['answer']}")
    log_debug(f"Attempts: {result['attempts']}")
    log_debug(f"Turns taken: {result['turns_taken']} / {result['max_turns'] if result['max_turns'] > 0 else 'unlimited'}")
    
    return result


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
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # If no arguments provided, show help
    if not any([args.config, args.query]):
        print("🤖 RoboData - AI-powered knowledge exploration system")
        print("\nUsage examples:")
        print("  python main.py -q 'What is climate change?'")
        print("  python main.py -c experiment.yaml")
        print("  python main.py -c experiment.yaml -q 'Who was Albert Einstein?'")
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
        
        # Determine query
        query = args.query or config.get("query", "")
        if not query:
            print("❌ No query specified. Use -q option or specify 'query' in config file.")
            return
        
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
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

