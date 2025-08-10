"""
Utilities for RoboData orchestrator execution

This module contains common utility functions for creating and configuring
orchestrators, agents, and toolboxes that are shared between interactive
and non-interactive execution modes.
"""

import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from settings import settings_manager
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
from core.orchestrator.hil_orchestrator import HILOrchestrator, AsyncQueueInputHandler
from core.memory import SimpleMemory
from core.toolbox.toolbox import Toolbox
from core.logging import log_debug


def generate_experiment_id(query: str) -> str:
    """
    Generate experiment ID with timestamp followed by first 8 words of query.
    
    Args:
        query: The query string
        
    Returns:
        Experiment ID in format: YYYYMMDD_HHMMSS_word1_word2_..._word8
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Clean and split query into words
    import re
    words = re.findall(r'\w+', query.lower())
    
    # Take first 8 words and join with underscores
    query_words = '_'.join(words[:8])
    
    return f"{timestamp}_{query_words}"


def _make_json_serializable(obj):
    """
    Convert objects to JSON-serializable format recursively.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        # Convert custom objects to dict representation
        return _make_json_serializable(obj.__dict__)
    elif hasattr(obj, 'isoformat'):
        # Handle datetime objects
        return obj.isoformat()
    else:
        # For other types, try to convert to string if not already serializable
        try:
            import json
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def create_agent(model_name: str, toolbox=None):
    """Create the appropriate agent based on model name.
    
    Args:
        model_name: Name of the model to use
        toolbox: Optional toolbox for local models
        
    Returns:
        Configured agent instance
    """
    if model_name == "local":
        print("Using local WatTool SLM Agent")
        return WatToolSLMAgent(toolbox=toolbox)
    elif model_name.startswith("gpt"):
        print(f"Using OpenAI Agent with model: {model_name}")
        return OpenAIAgent(model=model_name, toolbox=toolbox)
    else:
        # Default to OpenAI
        print(f"Using OpenAI Agent with model: {model_name}")
        return OpenAIAgent(model=model_name, toolbox=toolbox)


def create_configured_toolboxes() -> Tuple[Toolbox, Toolbox, Toolbox, Toolbox]:
    """Create and configure toolboxes for different orchestrator phases.
    
    Returns:
        Tuple of (local_exploration, remote_exploration, graph_update, evaluation) toolboxes
    """
    local_exploration_toolbox = create_local_exploration_toolbox()
    remote_exploration_toolbox = create_remote_exploration_toolbox()
    graph_update_toolbox = create_graph_update_toolbox()
    evaluation_toolbox = create_evaluation_toolbox()
    
    return local_exploration_toolbox, remote_exploration_toolbox, graph_update_toolbox, evaluation_toolbox


async def create_multi_stage_orchestrator(
    config: Dict[str, Any],
    experiment_id: Optional[str] = None,
    enable_question_decomposition: bool = False,
    enable_metacognition: bool = False,
    query: Optional[str] = None
) -> MultiStageOrchestrator:
    """Create a configured MultiStageOrchestrator.
    
    Args:
        config: Configuration dictionary
        experiment_id: Optional experiment identifier
        enable_question_decomposition: Enable question decomposition
        enable_metacognition: Enable metacognitive capabilities
        query: Optional query string for generating experiment ID
        
    Returns:
        Configured MultiStageOrchestrator instance
    """
    # Generate experiment ID if not provided
    if not experiment_id:
        if query:
            experiment_id = generate_experiment_id(query)
        else:
            experiment_id = config.get("experiment_id") or f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create toolboxes
    local_exploration_toolbox, remote_exploration_toolbox, graph_update_toolbox, evaluation_toolbox = create_configured_toolboxes()
    
    # Create agent
    orchestrator_config = config["orchestrator"]
    model_name = orchestrator_config.get("model", "gpt-4o")
    
    # Create agent based on model type
    if model_name == "local":
        agent = create_agent(model_name, remote_exploration_toolbox)
    else:
        agent = create_agent(model_name)
    
    # Get LLM settings from config or settings manager
    llm_settings = None
    if "llm" in config:
        # Create LLMSettings from config
        from settings import LLMSettings
        llm_config = config["llm"]
        llm_settings = LLMSettings(**llm_config)
    else:
        # Use global settings manager
        llm_settings = settings_manager.get_settings().llm

    # Create metacognition module if enabled
    metacognition = None
    if enable_metacognition:
        from core.metacognition.metacognition import Metacognition
        
        # Get metacognition descriptions from config - all required
        metacognition_config = orchestrator_config.get("metacognition", {})
        system_description = metacognition_config.get("system_description")
        agent_description = metacognition_config.get("agent_description")
        task_description = metacognition_config.get("task_description")
        
        if not all([system_description, agent_description, task_description]):
            raise ValueError("Metacognition enabled but missing required descriptions in config. Please provide system_description, agent_description, and task_description in orchestrator.metacognition section.")
        
        metacognition = Metacognition(
            agent=agent,
            system_description=system_description,
            agent_description=agent_description,
            task_description=task_description,
            llm_settings=llm_settings
        )
        log_debug("Metacognition module enabled", "METACOGNITION")
    
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
        experiment_id=experiment_id,
        enable_question_decomposition=enable_question_decomposition,
        metacognition=metacognition,
        llm_settings=llm_settings
    )
    
    return orchestrator


async def create_hil_orchestrator(
    config: Dict[str, Any],
    experiment_id: Optional[str] = None,
    clear_kg_on_start: bool = True,
    enable_question_decomposition: bool = False,
    enable_metacognition: bool = False,
    query: Optional[str] = None
) -> HILOrchestrator:
    """Create a HIL orchestrator with MultiStageOrchestrator as the wrapped component.
    
    Args:
        config: Configuration dictionary
        experiment_id: Optional experiment identifier for logging and tracking
        clear_kg_on_start: Whether to clear knowledge graph on startup
        enable_question_decomposition: Enable question decomposition
        enable_metacognition: Enable metacognitive capabilities
        query: Optional query string for generating experiment ID
        
    Returns:
        Configured HIL orchestrator ready for use
    """
    # Create the MultiStage orchestrator
    multi_stage_orchestrator = await create_multi_stage_orchestrator(
        config, experiment_id, enable_question_decomposition, enable_metacognition, query
    )
    
    # Create user input handler
    user_input_handler = AsyncQueueInputHandler()
    
    # Create HIL orchestrator
    hil_orchestrator = HILOrchestrator(
        wrapped_orchestrator=multi_stage_orchestrator,
        user_input_handler=user_input_handler,
        memory=SimpleMemory(max_slots=100),
        experiment_id=experiment_id,
        clear_knowledge_graph_on_start=clear_kg_on_start
    )
    
    return hil_orchestrator


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
            # Create a JSON-serializable copy of the result
            serializable_result = _make_json_serializable(result)
            json.dump({
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "result": serializable_result
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
            from core.toolbox.graph.visualization import GraphVisualizer
            
            # Create visualizer with experiment directory as output
            visualizer = GraphVisualizer(output_dir=str(results_dir))
            
            # Get current graph data
            graph_data = await knowledge_graph.get_whole_graph()
            
            if graph_data and (graph_data.nodes or graph_data.edges):
                # Create static visualization (PNG) with support set highlighting
                static_viz_path = visualizer.create_static_visualization(
                    graph_data,
                    title=f"Knowledge Graph: {experiment_id}",
                    filename="knowledge_graph_static.png",
                    final_answer=result.get('answer', '')
                )
                
                print(f"✅ Static graph visualization saved to {static_viz_path}")
                
                # Create interactive visualization (HTML)
                try:
                    dynamic_viz_path = visualizer.create_dynamic_visualization(
                        graph_data,
                        title=f"Interactive Knowledge Graph: {experiment_id}",
                        filename="knowledge_graph_interactive.html"
                    )
                    
                    print(f"✅ Interactive graph visualization saved to {dynamic_viz_path}")
                    
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


def validate_environment() -> bool:
    """Validate that the environment is properly configured.
    
    Returns:
        True if environment is valid, False otherwise
    """
    # Check for OpenAI API key if needed
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key before running this example")
        return False
    
    return True
