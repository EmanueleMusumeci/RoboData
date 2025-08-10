import asyncio
import glob
import json
import pprint
import time
import traceback
from typing import Any, List, Dict, Optional, Tuple, TYPE_CHECKING
from enum import Enum
import asyncio
import json
import traceback
import pprint

if TYPE_CHECKING:
    from ...metacognition.metacognition import Metacognition

from ..orchestrator import Orchestrator
from ...memory import Memory, SimpleMemory, SummaryMemory
from ...agents.agent import Query
from ...toolbox.toolbox import Toolbox
from ...knowledge_base.graph import KnowledgeGraph, get_knowledge_graph
from ...logging import setup_logger, log_prompt, log_tool_result, log_tool_error, log_debug, log_response, log_turn_separator, log_tool_results_summary
from ...toolbox.graph.visualization import GraphVisualizer
from .statistics import OrchestratorStatistics, create_statistics_collector
from .prompts import (
    PromptStructure,
    create_local_exploration_prompt,
    create_local_evaluation_prompt,
    create_remote_exploration_prompt,
    create_remote_evaluation_prompt,
    create_graph_update_prompt,
    create_answer_production_prompt,
    create_question_decomposition_prompt,
    format_tool_results_for_prompt
)
from .utils import (
    should_continue_local_exploration,
    has_sufficient_local_data,
    should_attempt_remote_exploration,
    remote_exploration_successful,
    is_remote_data_relevant,
    graph_update_successful,
    extract_partial_answer,

)
from ...knowledge_base.schema import Graph

class MultiStageOrchestrator(Orchestrator):
    """Multi-stage orchestrator with autonomous graph exploration capabilities.
    
    Features:
    - Optional question decomposition: Break complex queries into sub-questions
    - Autonomous local graph exploration
    - Remote data exploration and integration
    - Iterative knowledge graph building
    - Memory management with summary capabilities
    """
    
    class OrchestratorState(Enum):
        """Main orchestrator states."""
        READY = "ready"
        QUESTION_DECOMPOSITION = "question_decomposition"
        AUTONOMOUS = "autonomous"
        FINISHED = "finished"
    
    class AutonomousSubstate(Enum):
        """Substates within AUTONOMOUS state."""
        LOCAL_GRAPH_EXPLORATION = "local_graph_exploration"
        EVAL_LOCAL_DATA = "eval_local_data"
        PRODUCE_ANSWER = "produce_answer"
        REMOTE_GRAPH_EXPLORATION = "remote_graph_exploration"
        EVAL_REMOTE_DATA = "eval_remote_data"
        LOCAL_GRAPH_UPDATE = "local_graph_update"
    
    def __init__(self, 
                 agent: Any, 
                 knowledge_graph: KnowledgeGraph,
                 memory: Optional[Memory] = None, 
                 context_length: int = 128000,
                 local_exploration_toolbox: Optional[Toolbox] = None,
                 remote_exploration_toolbox: Optional[Toolbox] = None,
                 graph_update_toolbox: Optional[Toolbox] = None,
                 evaluation_toolbox: Optional[Toolbox] = None,
                 use_summary_memory: bool = True,
                 memory_max_slots: int = 20,  # Increased from 10 to provide more memory capacity
                 max_turns: int = -1,
                 experiment_id: Optional[str] = None,
                 enable_question_decomposition: bool = False,
                 metacognition: Optional['Metacognition'] = None,
                 llm_settings: Optional[Any] = None):
        # Set up logging first
        from datetime import datetime
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_id = experiment_id
        self.logger = setup_logger(experiment_id=experiment_id)
        
        # Use SummaryMemory by default for better memory management, allow SimpleMemory as option
        if memory is None:
            if use_summary_memory:
                from ...memory import SummaryMemory
                memory = SummaryMemory(max_slots=memory_max_slots, agent=agent)
            else:
                # Use SimpleMemory when explicitly not using summary memory
                memory = SimpleMemory(max_slots=memory_max_slots)
        super().__init__(agent, memory, metacognition)
        self.knowledge_graph = knowledge_graph
        self.context_length = context_length
        self.state = self.OrchestratorState.READY
        self.substate = None
        self.current_query = None
        
        # Store LLM settings for model selection
        self.llm_settings = llm_settings
        
        # Initialize statistics collector (replaces attempt_history)
        self.statistics = None  # Will be initialized when query is processed
        
        self.current_remote_data = None
        self.final_answer = None
        self.tool_call_results = []  # Store all tool call results
        self.remote_exploration_results = []  # Store only remote exploration results
        self.local_exploration_results = []  # Store only local exploration results
        self.max_turns = max_turns
        self.current_turn = 0
        
        # Question decomposition fields
        self.original_query = None
        self.sub_questions = []
        self.current_sub_question_index = 0
        self.sub_question_answers = []
        self.enable_question_decomposition = enable_question_decomposition
        
        # Store toolboxes for different phases
        self.local_exploration_toolbox = local_exploration_toolbox
        self.remote_exploration_toolbox = remote_exploration_toolbox
        self.graph_update_toolbox = graph_update_toolbox
        self.evaluation_toolbox = evaluation_toolbox
        
        # Initialize graph visualizer with experiment directory
        from pathlib import Path
        experiment_dir = Path("experiments") / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        self.graph_visualizer = GraphVisualizer(output_dir=str(experiment_dir))

    def _track_failure(self, failure_message: str):
        """Track a failure in statistics."""
        if self.statistics:
            self.statistics.attempt_history.failures.append(failure_message)
    
    async def _generate_turn_visualization(self):
        """Generate knowledge graph visualization for the current turn."""
        try:
            # Get current graph state
            graph = await self.knowledge_graph.get_whole_graph()
            
            # Skip visualization if graph is empty
            if not graph.nodes and not graph.edges:
                log_debug("Skipping visualization - graph is empty", "VISUALIZATION")
                return
            
            # Create filename with turn number and substate
            current_substate = self.substate.value if self.substate else "unknown"
            filename = f"turn_{self.current_turn:03d}_{current_substate}.png"
            
            # Create title with turn and state information
            title = f"Turn {self.current_turn} - {current_substate.replace('_', ' ').title()}"
            if self.current_query:
                title += f"\nQuery: {self.current_query.text[:60]}{'...' if len(self.current_query.text) > 60 else ''}"
            
            # Generate visualization
            image_path = self.graph_visualizer.create_static_visualization(
                graph=graph,
                title=title,
                filename=filename,
                figsize=(16, 12),
                node_size_multiplier=1.2
            )
            
            log_debug(f"Generated turn visualization: {image_path}", "VISUALIZATION")
            
        except Exception as e:
            log_debug(f"Failed to generate turn visualization: {e}", "VISUALIZATION_ERROR")
            # Don't let visualization errors break the orchestrator
            pass
    
    async def _generate_final_visualization(self):
        """Generate final knowledge graph visualization with support sets highlighted."""
        try:
            # Get current graph state
            graph = await self.knowledge_graph.get_whole_graph()
            
            # Skip visualization if graph is empty
            if not graph.nodes and not graph.edges:
                log_debug("Skipping final visualization - graph is empty", "VISUALIZATION")
                return
            
            # Create filename for final visualization
            filename = f"final_graph_with_answer.png"
            
            # Create title with final state information
            title = f"Final Graph - {self.current_turn} Turns Completed"
            if self.current_query:
                title += f"\nQuery: {self.current_query.text[:60]}{'...' if len(self.current_query.text) > 60 else ''}"
            
            # Generate visualization with support sets from final answer
            image_path = self.graph_visualizer.create_static_visualization(
                graph=graph,
                title=title,
                filename=filename,
                figsize=(20, 16),
                node_size_multiplier=1.5,
                final_answer=self.final_answer  # Include final answer for support set highlighting
            )
            
            log_debug(f"Generated final visualization with support sets: {image_path}", "VISUALIZATION")
            
        except Exception as e:
            log_debug(f"Failed to generate final visualization: {e}", "VISUALIZATION_ERROR")
            # Don't let visualization errors break the orchestrator
            pass
    
    async def generate_animation(self, duration_per_frame: float = 1.0, output_filename: str = "graph_evolution.gif"):
        """
        Generate an animated GIF showing the evolution of the knowledge graph across turns.
        
        Args:
            duration_per_frame: Duration in seconds for each frame
            output_filename: Name of the output GIF file
            
        Returns:
            Path to the generated animation file
        """
        try:
            import glob
            from PIL import Image
            
            # Get all turn visualization files
            pattern = str(self.graph_visualizer.output_dir / "turn_*.png")
            image_files = sorted(glob.glob(pattern))
            
            if not image_files:
                log_debug("No turn visualizations found for animation", "ANIMATION")
                return None
            
            # Load images
            images = []
            for image_file in image_files:
                img = Image.open(image_file)
                images.append(img)
            
            # Add final visualization if it exists
            final_image_path = self.graph_visualizer.output_dir / "final_graph_with_answer.png"
            if final_image_path.exists():
                final_img = Image.open(final_image_path)
                images.append(final_img)
            
            # Create animation
            animation_path = self.graph_visualizer.output_dir / output_filename
            images[0].save(
                animation_path,
                save_all=True,
                append_images=images[1:],
                duration=int(duration_per_frame * 1000),  # Convert to milliseconds
                loop=0  # Infinite loop
            )
            
            log_debug(f"Generated knowledge graph animation: {animation_path}", "ANIMATION")
            return str(animation_path)
            
        except ImportError:
            log_debug("PIL not available - cannot generate animation", "ANIMATION_ERROR")
            return None
        except Exception as e:
            log_debug(f"Failed to generate animation: {e}", "ANIMATION_ERROR")
            return None
    
    async def _create_visualization_index(self):
        """Create an HTML index file listing all generated visualizations."""
        try:
            import glob
            from pathlib import Path
            
            # Get all visualization files
            turn_images = sorted(glob.glob(str(self.graph_visualizer.output_dir / "turn_*.png")))
            final_image = self.graph_visualizer.output_dir / "final_graph_with_answer.png"
            animation_file = self.graph_visualizer.output_dir / "graph_evolution.gif"
            
            # Create HTML content
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Graph Evolution - {self.experiment_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .turn-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .turn-item {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .turn-item img {{ max-width: 100%; height: auto; }}
        .animation {{ text-align: center; margin: 30px 0; }}
        .final {{ text-align: center; margin: 30px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Knowledge Graph Evolution</h1>
        <h2>Query: {self.current_query.text if self.current_query else 'Unknown'}</h2>
        <p>Experiment ID: {self.experiment_id}</p>
        <p>Total Turns: {self.current_turn}</p>
    </div>
"""
            
            # Add animation section
            if animation_file.exists():
                html_content += f"""
    <div class="section animation">
        <h2>📽️ Complete Evolution Animation</h2>
        <img src="graph_evolution.gif" alt="Knowledge Graph Evolution Animation" style="max-width: 800px;">
    </div>
"""
            
            # Add turn-by-turn section
            if turn_images:
                html_content += """
    <div class="section">
        <h2>🔄 Turn-by-Turn Evolution</h2>
        <div class="turn-grid">
"""
                for i, image_path in enumerate(turn_images, 1):
                    image_name = Path(image_path).name
                    html_content += f"""
            <div class="turn-item">
                <h3>Turn {i}</h3>
                <img src="{image_name}" alt="Turn {i} Visualization">
            </div>
"""
                html_content += """
        </div>
    </div>
"""
            
            # Add final graph section
            if final_image.exists():
                html_content += f"""
    <div class="section final">
        <h2>🎯 Final Knowledge Graph</h2>
        <img src="final_graph_with_answer.png" alt="Final Knowledge Graph" style="max-width: 900px;">
    </div>
"""
            
            html_content += """
</body>
</html>"""
            
            # Write index file
            index_path = self.graph_visualizer.output_dir / "index.html"
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            log_debug(f"Created visualization index: {index_path}", "VISUALIZATION")
            return str(index_path)
            
        except Exception as e:
            log_debug(f"Failed to create visualization index: {e}", "VISUALIZATION_ERROR")
            return None
    
    async def _query_llm_with_timing(self, messages, tools=None, context_type="orchestrator", operation_type=None, **kwargs):
        """Query LLM and track timing and token usage statistics.
        
        Args:
            messages: Messages to send to the LLM
            tools: Optional tools to make available
            context_type: Type of context for logging
            operation_type: Type of operation for model selection ('metacognition', 'evaluation', 'exploration', 'update')
            **kwargs: Additional parameters
        """
        # Determine the appropriate model based on operation type
        model = None
        if self.llm_settings and operation_type:
            model = self.llm_settings.get_model_for_operation(operation_type)
        
        if not self.statistics:
            return await self.agent.query_llm(messages, tools, model=model, **kwargs)
        
        # Record inference start
        start_time = time.time()
        
        # Query the LLM with the appropriate model
        response = await self.agent.query_llm(messages, tools, model=model, **kwargs)
        
        # Calculate timing
        duration = time.time() - start_time
        
        # Record inference event
        current_state = self.state.value if self.state else "unknown"
        current_substate = self.substate.value if self.substate else None
        
        tokens = None
        if hasattr(response, 'usage') and response.usage:
            tokens = response.usage
        
        model = getattr(self.agent, 'model', None) if hasattr(self.agent, 'model') else None
        
        self.statistics.record_inference_event(
            state=current_state,
            duration=duration,
            tokens=tokens,
            model=model,
            substate=current_substate,
            context_type=context_type
        )
        
        return response
    
    async def _record_state_transition(self, from_state: str, to_state: str):
        """Record a state transition in statistics."""
        if self.statistics:
            self.statistics.record_state_transition(from_state, to_state)
            # Record graph statistics at each state transition
            await self._record_current_graph_statistics()
            
    async def _record_substate_transition(self, from_substate: Optional[str], to_substate: str):
        """Record a substate transition in statistics."""
        if self.statistics:
            self.statistics.record_substate_transition(from_substate, to_substate)
            # Record graph statistics at each substate transition
            await self._record_current_graph_statistics()
            
    async def _transition_to_substate(self, new_substate):
        """Transition to a new substate and record the transition with graph statistics."""
        old_substate = self.substate.value if self.substate else None
        self.substate = new_substate
        await self._record_substate_transition(old_substate, new_substate.value)
            
    async def _record_current_graph_statistics(self):
        """Record current graph statistics for the current state."""
        if self.statistics:
            await self.statistics.record_current_graph_statistics(self.knowledge_graph)
            
    async def _transition_to_state(self, new_state):
        """Transition to a new state and record the transition with graph statistics."""
        old_state = self.state.value if self.state else "unknown"
        self.state = new_state
        await self._record_state_transition(old_state, new_state.value)
            
    async def _connect_knowledge_graph(self):
        """Connect to the knowledge graph if not already connected."""
        if not await self.knowledge_graph.is_connected():
            await self.knowledge_graph.connect()
    
    async def _invoke_metacognition(self, current_state: str, 
                                   local_graph_data: Optional[str] = None,
                                   remote_data: Optional[List] = None,
                                   local_exploration_results: Optional[List] = None,
                                   next_step_tools: Optional[List] = None) -> Optional[str]:
        """
        Invoke metacognition before eval states if metacognition is enabled.
        Enhanced to receive the same information that evaluation prompts receive.
        
        Args:
            current_state: Current state/substate name for context
            local_graph_data: Current local graph data in readable format
            remote_data: Current remote exploration results
            local_exploration_results: Current local exploration results  
            next_step_tools: Tools available for the next stage
            
        Returns:
            Metacognitive suggestion if any, None otherwise
        """
        if not self.metacognition:
            return None
            
        try:
            # Build action sequence from FULL memory trace (no character limit)
            # Since SummaryMemory summarizes long-term past, it should never be too long
            memory_context = self.memory.read()  # No max_characters limit
            action_sequence = f"Full execution history in {current_state}:\n{memory_context}"
            
            # Get local graph data if not provided
            if local_graph_data is None:
                local_graph_data = await self.knowledge_graph.to_readable_format()
            
            # Build comprehensive task outcome with same information as evaluation prompts
            graph_status = "populated" if local_graph_data else "empty"
            local_explorations = self.statistics.attempt_history.local_explorations if self.statistics else 0
            remote_explorations = self.statistics.attempt_history.remote_explorations if self.statistics else 0
            
            # Include last LLM response for context
            last_llm_response = self.memory.get_last_llm_response()
            
            task_outcome = f"""Current state: {current_state}
Current query: {self.current_query.text if self.current_query else "Unknown"}
Graph status: {graph_status}
Attempts so far: {local_explorations} local, {remote_explorations} remote
Current turn: {self.current_turn}/{self.max_turns if self.max_turns > 0 else 'unlimited'}
Memory context available: {len(memory_context)} characters
Last LLM response: {last_llm_response or "None"}

CURRENT LOCAL GRAPH DATA:
##############
{local_graph_data or "No local graph data available."}
##############"""

            # Add local exploration results if available (similar to local evaluation prompt)
            if local_exploration_results:
                formatted_local_results = format_tool_results_for_prompt(local_exploration_results)
                task_outcome += f"""

LOCAL EXPLORATION RESULTS:
##############
{formatted_local_results}
##############"""

            # Add remote data if available (similar to remote evaluation prompt)
            if remote_data:
                formatted_remote_data = format_tool_results_for_prompt(remote_data)
                task_outcome += f"""

REMOTE DATA:
##############
{formatted_remote_data}
##############"""

            # Collect tool descriptions - use next_step_tools if provided, otherwise all tools
            if next_step_tools:
                # Focus on tools for the next stage (like evaluation prompts do)
                available_tools = {
                    "next_stage_tools": next_step_tools,
                    "evaluation": await self.get_evaluation_tools()
                }
            else:
                # Fallback to all available tools
                available_tools = {
                    "local_exploration": await self.get_local_exploration_tools(),
                    "remote_exploration": await self.get_remote_exploration_tools(),
                    "graph_update": await self.get_graph_update_tools(),
                    "evaluation": await self.get_evaluation_tools()
                }
            
            # Run metacognitive cycle with comprehensive context
            statistics_data = self.statistics.export_statistics() if self.statistics else None
            suggestion = await self.metacognition.process_metacognitive_cycle(
                action_sequence, task_outcome, self.current_query.text if self.current_query else "Unknown query", 
                self.current_turn, self.max_turns, statistics_data, available_tools
            )
            
            # Log but do NOT add to memory - metacognition should only appear in prompts
            if suggestion:
                log_debug(f"Metacognitive suggestion in {current_state}: {suggestion}", "METACOGNITION")
            
            return suggestion
            
        except Exception as e:
            log_debug(f"Error in metacognition invocation: {e}", "METACOGNITION_ERROR")
            return None
        
    async def _process_tool_calls(self, response: Any, context_name: str = "tool") -> int:
        """Process tool calls from LLM response and add results to tool_call_results list.
        
        Args:
            response: The LLM response containing tool calls
            context_name: Context name for logging (e.g., "local", "remote", "evaluation", "update")
            
        Returns:
            Number of successfully executed tool calls
        """
        tool_calls_executed = 0
        
        # Determine which toolbox to use based on context
        toolbox = None
        if context_name == "local":
            toolbox = self.local_exploration_toolbox
        elif context_name == "remote":
            toolbox = self.remote_exploration_toolbox
        elif context_name == "evaluation":
            toolbox = self.evaluation_toolbox
        elif context_name == "update":
            toolbox = self.graph_update_toolbox
            
        if response.tool_calls:
            log_debug(f"Processing tool calls in context: {context_name}")
            for tool_call in response.tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])

                log_debug(f"Tool call detected: {tool_name}")
                log_debug(f"Arguments for {tool_name}: {tool_args}")

                # Start timing the tool execution
                if self.statistics:
                    start_time = self.statistics.start_tool_execution(tool_name, tool_args, context_name)
                else:
                    start_time = time.time()

                try:
                    # Get tool from the appropriate toolbox
                    if toolbox:
                        tool = toolbox.get_tool(tool_name)
                        if not tool:
                            raise ValueError(f"Tool '{tool_name}' not found in the '{context_name}' toolbox.")
                    else:
                        raise ValueError(f"No toolbox found for context '{context_name}'")

                    if not tool:
                        raise ValueError(f"Tool '{tool_name}' not found in the '{context_name}' toolbox.")

                    # Execute the tool
                    result = await tool.execute(**tool_args)
                    
                    # Format and store result
                    formatted_result = tool.format_result(result)
                    log_tool_result(tool_name, tool_args, formatted_result, context_name)
                    
                    # Record successful tool execution in statistics
                    if self.statistics:
                        self.statistics.record_tool_execution(
                            tool_name, tool_args, start_time, formatted_result, 
                            context_name, success=True
                        )
                    
                    # Add to memory with more detailed information, especially for queries with no results or failures
                    self.memory.add(f"Tool {tool_name} executed with arguments {tool_args}. Result: {formatted_result}", "System")
                    tool_result_entry = {
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": formatted_result,
                        "context": context_name
                    }
                    self.tool_call_results.append(tool_result_entry)
                    
                    # Add to specific context result lists
                    if context_name == "remote":
                        self.remote_exploration_results.append(tool_result_entry)
                    elif context_name == "local":
                        self.local_exploration_results.append(tool_result_entry)
                    
                    tool_calls_executed += 1
                    
                except Exception as e:
                    traceback.print_exc()
                    error_message = f"Error: {str(e)}"
                    log_tool_error(tool_name, tool_args, error_message, context_name)
                    
                    # Record failed tool execution in statistics
                    if self.statistics:
                        self.statistics.record_tool_execution(
                            tool_name, tool_args, start_time, error_message, 
                            context_name, success=False, error_message=error_message
                        )
                    
                    self.memory.add(f"Tool {tool_name} failed with arguments {tool_args}: {error_message}", "System")
                    tool_error_entry = {
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "error": error_message,
                        "context": context_name
                    }
                    self.tool_call_results.append(tool_error_entry)
                    
                    # Add to specific context result lists (errors are important too!)
                    if context_name == "remote":
                        self.remote_exploration_results.append(tool_error_entry)
                    elif context_name == "local":
                        self.local_exploration_results.append(tool_error_entry)
        
        return tool_calls_executed



    async def get_local_exploration_tools(self) -> List[Dict]:
        """Return tools for local graph exploration."""
        return self.local_exploration_toolbox.get_openai_tools() if self.local_exploration_toolbox else []

    async def get_remote_exploration_tools(self) -> List[Dict]:
        """Return tools for remote graph exploration."""
        return self.remote_exploration_toolbox.get_openai_tools() if self.remote_exploration_toolbox else []

    async def get_graph_update_tools(self) -> List[Dict]:
        """Return tools for graph updates."""
        return self.graph_update_toolbox.get_openai_tools() if self.graph_update_toolbox else []

    async def get_evaluation_tools(self) -> List[Dict]:
        """Return tools for evaluation phases."""
        return self.evaluation_toolbox.get_openai_tools() if self.evaluation_toolbox else []
    
    async def get_all_available_tools(self) -> List[Dict]:
        """Return all available tools from all toolboxes for question decomposition."""
        all_tools = []
        
        # Add tools from all toolboxes
        if self.local_exploration_toolbox:
            all_tools.extend(self.local_exploration_toolbox.get_openai_tools())
        if self.remote_exploration_toolbox:
            all_tools.extend(self.remote_exploration_toolbox.get_openai_tools())
        if self.graph_update_toolbox:
            all_tools.extend(self.graph_update_toolbox.get_openai_tools())
        if self.evaluation_toolbox:
            all_tools.extend(self.evaluation_toolbox.get_openai_tools())
            
        return all_tools
    


    async def start(self) -> None:
        """Start the multi-stage orchestration."""
        self._running = True
        self._stop_event.clear()
        
        while self._running and not self._stop_event.is_set():
            try:
                # Check max_turns limit
                if self.max_turns > 0:
                    if self.current_turn >= self.max_turns:
                        self.memory.add(f"Reached maximum turns limit ({self.max_turns}), stopping orchestration", "System")
                        log_debug(f"Reached maximum turns limit ({self.max_turns}), stopping orchestration")
                        if not self.final_answer:
                            self.final_answer = "Maximum turns limit reached without finding a complete answer"
                        self.state = self.OrchestratorState.FINISHED
                        break
                    elif self.current_turn >= self.max_turns - 1:
                        # Force answer production on the last turn
                        log_debug(f"Reached second-to-last turn ({self.current_turn + 1}/{self.max_turns}), forcing answer production")
                        self.memory.add(f"Forcing answer production on turn {self.current_turn + 1} (last allowed turn)", "System")
                        if self.state == self.OrchestratorState.AUTONOMOUS:
                            await self._transition_to_substate(self.AutonomousSubstate.PRODUCE_ANSWER)
                
                if self.state == self.OrchestratorState.READY:
                    await self._handle_ready_state()
                elif self.state == self.OrchestratorState.QUESTION_DECOMPOSITION:
                    await self._handle_question_decomposition_state()
                elif self.state == self.OrchestratorState.AUTONOMOUS:
                    self.current_turn += 1
                    log_turn_separator(self.current_turn, "ORCHESTRATOR")
                    log_debug(f"Starting turn {self.current_turn}" + (f" of {self.max_turns}" if self.max_turns > 0 else ""))
                    await self._handle_autonomous_state()
                elif self.state == self.OrchestratorState.FINISHED:
                    await self._handle_finished_state()
                    break
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            
            except Exception as e:
                traceback.print_exc()
                self.memory.add(f"Error in orchestration: {str(e)}", "System")
                self.state = self.OrchestratorState.FINISHED
                break
    
    async def process_user_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return the result."""
        self.current_query = Query(text=query)
        self.original_query = Query(text=query)
        
        # Initialize statistics collector for this query
        self.statistics = create_statistics_collector(
            self.experiment_id, 
            query, 
            enable_metacognition=self.metacognition is not None
        )
        
        # Start with question decomposition if enabled, otherwise go directly to autonomous
        if self.enable_question_decomposition:
            await self._transition_to_state(self.OrchestratorState.QUESTION_DECOMPOSITION)
            self.substate = None
        else:
            await self._transition_to_state(self.OrchestratorState.AUTONOMOUS)
            await self._transition_to_substate(self.AutonomousSubstate.EVAL_LOCAL_DATA)
            
        # Reset query-specific variables
        self.tool_call_results = []  # Reset tool call results for new query
        self.remote_exploration_results = []  # Reset remote exploration results for new query
        self.local_exploration_results = []  # Reset local exploration results for new query
        self.current_turn = 0  # Reset turn counter for new query
        
        # Reset question decomposition fields
        self.sub_questions = []
        self.current_sub_question_index = 0
        self.sub_question_answers = []
        
        # Log query start
        self.memory.add(f"Starting query processing: {query}", "System")
        if self.enable_question_decomposition:
            log_debug("Question decomposition is enabled")
        else:
            log_debug("Question decomposition is disabled, proceeding directly to autonomous processing")
        
        # Ensure knowledge graph is connected before starting
        await self._connect_knowledge_graph()
        
        # Start the orchestration process
        await self.start()
        
        # Finalize statistics and save them
        if self.statistics:
            self.statistics.finalize(self.final_answer or "No answer generated", 
                                   success=bool(self.final_answer and not self.final_answer.startswith("Error")))
            # Save statistics to file
            try:
                stats_file = self.statistics.save_to_file()
                log_debug(f"Statistics saved to: {stats_file}")
            except Exception as e:
                log_debug(f"Failed to save statistics: {e}")
        
        # Get statistics data for return
        stats_data = {}
        if self.statistics:
            stats_data = {
                "remote_explorations": self.statistics.attempt_history.remote_explorations,
                "local_explorations": self.statistics.attempt_history.local_explorations,
                "failures": self.statistics.attempt_history.failures
            }
        else:
            stats_data = {"remote_explorations": 0, "local_explorations": 0, "failures": []}
        
        # Generate final visualization with the complete graph and final answer
        await self._generate_final_visualization()
        
        # Generate animation if multiple turns occurred
        animation_path = None
        if self.current_turn > 1:
            animation_path = await self.generate_animation()
        
        # Create visualization index
        index_path = await self._create_visualization_index()
        
        # Log visualization information
        if index_path:
            log_debug(f"📊 Knowledge graph visualizations available at: {index_path}", "VISUALIZATION")
            print(f"📊 View knowledge graph evolution at: {index_path}")
            print(f"📁 All visualizations saved in: {self.graph_visualizer.output_dir}")
        
        return {
            "answer": self.final_answer,
            "original_query": self.original_query.text,
            "sub_questions": self.sub_questions,
            "sub_question_answers": self.sub_question_answers,
            "question_decomposition_enabled": self.enable_question_decomposition,
            "attempts": stats_data,
            "final_state": self.state.value,
            "memory_summary": self.memory.read(max_characters=3000),  # Increased from 1000
            "tool_call_results": self.tool_call_results,
            "turns_taken": self.current_turn,
            "max_turns": self.max_turns,
            "statistics_file": self.statistics.save_to_file() if self.statistics else None,
            "visualizations": {
                "output_dir": str(self.graph_visualizer.output_dir),
                "animation_path": animation_path,
                "final_image": str(self.graph_visualizer.output_dir / "final_graph_with_answer.png") if (self.graph_visualizer.output_dir / "final_graph_with_answer.png").exists() else None,
                "index_html": index_path
            }
        }
    


    ###############
    # MAIN STATES #
    ###############

    #TODO Handle waiting for user input in READY state
    async def _handle_ready_state(self):
        """Handle READY state - wait for user input."""
        await asyncio.sleep(1)
    
    async def _handle_autonomous_state(self):
        """Handle AUTONOMOUS state and its substates."""
        
        if self.substate == self.AutonomousSubstate.LOCAL_GRAPH_EXPLORATION:
            await self._handle_local_graph_exploration()
        elif self.substate == self.AutonomousSubstate.EVAL_LOCAL_DATA:
            await self._handle_eval_local_data()
        elif self.substate == self.AutonomousSubstate.PRODUCE_ANSWER:
            await self._handle_produce_answer()
        elif self.substate == self.AutonomousSubstate.REMOTE_GRAPH_EXPLORATION:
            await self._handle_remote_graph_exploration()
        elif self.substate == self.AutonomousSubstate.EVAL_REMOTE_DATA:
            await self._handle_eval_remote_data()
        elif self.substate == self.AutonomousSubstate.LOCAL_GRAPH_UPDATE:
            await self._handle_local_graph_update()
        
        # Generate visualization at the end of each turn (after substate processing)
        await self._generate_turn_visualization()
    
    async def _handle_question_decomposition_state(self):
        """Handle QUESTION_DECOMPOSITION state - decompose complex query into sub-questions."""
        
        if not self.original_query:
            log_debug("No original query available")
            self.state = self.OrchestratorState.FINISHED
            return
        
        try:
            # Get all available tools to include in the prompt
            all_tools = await self.get_all_available_tools()
            
            # Create question decomposition prompt
            memory_context = self.memory.read(max_characters=8000)  # Increased from 2000
            decomposition_prompt = create_question_decomposition_prompt(
                self.original_query.text,
                memory_context,
                all_tools
            )
            
            log_prompt(
                decomposition_prompt.system,
                decomposition_prompt.user,
                decomposition_prompt.assistant,
                "question_decomposition",
                all_tools
            )
            
            # Use the prompt structure to get properly formatted messages
            messages = decomposition_prompt.get_messages()
            response = await self._query_llm_with_timing(messages, tools=None, context_type="orchestrator")  # No tools in decomposition phase
            
            log_response(response.content, "question_decomposition")
            
            # Add LLM response to memory
            if response.content and response.content.strip():
                self.memory.add(f"LLM_Agent: {response.content}", "LLM_Agent")
            
            # Extract sub-questions from response
            self._extract_sub_questions_from_response(response.content)
            
            # If we have sub-questions, start processing them, otherwise go directly to autonomous
            if self.sub_questions:
                log_debug(f"Generated {len(self.sub_questions)} sub-questions: {self.sub_questions}")
                self.memory.add(f"Generated {len(self.sub_questions)} sub-questions for processing", "System")
                self.current_sub_question_index = 0
                await self._process_next_sub_question()
            else:
                log_debug("No sub-questions generated, proceeding with original query")
                self.state = self.OrchestratorState.AUTONOMOUS
                await self._transition_to_substate(self.AutonomousSubstate.EVAL_LOCAL_DATA)
                
        except Exception as e:
            traceback.print_exc()
            # Track failure in statistics
            if self.statistics:
                self.statistics.attempt_history.failures.append(f"Question decomposition failed: {str(e)}")
            self.memory.add(f"Question decomposition failed: {str(e)}", "System")
            # Fall back to processing original query directly
            self.state = self.OrchestratorState.AUTONOMOUS
            await self._transition_to_substate(self.AutonomousSubstate.EVAL_LOCAL_DATA)

    def _extract_sub_questions_from_response(self, response_content: str):
        """Extract sub-questions from LLM response."""
        if not response_content:
            return
            
        lines = response_content.split('\n')
        sub_questions = []
        
        for line in lines:
            line = line.strip()
            # Look for numbered questions or bullet points
            if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.',
                                                          '•', '-', '*', 'Q1:', 'Q2:', 'Q3:', 'Q4:', 'Q5:']):
                # Clean up the question
                for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.',
                               '•', '-', '*', 'Q1:', 'Q2:', 'Q3:', 'Q4:', 'Q5:']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                
                if line and line.endswith('?'):
                    sub_questions.append(line)
        
        # If no structured questions found, try to find sentences ending with '?'
        if not sub_questions:
            sentences = response_content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence.endswith('?'):
                    sub_questions.append(sentence + '?')
        
        self.sub_questions = sub_questions

    async def _process_next_sub_question(self):
        """Process the next sub-question in the sequence."""
        if self.current_sub_question_index >= len(self.sub_questions):
            # All sub-questions processed, create final answer
            await self._create_final_answer_from_sub_answers()
            return
        
        # Get current sub-question
        current_sub_question = self.sub_questions[self.current_sub_question_index]
        log_debug(f"Processing sub-question {self.current_sub_question_index + 1}/{len(self.sub_questions)}: {current_sub_question}")
        
        # Build context from previous sub-question answers
        context = ""
        if self.sub_question_answers:
            context = "Previous sub-question answers:\n"
            for i, answer in enumerate(self.sub_question_answers):
                context += f"Q{i+1}: {self.sub_questions[i]}\nA{i+1}: {answer}\n\n"
            context += "Current sub-question:\n"
        
        # Set current query to the sub-question with context
        self.current_query = Query(text=f"{context}{current_sub_question}")
        
        # Reset autonomous state variables for this sub-question
        # (Statistics continue from main query, but we reset tool results)
        self.tool_call_results = []
        self.remote_exploration_results = []
        self.local_exploration_results = []
        self.current_turn = 0  # Reset turn counter for each sub-question
        
        # Start autonomous processing for this sub-question
        self.state = self.OrchestratorState.AUTONOMOUS
        await self._transition_to_substate(self.AutonomousSubstate.EVAL_LOCAL_DATA)

    async def _create_final_answer_from_sub_answers(self):
        """Create a final comprehensive answer from all sub-question answers."""
        if not self.sub_question_answers:
            self.final_answer = "No answers were generated for the sub-questions."
            self.state = self.OrchestratorState.FINISHED
            return
            
        # Combine all sub-answers into a comprehensive response
        original_text = self.original_query.text if self.original_query else "Unknown question"
        final_answer_parts = [f"Original question: {original_text}\n"]
        
        for i, (question, answer) in enumerate(zip(self.sub_questions, self.sub_question_answers)):
            final_answer_parts.append(f"Sub-question {i+1}: {question}")
            final_answer_parts.append(f"Answer {i+1}: {answer}\n")
        
        final_answer_parts.append("Comprehensive Answer:")
        final_answer_parts.append("Based on the analysis of the sub-questions above, " + 
                                 " ".join(self.sub_question_answers))
        
        self.final_answer = "\n".join(final_answer_parts)
        self.state = self.OrchestratorState.FINISHED
    

    #TODO Handle waiting for new user query
    async def _handle_finished_state(self):
        """Handle FINISHED state."""
        self._running = False
    


    ########################
    # AUTONOMOUS SUBSTATES #
    ########################

    async def _handle_eval_local_data(self):
        """Handle EVAL_LOCAL_DATA substate."""
        
        if not self.current_query:
            log_debug("No current query available")
            self.state = self.OrchestratorState.FINISHED
            return
        
        # Invoke metacognition before evaluation if enabled (but not on first turn)
        metacognitive_suggestion = None
        local_graph_data = None
        if self.current_turn > 1:
            # Get the same data that will be passed to the evaluation prompt
            local_graph_data = await self.knowledge_graph.to_readable_format()
            next_step_tools = await self.get_remote_exploration_tools()
            
            metacognitive_suggestion = await self._invoke_metacognition(
                "EVAL_LOCAL_DATA",
                local_graph_data=local_graph_data,
                local_exploration_results=self.local_exploration_results,
                next_step_tools=next_step_tools
            )
        
        # First, get and analyze local graph data (reuse if already fetched for metacognition)
        if local_graph_data is None:
            local_graph_data = await self.knowledge_graph.to_readable_format()
        log_debug(f"Local graph data: {local_graph_data}", "EVAL_LOCAL_DATA")

        # If local graph is empty, proceed based on turn number
        if not local_graph_data:
            if self.current_turn == 1:
                # On first turn, force local exploration instead of going to remote
                fictitious_response = f"The local graph is EMPTY. On the first turn, let's start with local graph exploration to gather initial information."
                self.memory.add(f"LLM_Agent: {fictitious_response}", "LLM_Agent")
                await self._transition_to_substate(self.AutonomousSubstate.LOCAL_GRAPH_EXPLORATION)
                return
            else:
                # On subsequent turns, allow remote exploration
                fictitious_response = f"The local graph is EMPTY. Let's proceed with remote graph exploration to find the necessary information."
                self.memory.add(f"LLM_Agent: {fictitious_response}", "LLM_Agent")
                await self._transition_to_substate(self.AutonomousSubstate.REMOTE_GRAPH_EXPLORATION)
                return
        
        try:
            # Get evaluation tools
            eval_tools = await self.get_evaluation_tools()

            # Create evaluation prompt with statistics attempt history and local graph data
            memory_context = self.memory.read(max_characters=10000)  # Increased from 3000
            eval_prompt = create_local_evaluation_prompt(
                self.current_query.text, 
                self.statistics.attempt_history if self.statistics else None, 
                memory_context,
                local_graph_data,
                self.memory.get_last_llm_response(),
                await self.get_remote_exploration_tools(),
                self.substate.value if self.substate else "Initial",
                self.local_exploration_results,
                strategy=metacognitive_suggestion  # Use metacognitive suggestion as strategy
            )
            
            log_prompt(
                eval_prompt.system, 
                eval_prompt.user, 
                eval_prompt.assistant, 
                "local_evaluation", 
                eval_tools
            )

            # Use the prompt structure to get properly formatted messages
            messages = eval_prompt.get_messages()
            response = await self._query_llm_with_timing(messages, tools=eval_tools, temperature=0.8, operation_type="evaluation")

            log_response(response.content, "local_evaluation")

            # Add LLM response to memory if not empty
            if response.content and response.content.strip():
                self.memory.add(f"Local evaluation response: {response.content}", "LLM_Agent")

            # Process tool calls if present
            await self._process_tool_calls(response, "evaluation")
                        
            # Check if LLM can provide final answer
            if await has_sufficient_local_data({"response": response.content}):
                await self._transition_to_substate(self.AutonomousSubstate.PRODUCE_ANSWER)
            else:
                # Decide whether to do remote exploration or go back to local
                if "REMOTE_GRAPH_EXPLORATION" in response.content:
                    await self._transition_to_substate(self.AutonomousSubstate.REMOTE_GRAPH_EXPLORATION)
                elif "LOCAL_GRAPH_EXPLORATION" in response.content:
                    await self._transition_to_substate(self.AutonomousSubstate.LOCAL_GRAPH_EXPLORATION)
                else:
                    # Could not find sufficient data, produce partial answer
                    self.memory.add("Could not find sufficient data, producing partial answer.", "System")
                    await self._transition_to_substate(self.AutonomousSubstate.PRODUCE_ANSWER)
                 
        except Exception as e:
            traceback.print_exc()
            self._track_failure(f"Local evaluation failed: {str(e)}")
            self.memory.add(f"Local evaluation failed: {str(e)}", "System")
            self.final_answer = f"Error during evaluation: {str(e)}"
            self.state = self.OrchestratorState.FINISHED

    async def _handle_local_graph_exploration(self):
        """Handle LOCAL_GRAPH_EXPLORATION substate."""
        
        if not self.current_query:
            print("No current query available")
            self.state = self.OrchestratorState.FINISHED
            return
            
        # Track local exploration attempt in statistics
        if self.statistics:
            self.statistics.attempt_history.local_explorations += 1
        
        # Clear previous local exploration results for a fresh start
        self.local_exploration_results = []
        
        try:
            # Get local graph data
            local_graph_data = await self.knowledge_graph.to_readable_format()
            
            # Get local exploration tools
            local_tools = await self.get_local_exploration_tools()
            
            # Let LLM decide on exploration using local exploration tools
            memory_context = self.memory.read(max_characters=8000)  # Increased from 2000
            exploration_prompt = create_local_exploration_prompt(
                self.current_query.text, 
                memory_context, 
                local_graph_data,
                self.memory.get_last_llm_response(),
                self.substate.value if self.substate else "Initial",
                local_tools,
                self.local_exploration_results
            )

            log_prompt(
                exploration_prompt.system,
                exploration_prompt.user,
                exploration_prompt.assistant,
                "local_exploration",
                local_tools
            )
            
            # Use the prompt structure to get properly formatted messages
            messages = exploration_prompt.get_messages()
            response = await self._query_llm_with_timing(messages, tools=local_tools, operation_type="exploration")
            
            log_response(response.content, "local_exploration")
            
            # Add LLM response to memory if not empty
            if response.content and response.content.strip():
                self.memory.add(f"Local exploration response: {response.content}", "LLM_Agent")

            # Process tool calls if present
            tool_calls_executed = await self._process_tool_calls(response, "local")
            
            # Safety check: if no tools were executed and no clear continuation signal,
            # exit local exploration to prevent infinite loops
            if tool_calls_executed == 0 and not response.tool_calls:
                log_debug("No tools executed in local exploration and no tool calls requested - moving to evaluation", "LOCAL_EXPLORATION")
                self.memory.add("No tools executed in local exploration, moving to evaluation phase", "System")
                await self._transition_to_substate(self.AutonomousSubstate.EVAL_LOCAL_DATA)
                return
                    
            # Check if LLM wants to continue exploration or move to evaluation
            if await should_continue_local_exploration({"response": response.content}):
                # Continue exploration - stay in current substate
                pass
            else:
                # Move to evaluation
                await self._transition_to_substate(self.AutonomousSubstate.EVAL_LOCAL_DATA)
                
        except Exception as e:
            traceback.print_exc()
            self._track_failure(f"Local exploration failed: {str(e)}")
            self.memory.add(f"Local exploration failed: {str(e)}", "System")
            await self._transition_to_substate(self.AutonomousSubstate.EVAL_LOCAL_DATA)
    
    async def _handle_remote_graph_exploration(self):
        """Handle REMOTE_GRAPH_EXPLORATION substate."""
        
        if not self.current_query:
            print("No current query available")
            self.state = self.OrchestratorState.FINISHED
            return

        # Track remote exploration attempt in statistics
        if self.statistics:
            self.statistics.attempt_history.remote_explorations += 1
        
        # Clear previous remote exploration results for a fresh start
        self.remote_exploration_results = []
        
        try:
            # Get local graph data
            local_graph_data = await self.knowledge_graph.to_readable_format()
            
            # Get remote exploration tools
            remote_tools = await self.get_remote_exploration_tools()
            
            # Create remote exploration prompt
            memory_context = self.memory.read(max_characters=8000)  # Increased from 2000
            remote_prompt = create_remote_exploration_prompt(
                self.current_query.text, 
                memory_context, 
                local_graph_data,
                self.memory.get_last_llm_response(),
                await self.get_remote_exploration_tools(),
                self.substate.value if self.substate else "Initial"
            )
            
            log_prompt(
                remote_prompt.system,
                remote_prompt.user,
                remote_prompt.assistant,
                "remote_exploration",
                remote_tools
            )
            
            # Use the prompt structure to get properly formatted messages
            messages = remote_prompt.get_messages()
            response = await self._query_llm_with_timing(messages, tools=remote_tools, temperature=0.8, operation_type="exploration")

            log_response(response.content, "remote_exploration")
            
            # Add LLM response to memory if not empty
            if response.content and response.content.strip():
                self.memory.add(f"Remote exploration response: {response.content}", "LLM_Agent")
            
            # Process tool calls if present
            tool_calls_executed = await self._process_tool_calls(response, "remote")
            
            # Print summary of all tool results if any were executed
            if tool_calls_executed > 0:
                log_tool_results_summary(self.tool_call_results, "remote_exploration")
            
            
            # Now evaluate the retrieved data
            self.current_remote_data = self.remote_exploration_results
            log_debug("Remote data collected:\n" + pprint.pformat(self.current_remote_data), "REMOTE_EXPLORATION")
            await self._transition_to_substate(self.AutonomousSubstate.EVAL_REMOTE_DATA)
                
        except Exception as e:
            traceback.print_exc()
            self._track_failure(f"Remote exploration error: {str(e)}")
            self.memory.add(f"Remote exploration error: {str(e)}", "System")
            await self._transition_to_substate(self.AutonomousSubstate.EVAL_LOCAL_DATA)
    
    async def _handle_eval_remote_data(self):
        """Handle EVAL_REMOTE_DATA substate."""
        
        if not self.current_query:
            print("No current query available")
            self.state = self.OrchestratorState.FINISHED
            return
            
        if not self.current_remote_data:
            print("No remote data to evaluate")
            await self._transition_to_substate(self.AutonomousSubstate.EVAL_LOCAL_DATA)
            return
        
        # Invoke metacognition before evaluation if enabled (but not on first turn)
        local_graph_data = None
        next_step_tools = None
        metacognitive_suggestion = None
        if self.current_turn > 1:
            # Get the same data that will be passed to the evaluation prompt
            local_graph_data = await self.knowledge_graph.to_readable_format()
            next_step_tools = await self.get_graph_update_tools()
            
            metacognitive_suggestion = await self._invoke_metacognition(
                "EVAL_REMOTE_DATA",
                local_graph_data=local_graph_data,
                remote_data=self.current_remote_data,
                next_step_tools=next_step_tools
            )
            
        try:
            # Get evaluation tools
            eval_tools = await self.get_evaluation_tools()

            memory_context = self.memory.read(max_characters=8000)  # Increased from 2000

            # Get data for evaluation prompt (reuse if already fetched for metacognition)
            if local_graph_data is None:
                local_graph_data = await self.knowledge_graph.to_readable_format()
            if next_step_tools is None:
                next_step_tools = await self.get_graph_update_tools()

            # Create evaluation prompt
            eval_prompt = create_remote_evaluation_prompt(
                self.current_query.text,
                memory_context,
                self.current_remote_data,
                local_graph_data,
                self.memory.get_last_llm_response(),
                next_step_tools,
                self.substate.value if self.substate else "Initial",
                strategy=metacognitive_suggestion  # Use metacognitive suggestion as strategy
            )

            log_prompt(
                eval_prompt.system,
                eval_prompt.user,
                eval_prompt.assistant,
                "remote_evaluation",
                eval_tools
            )

            # Use the prompt structure to get properly formatted messages
            messages = eval_prompt.get_messages()
            response = await self._query_llm_with_timing(messages, tools=eval_tools, temperature=0.8, operation_type="evaluation")

            log_response(response.content, "remote_evaluation")
            
            # Add LLM response to memory if not empty
            if response.content and response.content.strip():
                self.memory.add(f"Remote evaluation response: {response.content}", "LLM_Agent")
            
            # Process tool calls if present
            await self._process_tool_calls(response, "evaluation")
            
            
            # Check if remote data is relevant
            if await is_remote_data_relevant({"response": response.content}):
                await self._transition_to_substate(self.AutonomousSubstate.LOCAL_GRAPH_UPDATE)
            elif "REMOTE_GRAPH_EXPLORATION" in response.content:
                self.current_remote_data = None
                await self._transition_to_substate(self.AutonomousSubstate.REMOTE_GRAPH_EXPLORATION)
            else:
                # Remote data is irrelevant, go back to local evaluation
                self.current_remote_data = None
                await self._transition_to_substate(self.AutonomousSubstate.EVAL_LOCAL_DATA)
                
        except Exception as e:
            traceback.print_exc()
            self._track_failure(f"Remote evaluation failed: {str(e)}")
            self.memory.add(f"Remote evaluation failed: {str(e)}", "System")
            await self._transition_to_substate(self.AutonomousSubstate.EVAL_LOCAL_DATA)
    
    async def _handle_local_graph_update(self):
        """Handle LOCAL_GRAPH_UPDATE substate."""
        
        if not self.current_query:
            print("No current query available")
            self.state = self.OrchestratorState.FINISHED
            return
            
        try:
            # Get graph update tools
            update_tools = await self.get_graph_update_tools()
            
            memory_context = self.memory.read(max_characters=8000)  # Increased from 2000
                

            # Create graph update prompt
            graph_update_prompt = create_graph_update_prompt(
                self.current_query.text,
                memory_context,
                self.current_remote_data,
                await self.knowledge_graph.to_readable_format(),
                self.memory.get_last_llm_response(),
                await self.get_evaluation_tools(),
                self.substate.value if self.substate else "Initial"
            )
            
            log_prompt(
                graph_update_prompt.system,
                graph_update_prompt.user,
                graph_update_prompt.assistant,
                "graph_update",
                update_tools
            )
            
            # Use the prompt structure to get properly formatted messages
            messages = graph_update_prompt.get_messages()
            response = await self._query_llm_with_timing(messages, tools=update_tools, operation_type="update")

            log_response(response.content, "graph_update")
            
            # Add LLM response to memory if not empty
            if response.content and response.content.strip():
                self.memory.add(f"Graph update response: {response.content}", "LLM_Agent")
            
            # Process tool calls if present
            tool_calls_executed = await self._process_tool_calls(response, "update")
            
            # Now go to the eval step
            self.current_remote_data = None
            await self._transition_to_substate(self.AutonomousSubstate.EVAL_LOCAL_DATA)
            self.current_remote_data = None
            
        except Exception as e:
            traceback.print_exc()
            self._track_failure(f"Graph update failed: {str(e)}")
            self.memory.add(f"Graph update failed: {str(e)}", "System")
            await self._transition_to_substate(self.AutonomousSubstate.EVAL_LOCAL_DATA)



    async def _handle_produce_answer(self):
        """Handle PRODUCE_ANSWER substate: produce answer with proof sets."""
        
        if not self.current_query:
            print("No current query available")
            self.state = self.OrchestratorState.FINISHED
            return
            
        try:
            # Get local graph data
            local_graph_data = await self.knowledge_graph.to_readable_format()
            
            # Create answer production prompt
            memory_context = self.memory.read(max_characters=4000)
            answer_prompt = create_answer_production_prompt(
                self.current_query.text, 
                memory_context, 
                local_graph_data,
                self.memory.get_last_llm_response(),
                self.substate.value if self.substate else "Initial"
            )
            
            log_prompt(
                answer_prompt.system,
                answer_prompt.user,
                answer_prompt.assistant,
                "produce_answer"
            )

            messages = answer_prompt.get_messages()
            response = await self._query_llm_with_timing(messages, operation_type="evaluation")

            log_response(response.content, "produce_answer")

            # Add LLM response to memory if not empty
            if response.content and response.content.strip():
                self.memory.add(f"Answer production response: {response.content}", "LLM_Agent")

            # Process tool calls if present
            await self._process_tool_calls(response, "produce_answer")

            # Check if we're processing sub-questions
            if self.sub_questions and self.current_sub_question_index < len(self.sub_questions):
                # Store answer for current sub-question
                self.sub_question_answers.append(response.content)
                log_debug(f"Completed sub-question {self.current_sub_question_index + 1}/{len(self.sub_questions)}")
                
                # Move to next sub-question
                self.current_sub_question_index += 1
                await self._process_next_sub_question()
            else:
                # No sub-questions or all completed, store final answer and finish
                self.final_answer = response.content
                self.state = self.OrchestratorState.FINISHED
                self.memory.add("Final answer produced and orchestration finished.", "System")
                
        except Exception as e:
            traceback.print_exc()
            self._track_failure(f"Answer production failed: {str(e)}")
            self.memory.add(f"Answer production failed: {str(e)}", "System")
            self.final_answer = f"Error during answer production: {str(e)}"
            self.state = self.OrchestratorState.FINISHED
