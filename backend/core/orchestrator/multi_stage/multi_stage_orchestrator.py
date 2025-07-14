import asyncio
import json
import pprint
import traceback
from typing import Any, List, Dict, Optional, Tuple
from enum import Enum

from .formatting import (
    Colors,
    print_prompt, 
    print_tool_result, 
    print_tool_error, 
    print_debug, 
    print_response,
    print_tool_results_summary
)

from ..orchestrator import Orchestrator
from ...memory import Memory, SimpleMemory
from ...agents.agent import Query
from ...toolbox.toolbox import Toolbox
from ...knowledge_base.graph import KnowledgeGraph, get_knowledge_graph
from .prompts import (
    AttemptHistory,
    PromptStructure,
    create_local_exploration_prompt,
    create_local_evaluation_prompt,
    create_remote_exploration_prompt,
    create_remote_evaluation_prompt,
    create_graph_update_prompt,
    create_answer_production_prompt
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
    """Multi-stage orchestrator with autonomous graph exploration capabilities."""
    
    class OrchestratorState(Enum):
        """Main orchestrator states."""
        READY = "ready"
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
                 use_summary_memory: bool = False,
                 memory_max_slots: int = 10,
                 max_turns: int = -1):
        # Use SummaryMemory or SimpleMemory with larger capacity for complex orchestration
        if memory is None:
            if use_summary_memory:
                from ...memory import SummaryMemory
                memory = SummaryMemory(max_slots=memory_max_slots, agent=agent)
            else:
                memory = SimpleMemory(max_slots=memory_max_slots)
        super().__init__(agent, memory)
        self.knowledge_graph = knowledge_graph
        self.context_length = context_length
        self.state = self.OrchestratorState.READY
        self.substate = None
        self.current_query = None
        self.attempt_history = AttemptHistory()
        self.current_remote_data = None
        self.final_answer = None
        self.tool_call_results = []  # Store all tool call results
        self.max_turns = max_turns
        self.current_turn = 0
        
        # Store toolboxes for different phases
        self.local_exploration_toolbox = local_exploration_toolbox
        self.remote_exploration_toolbox = remote_exploration_toolbox
        self.graph_update_toolbox = graph_update_toolbox
        self.evaluation_toolbox = evaluation_toolbox

    async def _connect_knowledge_graph(self):
        """Connect to the knowledge graph if not already connected."""
        if not await self.knowledge_graph.is_connected():
            await self.knowledge_graph.connect()
        
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
            print_debug(f"Processing tool calls in context: {context_name}")
            for tool_call in response.tool_calls:
                print(tool_call)
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])

                print_debug(f"Tool call detected: {tool_name}")
                print_debug(f"Arguments for {tool_name}: {tool_args}")

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
                    print_tool_result(tool_name, tool_args, formatted_result, context_name)
                    
                    # Add to memory and results list
                    self.memory.add(f"Tool {tool_name} executed successfully in context '{context_name}'.", "System")
                    self.tool_call_results.append({
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": formatted_result,
                        "context": context_name
                    })
                    
                    tool_calls_executed += 1
                    
                except Exception as e:
                    traceback.print_exc()
                    error_message = f"Error: {str(e)}"
                    print_tool_error(tool_name, tool_args, error_message, context_name)
                    self.memory.add(f"Tool {tool_name} failed in context '{context_name}': {error_message}", "System")
                    self.tool_call_results.append({
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "error": error_message,
                        "context": context_name
                    })
        
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
    


    async def start(self) -> None:
        """Start the multi-stage orchestration."""
        self._running = True
        self._stop_event.clear()
        
        while self._running and not self._stop_event.is_set():
            try:
                # Check max_turns limit
                if self.max_turns > 0 and self.current_turn >= self.max_turns:
                    self.memory.add(f"Reached maximum turns limit ({self.max_turns}), stopping orchestration", "System")
                    print_debug(f"Reached maximum turns limit ({self.max_turns}), stopping orchestration")
                    if not self.final_answer:
                        self.final_answer = "Maximum turns limit reached without finding a complete answer"
                    self.state = self.OrchestratorState.FINISHED
                    break
                
                if self.state == self.OrchestratorState.READY:
                    await self._handle_ready_state()
                elif self.state == self.OrchestratorState.AUTONOMOUS:
                    self.current_turn += 1
                    print_debug(f"Starting turn {self.current_turn}" + (f" of {self.max_turns}" if self.max_turns > 0 else ""))
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
        self.state = self.OrchestratorState.AUTONOMOUS
        self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
        self.attempt_history = AttemptHistory()
        self.tool_call_results = []  # Reset tool call results for new query
        self.current_turn = 0  # Reset turn counter for new query
        
        # Log query start
        self.memory.add(f"Starting query processing: {query}", "System")
        
        # Ensure knowledge graph is connected before starting
        await self._connect_knowledge_graph()
        
        # Start the orchestration process
        await self.start()
        
        return {
            "answer": self.final_answer,
            "attempts": {
                "remote_explorations": self.attempt_history.remote_explorations,
                "local_explorations": self.attempt_history.local_explorations,
                "failures": self.attempt_history.failures
            },
            "final_state": self.state.value,
            "memory_summary": self.memory.read(max_characters=1000),
            "tool_call_results": self.tool_call_results,
            "turns_taken": self.current_turn,
            "max_turns": self.max_turns
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
            print_debug("No current query available")
            self.state = self.OrchestratorState.FINISHED
            return
        
        # First, get and analyze local graph data
        local_graph_data = await self.knowledge_graph.to_triples()
        print_debug(f"Local graph data: {local_graph_data}", "EVAL_LOCAL_DATA")
        
        try:
            # Get evaluation tools
            eval_tools = await self.get_evaluation_tools()
            
            # Create evaluation prompt with attempt history and local graph data
            memory_context = self.memory.read(max_characters=3000)
            eval_prompt = create_local_evaluation_prompt(
                self.current_query.text, 
                self.attempt_history, 
                memory_context,
                local_graph_data
            )
            
            print_prompt(
                eval_prompt.system, 
                eval_prompt.user, 
                eval_prompt.assistant, 
                "local_evaluation", 
                eval_tools
            )

            # Use the prompt structure to get properly formatted messages
            messages = eval_prompt.get_messages()
            response = await self.agent.query_llm(messages, tools=eval_tools)

            print_response(response.content, "local_evaluation")

            # Add LLM response to memory if not empty
            if response.content and response.content.strip():
                self.memory.add(f"Local evaluation response: {response.content}", "LLM_Agent")

            # Process tool calls if present
            await self._process_tool_calls(response, "evaluation")
                        
            # Check if LLM can provide final answer
            if await has_sufficient_local_data({"response": response.content}):
                self.substate = self.AutonomousSubstate.PRODUCE_ANSWER
                self.memory.add("Sufficient local data found, transitioning to PRODUCE_ANSWER", "System")
            else:
                # Check if we should attempt remote exploration
                if await should_attempt_remote_exploration({"response": response.content}, self.attempt_history.remote_explorations):
                    self.substate = self.AutonomousSubstate.REMOTE_GRAPH_EXPLORATION
                    self.memory.add("Insufficient local data, transitioning to REMOTE_GRAPH_EXPLORATION", "System")
                elif not local_graph_data or len(local_graph_data) == 0:
                    # No local data and decision is to explore locally first
                    self.substate = self.AutonomousSubstate.REMOTE_GRAPH_EXPLORATION
                    self.memory.add("Local graph is empty or insufficient, transitioning to REMOTE_GRAPH_EXPLORATION", "System")
                else:
                    # Provide partial answer or indicate impossibility
                    self.final_answer = await extract_partial_answer({"response": response.content})
                    self.state = self.OrchestratorState.FINISHED
                    self.memory.add("Cannot explore further, providing partial answer or indicating impossibility", "System")
                 
        except Exception as e:
            traceback.print_exc()
            self.attempt_history.failures.append(f"Local evaluation failed: {str(e)}")
            self.memory.add(f"Local evaluation failed: {str(e)}", "System")
            self.final_answer = f"Error during evaluation: {str(e)}"
            self.state = self.OrchestratorState.FINISHED

    async def _handle_local_graph_exploration(self):
        """Handle LOCAL_GRAPH_EXPLORATION substate."""
        
        if not self.current_query:
            print("No current query available")
            self.state = self.OrchestratorState.FINISHED
            return
            
        self.attempt_history.local_explorations += 1
        
        try:
            # Get local graph data
            local_graph_data = await self.knowledge_graph.to_triples()
            
            # Get local exploration tools
            local_tools = await self.get_local_exploration_tools()
            
            # Let LLM decide on exploration using local exploration tools
            memory_context = self.memory.read(max_characters=2000)
            exploration_prompt = create_local_exploration_prompt(self.current_query.text, memory_context, local_graph_data)

            print_prompt(
                exploration_prompt.system,
                exploration_prompt.user,
                exploration_prompt.assistant,
                "local_exploration",
                local_tools
            )
            
            # Use the prompt structure to get properly formatted messages
            messages = exploration_prompt.get_messages()
            response = await self.agent.query_llm(messages, tools=local_tools)
            
            print_response(response.content, "local_exploration")
            
            # Add LLM response to memory if not empty
            if response.content and response.content.strip():
                self.memory.add(f"Local exploration response: {response.content}", "LLM_Agent")

            # Process tool calls if present
            tool_calls_executed = await self._process_tool_calls(response, "local")
                    
            # Check if LLM wants to continue exploration or move to evaluation
            if await should_continue_local_exploration({"response": response.content}):
                # Continue exploration - stay in current substate
                pass
            else:
                # Move to evaluation
                self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
                
        except Exception as e:
            traceback.print_exc()
            self.attempt_history.failures.append(f"Local exploration failed: {str(e)}")
            self.memory.add(f"Local exploration failed: {str(e)}", "System")
            self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
    
    async def _handle_remote_graph_exploration(self):
        """Handle REMOTE_GRAPH_EXPLORATION substate."""
        
        if not self.current_query:
            print("No current query available")
            self.state = self.OrchestratorState.FINISHED
            return

        self.attempt_history.remote_explorations += 1
        
        try:
            # Get local graph data
            local_graph_data = await self.knowledge_graph.to_triples()
            
            # Get remote exploration tools
            remote_tools = await self.get_remote_exploration_tools()
            
            # Create remote exploration prompt
            memory_context = self.memory.read(max_characters=2000)
            remote_prompt = create_remote_exploration_prompt(self.current_query.text, memory_context, local_graph_data)
            
            print_prompt(
                remote_prompt.system,
                remote_prompt.user,
                remote_prompt.assistant,
                "remote_exploration",
                remote_tools
            )
            
            # Use the prompt structure to get properly formatted messages
            messages = remote_prompt.get_messages()
            response = await self.agent.query_llm(messages, tools=remote_tools)

            print_response(response.content, "remote_exploration")
            
            # Add LLM response to memory if not empty
            if response.content and response.content.strip():
                self.memory.add(f"Remote exploration response: {response.content}", "LLM_Agent")
            
            # Process tool calls if present
            tool_calls_executed = await self._process_tool_calls(response, "remote")
            
            # Print summary of all tool results if any were executed
            if tool_calls_executed > 0:
                print_tool_results_summary(self.tool_call_results, "remote_exploration")
            
            
            # Check if remote exploration was successful
            if await remote_exploration_successful({"response": response.content, "tool_calls_executed": tool_calls_executed}):
                self.current_remote_data = self.tool_call_results
                print_debug("Remote data collected:\n" + pprint.pformat(self.current_remote_data), "REMOTE_EXPLORATION")
                self.substate = self.AutonomousSubstate.EVAL_REMOTE_DATA
            else:
                # Failed, go back to local evaluation
                self.attempt_history.failures.append("Remote exploration failed")
                self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
                
        except Exception as e:
            traceback.print_exc()
            self.attempt_history.failures.append(f"Remote exploration error: {str(e)}")
            self.memory.add(f"Remote exploration error: {str(e)}", "System")
            self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
    
    async def _handle_eval_remote_data(self):
        """Handle EVAL_REMOTE_DATA substate."""
        
        if not self.current_query:
            print("No current query available")
            self.state = self.OrchestratorState.FINISHED
            return
            
        try:
            # Get local graph data
            local_graph_data = await self.knowledge_graph.to_triples()
            
            # Get evaluation tools
            eval_tools = await self.get_evaluation_tools()
            
            # Create remote data evaluation prompt
            eval_prompt = create_remote_evaluation_prompt(self.current_query.text, self.current_remote_data or [], local_graph_data)
            
            print_prompt(
                eval_prompt.system,
                eval_prompt.user,
                eval_prompt.assistant,
                "remote_evaluation",
                eval_tools
            )

            # Use the prompt structure to get properly formatted messages
            messages = eval_prompt.get_messages()
            response = await self.agent.query_llm(messages, tools=eval_tools)

            print_response(response.content, "remote_evaluation")
            
            # Add LLM response to memory if not empty
            if response.content and response.content.strip():
                self.memory.add(f"Remote evaluation response: {response.content}", "LLM_Agent")
            
            # Process tool calls if present
            await self._process_tool_calls(response, "evaluation")
            
            
            # Check if remote data is relevant
            if await is_remote_data_relevant({"response": response.content}):
                self.substate = self.AutonomousSubstate.LOCAL_GRAPH_UPDATE
            else:
                # Remote data is irrelevant, go back to local evaluation
                self.current_remote_data = None
                self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
                
        except Exception as e:
            traceback.print_exc()
            self.attempt_history.failures.append(f"Remote evaluation failed: {str(e)}")
            self.memory.add(f"Remote evaluation failed: {str(e)}", "System")
            self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
    
    async def _handle_local_graph_update(self):
        """Handle LOCAL_GRAPH_UPDATE substate."""
        
        if not self.current_query:
            print("No current query available")
            self.state = self.OrchestratorState.FINISHED
            return
            
        try:
            # Get local graph data
            local_graph_data = await self.knowledge_graph.to_triples()
            
            # Get graph update tools
            update_tools = await self.get_graph_update_tools()
            
            # Create update prompt
            update_prompt = create_graph_update_prompt(self.current_query.text, self.current_remote_data, local_graph_data)
            
            print_prompt(
                update_prompt.system,
                update_prompt.user,
                update_prompt.assistant,
                "graph_update",
                update_tools
            )
            
            # Use the prompt structure to get properly formatted messages
            messages = update_prompt.get_messages()
            response = await self.agent.query_llm(messages, tools=update_tools)

            print_response(response.content, "graph_update")
            
            # Add LLM response to memory if not empty
            if response.content and response.content.strip():
                self.memory.add(f"Graph update response: {response.content}", "LLM_Agent")
            
            # Process tool calls if present
            tool_calls_executed = await self._process_tool_calls(response, "update")
            
            
            # Check if graph update was successful using the new robust logic
            if await graph_update_successful({"response": response.content, "tool_calls_executed": tool_calls_executed}):
                
                # Clear remote data and go back to evaluating local data
                self.current_remote_data = None
                self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
            else:
                # If update fails, go back to local evaluation
                self.attempt_history.failures.append("Graph update failed after remote exploration.")
                self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
            
        except Exception as e:
            traceback.print_exc()
            self.attempt_history.failures.append(f"Graph update failed: {str(e)}")
            self.memory.add(f"Graph update failed: {str(e)}", "System")
            self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA



    async def _handle_produce_answer(self):
        """Handle PRODUCE_ANSWER substate: produce answer with proof sets."""
        
        if not self.current_query:
            print("No current query available")
            self.state = self.OrchestratorState.FINISHED
            return
            
        try:
            # Get local graph data
            local_graph_data = await self.knowledge_graph.to_triples()
            
            # Use the same tool set as local graph exploration
            local_tools = await self.get_local_exploration_tools()
            memory_context = self.memory.read(max_characters=3000)
            answer_prompt = create_answer_production_prompt(self.current_query.text, memory_context, local_graph_data)

            print_prompt(
                answer_prompt.system,
                answer_prompt.user,
                answer_prompt.assistant,
                "produce_answer",
                local_tools
            )

            messages = answer_prompt.get_messages()
            response = await self.agent.query_llm(messages, tools=local_tools)

            print_response(response.content, "produce_answer")

            # Add LLM response to memory if not empty
            if response.content and response.content.strip():
                self.memory.add(f"Answer production response: {response.content}", "LLM_Agent")

            # Process tool calls if present
            await self._process_tool_calls(response, "produce_answer")

            # Store the answer and finish
            self.final_answer = response.content
            self.state = self.OrchestratorState.FINISHED
            self.memory.add("Final answer produced and orchestration finished.", "System")
        except Exception as e:
            traceback.print_exc()
            self.attempt_history.failures.append(f"Answer production failed: {str(e)}")
            self.memory.add(f"Answer production failed: {str(e)}", "System")
            self.final_answer = f"Error during answer production: {str(e)}"
            self.state = self.OrchestratorState.FINISHED



if __name__ == "__main__":
    import asyncio
    import pprint
    import json
    import traceback
    from ...toolbox.toolbox import Toolbox
    from ...agents.openai import OpenAIAgent
    from ...agents.wattool_slm import WatToolSLMAgent
    from .toolboxes import (
        create_local_exploration_toolbox,
        create_remote_exploration_toolbox,
        create_graph_update_toolbox,
        create_evaluation_toolbox
    )
    
    
    # --- Agent Selection ---
    # Set to True to use the local watt-tool-8B SLM.
    # Make sure to run the model locally.
    USE_SLM = False
    
    # Create specialized toolboxes
    local_exploration_toolbox = create_local_exploration_toolbox()
    remote_exploration_toolbox = create_remote_exploration_toolbox()
    graph_update_toolbox = create_graph_update_toolbox()
    evaluation_toolbox = create_evaluation_toolbox()
    
    agent: Any
    if USE_SLM:
        print("Using local WatTool SLM Agent")
        # The toolbox needs to be passed to the agent
        agent = WatToolSLMAgent(toolbox=remote_exploration_toolbox)
    else:
        print("Using OpenAI Agent")
        agent = OpenAIAgent(model="gpt-4o")
    
    # Get knowledge graph instance
    knowledge_graph = get_knowledge_graph()
    
    orchestrator = MultiStageOrchestrator(
        agent, 
        knowledge_graph,
        context_length=8000,
        local_exploration_toolbox=local_exploration_toolbox,
        remote_exploration_toolbox=remote_exploration_toolbox,
        graph_update_toolbox=graph_update_toolbox,
        evaluation_toolbox=evaluation_toolbox,
        use_summary_memory=True,  # Enable SummaryMemory
        memory_max_slots=50,  # Customize memory size
        max_turns=10  # Limit to 10 turns
    )
    
    # Example usage
    async def main():
        #result = await orchestrator.process_user_query("Who was the mother of Douglas Adams?")
        #result = await orchestrator.process_user_query("What is the capital of France?")
        result = await orchestrator.process_user_query("What is the class of Douglas Adams?")
        print_debug(f"Answer: {result['answer']}")
        print_debug(f"Attempts: {result['attempts']}")
        print_debug(f"Turns taken: {result['turns_taken']} / {result['max_turns'] if result['max_turns'] > 0 else 'unlimited'}")
    
    asyncio.run(main())
