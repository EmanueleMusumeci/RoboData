import asyncio
from typing import Any, List, Dict, Optional, Tuple
from enum import Enum
import json
import pprint

from .formatting import (
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
    extract_final_answer,
    extract_partial_answer,
    extract_remote_data,
    fits_in_context,
    get_local_graph_data,
    send_graph_data_to_agent,
    send_entities_and_properties,
    update_local_graph
)

class MultiStageOrchestrator(Orchestrator):
    """Multi-stage orchestrator with autonomous graph exploration capabilities."""
    
    class OrchestratorState(Enum):
        """Main orchestrator states."""
        READY = "ready"
        AUTONOMOUS = "autonomous"
        FINISHED = "finished"
    
    class AutonomousSubstate(Enum):
        """Substates within AUTONOMOUS state."""
        SEND_LOCAL_DATA = "send_local_data"
        LOCAL_GRAPH_EXPLORATION = "local_graph_exploration"
        EVAL_LOCAL_DATA = "eval_local_data"
        PRODUCE_ANSWER = "produce_answer"
        REMOTE_GRAPH_EXPLORATION = "remote_graph_exploration"
        EVAL_REMOTE_DATA = "eval_remote_data"
        LOCAL_GRAPH_UPDATE = "local_graph_update"
    
    def __init__(self, 
                 agent: Any, 
                 memory: Optional[Memory] = None, 
                 context_length: int = 128000,
                 local_exploration_toolbox: Optional[Toolbox] = None,
                 remote_exploration_toolbox: Optional[Toolbox] = None,
                 graph_update_toolbox: Optional[Toolbox] = None,
                 evaluation_toolbox: Optional[Toolbox] = None):
        # Use SimpleMemory with larger capacity for complex orchestration
        if memory is None:
            memory = SimpleMemory(max_slots=200)
        super().__init__(agent, memory)
        self.context_length = context_length
        self.state = self.OrchestratorState.READY
        self.substate = None
        self.current_query = None
        self.attempt_history = AttemptHistory()
        self.current_remote_data = None
        self.final_answer = None
        self.tool_call_results = []  # Store all tool call results
        
        # Store toolboxes for different phases
        self.local_exploration_toolbox = local_exploration_toolbox
        self.remote_exploration_toolbox = remote_exploration_toolbox
        self.graph_update_toolbox = graph_update_toolbox
        self.evaluation_toolbox = evaluation_toolbox
        
    async def _process_tool_calls(self, response: Any, context_name: str = "tool") -> int:
        """Process tool calls from LLM response and add results to tool_call_results list.
        
        Args:
            response: The LLM response containing tool calls
            context_name: Context name for logging (e.g., "local", "remote", "evaluation", "update")
            
        Returns:
            Number of successfully executed tool calls
        """
        tool_calls_executed = 0
        
        if response.tool_calls:
            print_debug(f"Processing tool calls in context: {context_name}")
            for tool_call in response.tool_calls:
                tool_name = tool_call["function"]["name"]
                print_debug(f"Tool call detected: {tool_name}")
                arguments = {}
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                    print_debug(f"Arguments for {tool_name}: {arguments}")
                    result = await self.agent.toolbox.execute_tool(tool_name, **arguments)
                    print_tool_result(tool_name, arguments, result, context_name)
                    self.memory.add(f"{context_name.title()} tool {tool_name} executed: {result}")
                    tool_calls_executed += 1
                    
                    # Add to results list
                    self.tool_call_results.append({
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result
                    })
                except Exception as e:
                    print_tool_error(tool_name, arguments, str(e), context_name)
                    self.memory.add(f"{context_name.title()} tool {tool_name} failed: {str(e)}")
                    
                    # Add failed tool call to results list
                    self.tool_call_results.append({
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": f"Error: {str(e)}"
                    })
        
        return tool_calls_executed

    async def get_local_exploration_tools(self) -> List[Dict]:
        """Get tools for local graph exploration."""
        if self.local_exploration_toolbox:
            return self.local_exploration_toolbox.get_openai_tools()
        return []
    
    async def get_remote_exploration_tools(self) -> List[Dict]:
        """Get tools for remote graph exploration."""
        if self.remote_exploration_toolbox:
            return self.remote_exploration_toolbox.get_openai_tools()
        return []
    
    async def get_graph_update_tools(self) -> List[Dict]:
        """Get tools for graph updates."""
        if self.graph_update_toolbox:
            return self.graph_update_toolbox.get_openai_tools()
        return []
    
    async def get_evaluation_tools(self) -> List[Dict]:
        """Get tools for evaluation phases."""
        if self.evaluation_toolbox:
            return self.evaluation_toolbox.get_openai_tools()
        return []
    
    async def start(self) -> None:
        """Start the multi-stage orchestration."""
        self._running = True
        self._stop_event.clear()
        
        while self._running and not self._stop_event.is_set():
            try:
                if self.state == self.OrchestratorState.READY:
                    await self._handle_ready_state()
                elif self.state == self.OrchestratorState.AUTONOMOUS:
                    await self._handle_autonomous_state()
                elif self.state == self.OrchestratorState.FINISHED:
                    await self._handle_finished_state()
                    break
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.memory.add(f"Error in orchestration: {str(e)}")
                self.state = self.OrchestratorState.FINISHED
                break
    
    async def process_user_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return the result."""
        self.current_query = Query(text=query)
        self.state = self.OrchestratorState.AUTONOMOUS
        self.substate = self.AutonomousSubstate.SEND_LOCAL_DATA
        self.attempt_history = AttemptHistory()
        self.tool_call_results = []  # Reset tool call results for new query
        
        # Log query start
        self.memory.add(f"Starting query processing: {query}")
        
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
            "tool_call_results": self.tool_call_results
        }
    


    ###############
    # MAIN STATES #
    ###############

    async def _handle_ready_state(self):
        """Handle READY state - wait for user input."""
        # In a real implementation, this would wait for user input
        # For now, we'll just sleep
        await asyncio.sleep(1)
    
    async def _handle_autonomous_state(self):
        """Handle AUTONOMOUS state and its substates."""

        # Log state transitions
        #self.memory.add(f"Handling autonomous substate: {self.substate.value}")
        
        if self.substate == self.AutonomousSubstate.SEND_LOCAL_DATA:
            await self._handle_send_local_data()
        elif self.substate == self.AutonomousSubstate.LOCAL_GRAPH_EXPLORATION:
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
    
    async def _handle_finished_state(self):
        """Handle FINISHED state."""
        self._running = False
        self.memory.add(f"Orchestration finished with answer: {self.final_answer}")
    


    ########################
    # AUTONOMOUS SUBSTATES #
    ########################

    async def _handle_send_local_data(self):
        """Handle SEND_LOCAL_DATA substate."""
        # Get local graph data
        local_graph_data = await get_local_graph_data()
        print_debug(f"Local graph data: {local_graph_data}", "SEND_LOCAL_DATA")
        
        if not local_graph_data:
            # Local graph is empty
            self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
            self.memory.add("Local graph is empty")
        else:
            # Check if data fits in context
            if await fits_in_context(local_graph_data, self.context_length):
                # Send full graph data
                await send_graph_data_to_agent(local_graph_data)
                self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
                self.memory.add("Local graph sent")
            else:
                # Send only entities and properties
                await send_entities_and_properties(local_graph_data)
                self.substate = self.AutonomousSubstate.LOCAL_GRAPH_EXPLORATION
                self.memory.add("Local graph too large, sending entities/properties only")



    async def _handle_local_graph_exploration(self):
        """Handle LOCAL_GRAPH_EXPLORATION substate."""
            
        self.attempt_history.local_explorations += 1
        
        try:
            # Get local exploration tools
            local_tools = await self.get_local_exploration_tools()
            
            # Let LLM decide on exploration using local exploration tools
            memory_context = self.memory.read(max_characters=2000)
            exploration_prompt = create_local_exploration_prompt(self.current_query.text, memory_context)

            print_prompt(f"System: {exploration_prompt.system}\nUser: {exploration_prompt.user}\nAssistant: {exploration_prompt.assistant}", "local_exploration")
            
            # Use the prompt structure to get properly formatted messages
            messages = exploration_prompt.get_messages()
            response = await self.agent.query_llm(messages, tools=local_tools)
            
            print_response(response.content, "local_exploration")
            

            # Process tool calls if present
            tool_calls_executed = await self._process_tool_calls(response, "local")
            
            self.memory.add(f"Local exploration attempt {self.attempt_history.local_explorations} completed")
            
            # Check if LLM wants to continue exploration or move to evaluation
            if await should_continue_local_exploration({"response": response.content}):
                # Continue exploration
                self.memory.add("Continuing local graph exploration")
            else:
                # Move to evaluation
                self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
                self.memory.add("Local exploration complete, transitioning to EVAL_LOCAL_DATA")
                
        except Exception as e:
            self.attempt_history.failures.append(f"Local exploration failed: {str(e)}")
            self.memory.add(f"Local exploration failed: {str(e)}")
            self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
    


    async def _handle_eval_local_data(self):
        """Handle EVAL_LOCAL_DATA substate."""

        try:
            # Get evaluation tools
            eval_tools = await self.get_evaluation_tools()
            
            # Create evaluation prompt with attempt history
            memory_context = self.memory.read(max_characters=3000)
            eval_prompt = create_local_evaluation_prompt(self.current_query.text, self.attempt_history, memory_context)
            
            print_prompt(f"System: {eval_prompt.system}\nUser: {eval_prompt.user}\nAssistant: {eval_prompt.assistant}", "local_evaluation")

            # Use the prompt structure to get properly formatted messages
            messages = eval_prompt.get_messages()
            response = await self.agent.query_llm(messages, tools=eval_tools)

            print_response(response.content, "local_evaluation")

            # Process tool calls if present
            await self._process_tool_calls(response, "evaluation")
                        
            self.memory.add("Local data evaluation completed")
            
            # Check if LLM can provide final answer
            if await has_sufficient_local_data({"response": response.content}):
                self.substate = self.AutonomousSubstate.PRODUCE_ANSWER
                self.memory.add("Sufficient local data found, transitioning to PRODUCE_ANSWER")
            else:
                # Check if we should attempt remote exploration
                if await should_attempt_remote_exploration({"response": response.content}, self.attempt_history.remote_explorations):
                    self.substate = self.AutonomousSubstate.REMOTE_GRAPH_EXPLORATION
                    self.memory.add("Insufficient local data, transitioning to REMOTE_GRAPH_EXPLORATION")
                else:
                    # Provide partial answer or indicate impossibility
                    self.final_answer = await extract_partial_answer({"response": response.content})
                    self.state = self.OrchestratorState.FINISHED
                    self.memory.add("Cannot explore further, providing partial answer or indicating impossibility")
                    
        except Exception as e:
            self.attempt_history.failures.append(f"Local evaluation failed: {str(e)}")
            self.memory.add(f"Local evaluation failed: {str(e)}")
            self.final_answer = f"Error during evaluation: {str(e)}"
            self.state = self.OrchestratorState.FINISHED
    
    async def _handle_remote_graph_exploration(self):
        """Handle REMOTE_GRAPH_EXPLORATION substate."""

        self.attempt_history.remote_explorations += 1
        
        try:
            # Get remote exploration tools
            remote_tools = await self.get_remote_exploration_tools()
            
            # Create remote exploration prompt
            memory_context = self.memory.read(max_characters=2000)
            remote_prompt = create_remote_exploration_prompt(self.current_query.text, memory_context)
            
            print_prompt(f"System: {remote_prompt.system}\nUser: {remote_prompt.user}\nAssistant: {remote_prompt.assistant}", "remote_exploration")
            
            # Use the prompt structure to get properly formatted messages
            messages = remote_prompt.get_messages()
            response = await self.agent.query_llm(messages, tools=remote_tools)

            print_response(response.content, "remote_exploration")
            
            # Process tool calls if present
            tool_calls_executed = await self._process_tool_calls(response, "remote")
            
            # Print summary of all tool results if any were executed
            if tool_calls_executed > 0:
                print_tool_results_summary(self.tool_call_results, "remote_exploration")
            
            self.memory.add(f"Remote exploration attempt {self.attempt_history.remote_explorations} completed")
            
            # Check if remote exploration was successful
            if await remote_exploration_successful({"response": response.content, "tool_calls_executed": tool_calls_executed}):
                self.current_remote_data = self.tool_call_results
                print_debug(f"Remote data collected: {self.current_remote_data}", "REMOTE_EXPLORATION")
                self.substate = self.AutonomousSubstate.EVAL_REMOTE_DATA
                self.memory.add("Remote exploration successful, transitioning to EVAL_REMOTE_DATA")
            else:
                # Failed, go back to local evaluation
                self.attempt_history.failures.append("Remote exploration failed")
                self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
                self.memory.add("Remote exploration failed, returning to EVAL_LOCAL_DATA")
                
        except Exception as e:
            self.attempt_history.failures.append(f"Remote exploration error: {str(e)}")
            self.memory.add(f"Remote exploration error: {str(e)}")
            self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
    
    async def _handle_eval_remote_data(self):
        """Handle EVAL_REMOTE_DATA substate."""
        try:
            # Get evaluation tools
            eval_tools = await self.get_evaluation_tools()
            
            # Create remote data evaluation prompt
            eval_prompt = create_remote_evaluation_prompt(self.current_query.text, self.current_remote_data or [])
            
            print_prompt(f"System: {eval_prompt.system}\nUser: {eval_prompt.user}\nAssistant: {eval_prompt.assistant}", "remote_evaluation")

            # Use the prompt structure to get properly formatted messages
            messages = eval_prompt.get_messages()
            response = await self.agent.query_llm(messages, tools=eval_tools)

            print_response(response.content, "remote_evaluation")
            
            # Process tool calls if present
            await self._process_tool_calls(response, "evaluation")
            
            self.memory.add("Remote data evaluation completed")
            
            # Check if remote data is relevant
            if await is_remote_data_relevant({"response": response.content}):
                self.substate = self.AutonomousSubstate.LOCAL_GRAPH_UPDATE
                self.memory.add("Remote data is relevant, transitioning to LOCAL_GRAPH_UPDATE")
            else:
                # Remote data is irrelevant, go back to local evaluation
                self.current_remote_data = None
                self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
                self.memory.add("Remote data irrelevant, returning to EVAL_LOCAL_DATA")
                
        except Exception as e:
            self.attempt_history.failures.append(f"Remote evaluation failed: {str(e)}")
            self.memory.add(f"Remote evaluation failed: {str(e)}")
            self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
    


    async def _handle_local_graph_update(self):
        """Handle LOCAL_GRAPH_UPDATE substate."""
        try:
            # Get graph update tools
            update_tools = await self.get_graph_update_tools()
            
            # Create update prompt
            update_prompt = create_graph_update_prompt(self.current_query.text, self.current_remote_data)
            
            print_prompt(f"System: {update_prompt.system}\nUser: {update_prompt.user}\nAssistant: {update_prompt.assistant}", "graph_update")
            
            # Use the prompt structure to get properly formatted messages
            messages = update_prompt.get_messages()
            response = await self.agent.query_llm(messages, tools=update_tools)

            print_response(response.content, "graph_update")
            
            # Process tool calls if present
            tool_calls_executed = await self._process_tool_calls(response, "update")
            
            self.memory.add("Local graph update completed")
            
            # Check if graph update was successful using the new robust logic
            if await graph_update_successful({"response": response.content, "tool_calls_executed": tool_calls_executed}):
                # Update local graph with relevant data
                await update_local_graph({"response": response.content})
                
                # Clear remote data and go back to sending local data
                self.current_remote_data = None
                self.substate = self.AutonomousSubstate.SEND_LOCAL_DATA
                self.memory.add("Local graph updated successfully, returning to SEND_LOCAL_DATA")
            else:
                # Update failed, go back to evaluation
                self.current_remote_data = None
                self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
                self.memory.add("Graph update failed, returning to EVAL_LOCAL_DATA")
            
        except Exception as e:
            self.attempt_history.failures.append(f"Graph update failed: {str(e)}")
            self.memory.add(f"Graph update failed: {str(e)}")
            self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA



    async def _handle_produce_answer(self):
        """Handle PRODUCE_ANSWER substate: produce answer with proof sets."""
        try:
            # Use the same tool set as local graph exploration
            local_tools = await self.get_local_exploration_tools()
            memory_context = self.memory.read(max_characters=3000)
            answer_prompt = create_answer_production_prompt(self.current_query.text, memory_context)

            print_prompt(f"System: {answer_prompt.system}\nUser: {answer_prompt.user}\nAssistant: {answer_prompt.assistant}", "produce_answer")

            messages = answer_prompt.get_messages()
            response = await self.agent.query_llm(messages, tools=local_tools)

            print_response(response.content, "produce_answer")

            # Process tool calls if present
            await self._process_tool_calls(response, "produce_answer")

            # Store the answer and finish
            self.final_answer = response.content
            self.state = self.OrchestratorState.FINISHED
            self.memory.add("Final answer produced and orchestration finished.")
        except Exception as e:
            self.attempt_history.failures.append(f"Answer production failed: {str(e)}")
            self.memory.add(f"Answer production failed: {str(e)}")
            self.final_answer = f"Error during answer production: {str(e)}"
            self.state = self.OrchestratorState.FINISHED



if __name__ == "__main__":
    from ...toolbox.toolbox import Toolbox
    from ...agents.openai import OpenAIAgent
    from .toolboxes import (
        create_local_exploration_toolbox,
        create_remote_exploration_toolbox,
        create_graph_update_toolbox,
        create_evaluation_toolbox
    )
    
    # Create main agent toolbox with all tools
    main_toolbox = Toolbox()
    
    # Register all tools in main toolbox
    from ...toolbox.wikidata.base import GetEntityInfoTool, GetPropertyInfoTool, SearchEntitiesTool
    main_toolbox.register_tool(GetEntityInfoTool())
    main_toolbox.register_tool(GetPropertyInfoTool())
    main_toolbox.register_tool(SearchEntitiesTool())
    
    from ...toolbox.wikidata.queries import SPARQLQueryTool, SubclassQueryTool, SuperclassQueryTool, InstanceQueryTool, InstanceOfQueryTool
    main_toolbox.register_tool(SPARQLQueryTool())
    main_toolbox.register_tool(SubclassQueryTool())
    main_toolbox.register_tool(SuperclassQueryTool())
    main_toolbox.register_tool(InstanceQueryTool())
    main_toolbox.register_tool(InstanceOfQueryTool())
    
    from ...toolbox.wikidata.exploration import NeighborsExplorationTool, LocalGraphTool
    main_toolbox.register_tool(NeighborsExplorationTool())
    main_toolbox.register_tool(LocalGraphTool())
    
    from ...toolbox.graph.graph_tools import AddNodeTool, AddEdgeTool, GetNodeTool, QueryGraphTool, CypherQueryTool
    main_toolbox.register_tool(AddNodeTool())
    main_toolbox.register_tool(AddEdgeTool())
    main_toolbox.register_tool(GetNodeTool())
    main_toolbox.register_tool(QueryGraphTool())
    main_toolbox.register_tool(CypherQueryTool())
    
    # Create specialized toolboxes
    local_exploration_toolbox = create_local_exploration_toolbox()
    remote_exploration_toolbox = create_remote_exploration_toolbox()
    graph_update_toolbox = create_graph_update_toolbox()
    evaluation_toolbox = create_evaluation_toolbox()
    
    agent = OpenAIAgent(main_toolbox, model="gpt-4o")
    orchestrator = MultiStageOrchestrator(
        agent, 
        context_length=8000,
        local_exploration_toolbox=local_exploration_toolbox,
        remote_exploration_toolbox=remote_exploration_toolbox,
        graph_update_toolbox=graph_update_toolbox,
        evaluation_toolbox=evaluation_toolbox
    )
    
    # Example usage
    async def main():
        result = await orchestrator.process_user_query("What is the capital of France?")
        print_debug(f"Answer: {result['answer']}")
        print_debug(f"Attempts: {result['attempts']}")
    
    asyncio.run(main())
