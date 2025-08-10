import asyncio
from typing import Any, List, Dict, Optional, Tuple
from enum import Enum

from ..orchestrator import Orchestrator
from ...memory import Memory, SimpleMemory
from ...agents.agent import Query
from ...toolbox.toolbox import Toolbox
from .prompts import (
    AttemptHistory, 
    create_local_exploration_prompt,
    create_local_evaluation_prompt,
    create_remote_exploration_prompt,
    create_remote_evaluation_prompt,
    create_graph_update_prompt
)
from .utils import (
    should_continue_local_exploration,
    has_sufficient_local_data,
    should_attempt_remote_exploration,
    remote_exploration_successful,
    is_remote_data_relevant,
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
        
        # Store toolboxes for different phases
        self.local_exploration_toolbox = local_exploration_toolbox
        self.remote_exploration_toolbox = remote_exploration_toolbox
        self.graph_update_toolbox = graph_update_toolbox
        self.evaluation_toolbox = evaluation_toolbox
        
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
            "memory_summary": self.memory.read(max_characters=1000)
        }
    
    async def _handle_ready_state(self):
        """Handle READY state - wait for user input."""
        # In a real implementation, this would wait for user input
        # For now, we'll just sleep
        await asyncio.sleep(1)
    
    async def _handle_autonomous_state(self):
        """Handle AUTONOMOUS state and its substates."""
        # Log state transitions
        self.memory.add(f"Handling autonomous substate: {self.substate.value}")
        
        if self.substate == self.AutonomousSubstate.SEND_LOCAL_DATA:
            await self._handle_send_local_data()
        elif self.substate == self.AutonomousSubstate.LOCAL_GRAPH_EXPLORATION:
            await self._handle_local_graph_exploration()
        elif self.substate == self.AutonomousSubstate.EVAL_LOCAL_DATA:
            await self._handle_eval_local_data()
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
    
    async def _handle_send_local_data(self):
        """Handle SEND_LOCAL_DATA substate."""
        # Get local graph data
        local_graph_data = await get_local_graph_data()
        
        if not local_graph_data:
            # Local graph is empty
            self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
            self.memory.add("Local graph is empty, transitioning to EVAL_LOCAL_DATA")
        else:
            # Check if data fits in context
            if await fits_in_context(local_graph_data, self.context_length):
                # Send full graph data
                await send_graph_data_to_agent(local_graph_data)
                self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA
                self.memory.add("Local graph data sent to agent, transitioning to EVAL_LOCAL_DATA")
            else:
                # Send only entities and properties
                await send_entities_and_properties(local_graph_data)
                self.substate = self.AutonomousSubstate.LOCAL_GRAPH_EXPLORATION
                self.memory.add("Local graph too large, sending entities/properties only, transitioning to LOCAL_GRAPH_EXPLORATION")
    
    async def _handle_local_graph_exploration(self):
        """Handle LOCAL_GRAPH_EXPLORATION substate."""
        self.attempt_history.local_explorations += 1
        
        try:
            # Get local exploration tools
            local_tools = await self.get_local_exploration_tools()
            
            # Let LLM decide on exploration using local exploration tools
            memory_context = self.memory.read(max_characters=2000)
            exploration_prompt = create_local_exploration_prompt(self.current_query.text, memory_context)
            
            # Create messages for direct LLM query
            messages = [
                {"role": "user", "content": exploration_prompt}
            ]
            
            # Query LLM with specific tools for local exploration
            response = await self.agent.query_llm(messages, tools=local_tools)
            
            # Process tool calls if present
            if response.tool_calls:
                # Execute tool calls using the agent's full toolbox
                for tool_call in response.tool_calls:
                    tool_name = tool_call["function"]["name"]
                    try:
                        import json
                        arguments = json.loads(tool_call["function"]["arguments"])
                        result = await self.agent.toolbox.execute_tool(tool_name, **arguments)
                        self.memory.add(f"Tool {tool_name} executed: {result}")
                    except Exception as e:
                        self.memory.add(f"Tool {tool_name} failed: {str(e)}")
                        
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
            
            # Create messages for direct LLM query
            messages = [
                {"role": "user", "content": eval_prompt}
            ]
            
            # Query LLM with evaluation tools
            response = await self.agent.query_llm(messages, tools=eval_tools)
            
            # Process tool calls if present
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["function"]["name"]
                    try:
                        import json
                        arguments = json.loads(tool_call["function"]["arguments"])
                        result = await self.agent.toolbox.execute_tool(tool_name, **arguments)
                        self.memory.add(f"Evaluation tool {tool_name} executed: {result}")
                    except Exception as e:
                        self.memory.add(f"Evaluation tool {tool_name} failed: {str(e)}")
                        
            self.memory.add("Local data evaluation completed")
            
            # Check if LLM can provide final answer
            if await has_sufficient_local_data({"response": response.content}):
                self.final_answer = await extract_final_answer({"response": response.content})
                self.state = self.OrchestratorState.FINISHED
                self.memory.add("Sufficient local data found, providing final answer")
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
            
            # Create messages for direct LLM query
            messages = [
                {"role": "user", "content": remote_prompt}
            ]
            
            # Query LLM with remote exploration tools
            response = await self.agent.query_llm(messages, tools=remote_tools)
            
            # Process tool calls if present
            tool_calls_executed = 0
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["function"]["name"]
                    try:
                        import json
                        arguments = json.loads(tool_call["function"]["arguments"])
                        result = await self.agent.toolbox.execute_tool(tool_name, **arguments)
                        self.memory.add(f"Remote tool {tool_name} executed: {result}")
                        tool_calls_executed += 1
                    except Exception as e:
                        self.memory.add(f"Remote tool {tool_name} failed: {str(e)}")
                        
            self.memory.add(f"Remote exploration attempt {self.attempt_history.remote_explorations} completed")
            
            # Check if remote exploration was successful
            if await remote_exploration_successful({"response": response.content, "tool_calls_executed": tool_calls_executed}):
                self.current_remote_data = await extract_remote_data({"response": response.content})
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
            eval_prompt = create_remote_evaluation_prompt(self.current_query.text, self.current_remote_data)
            
            # Create messages for direct LLM query
            messages = [
                {"role": "user", "content": eval_prompt}
            ]
            
            # Query LLM with evaluation tools
            response = await self.agent.query_llm(messages, tools=eval_tools)
            
            # Process tool calls if present
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["function"]["name"]
                    try:
                        import json
                        arguments = json.loads(tool_call["function"]["arguments"])
                        result = await self.agent.toolbox.execute_tool(tool_name, **arguments)
                        self.memory.add(f"Remote evaluation tool {tool_name} executed: {result}")
                    except Exception as e:
                        self.memory.add(f"Remote evaluation tool {tool_name} failed: {str(e)}")
                        
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
            
            # Create messages for direct LLM query
            messages = [
                {"role": "user", "content": update_prompt}
            ]
            
            # Query LLM with graph update tools
            response = await self.agent.query_llm(messages, tools=update_tools)
            
            # Process tool calls if present
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["function"]["name"]
                    try:
                        import json
                        arguments = json.loads(tool_call["function"]["arguments"])
                        result = await self.agent.toolbox.execute_tool(tool_name, **arguments)
                        self.memory.add(f"Graph update tool {tool_name} executed: {result}")
                    except Exception as e:
                        self.memory.add(f"Graph update tool {tool_name} failed: {str(e)}")
                        
            self.memory.add("Local graph update completed")
            
            # Update local graph with relevant data
            await update_local_graph({"response": response.content})
            
            # Clear remote data and go back to sending local data
            self.current_remote_data = None
            self.substate = self.AutonomousSubstate.SEND_LOCAL_DATA
            self.memory.add("Local graph updated, returning to SEND_LOCAL_DATA")
            
        except Exception as e:
            self.attempt_history.failures.append(f"Graph update failed: {str(e)}")
            self.memory.add(f"Graph update failed: {str(e)}")
            self.substate = self.AutonomousSubstate.EVAL_LOCAL_DATA


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
    
    from ...toolbox.wikidata.queries import SPARQLQueryTool, SubclassQueryTool, SuperclassQueryTool, InstanceOfQueryTool
    main_toolbox.register_tool(SPARQLQueryTool())
    main_toolbox.register_tool(SubclassQueryTool())
    main_toolbox.register_tool(SuperclassQueryTool())
    main_toolbox.register_tool(InstanceOfQueryTool())
    
    from ...toolbox.wikidata.exploration import NeighborsExplorationTool, LocalGraphTool
    main_toolbox.register_tool(NeighborsExplorationTool())
    main_toolbox.register_tool(LocalGraphTool())
    
    from ...toolbox.graph.graph_tools import AddNodeTool, AddEdgeTool, GetNodeTool, CypherQueryTool
    main_toolbox.register_tool(AddNodeTool())
    main_toolbox.register_tool(AddEdgeTool())
    main_toolbox.register_tool(GetNodeTool())
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
        print(f"Answer: {result['answer']}")
        print(f"Attempts: {result['attempts']}")
    
    asyncio.run(main())
