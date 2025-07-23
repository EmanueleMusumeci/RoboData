"""
Human-in-the-Loop Orchestrator Module

This module provides a HIL (Human-in-the-Loop) orchestrator that wraps another orchestrator
and adds the capability to pause execution for user input, maintain query history across
sessions, and process user feedback as new queries while preserving the knowledge graph.

The HIL orchestrator runs two parallel state machines:
1. The autonomous orchestrator (wrapped orchestrator)
2. The user input monitoring system

Key features:
- Pause/resume autonomous execution based on user input
- Maintain conversation history across multiple queries
- Preserve knowledge graph state between queries
- Decoupled user input handling for flexible integration
"""

import asyncio
import json
from typing import Any, Optional, List, Dict
from abc import ABC, abstractmethod
from datetime import datetime

from .orchestrator import Orchestrator
from ..memory import Memory, SimpleMemory
from ..agents.agent import Query
from ..logging import setup_logger, log_debug


class UserInputHandler(ABC):
    """Abstract interface for handling user input in HIL orchestrator.
    
    This interface allows for different input sources (console, web UI, API, etc.)
    to be used with the HIL orchestrator without coupling the orchestrator
    to a specific input mechanism.
    """
    
    @abstractmethod
    async def wait_for_input(self) -> Optional[str]:
        """Wait for user input and return it.
        
        Returns:
            User input string if available, None if no input received within timeout
        """
        pass
    
    @abstractmethod
    def has_pending_input(self) -> bool:
        """Check if there's pending user input.
        
        Returns:
            True if input is available, False otherwise
        """
        pass


class AsyncQueueInputHandler(UserInputHandler):
    """Async queue-based input handler for testing and basic functionality.
    
    This implementation uses an asyncio.Queue to handle input from external sources.
    It's suitable for testing, API integration, or any scenario where input
    comes from async sources rather than direct console input.
    """
    
    def __init__(self):
        self._input_queue = asyncio.Queue()
        self._has_pending = False
    
    async def send_input(self, user_input: str) -> None:
        """Send input to the handler.
        
        This method is typically called from external sources like web APIs,
        other orchestrators, or test scripts.
        
        Args:
            user_input: The input string from the user
        """
        await self._input_queue.put(user_input)
        self._has_pending = True
    
    async def wait_for_input(self) -> Optional[str]:
        """Wait for user input with a short timeout to avoid blocking."""
        try:
            user_input = await asyncio.wait_for(self._input_queue.get(), timeout=0.1)
            self._has_pending = False
            return user_input
        except asyncio.TimeoutError:
            return None
    
    def has_pending_input(self) -> bool:
        """Check if there's pending user input."""
        return self._has_pending or not self._input_queue.empty()


def format_query_history(queries: List[Dict[str, Any]]) -> str:
    """Format query history for prompt inclusion.
    
    Args:
        queries: List of query dictionaries with 'query', 'response', and 'timestamp'
        
    Returns:
        Formatted string suitable for including in LLM prompts
    """
    if not queries:
        return ""
    
    history_lines = ["Previous queries (in chronological order):"]
    for i, entry in enumerate(queries, 1):
        history_lines.append(f"{i}. Query: {entry['query']}")
        history_lines.append(f"   Response: {entry['response']}")
        history_lines.append("")  # Empty line for readability
    
    return "\n".join(history_lines)


def create_formatted_query_with_history(current_query: str, query_history: List[Dict[str, Any]]) -> str:
    """Create a formatted query string that includes historical context.
    
    Args:
        current_query: The current user query
        query_history: List of previous query-response pairs
        
    Returns:
        Formatted query string with history context
    """
    history = format_query_history(query_history)
    
    if history:
        return f"{history}\nCurrent query: {current_query}"
    else:
        return current_query


class QueryHistory:
    """Manages the history of queries and their responses.
    
    This class maintains a chronological record of all user queries and their
    corresponding responses, which is used to provide context for subsequent
    queries in the same session.
    """
    
    def __init__(self):
        self.queries: List[Dict[str, Any]] = []
    
    def add_query(self, query: str, response: str) -> None:
        """Add a query-response pair to history.
        
        Args:
            query: The user query string
            response: The response from the orchestrator
        """
        self.queries.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_formatted_history(self) -> str:
        """Get formatted history for prompt inclusion."""
        return format_query_history(self.queries)
    
    def clear(self) -> None:
        """Clear all query history."""
        self.queries.clear()
    
    def get_queries_copy(self) -> List[Dict[str, Any]]:
        """Get a copy of the queries list."""
        return self.queries.copy()


async def clear_knowledge_graph_if_available(orchestrator: Orchestrator) -> bool:
    """Clear the knowledge graph if the orchestrator has one.
    
    Args:
        orchestrator: The orchestrator to check for knowledge graph
        
    Returns:
        True if knowledge graph was cleared, False otherwise
    """
    try:
        if hasattr(orchestrator, 'knowledge_graph'):
            kg = getattr(orchestrator, 'knowledge_graph')
            if hasattr(kg, 'clear'):
                await kg.clear()
                log_debug("Knowledge graph cleared")
                return True
    except Exception as e:
        log_debug(f"Could not clear knowledge graph: {str(e)}")
    return False


class HILOrchestrator(Orchestrator):
    """Human-in-the-Loop Orchestrator.
    
    This orchestrator wraps another orchestrator and adds human-in-the-loop capabilities:
    - Pause/resume autonomous execution based on user input
    - Maintain query history across multiple interactions
    - Preserve knowledge graph state between queries
    - Handle user feedback and new questions as contextual queries
    
    The HIL orchestrator runs two parallel state machines:
    1. User input monitoring (always active)
    2. Autonomous processing (started/stopped based on user input)
    """
    
    def __init__(self, 
                 wrapped_orchestrator: Orchestrator,
                 user_input_handler: UserInputHandler,
                 memory: Optional[Memory] = None,
                 experiment_id: Optional[str] = None,
                 clear_knowledge_graph_on_start: bool = True):
        """Initialize HIL Orchestrator.
        
        Args:
            wrapped_orchestrator: The orchestrator to wrap (e.g., MultiStageOrchestrator)
            user_input_handler: Handler for user input
            memory: Memory instance (uses SimpleMemory if None)
            experiment_id: Experiment identifier for logging
            clear_knowledge_graph_on_start: Whether to clear knowledge graph on first launch
        """
        super().__init__(wrapped_orchestrator.agent, memory or SimpleMemory(max_slots=200))
        
        # Set up experiment tracking and logging
        self.experiment_id = experiment_id or f"hil_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = setup_logger(experiment_id=self.experiment_id)
        
        # Store wrapped components
        self.wrapped_orchestrator = wrapped_orchestrator
        self.user_input_handler = user_input_handler
        
        # Query history management
        self.query_history = QueryHistory()
        self.clear_knowledge_graph_on_start = clear_knowledge_graph_on_start
        self._first_launch = True
        
        # State management for parallel execution
        self._autonomous_task = None
        self._waiting_for_user = True
        self._paused = False
        
        # Storage for autonomous results
        self._last_autonomous_result = None
        self._last_original_query = None
        
        log_debug("HIL Orchestrator initialized")
    
    async def start(self) -> None:
        """Start the HIL orchestrator main loop."""
        self._running = True
        self._stop_event.clear()
        
        log_debug("Starting HIL Orchestrator")
        
        # Clear knowledge graph on first launch if requested
        if self._first_launch and self.clear_knowledge_graph_on_start:
            if await clear_knowledge_graph_if_available(self.wrapped_orchestrator):
                self.memory.add("Knowledge graph cleared for new session", "System")
            self._first_launch = False
        
        # Main HIL loop - runs both state machines in parallel
        while self._running and not self._stop_event.is_set():
            try:
                if self._waiting_for_user:
                    await self._handle_user_input_phase()
                elif self._autonomous_task and not self._autonomous_task.done():
                    await self._handle_autonomous_phase()
                else:
                    # Autonomous task finished, process results and return to user input
                    if self._autonomous_task:
                        await self._process_autonomous_completion()
                    self._waiting_for_user = True
                
                await asyncio.sleep(0.1)  # Prevent busy waiting
                
            except Exception as e:
                log_debug(f"Error in HIL orchestrator main loop: {str(e)}")
                self.memory.add(f"Error in HIL orchestration: {str(e)}", "System")
                break
        
        log_debug("HIL Orchestrator stopped")
    
    async def send_user_input(self, user_input: str) -> None:
        """Send user input to the orchestrator (external interface).
        
        Args:
            user_input: The input string from the user
        """
        if isinstance(self.user_input_handler, AsyncQueueInputHandler):
            await self.user_input_handler.send_input(user_input)
        else:
            log_debug("User input handler does not support sending input")
    
    async def pause_autonomous_execution(self) -> None:
        """Pause the current autonomous execution."""
        if self._autonomous_task and not self._autonomous_task.done():
            self._paused = True
            self.wrapped_orchestrator.stop()
            log_debug("Autonomous execution paused")
            self.memory.add("Autonomous execution paused by user", "System")
    
    async def resume_autonomous_execution(self) -> None:
        """Resume autonomous execution."""
        self._paused = False
        log_debug("Autonomous execution resumed")
        self.memory.add("Autonomous execution resumed", "System")
    
    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get the current query history.
        
        Returns:
            Copy of the query history list
        """
        return self.query_history.get_queries_copy()
    
    def clear_query_history(self) -> None:
        """Clear the query history."""
        self.query_history.clear()
        log_debug("Query history cleared")
        self.memory.add("Query history cleared", "System")
    
    def is_autonomous_running(self) -> bool:
        """Check if autonomous processing is currently running."""
        return (self._autonomous_task is not None and 
                not self._autonomous_task.done() and 
                not self._paused)
    
    def is_paused(self) -> bool:
        """Check if the orchestrator is paused."""
        return self._paused
    
    def is_waiting_for_user(self) -> bool:
        """Check if the orchestrator is waiting for user input."""
        return self._waiting_for_user
    
    async def _handle_user_input_phase(self) -> None:
        """Handle the user input phase of the state machine."""
        user_input = await self.user_input_handler.wait_for_input()
        
        if user_input:
            log_debug(f"Received user input: {user_input}")
            self.memory.add(f"User input received: {user_input}", "User")
            
            # Process special commands
            await self._process_user_command(user_input)
    
    async def _process_user_command(self, user_input: str) -> None:
        """Process user commands and determine appropriate action.
        
        Args:
            user_input: The user input string to process
        """
        command = user_input.lower().strip()
        
        if command in ['exit', 'quit', 'stop']:
            self.stop()
        elif command == 'pause' and self._autonomous_task:
            await self.pause_autonomous_execution()
        elif command == 'resume' and self._paused:
            await self.resume_autonomous_execution()
        else:
            # Process as a new query
            await self._start_autonomous_processing(user_input)
    
    async def _handle_autonomous_phase(self) -> None:
        """Handle the autonomous processing phase of the state machine."""
        # Check for user input to potentially pause
        if self.user_input_handler.has_pending_input():
            user_input = await self.user_input_handler.wait_for_input()
            if user_input and user_input.lower().strip() == 'pause':
                await self.pause_autonomous_execution()
        
        # Let the autonomous task continue (it runs in background)
        await asyncio.sleep(0.1)
    
    async def _start_autonomous_processing(self, query: str) -> None:
        """Start autonomous processing of a query.
        
        Args:
            query: The user query to process
        """
        self._waiting_for_user = False
        self._paused = False
        
        # Prepare the query with historical context
        formatted_query = create_formatted_query_with_history(query, self.query_history.queries)
        
        # Create autonomous task
        self._autonomous_task = asyncio.create_task(
            self._run_autonomous_query(formatted_query, original_query=query)
        )
        
        log_debug(f"Started autonomous processing for query: {query}")
    
    async def _run_autonomous_query(self, formatted_query: str, original_query: str) -> Dict[str, Any]:
        """Run the autonomous query processing.
        
        Args:
            formatted_query: The query with historical context
            original_query: The original user query without context
            
        Returns:
            Dictionary containing the processing result
        """
        try:
            # Use the standardized process_user_query method
            result = await self.wrapped_orchestrator.process_user_query(formatted_query)
            
            # Store the result for later processing
            self._last_autonomous_result = result
            self._last_original_query = original_query
            
            return result
            
        except Exception as e:
            log_debug(f"Error in autonomous processing: {str(e)}")
            self.memory.add(f"Error in autonomous processing: {str(e)}", "System")
            self._last_autonomous_result = {"error": str(e)}
            self._last_original_query = original_query
            return {"error": str(e)}
    
    async def _process_autonomous_completion(self) -> None:
        """Process the completion of autonomous execution."""
        try:
            result = self._last_autonomous_result or {}
            original_query = self._last_original_query or "Unknown query"
            
            # Extract answer from result
            answer = result.get('answer', result.get('error', 'No answer provided'))
            
            # Add to query history
            self.query_history.add_query(original_query, answer)
            
            # Log completion
            log_debug(f"Autonomous processing completed for query: {original_query}")
            self.memory.add(f"Query processed successfully: {original_query}", "System")
            self.memory.add(f"Answer: {answer}", "LLM_Agent")
            
            # Clean up
            self._autonomous_task = None
            self._last_autonomous_result = None
            self._last_original_query = None
            
        except Exception as e:
            log_debug(f"Error processing autonomous completion: {str(e)}")
            self.memory.add(f"Error processing completion: {str(e)}", "System")


def create_mock_orchestrator() -> Orchestrator:
    """Create a mock orchestrator for testing and demonstration purposes.
    
    Returns:
        A mock orchestrator that simulates processing with delays
    """
    class MockOrchestrator(Orchestrator):
        def __init__(self):
            super().__init__(agent=None, memory=SimpleMemory(max_slots=10))
            self.current_query = None
        
        async def start(self):
            print("    Mock orchestrator: Starting processing...")
            await asyncio.sleep(2)  # Simulate processing time
            print("    Mock orchestrator: Processing completed")
        
        async def process_user_query(self, query: str):
            print(f"    Mock orchestrator: Processing query: {query[:50]}...")
            await asyncio.sleep(1)  # Simulate processing time
            return {"answer": f"Mock answer for query about: {query.split()[-10:]}"}
    
    return MockOrchestrator()


async def run_hil_demo():
    """Run a demonstration of the HIL orchestrator functionality.
    
    This demo shows:
    - Creating and configuring a HIL orchestrator
    - Processing multiple queries with history preservation
    - Query history management
    - Graceful shutdown
    """
    print("=== HIL Orchestrator Demo ===")
    print("Testing Human-in-the-Loop orchestrator with mock backend")
    print()
    
    # Create components
    input_handler = AsyncQueueInputHandler()
    mock_orchestrator = create_mock_orchestrator()
    
    # Create HIL orchestrator
    hil = HILOrchestrator(
        wrapped_orchestrator=mock_orchestrator,
        user_input_handler=input_handler,
        experiment_id="demo_hil"
    )
    
    # Start HIL orchestrator in background
    print("Starting HIL orchestrator...")
    hil_task = asyncio.create_task(hil.start())
    
    try:
        # Simulate user interactions
        await asyncio.sleep(0.5)
        
        print("Sending first query: 'What is the weather today?'")
        await input_handler.send_input("What is the weather today?")
        await asyncio.sleep(3)  # Let autonomous processing finish
        
        print("Sending second query: 'Tell me about machine learning'")
        await input_handler.send_input("Tell me about machine learning")
        await asyncio.sleep(3)  # Let autonomous processing finish
        
        print("Sending third query with context: 'How does it relate to AI?'")
        await input_handler.send_input("How does it relate to AI?")
        await asyncio.sleep(3)  # Let autonomous processing finish
        
        # Display query history
        history = hil.get_query_history()
        print(f"\nQuery History ({len(history)} entries):")
        for i, entry in enumerate(history, 1):
            print(f"  {i}. Q: {entry['query']}")
            print(f"     A: {entry['response'][:80]}{'...' if len(entry['response']) > 80 else ''}")
            print(f"     Time: {entry['timestamp']}")
        
        # Test pause/resume functionality
        print("\nTesting pause functionality...")
        await input_handler.send_input("This is a test query for pause functionality")
        await asyncio.sleep(0.5)  # Let it start
        
        if hil.is_autonomous_running():
            print("Autonomous processing is running, sending pause command...")
            await input_handler.send_input("pause")
            await asyncio.sleep(1)
            
            if hil.is_paused():
                print("Successfully paused! Resuming...")
                await input_handler.send_input("resume")
                await asyncio.sleep(2)
        
        # Graceful shutdown
        print("\nShutting down gracefully...")
        await input_handler.send_input("exit")
        await asyncio.wait_for(hil_task, timeout=5.0)
        
    except asyncio.TimeoutError:
        print("Timeout waiting for graceful shutdown, forcing stop...")
        hil.stop()
        hil_task.cancel()
    except Exception as e:
        print(f"Error during demo: {e}")
        hil.stop()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    """Demonstrate HIL orchestrator functionality with a mock backend."""
    asyncio.run(run_hil_demo())
