import asyncio
from typing import Any, Optional, Dict
from abc import ABC, abstractmethod

from ..memory import Memory, SimpleMemory
from ..metacognition import Metacognition

class Orchestrator(ABC):
    """Abstract base class for orchestrators."""
    
    def __init__(self, agent: Any, memory: Optional[Memory] = None, metacognition: Optional[Metacognition] = None):
        self.agent = agent
        self.memory = memory or SimpleMemory(max_slots=100)
        self.metacognition = metacognition
        self._running = False
        self._stop_event = asyncio.Event()
    
    @abstractmethod
    async def start(self) -> None:
        """Start the orchestrator."""
        pass
    
    def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False
        self._stop_event.set()
    
    def is_running(self) -> bool:
        """Check if orchestrator is running."""
        return self._running
    
    async def process_user_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return the result.
        
        Default implementation for orchestrators that don't have specialized query processing.
        Subclasses should override this method to provide their own query processing logic.
        
        Args:
            query: The user query string
            
        Returns:
            Dictionary containing the processing result
        """
        return {"answer": "Query processing not implemented", "status": "not_implemented"}
