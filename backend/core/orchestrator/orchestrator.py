import asyncio
from typing import Any, Optional
from abc import ABC, abstractmethod

from ..memory import Memory, SimpleMemory

class Orchestrator(ABC):
    """Abstract base class for orchestrators."""
    
    def __init__(self, agent: Any, memory: Optional[Memory] = None):
        self.agent = agent
        self.memory = memory or SimpleMemory(max_slots=100)
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
