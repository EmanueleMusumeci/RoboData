from abc import ABC, abstractmethod
from typing import List, Optional
from collections import deque

# Import print_memory_entry for console output
try:
    from ..orchestrator.multi_stage.formatting import print_memory_entry
except ImportError:
    # Fallback if formatting module is not available
    def print_memory_entry(message: str):
        print(f"MEMORY: {message}")

class Memory(ABC):
    """Abstract base class for memory storage in orchestrators."""
    
    @abstractmethod
    def add(self, content: str) -> None:
        """Add content to memory."""
        pass
    
    @abstractmethod
    def read(self, max_characters: Optional[int] = None) -> str:
        """Read from memory, newest to oldest, up to max_characters."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memory content."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get current memory size."""
        pass

class SimpleMemory(Memory):
    """Simple FIFO memory implementation with max_slots limit."""
    
    def __init__(self, max_slots: int = 100):
        """Initialize SimpleMemory with maximum number of slots.
        
        Args:
            max_slots: Maximum number of memory entries to store
        """
        self.max_slots = max_slots
        self._memory = deque(maxlen=max_slots)
    
    def add(self, content: str) -> None:
        """Add content to memory. If at capacity, removes oldest entry."""
        self._memory.append(content)
        print_memory_entry(content)
    
    def read(self, max_characters: Optional[int] = None) -> str:
        """Read from memory, newest to oldest, up to max_characters.
        
        Args:
            max_characters: Maximum characters to read. If None, read all.
            
        Returns:
            String containing memory content, newest entries first
        """
        if not self._memory:
            return ""
        
        # Read from newest to oldest
        memory_items = list(reversed(self._memory))
        
        if max_characters is None:
            return "\n".join(memory_items)
        
        # Build result respecting max_characters limit
        result = []
        total_chars = 0
        
        for item in memory_items:
            # Check if adding this item would exceed the limit
            item_length = len(item) + 1  # +1 for newline
            if total_chars + item_length > max_characters:
                # If we haven't added anything yet, add truncated version
                if not result:
                    remaining_chars = max_characters - total_chars
                    if remaining_chars > 0:
                        result.append(item[:remaining_chars])
                break
            
            result.append(item)
            total_chars += item_length
        
        return "\n".join(result)
    
    def clear(self) -> None:
        """Clear all memory content."""
        self._memory.clear()
    
    def size(self) -> int:
        """Get current memory size."""
        return len(self._memory)
    
    def get_recent(self, count: int) -> List[str]:
        """Get the most recent N entries.
        
        Args:
            count: Number of recent entries to retrieve
            
        Returns:
            List of recent entries, newest first
        """
        if count <= 0:
            return []
        
        recent_items = list(self._memory)[-count:]
        return list(reversed(recent_items))
    
    def get_oldest(self, count: int) -> List[str]:
        """Get the oldest N entries.
        
        Args:
            count: Number of oldest entries to retrieve
            
        Returns:
            List of oldest entries, oldest first
        """
        if count <= 0:
            return []
        
        return list(self._memory)[:count]
