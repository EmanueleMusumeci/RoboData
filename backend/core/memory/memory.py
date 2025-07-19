from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from collections import deque
import re
import json

# Import print_memory_entry for console output
try:
    from ..orchestrator.multi_stage.formatting import print_memory_entry
except ImportError:
    # Fallback if formatting module is not available
    def print_memory_entry(message: str, role: str = "System"):
        print(f"MEMORY [{role}]: {message}")

class MemoryEntry:
    """A single memory entry with role tracking."""
    
    def __init__(self, content: str, role: str = "System"):
        """Initialize memory entry.
        
        Args:
            content: The content of the memory entry
            role: The role of the message sender ("User", "System", "LLM_Agent")
        """
        self.content = content
        self.role = role
    
    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"

class Memory(ABC):
    """Abstract base class for memory storage in orchestrators."""
    
    @abstractmethod
    def add(self, content: str, role: str = "System") -> None:
        """Add content to memory with role tracking."""
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

    @abstractmethod
    def get_last_llm_response(self) -> Optional[str]:
        """Get the last response from the LLM agent."""
        pass

class SimpleMemory(Memory):
    """Simple FIFO memory implementation with max_slots limit and role tracking."""
    
    def __init__(self, max_slots: int = 10):
        """Initialize SimpleMemory with maximum number of slots.
        
        Args:
            max_slots: Maximum number of memory entries to store
        """
        self.max_slots = max_slots
        self._memory = deque(maxlen=max_slots)
    
    def add(self, content: str, role: str = "System") -> None:
        """Add content to memory with role tracking. If at capacity, removes oldest entry."""
        entry = MemoryEntry(content, role)
        self._memory.append(entry)
        print_memory_entry(content, role)
    
    def get_last_llm_response(self) -> Optional[str]:
        """Get the last response from the LLM agent."""
        for entry in reversed(self._memory):
            if entry.role == "LLM_Agent":
                return entry.content
        return None

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
        memory_items = [str(entry) for entry in reversed(self._memory)]
        
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
    
    def get_recent(self, count: int) -> List[MemoryEntry]:
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
    
    def get_oldest(self, count: int) -> List[MemoryEntry]:
        """Get the oldest N entries.
        
        Args:
            count: Number of oldest entries to retrieve
            
        Returns:
            List of oldest entries, oldest first
        """
        if count <= 0:
            return []
        
        return list(self._memory)[:count]

class SummaryMemory(Memory):
    """Memory implementation that summarizes old entries when reaching capacity."""
    
    def __init__(self, max_slots: int = 20, agent: Optional[Any] = None):
        """Initialize SummaryMemory with maximum number of slots.
        
        Args:
            max_slots: Maximum number of memory entries to store before summarization
            agent: LLM agent to use for summarization
        """
        self.max_slots = max_slots
        self.agent = agent
        self._memory = deque()
        self._summary_count = 0  # Track how many summaries have been created
    
    def add(self, content: str, role: str = "System") -> None:
        """Add content to memory. Summarizes oldest entries if at capacity."""
        entry = MemoryEntry(content, role)
        self._memory.append(entry)
        print_memory_entry(content, role)
        
        # Check if we need to summarize
        if len(self._memory) > self.max_slots:
            # For now, just remove oldest entries if no agent is available
            # In practice, this should be called with await from an async context
            if not self.agent:
                entries_to_remove = len(self._memory) // 2
                for _ in range(entries_to_remove):
                    if self._memory:
                        self._memory.popleft()
            else:
                # Schedule summarization - in practice this needs to be handled
                # by calling add_async instead
                print_memory_entry("Warning: Summarization needed but add() is not async. Use add_async() instead.", "System")
    
    async def add_async(self, content: str, role: str = "System") -> None:
        """Add content to memory with async summarization support."""
        entry = MemoryEntry(content, role)
        self._memory.append(entry)
        print_memory_entry(content, role)
        
        # Check if we need to summarize
        if len(self._memory) > self.max_slots:
            await self._summarize_old_entries()
    
    async def _summarize_old_entries(self) -> None:
        """Summarize the oldest half of memory entries using LLM."""
        if not self.agent:
            # If no agent available, just remove oldest entries
            entries_to_remove = len(self._memory) // 2
            for _ in range(entries_to_remove):
                if self._memory:
                    self._memory.popleft()
            return
        
        # Get oldest half of entries for summarization
        entries_to_summarize = len(self._memory) // 2
        old_entries = []
        
        for _ in range(entries_to_summarize):
            if self._memory:
                old_entries.append(self._memory.popleft())
        
        if not old_entries:
            return
        
        # Extract entity names, property names, and tool names from old entries
        entities = set()
        properties = set()
        tools = set()
        
        for entry in old_entries:
            content = entry.content.lower()
            
            # Extract entity patterns (Q followed by digits)
            entity_matches = re.findall(r'\bq\d+\b', content)
            entities.update(entity_matches)
            
            # Extract property patterns (P followed by digits)
            property_matches = re.findall(r'\bp\d+\b', content)
            properties.update(property_matches)
            
            # Extract tool names (look for common patterns)
            tool_patterns = [
                r'tool (\w+) executed',
                r'tool (\w+) failed',
                r'(\w+Tool)',
                r'execute_tool\(["\'](\w+)["\']',
                r'tool[_\s]+(\w+)'
            ]
            
            for pattern in tool_patterns:
                tool_matches = re.findall(pattern, content, re.IGNORECASE)
                tools.update(tool_matches)
        
        # Create summary content
        entries_text = "\n".join([str(entry) for entry in old_entries])
        
        summary_prompt = f"""
        Please summarize the following memory entries while preserving important information:
        - Keep track of key findings and decisions made
        - Preserve any error states or failures
        - Maintain context about the exploration process
        - Retain the following entities if mentioned: {', '.join(entities) if entities else 'none'}
        - Retain the following properties if mentioned: {', '.join(properties) if properties else 'none'}
        - Retain the following tools if mentioned: {', '.join(tools) if tools else 'none'}
        
        Memory entries to summarize:
        {entries_text}
        
        Provide a concise summary that captures the essential information while being much shorter than the original.
        """
        
        try:
            # Use the agent to create summary
            from ..agents.agent import LLMMessage
            messages = [LLMMessage(role="user", content=summary_prompt)]
            response = await self.agent.query_llm(messages)
            
            # Create summary entry
            self._summary_count += 1
            summary_content = f"SUMMARY #{self._summary_count}: {response.content}"
            if entities:
                summary_content += f" [Entities: {', '.join(entities)}]"
            if properties:
                summary_content += f" [Properties: {', '.join(properties)}]"
            if tools:
                summary_content += f" [Tools: {', '.join(tools)}]"
            
            summary_entry = MemoryEntry(summary_content, "System")
            self._memory.appendleft(summary_entry)  # Add summary to the front
            
        except Exception as e:
            # If summarization fails, create a simple summary
            self._summary_count += 1
            simple_summary = f"SUMMARY #{self._summary_count}: Processed {len(old_entries)} memory entries"
            if entities:
                simple_summary += f" [Entities: {', '.join(entities)}]"
            if properties:
                simple_summary += f" [Properties: {', '.join(properties)}]"
            if tools:
                simple_summary += f" [Tools: {', '.join(tools)}]"
            
            summary_entry = MemoryEntry(simple_summary, "System")
            self._memory.appendleft(summary_entry)
    
    def read(self, max_characters: Optional[int] = None) -> str:
        """Read from memory, newest to oldest, up to max_characters."""
        if not self._memory:
            return ""
        
        # Read from newest to oldest (do not reverse)
        memory_items = [str(entry) for entry in self._memory]
        
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
        self._summary_count = 0
    
    def size(self) -> int:
        """Get current memory size."""
        return len(self._memory)
    
    def get_last_llm_response(self) -> Optional[str]:
        """Get the last response from the LLM agent."""
        for entry in reversed(self._memory):
            if entry.role == "LLM_Agent":
                return entry.content
        return None

    def get_recent(self, count: int) -> List[MemoryEntry]:
        """Get the most recent N entries."""
        if count <= 0:
            return []
        
        recent_items = list(self._memory)[-count:]
        return list(reversed(recent_items))
    
    def get_oldest(self, count: int) -> List[MemoryEntry]:
        """Get the oldest N entries."""
        if count <= 0:
            return []
        
        return list(self._memory)[:count]
