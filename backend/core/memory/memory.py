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
    
    def _summarize_long_entry(self, content: str, role: str) -> MemoryEntry:
        """Summarize entries longer than 2000 characters.
        
        Args:
            content: Original content to summarize
            role: Role of the message sender
            
        Returns:
            MemoryEntry with summarized content if needed
        """
        if len(content) <= 2000:
            return MemoryEntry(content, role)
        
        # For SimpleMemory, use a basic summarization approach
        lines = content.split('\n')
        important_lines = []
        
        # Keep first few lines (context)
        important_lines.extend(lines[:5])
        
        # Keep lines with key indicators
        for line in lines[5:]:
            if any(keyword in line.lower() for keyword in 
                   ['error', 'failed', 'success', 'found', 'q', 'p', 'entity', 'property']):
                important_lines.append(line)
                if len(important_lines) >= 15:  # Limit for SimpleMemory
                    break
        
        # Keep last few lines
        if len(lines) > len(important_lines):
            important_lines.extend(lines[-3:])
        
        summarized_content = '\n'.join(important_lines)
        
        # If still too long, truncate
        if len(summarized_content) > 1800:
            summarized_content = summarized_content[:1800] + "..."
        
        final_content = f"[SUMMARIZED - Original: {len(content)} chars] {summarized_content}"
        return MemoryEntry(final_content, role)

    def add(self, content: str, role: str = "System") -> None:
        """Add content to memory with role tracking. Summarizes long entries and removes oldest if at capacity."""
        # Summarize long entries before adding
        entry = self._summarize_long_entry(content, role)
        self._memory.append(entry)
        print_memory_entry(entry.content, role)
    
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
        
        # Build result respecting max_characters limit - never truncate entries
        result = []
        total_chars = 0
        
        for item in memory_items:
            # Check if adding this item would exceed the limit
            item_length = len(item) + 1  # +1 for newline
            if total_chars + item_length > max_characters:
                # Never truncate entries - if it doesn't fit entirely, exclude it
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
    
    def _create_comprehensive_summary(self, old_entries: List[MemoryEntry], summary_count: int) -> str:
        """Create a comprehensive summary of old entries (max 5000 chars).
        
        Args:
            old_entries: List of memory entries to summarize
            summary_count: Current summary count
            
        Returns:
            Comprehensive summary string, guaranteed to be <= 5000 characters
        """
        # Extract key information from removed entries
        entities = set()
        properties = set()
        tools = set()
        key_actions = []
        errors = []
        successes = []
        
        for entry in old_entries:
            content_lower = entry.content.lower()
            
            # Extract entity patterns (Q followed by digits)
            entity_matches = re.findall(r'\bq\d+\b', content_lower)
            entities.update(entity_matches)
            
            # Extract property patterns (P followed by digits)
            property_matches = re.findall(r'\bp\d+\b', content_lower)
            properties.update(property_matches)
            
            # Extract tool names
            if "tool" in content_lower and ("executed" in content_lower or "failed" in content_lower):
                tool_match = re.search(r'tool (\w+)', content_lower)
                if tool_match:
                    tools.add(tool_match.group(1))
            
            # Categorize important events
            if any(keyword in content_lower for keyword in ["error", "failed", "exception"]):
                # Keep important error information (up to 200 chars)
                error_summary = entry.content[:200] if len(entry.content) > 200 else entry.content
                errors.append(f"[{entry.role}] {error_summary}")
            elif any(keyword in content_lower for keyword in ["success", "found", "completed", "finished"]):
                # Keep important success information (up to 150 chars)
                success_summary = entry.content[:150] if len(entry.content) > 150 else entry.content
                successes.append(f"[{entry.role}] {success_summary}")
            elif any(keyword in content_lower for keyword in ["started", "begin", "searching", "querying"]):
                # Keep key action information (up to 100 chars)
                action_summary = entry.content[:100] if len(entry.content) > 100 else entry.content
                key_actions.append(f"[{entry.role}] {action_summary}")
        
        # Build comprehensive summary within 5000 char limit
        summary_parts = []
        summary_parts.append(f"SUMMARY #{summary_count}: Processed {len(old_entries)} memory entries.")
        
        # Add metadata (entities, properties, tools)
        if entities:
            entities_str = f"Entities: {', '.join(sorted(entities))}"
            summary_parts.append(entities_str)
        
        if properties:
            properties_str = f"Properties: {', '.join(sorted(properties))}"
            summary_parts.append(properties_str)
            
        if tools:
            tools_str = f"Tools used: {', '.join(sorted(tools))}"
            summary_parts.append(tools_str)
        
        # Add categorized events with priority: errors > successes > actions
        current_length = len(" | ".join(summary_parts))
        
        # Add errors (highest priority)
        if errors and current_length < 4000:  # Leave room for other content
            errors_section = "ERRORS: " + "; ".join(errors[:5])  # Max 5 errors
            if current_length + len(errors_section) < 4500:
                summary_parts.append(errors_section)
                current_length += len(errors_section)
        
        # Add successes
        if successes and current_length < 4000:
            successes_section = "SUCCESSES: " + "; ".join(successes[:3])  # Max 3 successes
            if current_length + len(successes_section) < 4500:
                summary_parts.append(successes_section)
                current_length += len(successes_section)
        
        # Add key actions
        if key_actions and current_length < 4000:
            actions_section = "KEY ACTIONS: " + "; ".join(key_actions[:3])  # Max 3 actions
            if current_length + len(actions_section) < 4500:
                summary_parts.append(actions_section)
        
        summary_content = " | ".join(summary_parts)
        
        # Ensure we never exceed 5000 characters
        if len(summary_content) > 5000:
            summary_content = summary_content[:4997] + "..."
        
        return summary_content

    def _summarize_long_entry(self, content: str, role: str) -> MemoryEntry:
        """Summarize entries longer than 2000 characters.
        
        Args:
            content: Original content to summarize
            role: Role of the message sender
            
        Returns:
            MemoryEntry with summarized content
        """
        if len(content) <= 2000:
            return MemoryEntry(content, role)
        
        # Extract key information from the long entry
        entities = set(re.findall(r'\bq\d+\b', content.lower()))
        properties = set(re.findall(r'\bp\d+\b', content.lower()))
        
        # Create a structured summary preserving important parts
        lines = content.split('\n')
        important_lines = []
        
        # Keep first few lines (usually contain context)
        important_lines.extend(lines[:3])
        
        # Keep lines with entities, properties, or error indicators
        for line in lines[3:]:
            if any(keyword in line.lower() for keyword in 
                   ['error', 'failed', 'success', 'found', 'entity', 'property', 'q', 'p']):
                important_lines.append(line)
                if len(important_lines) >= 10:  # Limit to prevent too long summaries
                    break
        
        # Keep last few lines (usually contain conclusions)
        if len(lines) > len(important_lines):
            important_lines.extend(lines[-2:])
        
        summarized_content = '\n'.join(important_lines)
        
        # If still too long, truncate to 1500 chars to leave room for metadata
        if len(summarized_content) > 1500:
            summarized_content = summarized_content[:1500] + "..."
        
        # Add metadata
        metadata = []
        if entities:
            metadata.append(f"Entities: {', '.join(sorted(entities))}")
        if properties:
            metadata.append(f"Properties: {', '.join(sorted(properties))}")
        
        final_content = f"[SUMMARIZED - Original: {len(content)} chars] {summarized_content}"
        if metadata:
            final_content += f" | {' | '.join(metadata)}"
        
        return MemoryEntry(final_content, role)

    def add(self, content: str, role: str = "System") -> None:
        """Add content to memory. Summarizes entries longer than 2000 chars and old entries if at capacity."""
        # Summarize long entries before adding
        entry = self._summarize_long_entry(content, role)
        self._memory.append(entry)
        print_memory_entry(entry.content, role)
        
        # Check if we need to summarize old entries
        if len(self._memory) > self.max_slots:
            # Always use simple removal strategy for synchronous operation
            # This ensures consistent behavior whether agent is available or not
            entries_to_remove = len(self._memory) // 2
            old_entries = []
            
            # Collect entries to be removed for summarization
            for _ in range(entries_to_remove):
                if self._memory:
                    old_entries.append(self._memory.popleft())
            
            # Create a comprehensive summary entry (max 5000 chars)
            self._summary_count += 1
            summary_content = self._create_comprehensive_summary(old_entries, self._summary_count)
            summary_entry = MemoryEntry(summary_content, "System")
            self._memory.appendleft(summary_entry)
    
    async def add_async(self, content: str, role: str = "System") -> None:
        """Add content to memory with async LLM-based summarization support.
        
        This method provides more sophisticated summarization using the LLM agent
        when compared to the synchronous add() method which uses simple removal.
        """
        # Summarize long entries before adding
        entry = self._summarize_long_entry(content, role)
        self._memory.append(entry)
        print_memory_entry(entry.content, role)
        
        # Check if we need to summarize
        if len(self._memory) > self.max_slots:
            await self._summarize_old_entries()
    
    async def _summarize_old_entries(self) -> None:
        """Summarize the oldest half of memory entries using LLM."""
        if not self.agent:
            # If no agent available, use the comprehensive summary method
            entries_to_remove = len(self._memory) // 2
            old_entries = []
            
            for _ in range(entries_to_remove):
                if self._memory:
                    old_entries.append(self._memory.popleft())
            
            if old_entries:
                self._summary_count += 1
                summary_content = self._create_comprehensive_summary(old_entries, self._summary_count)
                summary_entry = MemoryEntry(summary_content, "System")
                self._memory.appendleft(summary_entry)
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
        Please summarize the following memory entries while preserving important information.
        The summary MUST be under 4500 characters to allow for metadata.
        Focus on:
        - Key findings and decisions made
        - Any error states or failures
        - Context about the exploration process
        - Entities mentioned: {', '.join(entities) if entities else 'none'}
        - Properties mentioned: {', '.join(properties) if properties else 'none'}
        - Tools mentioned: {', '.join(tools) if tools else 'none'}
        
        Memory entries to summarize:
        {entries_text}
        
        Provide a concise summary under 4500 characters that captures essential information.
        """
        
        # Increment counter once for this summarization
        self._summary_count += 1
        
        try:
            # Use the agent to create summary
            from ..agents.agent import LLMMessage
            messages = [LLMMessage(role="user", content=summary_prompt)]
            response = await self.agent.query_llm(messages)
            
            # Create summary entry using LLM response with metadata
            summary_content = f"SUMMARY #{self._summary_count}: {response.content}"
            
            # Add metadata
            metadata = []
            if entities:
                metadata.append(f"Entities: {', '.join(sorted(entities))}")
            if properties:
                metadata.append(f"Properties: {', '.join(sorted(properties))}")
            if tools:
                metadata.append(f"Tools: {', '.join(sorted(tools))}")
            
            if metadata:
                summary_content += f" | {' | '.join(metadata)}"
            
            # Ensure we never exceed 5000 characters
            if len(summary_content) > 5000:
                summary_content = summary_content[:4997] + "..."
            
            summary_entry = MemoryEntry(summary_content, "System")
            self._memory.appendleft(summary_entry)  # Add summary to the front
            
        except Exception as e:
            # If summarization fails, use the comprehensive summary method
            summary_content = self._create_comprehensive_summary(old_entries, self._summary_count)
            summary_entry = MemoryEntry(summary_content, "System")
            self._memory.appendleft(summary_entry)
    
    def read(self, max_characters: Optional[int] = None) -> str:
        """Read from memory, balancing recent entries with summaries, up to max_characters.
        Never truncates entries - includes them entirely or excludes them.
        """
        if not self._memory:
            return ""
        
        if max_characters is None:
            # Read from newest to oldest (reversed deque)
            memory_items = [str(entry) for entry in reversed(self._memory)]
            return "\n".join(memory_items)
        
        # Strategy: Prioritize summaries (they contain excluded content) + recent entries
        # Never truncate any entry - include entirely or exclude
        
        all_entries = list(self._memory)
        summaries = []
        recent_entries = []
        
        # Separate summaries from regular entries
        for entry in all_entries:
            if entry.content.startswith("SUMMARY #"):
                summaries.append(str(entry))
            else:
                recent_entries.append(str(entry))
        
        # Reverse recent entries to get newest first
        recent_entries = list(reversed(recent_entries))
        
        # Build result ensuring summaries are included entirely (they contain excluded content)
        result = []
        total_chars = 0
        
        # First, include all summaries (they're guaranteed to be <= 5000 chars each)
        # This ensures excluded content is always represented
        for summary in summaries:
            summary_length = len(summary) + 1  # +1 for newline
            if total_chars + summary_length <= max_characters:
                result.append(summary)
                total_chars += summary_length
            # If summary doesn't fit, we still continue to try recent entries
        
        # Then add recent entries (newest first) without truncation
        for item in recent_entries:
            item_length = len(item) + 1  # +1 for newline
            if total_chars + item_length <= max_characters:
                result.append(item)
                total_chars += item_length
            else:
                # Never truncate - if it doesn't fit entirely, exclude it
                break
        
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
