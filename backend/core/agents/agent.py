from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class LLMMessage(BaseModel):
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

class LLMResponse(BaseModel):
    content: str
    tool_calls: Optional[List[Dict]] = None
    usage: Optional[Dict] = None

class BaseAgent(ABC):
    """Abstract base class for LLM agents with tool calling capabilities."""
    
    def __init__(self):
        self.conversation_history: List[LLMMessage] = []
        
    @abstractmethod
    async def query_llm(self, messages: List[LLMMessage], tools: Optional[List[Dict]] = None) -> LLMResponse:
        """Send messages to LLM and get response with optional tool calling."""
        pass
    
    @abstractmethod
    def process_tool_call(self, tool_name: str, parameters: Dict) -> Any:
        """Process a tool call and return the result."""
        pass
    
    def add_message(self, role: str, content: str, tool_calls: Optional[List[Dict]] = None):
        """Add a message to conversation history."""
        message = LLMMessage(role=role, content=content, tool_calls=tool_calls)
        self.conversation_history.append(message)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
