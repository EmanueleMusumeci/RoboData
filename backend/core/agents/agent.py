from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class Query(BaseModel):
    text: str
    entity_id: Optional[str] = None
    type: str = "general"  # general, property, navigation, query

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
    
    def __init__(self, toolbox=None):
        self.conversation_history: List[LLMMessage] = []
        self.toolbox = toolbox
        
    @abstractmethod
    async def query_llm(self, messages: List[LLMMessage], tools: Optional[List[Dict]] = None, 
                        model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Send messages to LLM and get response with optional tool calling.
        
        Args:
            messages: List of conversation messages
            tools: Optional list of available tools
            model: Optional model override for this specific call
            **kwargs: Hyperparameters like temperature, max_tokens, top_p, etc.
        """
        pass
    
    def process_tool_call(self, tool_name: str, parameters: Dict) -> Any:
        """Process a tool call using the toolbox."""
        if self.toolbox is None:
            raise ValueError("No toolbox available for tool execution")
        return self.toolbox.execute_tool(tool_name, **parameters)
    
    def add_message(self, role: str, content: str, tool_calls: Optional[List[Dict]] = None, tool_call_id: Optional[str] = None):
        """Add a message to conversation history."""
        message = LLMMessage(role=role, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id)
        self.conversation_history.append(message)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    async def process_query(self, query: Query) -> Dict:
        """Process a natural language query using tool-calling."""
        # Add user message
        self.add_message("user", query.text)
        
        # Get available tools from the toolbox
        tools = self.toolbox.get_openai_tools() if self.toolbox else None
        
        # Query LLM
        response = await self.query_llm(self.conversation_history, tools)
        
        # Handle tool calls if present
        if response.tool_calls:
            result = await self._execute_tool_calls(response.tool_calls)
        else:
            # Fall back to heuristic-based tool selection
            result = await self._determine_and_execute_tool(query)
        
        # Add assistant response
        self.add_message("assistant", str(result))
        
        return result
    
    async def _execute_tool_calls(self, tool_calls: List[Dict]) -> Dict:
        """Execute tool calls returned by the LLM."""
        results = {}
        for tool_call in tool_calls:
            tool_name = tool_call.get('function', {}).get('name')
            parameters = tool_call.get('function', {}).get('arguments', {})
            if isinstance(parameters, str):
                import json
                parameters = json.loads(parameters)
            
            try:
                result = await self.toolbox.execute_tool(tool_name, **parameters)
                results[tool_name] = result
            except Exception as e:
                results[tool_name] = {"error": str(e)}
        
        return results
    
    async def _determine_and_execute_tool(self, query: Query) -> Dict:
        """Determine which tool to use based on query content (fallback method)."""
        if not self.toolbox:
            return {"message": "No toolbox available"}
            
        text_lower = query.text.lower()
        
        if "subclass" in text_lower and query.entity_id:
            return await self.toolbox.execute_tool("query_subclasses", entity_id=query.entity_id)
        elif "superclass" in text_lower and query.entity_id:
            return await self.toolbox.execute_tool("query_superclasses", entity_id=query.entity_id)
        elif "instance" in text_lower and query.entity_id:
            return await self.toolbox.execute_tool("query_instances", class_id=query.entity_id)
        elif "explore" in text_lower and query.entity_id:
            return await self.toolbox.execute_tool("explore_entity", entity_id=query.entity_id)
        elif "path" in text_lower:
            return {"message": "Path finding requires two entity IDs"}
        elif "graph" in text_lower and query.entity_id:
            return await self.toolbox.execute_tool("build_local_graph", center_entity=query.entity_id)
        else:
            return {"message": "Query not understood. Try asking about subclasses, superclasses, instances, or exploration."}
