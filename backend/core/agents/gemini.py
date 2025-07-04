from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

from .agent import BaseAgent, LLMMessage, LLMResponse
from ..toolbox.toolbox import Toolbox

load_dotenv()

class Query(BaseModel):
    text: str
    entity_id: str = None
    type: str = "general"  # general, property, navigation, query

class GeminiAgent(BaseAgent):
    """Gemini-based LLM agent with Wikidata tools."""
    
    def __init__(self, toolbox: Toolbox):
        super().__init__()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        
        assert self.model is not None, "Gemini model initialization failed. Check your API key."

        # Use provided toolbox or create empty one
        self.toolbox = toolbox

        assert self.toolbox is not None, "Toolbox must be provided to GeminiAgent."
    
    async def query_llm(self, messages: List[LLMMessage], tools: Optional[List[Dict]] = None) -> LLMResponse:
        """Send messages to Gemini and get response."""
        # Convert messages to Gemini format
        prompt = self._format_messages_for_gemini(messages)
        
        # Add tool information if provided
        if tools:
            tool_descriptions = "\n".join([
                f"- {tool['function']['name']}: {tool['function']['description']}"
                for tool in tools
            ])
            prompt += f"\n\nAvailable tools:\n{tool_descriptions}"
        
        try:
            response = self.model.generate_content(prompt)
            return LLMResponse(
                content=response.text,
                tool_calls=None,  # Gemini doesn't have native function calling like OpenAI
                usage=None
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                tool_calls=None,
                usage=None
            )
    
    def _format_messages_for_gemini(self, messages: List[LLMMessage]) -> str:
        """Convert message history to Gemini prompt format."""
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"System: {msg.content}\n"
            elif msg.role == "user":
                formatted += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\n"
        return formatted
    
    def process_tool_call(self, tool_name: str, parameters: Dict) -> Any:
        """Process a tool call using the toolbox."""
        return self.toolbox.execute_tool(tool_name, **parameters)
    
    async def process_query(self, query: Query) -> Dict:
        """Process a natural language query using tool-calling."""
        # Add user message
        self.add_message("user", query.text)
        
        # Get available tools from the toolbox
        tools = self.toolbox.get_openai_tools()
        
        # Query LLM
        response = await self.query_llm(self.conversation_history, tools)
        
        # For now, use simple heuristics to determine tool usage
        # In a full implementation, you'd parse the LLM response for tool calls
        result = await self._determine_and_execute_tool(query)
        
        # Add assistant response
        self.add_message("assistant", str(result))
        
        return result
    
    async def _determine_and_execute_tool(self, query: Query) -> Dict:
        """Determine which tool to use based on query content."""
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
            # This would need entity extraction from query
            return {"message": "Path finding requires two entity IDs"}
        elif "graph" in text_lower and query.entity_id:
            return await self.toolbox.execute_tool("build_local_graph", center_entity=query.entity_id)
        else:
            return {"message": "Query not understood. Try asking about subclasses, superclasses, instances, or exploration."}

# Alias for backward compatibility
LLM_Agent = GeminiAgent
