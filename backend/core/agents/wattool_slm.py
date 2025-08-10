"""
This agent is designed to work with a local GGUF SLM like eaddario/Watt-Tool-8B-GGUF,
assuming it's served via an OpenAI-compatible API (e.g., using llama-cpp-python's server).
The user is responsible for running the model locally.

Example using llama-cpp-python server:
python3 -m llama_cpp.server --model <path_to_gguf_model> --chat_format functionary
"""
from typing import Dict, Any, List, Optional
import openai
import os
from dotenv import load_dotenv
import json

from .agent import BaseAgent, LLMMessage, LLMResponse, Query
from ..toolbox.toolbox import Toolbox

load_dotenv()

class WatToolSLMAgent(BaseAgent):
    """Agent for a locally-hosted GGUF SLM with tool-calling capabilities."""
    
    def __init__(self, toolbox: Optional[Toolbox] = None, model: str = "eaddario/Watt-Tool-8B-GGUF"):
        super().__init__(toolbox)
        # Assumes the local model is served at an OpenAI-compatible endpoint
        self.client = openai.AsyncOpenAI(
            base_url=os.getenv("WATT_TOOL_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("WATT_TOOL_API_KEY", "no-key-needed")
        )
        self.model = model
    
    async def query_llm(self, messages: List[LLMMessage], tools: Optional[List[Dict]] = None, 
                        model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Send messages to the local SLM and get response with tool calling support."""
        # Use the provided model or fall back to the agent's default model
        selected_model = model or self.model
        
        if messages and isinstance(messages[0], LLMMessage):
            openai_messages = self._format_messages_for_openai(messages)
        else:
            openai_messages = messages
        
        try:
            request_params = {
                "model": selected_model,
                "messages": openai_messages,
                "temperature": kwargs.get("temperature", 0.1),
            }
            
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            response = await self.client.chat.completions.create(**request_params)
            
            message = response.choices[0].message
            content = message.content or ""
            tool_calls = None
            
            if message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                tool_calls=None,
                usage=None
            )
    
    def _format_messages_for_openai(self, messages: List[LLMMessage]) -> List[Dict]:
        """Convert message history to OpenAI format."""
        openai_messages = []
        for msg in messages:
            message_dict: Dict[str, Any] = {
                "role": msg.role,
                "content": msg.content
            }
            
            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls
            
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
            
            openai_messages.append(message_dict)
        
        return openai_messages
    
    async def process_query(self, query: Query) -> Dict:
        """Process a natural language query using the local SLM's tool calling."""
        self.add_message("user", query.text)
        
        tools = self.toolbox.get_openai_tools() if self.toolbox is not None else None
        
        response = await self.query_llm(self.conversation_history, tools)
        
        if response.tool_calls:
            self.add_message("assistant", response.content or "", tool_calls=response.tool_calls)
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["function"]["name"]
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                    result = await self.toolbox.execute_tool(tool_name, **arguments)
                    
                    self.add_message(
                        "tool",
                        json.dumps(result),
                        tool_call_id=tool_call["id"]
                    )
                except Exception as e:
                    self.add_message(
                        "tool", 
                        json.dumps({"error": str(e)}),
                        tool_call_id=tool_call["id"]
                    )
            
            final_response = await self.query_llm(self.conversation_history, tools)
            self.add_message("assistant", final_response.content)
            
            return {"response": final_response.content, "tool_calls_executed": len(response.tool_calls)}
        else:
            self.add_message("assistant", response.content)
            return {"response": response.content}
