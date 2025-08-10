from typing import Dict, Any, List, Optional
import openai
import os
from dotenv import load_dotenv
import json
import pprint

from .agent import BaseAgent, LLMMessage, LLMResponse, Query
from ..toolbox.toolbox import Toolbox

load_dotenv()



class OpenAIAgent(BaseAgent):
    """OpenAI-based LLM agent with native tool calling support."""
    
    def __init__(self, toolbox: Optional[Toolbox] = None, model: str = "gpt-4o"):
        super().__init__(toolbox)
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

        
        assert self.client.api_key is not None, "OpenAI API key not found. Check your environment variables."
    
    async def query_llm(self, messages: List[LLMMessage], tools: Optional[List[Dict]] = None, 
                        model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Send messages to OpenAI and get response with tool calling support.
        
        Args:
            messages: List of conversation messages (LLMMessage objects or dicts)
            tools: Optional list of available tools
            model: Optional model override for this specific call
            **kwargs: Hyperparameters (temperature, max_tokens, top_p, frequency_penalty, presence_penalty, etc.)
        """
        # Use the provided model or fall back to the agent's default model
        selected_model = model or self.model

        if model=="gpt-5":
            temperature = 1.0
        else:
            temperature = None
        
        # Convert messages to OpenAI format if they're LLMMessage objects
        if messages and isinstance(messages[0], LLMMessage):
            openai_messages = self._format_messages_for_openai(messages)
        else:
            # Already in OpenAI format (for direct usage)
            openai_messages = messages
        
        try:
            # Prepare request parameters with defaults
            request_params = {
                "model": selected_model,
                "messages": openai_messages,
                "temperature": temperature if temperature is not None else kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", None),
                "top_p": kwargs.get("top_p", 1.0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0),
                "stop": kwargs.get("stop", None)
            }
            
            # Remove None values
            request_params = {k: v for k, v in request_params.items() if v is not None}
            
            # Add tools if provided
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

            response = await self.client.chat.completions.create(**request_params)
            
            # Extract response content and tool calls
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
                }
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
            message_dict = {
                "role": msg.role,
                "content": msg.content
            }
            
            # Add tool calls if present
            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls
            
            # Add tool call ID for tool responses
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
            
            openai_messages.append(message_dict)
        
        return openai_messages
    
    async def process_query(self, query: Query) -> Dict:
        """Process a natural language query using OpenAI's native tool calling."""
        # Add user message
        self.add_message("user", query.text)
        
        # Get available tools from the toolbox
        tools = self.toolbox.get_openai_tools() if self.toolbox is not None and self.toolbox else None
        
        # Query LLM
        response = await self.query_llm(self.conversation_history, tools)
        
        # Handle tool calls if present
        if response.tool_calls:
            # Add assistant message with tool calls
            self.add_message("assistant", response.content or "", response.tool_calls)
            
            # Execute tool calls and add tool responses
            for tool_call in response.tool_calls:
                tool_name = tool_call["function"]["name"]
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                    result = await self.toolbox.execute_tool(tool_name, **arguments)
                    
                    # Add tool response to conversation
                    self.add_message(
                        "tool",
                        json.dumps(result),
                        tool_call_id=tool_call["id"]
                    )
                except Exception as e:
                    # Add error response
                    self.add_message(
                        "tool", 
                        json.dumps({"error": str(e)}),
                        tool_call_id=tool_call["id"]
                    )
            
            # Get final response after tool execution
            final_response = await self.query_llm(self.conversation_history)
            self.add_message("assistant", final_response.content)
            
            return {"response": final_response.content, "tool_calls_executed": len(response.tool_calls)}
        else:
            # No tool calls, add response and optionally fall back to heuristics
            self.add_message("assistant", response.content)
            if not response.content or "I don't have" in response.content:
                # Fall back to heuristic tool selection
                heuristic_result = await self._determine_and_execute_tool(query)
                return heuristic_result
            return {"response": response.content}
