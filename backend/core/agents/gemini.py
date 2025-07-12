from typing import Dict, Any, List, Optional
import google.generativeai as genai
import os
from dotenv import load_dotenv

from .agent import BaseAgent, LLMMessage, LLMResponse, Query
from ..toolbox.toolbox import Toolbox

load_dotenv()

class GeminiAgent(BaseAgent):
    """Gemini-based LLM agent with Wikidata tools."""
    
    def __init__(self, toolbox: Toolbox):
        super().__init__(toolbox)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        
        assert self.model is not None, "Gemini model initialization failed. Check your API key."
        assert self.toolbox is not None, "Toolbox must be provided to GeminiAgent."
    
    async def query_llm(self, messages: List[LLMMessage], tools: Optional[List[Dict]] = None, **kwargs) -> LLMResponse:
        """Send messages to Gemini and get response.
        
        Args:
            messages: List of conversation messages (LLMMessage objects or dicts)
            tools: Optional list of available tools
            **kwargs: Hyperparameters (temperature, max_output_tokens, top_p, top_k, etc.)
        """
        # Convert messages to Gemini format
        if messages and isinstance(messages[0], LLMMessage):
            prompt = self._format_messages_for_gemini(messages)
        else:
            # Already in dict format, convert to string
            prompt = ""
            for msg in messages:
                if msg.get("role") == "user":
                    prompt += f"User: {msg.get('content', '')}\n"
                elif msg.get("role") == "assistant":
                    prompt += f"Assistant: {msg.get('content', '')}\n"
                elif msg.get("role") == "system":
                    prompt += f"System: {msg.get('content', '')}\n"
        
        # Add tool information if provided
        if tools:
            tool_descriptions = "\n".join([
                f"- {tool['function']['name']}: {tool['function']['description']}"
                for tool in tools
            ])
            prompt += f"\n\nAvailable tools:\n{tool_descriptions}"
        
        try:
            # Configure generation parameters
            generation_config = {
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 1.0),
                "top_k": kwargs.get("top_k", 40),
                "max_output_tokens": kwargs.get("max_output_tokens", None),
                "stop_sequences": kwargs.get("stop_sequences", None)
            }
            
            # Remove None values
            generation_config = {k: v for k, v in generation_config.items() if v is not None}
            
            response = self.model.generate_content(prompt, generation_config=generation_config)
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

# Alias for backward compatibility
LLM_Agent = GeminiAgent
