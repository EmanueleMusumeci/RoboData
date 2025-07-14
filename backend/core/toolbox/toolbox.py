from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import inspect

class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    items: Optional[Dict[str, Any]] = None  # For array types

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]
    return_type: str
    return_description: str

class Tool(ABC):
    """Abstract base class for tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._definition = None
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool's functionality."""
        pass
    
    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Get the tool's definition for registration."""
        pass
    
    @abstractmethod
    def format_result(self, result: Any) -> str:
        """Format the result into a readable, concise string."""
        pass
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert tool definition to OpenAI function calling format."""
        definition = self.get_definition()
        
        properties: Dict[str, Any] = {}
        required = []
        
        for param in definition.parameters:
            param_schema: Dict[str, Any] = {
                "type": param.type,
                "description": param.description
            }
            
            # Handle array types with items schema
            if param.type == "array":
                if param.items:
                    param_schema["items"] = param.items
                else:
                    # Default items schema for arrays without specification
                    param_schema["items"] = {"type": "string"}
            
            properties[param.name] = param_schema
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": definition.name,
                "description": definition.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def __repr__(self):
        return f"Tool(name={self.name}, description={self.description})"

class ToolRegistry:
    """Registry for managing tools with metadata."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.tool_definitions: Dict[str, ToolDefinition] = {}

class Toolbox:
    """Enhanced toolbox with dynamic tool registration and management."""

    def __init__(self):
        self.registry = ToolRegistry()

    def register_tool(self, tool: Tool):
        """Register a tool with validation."""
        if tool.name in self.registry.tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        
        # Get and validate tool definition
        definition = tool.get_definition()
        
        # Store tool and definition
        self.registry.tools[tool.name] = tool
        self.registry.tool_definitions[tool.name] = definition
        
        print(f"Registered tool: {tool.name}")

    def unregister_tool(self, name: str):
        """Unregister a tool by name."""
        if name not in self.registry.tools:
            raise ValueError(f"Tool '{name}' is not registered.")
        
        del self.registry.tools[name]
        del self.registry.tool_definitions[name]
        print(f"Unregistered tool: {name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Retrieve a registered tool by its name."""
        return self.registry.tools.get(name)

    def get_tool_definition(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name."""
        return self.registry.tool_definitions.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.registry.tools.keys())

    def get_all_definitions(self) -> List[ToolDefinition]:
        """Get all tool definitions."""
        return list(self.registry.tool_definitions.values())

    def get_openai_tools(self) -> List[Dict]:
        """Get all tools in OpenAI function calling format."""
        return [tool.to_openai_format() for tool in self.registry.tools.values()]

    async def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with given parameters."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found.")
        
        return await tool.execute(**kwargs)

    def search_tools(self, query: str) -> List[Tool]:
        """Search tools by query string (simple text matching)."""
        results = []
        query_lower = query.lower()
        
        for tool in self.registry.tools.values():
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                results.append(tool)
        
        return results


    def validate_tool_call(self, tool_name: str, parameters: Dict) -> bool:
        """Validate parameters for a tool call."""
        definition = self.get_tool_definition(tool_name)
        if not definition:
            return False
        
        # Check required parameters
        required_params = {p.name for p in definition.parameters if p.required}
        provided_params = set(parameters.keys())
        
        if not required_params.issubset(provided_params):
            missing = required_params - provided_params
            raise ValueError(f"Missing required parameters: {missing}")
        
        return True

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the tool registry."""
        return {
            "total_tools": len(self.registry.tools),
            "tool_names": list(self.registry.tools.keys()),
            "avg_parameters": sum(len(d.parameters) for d in self.registry.tool_definitions.values()) / len(self.registry.tools) if self.registry.tools else 0
        }