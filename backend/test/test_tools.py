import pytest
import asyncio
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..core.toolbox.toolbox import Toolbox, Tool, ToolDefinition, ToolParameter
from ..core.toolbox.wikidata import (
    get_entity_info, get_property_info, search_entities,
    SPARQLQueryTool, SubclassQueryTool, SuperclassQueryTool, GetInstancesQueryTool
)

class MockTool(Tool):
    """Mock tool for testing."""
    
    def __init__(self):
        super().__init__("mock_tool", "A mock tool for testing")
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="test_param",
                    type="string",
                    description="A test parameter"
                )
            ],
            return_type="string",
            return_description="A test result"
        )
    
    async def execute(self, test_param: str) -> str:
        return f"Mock result for: {test_param}"

class TestToolbox:
    """Test the Toolbox class."""
    
    def test_toolbox_initialization(self):
        """Test toolbox initialization."""
        toolbox = Toolbox()
        assert len(toolbox.list_tools()) == 0
    
    def test_tool_registration(self):
        """Test tool registration."""
        toolbox = Toolbox()
        mock_tool = MockTool()
        
        toolbox.register_tool(mock_tool)
        
        assert "mock_tool" in toolbox.list_tools()
        assert toolbox.get_tool("mock_tool") == mock_tool
        assert toolbox.get_tool_definition("mock_tool") is not None
    
    def test_duplicate_tool_registration(self):
        """Test that registering duplicate tools raises an error."""
        toolbox = Toolbox()
        mock_tool1 = MockTool()
        mock_tool2 = MockTool()
        
        toolbox.register_tool(mock_tool1)
        
        with pytest.raises(ValueError, match="already registered"):
            toolbox.register_tool(mock_tool2)
    
    def test_tool_unregistration(self):
        """Test tool unregistration."""
        toolbox = Toolbox()
        mock_tool = MockTool()
        
        toolbox.register_tool(mock_tool)
        toolbox.unregister_tool("mock_tool")
        
        assert "mock_tool" not in toolbox.list_tools()
        assert toolbox.get_tool("mock_tool") is None
    
    def test_unregister_nonexistent_tool(self):
        """Test that unregistering non-existent tool raises an error."""
        toolbox = Toolbox()
        
        with pytest.raises(ValueError, match="not registered"):
            toolbox.unregister_tool("nonexistent_tool")
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test tool execution."""
        toolbox = Toolbox()
        mock_tool = MockTool()
        
        toolbox.register_tool(mock_tool)
        
        result = await toolbox.execute_tool("mock_tool", test_param="hello")
        assert result == "Mock result for: hello"
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """Test executing non-existent tool raises an error."""
        toolbox = Toolbox()
        
        with pytest.raises(ValueError, match="not found"):
            await toolbox.execute_tool("nonexistent_tool")
    
    def test_openai_format_conversion(self):
        """Test conversion to OpenAI format."""
        mock_tool = MockTool()
        openai_format = mock_tool.to_openai_format()
        
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "mock_tool"
        assert "test_param" in openai_format["function"]["parameters"]["properties"]
    
    def test_tool_search(self):
        """Test tool search functionality."""
        toolbox = Toolbox()
        mock_tool = MockTool()
        
        toolbox.register_tool(mock_tool)
        
        results = toolbox.search_tools("mock")
        assert len(results) == 1
        assert results[0] == mock_tool
        
        results = toolbox.search_tools("testing")
        assert len(results) == 1  # Should find by description
        
        results = toolbox.search_tools("nonexistent")
        assert len(results) == 0

class TestWikidataBaseFunctions:
    """Test basic Wikidata functions with real API calls."""
    
    @pytest.mark.asyncio
    async def test_get_entity_info_douglas_adams(self):
        """Test getting real entity info for Douglas Adams (Q42)."""
        entity = await get_entity_info('Q42')
        
        assert entity.id == 'Q42'
        assert entity.label == 'Douglas Adams'
        assert 'British' in entity.description or 'English' in entity.description
        assert len(entity.aliases) > 0
        assert 'P31' in entity.properties  # instance of
        assert 'Q5' in entity.properties['P31']  # human
    
    @pytest.mark.asyncio
    async def test_get_property_info_instance_of(self):
        """Test getting real property info for P31 (instance of)."""
        property_info = await get_property_info('P31')
        
        assert property_info.id == 'P31'
        assert 'instance of' in property_info.label.lower()
        assert 'wikibase-item' in property_info.datatype
    
    @pytest.mark.asyncio
    async def test_search_entities_douglas_adams(self):
        """Test entity search with real API."""
        entities = await search_entities('Douglas Adams', limit=5)
        
        assert len(entities) >= 1
        assert len(entities) <= 5
        assert any(e.id == 'Q42' and 'Douglas Adams' in e.label for e in entities)
    
    @pytest.mark.asyncio
    async def test_search_entities_empty_result(self):
        """Test entity search with query that returns no results."""
        entities = await search_entities('xyzabcnonexistentquery123', limit=1)
        
        assert isinstance(entities, list)
        assert len(entities) == 0 or len(entities) < 2

class TestWikidataTools:
    """Test Wikidata-specific tools."""
    
    def test_sparql_query_tool_definition(self):
        """Test SPARQL query tool definition."""
        tool = SPARQLQueryTool()
        definition = tool.get_definition()
        
        assert definition.name == "sparql_query"
        assert "SPARQL" in definition.description
        assert len(definition.parameters) >= 1
        assert any(p.name == "query" for p in definition.parameters)
    
    def test_subclass_query_tool_definition(self):
        """Test subclass query tool definition."""
        tool = SubclassQueryTool()
        definition = tool.get_definition()
        
        assert definition.name == "query_subclasses"
        assert "subclass" in definition.description.lower()
        assert any(p.name == "entity_id" for p in definition.parameters)
    
    @pytest.mark.asyncio
    async def test_sparql_tool_execution_real(self):
        """Test SPARQL tool execution with real API."""
        tool = SPARQLQueryTool()
        
        # Simple query to get one human
        query = "SELECT ?item ?itemLabel WHERE { ?item wdt:P31 wd:Q5 } LIMIT 1"
        result = await tool.execute(query)
        
        assert 'results' in result
        assert 'bindings' in result['results']
    
    @pytest.mark.asyncio
    async def test_subclass_query_tool_execution_real(self):
        """Test subclass query tool execution with real API."""
        tool = SubclassQueryTool()
        
        # Query subclasses of person (Q215627)
        result = await tool.execute("Q215627", limit=5)
        
        assert isinstance(result, list)
        assert len(result) <= 5

@pytest.mark.asyncio
async def test_integration_toolbox_with_wikidata_tools():
    """Integration test: toolbox with basic Wikidata tools."""
    toolbox = Toolbox()
    
    # Register basic tools (without exploration tools)
    tools = [
        SPARQLQueryTool(),
        SubclassQueryTool(),
        SuperclassQueryTool(),
        GetInstancesQueryTool()
    ]
    
    for tool in tools:
        toolbox.register_tool(tool)
    
    # Verify all tools are registered
    tool_names = toolbox.list_tools()
    expected_tools = [
        "sparql_query", "query_subclasses", "query_superclasses",
        "query_instances"
    ]
    
    for expected_tool in expected_tools:
        assert expected_tool in tool_names
    
    # Verify OpenAI format conversion works for all tools
    openai_tools = toolbox.get_openai_tools()
    assert len(openai_tools) == len(tools)
    
    for openai_tool in openai_tools:
        assert "type" in openai_tool
        assert openai_tool["type"] == "function"
        assert "function" in openai_tool
    
    # Test actual tool execution through toolbox
    result = await toolbox.execute_tool("sparql_query", 
                                      query="SELECT ?item WHERE { ?item wdt:P31 wd:Q5 } LIMIT 1")
    assert 'results' in result

def test_agent_initialization_with_toolbox():
    """Test that agent can be initialized with toolbox."""
    toolbox = Toolbox()
    
    try:
        from core.agents.gemini import GeminiAgent
        agent = GeminiAgent(toolbox=toolbox)
        assert agent.toolbox == toolbox
        assert len(agent.toolbox.list_tools()) == 0
    except Exception as e:
        # Skip if API key not available in test environment
        if "api_key" in str(e).lower():
            pytest.skip("API key not available in test environment")
        else:
            raise

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])