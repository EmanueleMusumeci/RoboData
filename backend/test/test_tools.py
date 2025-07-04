import pytest
import asyncio
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..core.toolbox.toolbox import Toolbox, Tool, ToolDefinition, ToolParameter
from ..core.toolbox.wikidata import (
    get_entity_info, get_property_info, search_entities,
    SPARQLQueryTool, SubclassQueryTool, SuperclassQueryTool, InstanceQueryTool,
    EntityExplorationTool, PathFindingTool, LocalGraphTool
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
    """Test basic Wikidata functions."""
    
    @patch('core.toolbox.wikidata.base.requests.get')
    def test_get_entity_info_success(self, mock_get):
        """Test successful entity info retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'entities': {
                'Q42': {
                    'labels': {'en': {'value': 'Douglas Adams'}},
                    'descriptions': {'en': {'value': 'British author'}},
                    'aliases': {'en': [{'value': 'Douglas Noel Adams'}]},
                    'claims': {
                        'P31': [{
                            'mainsnak': {
                                'datavalue': {'value': {'id': 'Q5'}}
                            }
                        }]
                    }
                }
            }
        }
        mock_get.return_value = mock_response
        
        entity = get_entity_info('Q42')
        
        assert entity.id == 'Q42'
        assert entity.label == 'Douglas Adams'
        assert entity.description == 'British author'
        assert 'Douglas Noel Adams' in entity.aliases
        assert 'P31' in entity.properties
        assert 'Q5' in entity.properties['P31']
    
    @patch('core.toolbox.wikidata.base.requests.get')
    def test_get_entity_info_not_found(self, mock_get):
        """Test entity not found error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'entities': {}}
        mock_get.return_value = mock_response
        
        with pytest.raises(ValueError, match="not found"):
            get_entity_info('Q999999')
    
    @patch('core.toolbox.wikidata.base.requests.get')
    def test_search_entities(self, mock_get):
        """Test entity search."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'search': [
                {
                    'id': 'Q42',
                    'label': 'Douglas Adams',
                    'description': 'British author'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        entities = search_entities('Douglas Adams')
        
        assert len(entities) == 1
        assert entities[0].id == 'Q42'
        assert entities[0].label == 'Douglas Adams'

class TestWikidataTools:
    """Test Wikidata-specific tools."""
    
    @pytest.mark.asyncio
    async def test_sparql_query_tool_definition(self):
        """Test SPARQL query tool definition."""
        tool = SPARQLQueryTool()
        definition = tool.get_definition()
        
        assert definition.name == "sparql_query"
        assert "SPARQL" in definition.description
        assert len(definition.parameters) >= 1
        assert any(p.name == "query" for p in definition.parameters)
    
    @pytest.mark.asyncio
    async def test_subclass_query_tool_definition(self):
        """Test subclass query tool definition."""
        tool = SubclassQueryTool()
        definition = tool.get_definition()
        
        assert definition.name == "query_subclasses"
        assert "subclass" in definition.description.lower()
        assert any(p.name == "entity_id" for p in definition.parameters)
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_sparql_tool_execution(self, mock_get):
        """Test SPARQL tool execution."""
        # Mock successful SPARQL response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            'results': {'bindings': []}
        }
        
        mock_context = MagicMock()
        mock_context.__aenter__.return_value = mock_response
        mock_get.return_value = mock_context
        
        tool = SPARQLQueryTool()
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value = mock_context
            
            result = await tool.execute("SELECT ?item WHERE { ?item wdt:P31 wd:Q5 } LIMIT 1")
            
            assert 'results' in result
            assert 'bindings' in result['results']

@pytest.mark.asyncio
async def test_integration_toolbox_with_wikidata_tools():
    """Integration test: toolbox with Wikidata tools."""
    toolbox = Toolbox()
    
    # Register tools manually (as would be done in main.py)
    tools = [
        SPARQLQueryTool(),
        SubclassQueryTool(),
        SuperclassQueryTool(),
        InstanceQueryTool(),
        EntityExplorationTool(),
        PathFindingTool(),
        LocalGraphTool()
    ]
    
    for tool in tools:
        toolbox.register_tool(tool)
    
    # Test that agent can be initialized with toolbox
    from core.agents.gemini import GeminiAgent
    
    # This should not fail even without API key in tests
    try:
        agent = GeminiAgent(toolbox=toolbox)
        assert agent.toolbox == toolbox
        assert len(agent.toolbox.list_tools()) == len(tools)
    except Exception as e:
        # Skip if API key not available in test environment
        if "api_key" in str(e).lower():
            pytest.skip("API key not available in test environment")
        else:
            raise
    
    # Verify all tools are registered
    tool_names = toolbox.list_tools()
    expected_tools = [
        "sparql_query", "query_subclasses", "query_superclasses",
        "query_instances", "explore_entity", "find_path", "build_local_graph"
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

def test_agent_initialization_with_empty_toolbox():
    """Test that agent can be initialized with empty toolbox."""
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