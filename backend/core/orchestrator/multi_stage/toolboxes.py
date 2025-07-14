from typing import List, Dict, Any
from ...toolbox.toolbox import Toolbox

def create_local_exploration_toolbox() -> Toolbox:
    """Create toolbox for local graph exploration."""
    toolbox = Toolbox()
    
    # Graph database tools for local exploration
    from ...toolbox.graph.graph_tools import GetNodeTool, CypherQueryTool
    toolbox.register_tool(GetNodeTool())
    toolbox.register_tool(CypherQueryTool())
    
    return toolbox

def create_remote_exploration_toolbox() -> Toolbox:
    """Create toolbox for remote graph exploration."""
    toolbox = Toolbox()
    
    # Wikidata base tools
    from ...toolbox.wikidata.base import GetEntityInfoTool, GetPropertyInfoTool, SearchEntitiesTool
    toolbox.register_tool(GetEntityInfoTool())
    toolbox.register_tool(GetPropertyInfoTool())
    toolbox.register_tool(SearchEntitiesTool())
    
    # Wikidata query tools
    from ...toolbox.wikidata.queries import SPARQLQueryTool, SubclassQueryTool, SuperclassQueryTool, GetInstancesQueryTool, InstanceOfQueryTool
    toolbox.register_tool(SPARQLQueryTool())
    toolbox.register_tool(SubclassQueryTool())
    toolbox.register_tool(SuperclassQueryTool())
    toolbox.register_tool(GetInstancesQueryTool())
    toolbox.register_tool(InstanceOfQueryTool())
    
    # Wikidata exploration tools
    from ...toolbox.wikidata.exploration import NeighborsExplorationTool, LocalGraphTool
    toolbox.register_tool(NeighborsExplorationTool())
    toolbox.register_tool(LocalGraphTool())
    
    return toolbox

def create_graph_update_toolbox() -> Toolbox:
    """Create toolbox for updating local graph."""
    toolbox = Toolbox()
    
    # Graph database tools for adding/updating data
    from ...toolbox.graph.graph_tools import AddNodeTool, AddEdgeTool, GetNodeTool
    toolbox.register_tool(AddNodeTool())
    toolbox.register_tool(AddEdgeTool())
    toolbox.register_tool(GetNodeTool())
    
    return toolbox

def create_evaluation_toolbox() -> Toolbox:
    """Create toolbox for evaluation phases (minimal tools)."""
    toolbox = Toolbox()
    
    # Basic query tools for checking local data
    from ...toolbox.graph.graph_tools import GetNodeTool
    toolbox.register_tool(GetNodeTool())
    
    return toolbox
