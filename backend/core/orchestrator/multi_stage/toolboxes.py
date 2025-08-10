from typing import List, Dict, Any
from ...toolbox.toolbox import Toolbox

def create_local_exploration_toolbox() -> Toolbox:
    """Create toolbox for local graph exploration."""
    toolbox = Toolbox()
    
    # Graph database tools for local exploration
    from ...toolbox.graph.graph_tools import (
        GetNodeTool, GetEdgeTool, CypherQueryTool, FindNodesTool, FindEdgesTool, 
        GetNeighborsTool, GetSubgraphTool, GetGraphStatsTool
    )
    toolbox.register_tool(GetNodeTool())
    toolbox.register_tool(GetEdgeTool())
    toolbox.register_tool(CypherQueryTool())
    #toolbox.register_tool(FindNodesTool())
    #toolbox.register_tool(FindEdgesTool())
    #toolbox.register_tool(GetNeighborsTool())
    #toolbox.register_tool(GetSubgraphTool())
    #toolbox.register_tool(GetGraphStatsTool())
    
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
    #toolbox.register_tool(SubclassQueryTool())
    #toolbox.register_tool(SuperclassQueryTool())
    #toolbox.register_tool(GetInstancesQueryTool())
    #toolbox.register_tool(InstanceOfQueryTool())
    
    # Wikidata exploration tools
    from ...toolbox.wikidata.exploration import NeighborsExplorationTool, LocalGraphTool
    toolbox.register_tool(NeighborsExplorationTool())
    toolbox.register_tool(LocalGraphTool())
    
    return toolbox

def create_graph_update_toolbox() -> Toolbox:
    """Create toolbox for updating local graph."""
    toolbox = Toolbox()
    
    # Graph database tools for adding/updating data
    from ...toolbox.graph.graph_tools import (
        AddNodeTool, AddEdgeTool, GetNodeTool, GetEdgeTool, RemoveNodeTool, 
        RemoveEdgeTool, FindNodesTool, FindEdgesTool, FetchNodeTool, FetchRelationshipTool,
        FetchReverseRelationshipTool
    )
    toolbox.register_tool(FetchNodeTool())
    toolbox.register_tool(FetchRelationshipTool())
    toolbox.register_tool(FetchReverseRelationshipTool())
    toolbox.register_tool(RemoveNodeTool())
    toolbox.register_tool(RemoveEdgeTool())
    
    return toolbox

def create_evaluation_toolbox() -> Toolbox:
    """Create toolbox for evaluation phases (minimal tools)."""
    toolbox = Toolbox()
    
    # Basic query tools for checking local data
    from ...toolbox.graph.graph_tools import GetNodeTool, GetEdgeTool, CypherQueryTool, GetGraphStatsTool
    toolbox.register_tool(GetNodeTool())
    toolbox.register_tool(GetEdgeTool())
    toolbox.register_tool(CypherQueryTool())
    
    return toolbox
