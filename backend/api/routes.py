from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from app.core.orchestrator import Orchestrator
from app.core.llm_agent import LLM_Agent

router = APIRouter()

# Initialize core components
orchestrator = Orchestrator()
llm_agent = LLM_Agent()

class QueryRequest(BaseModel):
    query: str
    entity_id: Optional[str] = None

class QueryResponse(BaseModel):
    result: str
    explored_nodes: List[str]
    query_path: List[str]
    graph_data: Dict

class EntityRequest(BaseModel):
    entity_id: str

class EntityResponse(BaseModel):
    entity: Dict
    neighbors: List[str]
    graph_data: Dict

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a natural language query about a Wikidata entity."""
    try:
        # Process query through LLM agent
        query_result = llm_agent.process_query(
            Query(text=request.query, entity_id=request.entity_id)
        )
        
        # Update orchestrator state
        if request.entity_id:
            orchestrator.set_current_entity(request.entity_id)
            neighbors = orchestrator.explore_neighbors(request.entity_id)
        
        # Get updated graph data
        graph_data = orchestrator.get_graph_data()
        
        return QueryResponse(
            result=str(query_result),
            explored_nodes=list(orchestrator.explored_nodes),
            query_path=[],  # TODO: Implement path finding
            graph_data=graph_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entity/{entity_id}", response_model=EntityResponse)
async def get_entity_info(entity_id: str):
    """Get information about a specific Wikidata entity."""
    try:
        # Get entity info
        entity = orchestrator.get_entity_info(entity_id)
        
        # Get neighbors
        neighbors = orchestrator.explore_neighbors(entity_id)
        
        # Get graph data
        graph_data = orchestrator.get_graph_data()
        
        return EntityResponse(
            entity=entity.dict(),
            neighbors=neighbors,
            graph_data=graph_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
