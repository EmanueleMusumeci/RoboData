from typing import Dict, List, Optional
import json

from ...knowledge_base.graph import get_knowledge_graph

async def should_continue_local_exploration(result: Dict) -> bool:
    """Check if local exploration should continue based on explicit OPTION decision."""
    response = result.get("response", "")
    
    # Check for explicit option selections from local exploration prompt
    if "LOCAL_GRAPH_EXPLORATION" in response:  # Continue exploration
        return True
    elif "EVAL_LOCAL_DATA" in response:  # Ready for evaluation
        return False
        
    # Final fallback - be conservative and exit exploration if no clear decision
    # This prevents getting stuck in exploration when tools fail or LLM is confused
    return False

async def has_sufficient_local_data(result: Dict) -> bool:
    """Check if local data is sufficient for final answer."""
    response = result.get("response", "")
    # Check for explicit option selections from local evaluation prompt
    if "PRODUCE_ANSWER" in response:
        return True
    else:
        return False

async def should_attempt_remote_exploration(result: Dict, remote_explorations: int, max_attempts: int = 3) -> bool:
    """Check if remote exploration should be attempted."""
    response = result.get("response", "")
    # Check for explicit option selection for remote exploration
    if "REMOTE_GRAPH_EXPLORATION" in response and remote_explorations < max_attempts:
        return True
    # Fallback to old logic if no explicit decision found
    return ("remote" in response.lower() or "explore" in response.lower()) and \
           remote_explorations < max_attempts

async def remote_exploration_successful(result: Dict) -> bool:
    """Check if remote exploration was successful based on explicit OPTION decision."""
    response = result.get("response", "")
    
    # Check for explicit option selections from remote exploration prompt
    if "EVAL_REMOTE_DATA" in response:  # Exploration successful with relevant data
        return True
    elif "EVAL_LOCAL_DATA" in response:  # Exploration failed or no relevant data found
        return False
        
    # Final fallback - check if tools were executed
    return result.get("tool_calls_executed", 0) > 0

async def is_remote_data_relevant(result: Dict) -> bool:
    """Check if remote data is relevant based on explicit OPTION decision."""
    response = result.get("response", "")
    
    # Check for explicit option selections from remote evaluation prompt
    if "LOCAL_GRAPH_UPDATE" in response:  # Data is relevant and useful
        return True
    elif "EVAL_LOCAL_DATA" in response:  # Data is not relevant or useful
        return False
    
    # Final fallback
    return "relevant" in response

async def graph_update_successful(result: Dict) -> bool:
    """Check if graph update was successful based on explicit OPTION decision."""
    response = result.get("response", "")
    
    # Check for explicit option selections from graph update prompt
    if "EVAL_LOCAL_DATA" in response:  # Graph update successful or failed - both transition to eval
        return True
    
    # Fallback to old logic if no explicit decision found (for backward compatibility)
    if "UPDATE_COMPLETE" in response:
        return True
    elif "UPDATE_FAILED" in response:
        return False
    
    # Final fallback - check if tools were executed
    return result.get("tool_calls_executed", 0) > 0

async def extract_final_answer(result: Dict) -> str:
    """Extract final answer from result."""
    return result.get("response", "Unable to provide answer")

async def extract_partial_answer(result: Dict) -> str:
    """Extract partial answer from result."""
    return result.get("response", "Unable to provide complete answer")

async def extract_remote_data(result: Dict) -> Dict:
    """Extract remote data from exploration result."""
    return {"data": result.get("response", "")}

async def fits_in_context(data: List[Dict], context_length: int) -> bool:
    """Check if data fits within context length."""
    # Estimate token count and compare with context_length
    estimated_tokens = len(json.dumps(data)) * 0.75  # Rough estimation
    return estimated_tokens < context_length

