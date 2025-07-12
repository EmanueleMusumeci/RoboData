from typing import Dict, List, Optional
import json

from ...knowledge_base.graph import get_knowledge_graph

async def should_continue_local_exploration(result: Dict) -> bool:
    """Check if local exploration should continue based on explicit OPTION decision."""
    response = result.get("response", "")
    
    # Check for explicit option selections from local exploration prompt
    if "OPTION_A" in response:  # Continue exploration
        return True
    elif "OPTION_B" in response:  # Ready for evaluation
        return False
    
    # Fallback to old logic if no explicit decision found (for backward compatibility)
    if "TRANSITION_DECISION: READY_FOR_EVALUATION" in response:
        return False
    elif "TRANSITION_DECISION: CONTINUE_EXPLORATION" in response:
        return True
    
    # Final fallback
    return "READY_FOR_EVALUATION" not in response

async def has_sufficient_local_data(result: Dict) -> bool:
    """Check if local data is sufficient for final answer."""
    response = result.get("response", "")
    # Check for explicit option selections from local evaluation prompt
    if "OPTION_A" in response or "OPTION_B" in response:
        return True
    # Check for explicit final answer indicators
    return "sufficient" in response.lower() or "final answer" in response.lower()

async def should_attempt_remote_exploration(result: Dict, remote_explorations: int, max_attempts: int = 3) -> bool:
    """Check if remote exploration should be attempted."""
    response = result.get("response", "")
    # Check for explicit option selection for remote exploration
    if "OPTION_C" in response and remote_explorations < max_attempts:
        return True
    # Fallback to old logic if no explicit decision found
    return ("remote" in response.lower() or "explore" in response.lower()) and \
           remote_explorations < max_attempts

async def remote_exploration_successful(result: Dict) -> bool:
    """Check if remote exploration was successful based on explicit OPTION decision."""
    response = result.get("response", "")
    
    # Check for explicit option selections from remote exploration prompt
    if "OPTION_A" in response:  # Exploration successful with relevant data
        return True
    elif "OPTION_B" in response:  # Exploration failed or no relevant data found
        return False
    
    # Fallback to old logic if no explicit decision found (for backward compatibility)
    if "EXPLORATION_COMPLETE" in response:
        return True
    elif "EXPLORATION_FAILED" in response:
        return False
    
    # Final fallback - check if tools were executed
    return result.get("tool_calls_executed", 0) > 0

async def is_remote_data_relevant(result: Dict) -> bool:
    """Check if remote data is relevant based on explicit OPTION decision."""
    response = result.get("response", "")
    
    # Check for explicit option selections from remote evaluation prompt
    if "OPTION_A" in response:  # Data is relevant and useful
        return True
    elif "OPTION_B" in response:  # Data is not relevant or useful
        return False
    
    # Fallback to old logic if no explicit decision found (for backward compatibility)
    if "DATA_RELEVANT" in response:
        return True
    elif "DATA_IRRELEVANT" in response:
        return False
    
    # Final fallback
    return "RELEVANT" in response

async def graph_update_successful(result: Dict) -> bool:
    """Check if graph update was successful based on explicit OPTION decision."""
    response = result.get("response", "")
    
    # Check for explicit option selections from graph update prompt
    if "OPTION_A" in response:  # Graph update successful
        return True
    elif "OPTION_B" in response:  # Graph update failed
        return False
    
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

async def get_local_graph_data() -> Optional[List[Dict]]:
    """Get local graph data as triples."""
    
    try:
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        # Get the entire graph (nodes and relationships)
        graph_data = await graph.get_whole_graph()
        
        if not graph_data:
            return None
        
        # Convert to triples format
        triples = []
        
        # Process nodes
        nodes = graph_data.get('nodes', [])
        if not nodes:
            return None  # Empty graph
            
        for node in nodes:
            if not node:  # Skip null nodes
                continue
                
            # Add node properties as triples
            node_id = node.get('id')
            if not node_id:
                continue
                
            labels = node.get('labels', [])
            for label in labels:
                if label:  # Skip empty labels
                    triples.append({
                        'subject': node_id,
                        'predicate': 'rdf:type',
                        'object': label
                    })
            
            # Add properties
            for key, value in node.items():
                if key not in ['id', 'labels'] and value is not None:
                    triples.append({
                        'subject': node_id,
                        'predicate': key,
                        'object': value
                    })
        
        # Process relationships
        relationships = graph_data.get('relationships', [])
        for rel in relationships:
            if not rel:  # Skip null relationships
                continue
                
            source_id = rel.get('source_id') or rel.get('start_node_id')
            target_id = rel.get('target_id') or rel.get('end_node_id')
            rel_type = rel.get('type') or rel.get('relationship_type')
            
            if source_id and target_id and rel_type:
                triples.append({
                    'subject': source_id,
                    'predicate': rel_type,
                    'object': target_id
                })
                
                # Add relationship properties as additional triples
                for key, value in rel.items():
                    if key not in ['source_id', 'target_id', 'start_node_id', 'end_node_id', 'type', 'relationship_type', 'id'] and value is not None:
                        triples.append({
                            'subject': f"{source_id}_{rel_type}_{target_id}",
                            'predicate': key,
                            'object': value
                        })
        
        return triples if triples else None
        
    except Exception as e:
        # Log error if possible, but don't raise
        print(f"Error getting local graph data: {e}")
        return None

async def send_graph_data_to_agent(data: List[Dict]):
    """Send full graph data to the agent."""
    # Add graph data to agent's context
    pass

async def send_entities_and_properties(data: List[Dict]):
    """Send only entities and properties to the agent."""
    # Extract and send entities/properties
    pass

async def update_local_graph(result: Dict):
    """Update local graph with new data."""
    # This would use graph update tools
    pass
