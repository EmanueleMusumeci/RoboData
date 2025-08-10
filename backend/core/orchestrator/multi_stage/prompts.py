from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

# Import AttemptHistory from statistics module
from .statistics import AttemptHistory

def format_tool_results_for_prompt(tool_results: List[Dict]) -> str:
    """Format tool results into a readable format for LLM prompts."""
    if not tool_results:
        return "No tool results available."
    
    formatted_parts = []
    for i, result in enumerate(tool_results, 1):
        tool_name = result.get("tool_name", "Unknown")
        context = result.get("context", "")
        
        # Format the header
        header = f"TOOL RESULT {i}: {tool_name}"
        if context:
            header += f" (Context: {context})"
        
        formatted_parts.append(header)
        formatted_parts.append("=" * len(header))
        
        # Add arguments if available
        if "arguments" in result:
            formatted_parts.append(f"Arguments: {result['arguments']}")
        
        # Add result or error
        if "result" in result:
            formatted_parts.append(f"Result:\n{result['result']}")
        elif "error" in result:
            formatted_parts.append(f"Error: {result['error']}")
        
        formatted_parts.append("")  # Empty line between results
    
    return "\n".join(formatted_parts)

@dataclass
class PromptStructure:
    """Structure for organizing prompts by role."""
    system: str
    user: Optional[str] = None
    assistant: Optional[str] = None
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages in the order: SYSTEM, ASSISTANT, USER."""
        messages = []
        
        # 1. System message
        if self.system:
            messages.append({"role": "system", "content": self.system})
        
        # 2. Assistant message (if provided)
        if self.assistant:
            messages.append({"role": "assistant", "content": self.assistant})
        
        # 3. User message
        if self.user:
            messages.append({"role": "user", "content": self.user})
        
        return messages

def create_local_exploration_prompt(query_text: str, memory_context: str, local_graph_data: Optional[str] = None, last_llm_response: Optional[str] = None, last_stage_name: str = "Initial", next_step_tools: Optional[List] = None, local_exploration_results: Optional[List] = None) -> PromptStructure:
    """Create prompt for local graph exploration."""
    
    tools_text = ""
    if next_step_tools:
        tools_text = "\n".join([f"\t\t- TOOL {tool['function']['name']}: {tool['function']['description']}" for tool in next_step_tools])

    # SHORTENED SYSTEM PROMPT
    system_prompt = (
        """
Explore the local knowledge graph to answer the query. Use the current graph data and available tools to find relevant information. Avoid redundancy. After each step, decide:
- "LOCAL_GRAPH_EXPLORATION" (keep exploring)
- "EVAL_LOCAL_DATA" (ready to evaluate)
Base your choice on the data found and the query. Your response must end with the action name (the choice token) on its own line. Give clear instructions for the next stage. You may suggest using multiple tools at once. Explain your reasoning for the next step.
"""
    )

    user_prompt = f'Please explore the local knowledge graph to answer: "{query_text}"'
    
    # Include local graph data in the assistant prompt
    graph_data_text = ""
    if local_graph_data:
        graph_data_text = f"\n\nCURRENT LOCAL GRAPH DATA:\n##############\n{local_graph_data}\n##############\n"
    else:
        graph_data_text = "\n\nCURRENT LOCAL GRAPH DATA:\n##############\nNo local graph data available (empty graph).\n##############\n"
    
    # Include local exploration results if available
    local_exploration_text = ""
    if local_exploration_results:
        formatted_local_results = format_tool_results_for_prompt(local_exploration_results)
        local_exploration_text = f"\n\nRECENT LOCAL EXPLORATION RESULTS:\n##############\n{formatted_local_results}\n##############\n"
    else:
        local_exploration_text = "\n\nRECENT LOCAL EXPLORATION RESULTS:\n##############\nNo recent local exploration results available.\n##############\n"
    
    assistant_prompt = f"""
    PREVIOUS THOUGHT FROM THE {last_stage_name} STAGE:
    <thoughts>
    {last_llm_response or "I'll start exploring the local knowledge graph to find relevant information."}
    </thoughts>

    RECENT MEMORY:
    <memory>
    {memory_context}
    </memory>

    
    {graph_data_text}
    

    <local_exploration_results>
    {local_exploration_text}
    </local_exploration_results>
    """

    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )

def create_local_evaluation_prompt(query_text: str, attempt_history: Optional['AttemptHistory'], memory_context: str, local_graph_data: Optional[str] = None, last_llm_response: Optional[str] = None, next_step_tools: Optional[List] = None, last_stage_name: str = "Unknown", local_exploration_results: Optional[List] = None, strategy: Optional[str] = "") -> PromptStructure:
    """Create prompt for local data evaluation with optional graph data and strategy."""
    
    tools_text = ""
    if next_step_tools:
        tools_text = "\n".join([f"\t\t- TOOL {tool['function']['name']}: {tool['function']['description']}" for tool in next_step_tools])

    # SYSTEM PROMPT with separated instructions and strategy adherence
    system_prompt = (
        """
Evaluate if the local graph data is enough to answer the query. 

NOTICE: DESCRIPTION fields are only a textual description but you still need triples/relations in the local graph to support your answer.
DESCRIPTION ALONE IS NOT ENOUGH TO PRODUCE AN ANSWER.
Example:
If you have
DESCRIPTION: "A has some relationship with B"
but  does not contain the triple
<A> <some_relationship> <B> 
you can not go to PRODUCE_ANSWER. YOU MUST go to REMOTE_GRAPH_EXPLORATION.


Keep considering remote exploration if:
- the graph is not connected
- entities referenced in the query do not explicitly appear in the graph as nodes BUT they exist in the remote ontology
- relationships referenced in the query do not explicitly appear in the graph as nodes BUT they exist in the remote ontology

List key entities/relationships from the query and check if all are present in the graph. 
If you feel that key relationships are missing from the local graph, you can suggest using remote tools to gather more data and go back to remote exploration.

REMOTE EXPLORATION STRATEGIES TO CONSIDER:
- Fetch individual entities and their direct relationships
- Fetch all relationships of a specific property from entities
- Use reverse relationship discovery to find entities that reference key concepts in your query (e.g., all people born in a place, all works by an author)

If yes, end with "PRODUCE_ANSWER". If not, choose:
- "REMOTE_GRAPH_EXPLORATION" (explore remotely)
- "LOCAL_GRAPH_EXPLORATION" (explore locally)
Your response must end with the action name (the choice token) on its own line.

AVOID DISCONNECTED COMPONENTS AT ALL COSTS!!! ALL THE RELEVANT INFORMATION MUST BE CONNECTED IN THE LOCAL GRAPH TO THE RELEVANT ENTITIES/RELATIONSHIPS.
"""
    )
    # Add instructions for next stage, with strategy adherence if provided
    if strategy:
        system_prompt += (
            """
When providing instructions for the next stage, you MUST adhere to the STRATEGY provided in the assistant prompt section below.
Give clear instructions for the next stage. You may suggest using multiple tools at once. Explain your reasoning for the next step.
"""
        )
    else:
        system_prompt += (
            """
Give clear instructions for the next stage. You may suggest using multiple tools at once. Explain your reasoning for the next step.
"""
        )

    user_prompt = f'Evaluate whether the local data is sufficient to answer: "{query_text}"'
    
    # Include local graph data in the assistant prompt if available
    graph_data_text = ""
    if local_graph_data:
        graph_data_text = f"\n\nLOCAL GRAPH DATA:\n##############\n{local_graph_data}\n##############\n"
    else:
        graph_data_text = "\n\nLOCAL GRAPH DATA:\n##############\nNo local graph data available.\n##############\n"

    # Include local exploration results if available
    local_exploration_text = ""
    if local_exploration_results:
        formatted_local_results = format_tool_results_for_prompt(local_exploration_results)
        local_exploration_text = f"\n\nLOCAL EXPLORATION RESULTS:\n##############\n{formatted_local_results}\n##############\n"
    else:
        local_exploration_text = "\n\nLOCAL EXPLORATION RESULTS:\n##############\nNo local exploration results available.\n##############\n"

    assistant_data = f"""
    PREVIOUS THOUGHT FROM THE {last_stage_name} STAGE:
    <thoughts>
    {last_llm_response or "Let's evaluate the available local data."}
    </thoughts>

    RECENT MEMORY:
    <memory>
    {memory_context}
    </memory>

    
    {graph_data_text}
    

    <local_exploration_results>
    {local_exploration_text}
    </local_exploration_results>
"""
    # Add STRATEGY section if strategy is provided
    if strategy:
        assistant_data += f"""
    <STRATEGY: FOLLOW THIS STRATEGY CLOSELY>
    {strategy}
    </STRATEGY: FOLLOW THIS STRATEGY CLOSELY>
"""

    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_data
    )

def create_remote_exploration_prompt(query_text: str, memory_context: str, local_graph_data: Optional[str] = None, last_llm_response: Optional[str] = None, next_step_tools: Optional[List] = None, last_stage_name: str = "Unknown") -> PromptStructure:
    """Create prompt for remote graph exploration."""
    
    tools_text = ""
    if next_step_tools:
        tools_text = "\n".join([f"\t\t- TOOL {tool['function']['name']}: {tool['function']['description']}" for tool in next_step_tools])

    # SHORTENED SYSTEM PROMPT
    system_prompt = (
        """
The local graph is insufficient. Use remote tools to gather relevant data that fills gaps in the local graph. Avoid redundancy. Focus on entities/relationships related to the query.

EXPLORATION STRATEGIES:
- Search for specific entities and their properties
- Find relationships between entities using SPARQL queries  
- Explore bidirectional relationships (both subject->object and object<-subject patterns)
- Use reverse relationship discovery to find all entities that reference a target entity

Your response must end with the action name (the choice token) on its own line. Give clear instructions for the next stage. Explain your reasoning for the next step.
"""
    )

    user_prompt = f'Explore remote knowledge sources to answer: "{query_text}"'
    
    # Include local graph data in the assistant prompt
    graph_data_text = ""
    if local_graph_data:
        graph_data_text = f"\n\nCurrent local graph data:\n{local_graph_data}"
    else:
        graph_data_text = "\n\nCurrent local graph data is empty."
    
    assistant_prompt = f"""
    PREVIOUS THOUGHT FROM THE {last_stage_name} STAGE:
    <thoughts>
    {last_llm_response or "Let's evaluate the available local data."}
    </thoughts>

    RECENT MEMORY:
    <memory>
    {memory_context}
    </memory>

    
    {graph_data_text}
    
"""
    
    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )

def create_remote_evaluation_prompt(query_text: str, memory_context: str, remote_data: list, local_graph_data: Optional[str] = None, last_llm_response: Optional[str] = None, next_step_tools: Optional[List] = None, last_stage_name: str = "Unknown", strategy: Optional[str] = "") -> PromptStructure:
    """Create prompt for remote data evaluation with optional strategy."""
    
    tools_text = ""
    if next_step_tools:
        tools_text = "\n".join([f"\t\t- TOOL {tool['function']['name']}: {tool['function']['description']}" for tool in next_step_tools])

    # SYSTEM PROMPT with separated instructions and strategy adherence
    system_prompt = (
        """
Evaluate if remote data is relevant for building a knowledge graph for the query. Data is relevant if it contains related entities/relationships. Compare with local graph.

NEXT STEPS:
- "LOCAL_GRAPH_UPDATE" (add to graph) - Use available tools like fetch_node, fetch_relationship_from_node, or fetch_relationship_to_node
- "EVAL_LOCAL_DATA" (discard)
- "REMOTE_GRAPH_EXPLORATION" (need more data, correct tool usage) - Consider reverse relationship patterns

Your response must end with the action name (the choice token) on its own line.
"""
    )
    if strategy:
        system_prompt += (
            """
When providing instructions for the next stage, you MUST adhere to the STRATEGY provided in the assistant prompt section below.
Give clear instructions for the next stage. You may suggest using multiple tools at once. Explain your reasoning for the next step. If some tool usage needs to be refined, provide an analysis of the current tool usage and suggest how to improve it.
"""
        )
    else:
        system_prompt += (
            """
Give clear instructions for the next stage. You may suggest using multiple tools at once. Explain your reasoning for the next step.
"""
        )

    user_prompt = f'Evaluate whether the remote data may be useful for answering: "{query_text}"'
    
    # Include local graph data in the assistant prompt
    graph_data_text = ""
    if local_graph_data:
        graph_data_text = f"\n\nCURRENT LOCAL GRAPH DATA:\n##############\n{local_graph_data}\n##############\n"
    else:
        graph_data_text = "\n\nCURRENT LOCAL GRAPH DATA:\n##############\nNo local graph data available (empty graph).\n##############\n"
    if remote_data:
        formatted_remote_data = format_tool_results_for_prompt(remote_data)
        remote_graph_data_text = f"REMOTE DATA:\n##############\n{formatted_remote_data}\n##############\n"
    else:
        remote_graph_data_text = "REMOTE DATA:\n##############\nNo remote data available.\n##############\n"

    assistant_prompt = f"""
    PREVIOUS THOUGHT FROM THE {last_stage_name} STAGE:
    <thoughts>
    {last_llm_response or "Let's evaluate the relevance of this remote data."}
    </thoughts>

    
    {graph_data_text}
    

    
    {remote_graph_data_text}
    

    RECENT MEMORY:
    <memory>
    {memory_context}
    </memory>
"""
    if strategy:
        assistant_prompt += f"""
    <STRATEGY: FOLLOW THIS STRATEGY CLOSELY>
    {strategy}
    </STRATEGY: FOLLOW THIS STRATEGY CLOSELY>
"""
    
    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )

def create_graph_update_prompt(query_text: str, memory_context: str, remote_data: Optional[List], local_graph_data: Optional[str] = None, last_llm_response: Optional[str] = None, next_step_tools: Optional[List] = None, last_stage_name: str = "Unknown") -> PromptStructure:
    """Create prompt for graph update."""
    
    tools_text = ""
    if next_step_tools:
        tools_text = "\n".join([f"\t\t- TOOL {tool['function']['name']}: {tool['function']['description']}" for tool in next_step_tools])

    # SHORTENED SYSTEM PROMPT
    system_prompt = (
        """
Update the local graph with relevant remote data. Only add entities/relationships that are directly related to the query. Avoid isolated nodes and redundancy. Always connect new entities. Mirror remote relationships in the local graph.

AVAILABLE GRAPH UPDATE STRATEGIES:
- Use fetch_node to add individual Wikidata entities to the local graph
- Use fetch_relationship_from_node to get all relationships of a specific property from a subject entity
- Use fetch_relationship_to_node to find all entities that reference a specific object through a particular property (e.g., all people born in a city, all works by an author)

YOU SHOULD ALWAYS TRY TO CONNECT NEW ENTITIES TO THE LOCAL GRAPH, AVOIDING ISOLATED NODES.

Your response must end with the action name (the choice token) on its own line. Give clear instructions for the next stage. Explain your reasoning for the next step.
"""
    )

    user_prompt = f'Update the local knowledge graph with relevant remote data for query: "{query_text}"'
    
    # Include local graph data in the assistant prompt
    graph_data_text = ""
    if local_graph_data:
        graph_data_text = f"\n\nCURRENT LOCAL GRAPH DATA:\n##############\n{local_graph_data}\n##############\n"
    else:
        graph_data_text = "\n\nCURRENT LOCAL GRAPH DATA:\n##############\nNo local graph data available (empty graph).\n##############\n"
    if remote_data:
        formatted_remote_data = format_tool_results_for_prompt(remote_data)
        remote_graph_data_text = f"REMOTE DATA:\n##############\n{formatted_remote_data}\n##############\n"
    else:
        remote_graph_data_text = "REMOTE DATA:\n##############\nNo remote data available.\n##############\n"

    assistant_prompt = f"""
    PREVIOUS THOUGHT FROM THE {last_stage_name} STAGE:
    <thoughts>
    {last_llm_response or "I'll systematically update the local knowledge graph with this remote data."}
    </thoughts>

    
    {graph_data_text}
    

    
    {remote_graph_data_text}
    

    RECENT MEMORY:
    <memory>
    {memory_context}
    </memory>

    """

    
    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )

def create_answer_production_prompt(query_text: str, memory_context: str, local_graph_data: Optional[str] = None, last_llm_response: Optional[str] = None, last_stage_name: str = "Unknown") -> PromptStructure:
    """Create prompt for producing answers with proof sets from local graph data."""
    
    system_prompt = """You are an expert agent that produces comprehensive answers using data from a local knowledge graph. Your task is to create a final answer where each sentence is backed by specific evidence from the local graph.

CRITICAL INSTRUCTIONS FOR ANSWER CONSTRUCTION:

1. EVIDENCE-BACKED SENTENCES: Every sentence in your answer MUST be backed by specific data from the local knowledge graph. You cannot include any sentence without explicit graph evidence.

2. USE LOCAL GRAPH DATA: 
   - The current local graph data shows all entities, relationships, and facts that are relevant to the query
   - Extract information directly from this data to construct your answer
   - Each claim must be traceable to specific nodes, edges, or subgraphs
   - Do not make assumptions beyond what is explicitly present in the local graph

3. PROOF SET STRUCTURE: For each sentence, you must create a "proof set" that contains:
   - A unique proof set ID (format: PS_001, PS_002, etc.)
   - The specific nodes, edges, or subgraphs from the local graph that support the sentence
   - A clear explanation of how the graph data supports the sentence

4. SENTENCE-BY-SENTENCE CONSTRUCTION: Build your answer one sentence at a time, where each sentence is:
   - A factual statement that directly contributes to answering the query
   - Supported by a specific proof set from the local graph
   - Connected logically to form a comprehensive answer

5. PROOF SET TYPES: A proof set can contain:
   - Single nodes (entities with their properties)
   - Single edges/relationships (triples: subject-property-object)
   - Connected subgraphs (multiple nodes and relationships that together support the sentence)

6. NO HALLUCINATION: You cannot invent facts or make claims not explicitly supported by the local graph data. If the graph doesn't contain sufficient data for a complete answer, state what you can prove and what is missing.

7. ANSWER FORMAT: Structure your response as:
   ```
   SENTENCE 1: [Your first sentence]
   SUPPORT SET 001: [ONLY USE TRIPLES FOR EDGES
   (subject_id, relationship_id, object_id)
   AND NODE IDS FOR SINGLE NODES
   (node_id)
   IF SOMETHING CAN BE EXPLAINED WITH AN EDGE, USE AN EDGE, NOT A NODE!]
   
   SENTENCE 2: [Your second sentence] 
   SUPPORT SET 002: [SAME AS ABOVE]

   ... continue for all sentences ...

   FINAL ANSWER: [Complete answer containing all sentences, each backed by a support set]. 
   ```
"""

    user_prompt = f'Produce a comprehensive answer for: "{query_text}" using only data from the local knowledge graph, with each sentence backed by specific proof sets.'
    
    # Include local graph data in the assistant prompt
    graph_data_text = ""
    if local_graph_data:
        graph_data_text = f"\n\nCURRENT LOCAL GRAPH DATA:\n##############\n{local_graph_data}\n##############\n"
    else:
        graph_data_text = "\n\nCURRENT LOCAL GRAPH DATA:\n##############\nNo local graph data available (empty graph).\n##############\n"
    
    assistant_prompt = f"""HI! I AM THE ASSISTANT! I PROVIDE USEFUL DATA!
    PREVIOUS THOUGHT FROM THE {last_stage_name} STAGE:
    <thoughts>
    {last_llm_response or "I'll examine the local knowledge graph and construct an evidence-backed answer with proof sets for each sentence."}
    </thoughts>

    
    {graph_data_text}
    

    RECENT MEMORY:
    <memory>
    {memory_context}
    </memory>
"""
    
    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )

def create_question_decomposition_prompt(query_text: str, memory_context: str, available_tools: Optional[List] = None) -> PromptStructure:
    """Create prompt for decomposing complex questions into sub-questions."""
    
    tools_text = ""
    if available_tools:
        tools_text = "\n".join([f"\t\t- TOOL {tool['function']['name']}: {tool['function']['description']}" for tool in available_tools])

    # SHORTENED SYSTEM PROMPT
    system_prompt = f"""
Analyze the question and, if helpful, break it into up to 5 focused sub-questions that together cover all aspects. Each sub-question should be self-contained, specific, and logically ordered. Use available tools:
{tools_text}
If decomposition is useful, output:
1. ...
2. ...
...etc.
"""
    user_prompt = f'Please analyze this question and determine if it should be decomposed into sub-questions: "{query_text}"'
    
    assistant_prompt = f"""

    RECENT MEMORY:
    <memory>
    {memory_context}
    </memory>

    Let's analyze the complexity and scope of this question to determine if breaking it down into sub-questions would be beneficial for providing a comprehensive answer.
"""
    
    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )
