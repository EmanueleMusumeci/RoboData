from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class AttemptHistory:
    """Track previous attempts and their outcomes."""
    remote_explorations: int = 0
    local_explorations: int = 0
    failures: List[str] = None
    
    def __post_init__(self):
        if self.failures is None:
            self.failures = []

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

def create_local_exploration_prompt(query_text: str, memory_context: str, local_graph_data: Optional[List] = None) -> PromptStructure:
    """Create prompt for local graph exploration."""
    
    system_prompt = """You are an expert agent exploring a local knowledge graph to find information to answer a specific query.

Your task is to systematically explore the available knowledge graph to find relevant information. 

USING LOCAL GRAPH DATA:
- You have access to current local graph data that shows existing nodes and relationships
- Use this data to understand what's already in the graph and identify gaps
- Build upon existing data when exploring new connections
- Avoid redundant exploration of already-known information
- If the graph is too big, only the entities and relationships relevant to the query will be shown

NOTICE: The graph might be empty or incomplete. Try to determine this in the fewest steps possible.

DECISION CRITERIA:
After each exploration step, you must decide whether to:
A) Continue exploring the local graph because you found promising leads -> ANSWER "LOCAL_GRAPH_EXPLORATION"  
B) Stop exploring and evaluate what you've found -> ANSWER "EVAL_LOCAL_DATA"

Make your decision based on:
- Whether you found relevant entities/relationships that warrant further exploration
- Whether the graph appears to be empty or exhausted
- Whether you have gathered sufficient local information to attempt an evaluation
- How the current local graph data relates to your query

You MUST end your response with exactly one of:
- "LOCAL_GRAPH_EXPLORATION" - Continue local exploration
- "EVAL_LOCAL_DATA" - Ready for evaluation"""

    user_prompt = f'Please explore the local knowledge graph to answer: "{query_text}"'
    
    # Include local graph data in the assistant prompt
    graph_data_text = ""
    if local_graph_data:
        graph_data_text = f"\n\nCURRENT LOCAL GRAPH DATA:\n##############\n{local_graph_data}\n##############\n"
    else:
        graph_data_text = "\n\nCURRENT LOCAL GRAPH DATA:\n##############\nNo local graph data available (empty graph).\n##############\n"
    
    assistant_prompt = f"""Recent memory context:
{memory_context}{graph_data_text}

I'll start exploring the local knowledge graph to find relevant information.""" if memory_context else f"""{graph_data_text}"""

    
    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )

def create_local_evaluation_prompt(query_text: str, attempt_history: 'AttemptHistory', memory_context: str, local_graph_data: Optional[List] = None) -> PromptStructure:
    """Create prompt for local data evaluation with optional graph data."""
    
    system_prompt = """You are an expert agent evaluating the available data in a local knowledge graph to find information to answer a specific query. 
    You need to evaluate whether the local data is sufficient to answer the user's query.

USING LOCAL GRAPH DATA:
- Analyze the current local graph data to understand available entities, relationships, and facts
- Determine if this data directly answers the query or provides sufficient context
- Consider the completeness and relevance of the local data for the specific question
- Base your decision on concrete evidence present in the local graph
- If the graph is too big, only the entities and relationships relevant to the query will be shown

If you have sufficient data -> ANSWER \"PRODUCE_ANSWER\"

If not, decide whether to:
A) Give a partial answer with available data -> ANSWER \"PRODUCE_ANSWER\"
B) Attempt remote graph exploration (if previous attempts allow) -> ANSWER \"REMOTE_GRAPH_EXPLORATION\" """

    user_prompt = f'Evaluate whether the local data is sufficient to answer: "{query_text}"'
    
    # Include local graph data in the assistant prompt if available
    graph_data_text = ""
    if local_graph_data:
        graph_data_text = f"\n\nLOCAL GRAPH DATA:\n##############\n{local_graph_data}\n##############\n"
    else:
        graph_data_text = "\n\nLOCAL GRAPH DATA:\n##############\nNo local graph data available.\n##############\n"

    assistant_data = f"""Previous attempts:
- Remote explorations: {attempt_history.remote_explorations}
- Local explorations: {attempt_history.local_explorations}
- Failures: {attempt_history.failures}

Recent memory context:
{memory_context}{graph_data_text}

Let me evaluate the available local data."""
    
    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_data
    )

def create_remote_exploration_prompt(query_text: str, memory_context: str, local_graph_data: Optional[List] = None) -> PromptStructure:
    """Create prompt for remote graph exploration."""
    
    system_prompt = """You are exploring remote knowledge sources to gather relevant data from external sources.

The local graph does not contain sufficient information. Use the available remote exploration tools to gather relevant data.

USING LOCAL GRAPH DATA:
- Review the current local graph data to understand what information is already available
- Identify specific gaps and missing entities/relationships that need to be filled
- Target your remote exploration to complement existing local data
- Avoid gathering redundant information that's already in the local graph
- If the graph is too big, only the entities and relationships relevant to the query will be shown

EXPLORATION STRATEGY:
1. SYSTEMATICALLY EXPLORE: Use the remote exploration tools to systematically gather data related to the query.
2. FOCUS ON RELEVANCE: Prioritize data that contains entities, relationships, or facts related to the query topic.
3. COMPLEMENT LOCAL DATA: Focus on filling gaps identified in the local graph data. Start from existing entities and relationships to find new connections and neighboring data.
4. USE TOOLS EFFECTIVELY: Execute the remote exploration tools to gather data, ensuring you capture all relevant information.
5. AVOID HALLUCINATION: Only gather data that is explicitly available through the remote tools.

DECISION CRITERIA:
After your exploration attempts, you must decide whether:
A) You successfully gathered relevant entity and relationship data -> ANSWER "EVAL_REMOTE_DATA"
B) The exploration failed or you couldn't find relevant data -> ANSWER "EVAL_LOCAL_DATA"
C) You need to repeat remote exploration with a different or more comprehensive tool or query -> ANSWER "REMOTE_GRAPH_EXPLORATION"

Make your decision based on:
- Whether you successfully executed tools and received meaningful data
- Whether the data you found is related to the query topic
- Whether you encountered errors that prevent further progress
- How the new data complements the existing local graph data

You MUST end your response with exactly one of:
- "EVAL_REMOTE_DATA" - Exploration successful with relevant data to be added to the local knowledge graph
- "REMOTE_GRAPH_EXPLORATION" - Results are relevant to the query but not complete (repeat remote exploration with a different tool or query)
- "EVAL_LOCAL_DATA" - Exploration failed or no relevant data found (go back to evaluating local data to determine next steps)
"""

    user_prompt = f'Explore remote knowledge sources to answer: "{query_text}"'
    
    # Include local graph data in the assistant prompt
    graph_data_text = ""
    if local_graph_data:
        graph_data_text = f"\n\nCURRENT LOCAL GRAPH DATA:\n##############\n{local_graph_data}\n##############\n"
    else:
        graph_data_text = "\n\nCURRENT LOCAL GRAPH DATA:\n##############\nNo local graph data available (empty graph).\n##############\n"
    
    assistant_prompt = f"""Recent memory context:
{memory_context}{graph_data_text}

I'll start exploring remote knowledge sources to gather relevant data.""" if memory_context else f"""{graph_data_text}

I'll start exploring remote knowledge sources to gather relevant data."""
    
    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )
    
    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )

def create_remote_evaluation_prompt(query_text: str, remote_data: list, local_graph_data: Optional[List] = None) -> PromptStructure:
    """Create prompt for remote data evaluation."""
    
    system_prompt = """You are an expert agent evaluating remote data to decide if it's relevant for a user's query.
Your main goal is to incrementally build a knowledge graph. You must determine if the gathered remote data, even if incomplete, is related to the query and can contribute to the graph.

EVALUATION CRITERIA:
- The data does NOT need to directly answer the query.
- The data IS RELEVANT if it contains entities, relationships, or facts related to the query topic.
- Your task is to collect pieces of information that, step-by-step, will build a graph to answer the query.
- If the data is on-topic, it should be added to the graph.

USING LOCAL GRAPH DATA:
- Compare remote data with the existing local graph to see what's new.
- The remote data should complement or expand the local graph.
- If the graph is too big, only the entities and relationships relevant to the query will be shown.

DECISION CRITERIA:
Based on the relevance to the query topic, you must decide:
A) The data is relevant and on-topic -> ANSWER "LOCAL_GRAPH_UPDATE"
B) The data is completely irrelevant and off-topic -> ANSWER "EVAL_LOCAL_DATA"

You MUST end your response with exactly one of:
- "LOCAL_GRAPH_UPDATE" - Data is relevant and should be added to the graph.
- "EVAL_LOCAL_DATA" - Data is not relevant and should be discarded."""

    user_prompt = f'Evaluate whether the remote data may be useful for answering: "{query_text}"'
    
    # Include local graph data in the assistant prompt
    graph_data_text = ""
    if local_graph_data:
        graph_data_text = f"\n\nCURRENT LOCAL GRAPH DATA:\n##############\n{local_graph_data}\n##############\n"
    else:
        graph_data_text = "\n\nCURRENT LOCAL GRAPH DATA:\n##############\nNo local graph data available (empty graph).\n##############\n"
    
    assistant_prompt = f"""Remote data (each item includes tool_name, arguments, and result or error):
{remote_data}{graph_data_text}

Let me evaluate the relevance of this remote data."""
    
    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )

def create_graph_update_prompt(query_text: str, remote_data: Optional[List], local_graph_data: Optional[List] = None) -> PromptStructure:
    """Create prompt for graph update."""
    
    system_prompt = """Update the local knowledge graph with relevant remote data.

CRITICAL INSTRUCTIONS FOR GRAPH CONSTRUCTION:

1. BUILD A CONNECTED GRAPH: Every new entity must be connected to at least one other entity through relationships.
   Do NOT add isolated nodes unless absolutely necessary.

2. RELATIONSHIP-FIRST APPROACH: When you find related entities in the remote data:
   - First add the entities using add_node
   - Immediately add relationships using add_edge to connect them
   - Look for explicit relationships in the remote data

3. PRESERVE DATA STRUCTURE: If the remote data shows relationships between existing entities in the local graph translate them directly into graph relationships. 
The local graph should mirror the remote structure as closely as possible. Each local relationship MUST correspond to a remote relationship.

4. INTEGRATE WITH EXISTING DATA: 
   - Review the current local graph data to understand existing structure
   - Connect new entities to existing entities when relationships exist
   - Avoid duplicating existing nodes and relationships
   - Maintain consistency with the existing graph schema

5. AVOID HALLUCINATION: Only create relationships that are explicitly present in the remote data.
   Do not invent relationships that aren't supported by the data.

6. WORK SYSTEMATICALLY:
   a) First, identify all entities mentioned in the remote data
   b) Check if any already exist in the local graph
   c) Add new entities as nodes with their properties
   d) Then systematically add all relationships found in the data

DECISION CRITERIA:
After attempting to update the graph, you must decide:
A) You successfully updated the graph with the remote data -> ANSWER "EVAL_LOCAL_DATA"
B) The update failed due to errors or data issues -> ANSWER "EVAL_LOCAL_DATA"

Make your decision based on:
- Whether you successfully executed graph update tools
- Whether the data was properly structured for graph integration
- Whether you encountered errors during the update process

You MUST end your response with exactly one of:
- "EVAL_LOCAL_DATA" - Graph update successful
- "EVAL_LOCAL_DATA" - Graph update failed

Use the graph tools to build a coherent, connected knowledge subgraph that directly supports answering the query."""

    user_prompt = f'Update the local knowledge graph with relevant remote data for query: "{query_text}"'
    
    # Include local graph data in the assistant prompt
    graph_data_text = ""
    if local_graph_data:
        graph_data_text = f"\n\nCURRENT LOCAL GRAPH DATA:\n##############\n{local_graph_data}\n##############\n"
    else:
        graph_data_text = "\n\nCURRENT LOCAL GRAPH DATA:\n##############\nNo local graph data available (empty graph).\n##############\n"
    
    assistant_prompt = f"""Remote data collected: {remote_data}{graph_data_text}

I'll systematically update the local knowledge graph with this remote data."""
    
    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )

def create_answer_production_prompt(query_text: str, memory_context: str, local_graph_data: Optional[List] = None) -> PromptStructure:
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
   PROOF SET PS_001: [Specific graph elements (nodes and relationships) that support sentence 1, specified by triples ID] -> EXAMPLE: (subject_id, relationship_id, object_id)

   SENTENCE 2: [Your second sentence] 
   PROOF SET PS_002: [Specific graph elements (nodes and relationships) that support sentence 2]

   ... continue for all sentences ...
   
   FINAL ANSWER: [Complete answer combining all sentences]
   ```

Use the local graph exploration tools to examine the current state of the knowledge graph and construct your evidence-backed answer."""

    user_prompt = f'Produce a comprehensive answer for: "{query_text}" using only data from the local knowledge graph, with each sentence backed by specific proof sets.'
    
    # Include local graph data in the assistant prompt
    graph_data_text = ""
    if local_graph_data:
        graph_data_text = f"\n\nCURRENT LOCAL GRAPH DATA:\n##############\n{local_graph_data}\n##############\n"
    else:
        graph_data_text = "\n\nCURRENT LOCAL GRAPH DATA:\n##############\nNo local graph data available (empty graph).\n##############\n"
    
    assistant_prompt = f"""Recent memory context:
{memory_context}{graph_data_text}

I'll examine the local knowledge graph and construct an evidence-backed answer with proof sets for each sentence.""" if memory_context else f"""{graph_data_text}"""

    
    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )
