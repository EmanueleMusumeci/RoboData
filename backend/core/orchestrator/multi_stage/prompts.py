from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

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
class AttemptHistory:
    """Track previous attempts and their outcomes."""
    remote_explorations: int = 0
    local_explorations: int = 0
    failures: List[str] = field(default_factory=list)

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

    system_prompt = f"""
You are an expert agent exploring a local knowledge graph to find information to answer a specific query.
Your task is to systematically explore the available knowledge graph to find relevant information. 


USING LOCAL GRAPH DATA:
- Local graph data shows existing nodes and relationships
- The graph might be empty or incomplete
- Use this data to understand what's already in the graph and identify gaps that need to be filled
- Build upon existing data when exploring new connections
- Avoid redundant exploration of already-known information

NOTICE: 
If the graph is too big, only the entities and relationships relevant to the query will be shown.
In this case you can use the provided tools to explore the graph and gather relevant data.

DECISION CRITERIA:
After each exploration step, you must decide whether to:
A) Continue exploring the local graph because you found promising leads -> ANSWER "LOCAL_GRAPH_EXPLORATION" 
D) Stop exploring and evaluate what you've found -> ANSWER "EVAL_LOCAL_DATA"

Make your decision based on:
- Whether you found relevant entities/relationships that warrant further exploration
- Whether you have gathered sufficient local information to attempt an evaluation
- How the current local graph data relates to your query
- The DATA PROVIDED BY THE "ASSISTANT"

You MUST end your response with exactly one of:
- "LOCAL_GRAPH_EXPLORATION" - Continue local exploration
- "EVAL_LOCAL_DATA" - Ready for evaluation of the data contained in the local graph

IF YOU HAVE CHOSEN AGAIN "REMOTE_GRAPH_EXPLORATION" or "LOCAL_GRAPH_EXPLORATION", you should provide a clear explanation of why you need to explore remote data, including any changes to the query or tools used.
- Evaluation: a summary or evaluation of the current situation and/or the tools executed (tool_name, tool_parameters, why the tool was executed)
- Your instructions of what to do next as if they were an order "You should ....", explicitly stating that these instructions should be based on the available tools, which should be listed below as:
- tool_name: "tool_description", "explanation of why this tool is needed" USING THE FOLLOWING LOCAL_GRAPH_EXPLORATION TOOLS:
{tools_text}
- A final invitation to use more tools at the same time if necessary
- The final choice token
"""

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
    
    assistant_prompt = f"""HI! I AM THE ASSISTANT! I PROVIDE USEFUL DATA!
    PREVIOUS THOUGHT FROM THE {last_stage_name} STAGE:
    <thoughts>
    {last_llm_response or "I'll start exploring the local knowledge graph to find relevant information."}
    </thoughts>

    RECENT MEMORY:
    <memory>
    {memory_context}
    </memory>

    <local_graph_data>
    {graph_data_text}
    </local_graph_data>

    <local_exploration_results>
    {local_exploration_text}
    </local_exploration_results>
    """

    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )

def create_local_evaluation_prompt(query_text: str, attempt_history: 'AttemptHistory', memory_context: str, local_graph_data: Optional[str] = None, last_llm_response: Optional[str] = None, next_step_tools: Optional[List] = None, last_stage_name: str = "Unknown", local_exploration_results: Optional[List] = None) -> PromptStructure:
    """Create prompt for local data evaluation with optional graph data."""
    
    tools_text = ""
    if next_step_tools:
        tools_text = "\n".join([f"\t\t- TOOL {tool['function']['name']}: {tool['function']['description']}" for tool in next_step_tools])

    system_prompt = f"""You are an expert agent evaluating the available data in a local knowledge graph to find information to answer a specific query. 
    You need to evaluate whether the local data is sufficient to answer the user's query.

FIRST, EXTRACT RELEVANT ENTITIES AND RELATIONSHIPS FROM THE QUERY
- Identify key entities, relationships, and facts mentioned in the query.
- List them clearly
- Make sure each entity and relationship is backed by at least one node or edge in the local graph

THEN, USING LOCAL GRAPH DATA:
- Analyze the current local graph data to understand available entities, relationships, and facts
- Determine if this data directly answers the query or provides sufficient context
- Consider the completeness and relevance of the local data for the specific question
- Base your decision on concrete evidence present in the local graph
- If the graph is too big, only the entities and relationships relevant to the query will be shown

DECISION CRITERIA:
You have sufficient local data to answer the query if:
- The local graph contains relevant entities and relationships that directly address the query.
- Each relevant extracted entity and relationship is supported by at least one node or edge in the local graph.

If you have sufficient local data -> ANSWER "PRODUCE_ANSWER"

YOU DON'T HAVE SUFFICIENT LOCAL DATA TO ANSWER THE QUERY IF:
- The local graph lacks relevant entities or relationships to address the query.
- Key entities or relationships mentioned in the query are absent from the local graph.
- The supporting evidence in the local graph is incomplete or insufficient.
- At least one of the relevant entities or relationships is not present in the local graph.

Example:
    Description of entity A mentions entity B but entity B is not available in the local graph as a node or an edge.
    Entity B is not present in the local graph as a node or an edge therefore you cannot answer the query. 
    You need to further explore the remote graph to find entity B.


If you don't have sufficient local data, decide whether to:
A) Give a partial answer with available data -> "PRODUCE_ANSWER"
B) Attempt remote graph exploration (if previous attempts allow) -> "REMOTE_GRAPH_EXPLORATION"
C) Go back to local exploration to find more details -> "LOCAL_GRAPH_EXPLORATION"

You MUST end your response with exactly one of:
- "PRODUCE_ANSWER"
- "REMOTE_GRAPH_EXPLORATION"
- "LOCAL_GRAPH_EXPLORATION"

IF YOU HAVE CHOSEN AGAIN "REMOTE_GRAPH_EXPLORATION", you should provide a clear explanation of why you need to explore remote data, including any changes to the query or tools used.
- Evaluation: a summary or evaluation of the current situation and/or the tools executed (tool_name, tool_parameters, why the tool was executed)
- Your instructions of what to do next as if they were an order "You should ....", explicitly stating that these instructions should be based on the available tools, which should be listed below as:
- tool_name: "tool_description", "explanation of why this tool is needed" USING THE FOLLOWING REMOTE_GRAPH_EXPLORATION TOOLS:
{tools_text}
- A final invitation to use more tools at the same time if necessary
- The final choice token"""

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

    assistant_data = f"""HI! I AM THE ASSISTANT! I PROVIDE USEFUL DATA!
    PREVIOUS THOUGHT FROM THE {last_stage_name} STAGE:
    <thoughts>
    {last_llm_response or "Let's evaluate the available local data."}
    </thoughts>

    RECENT MEMORY:
    <memory>
    {memory_context}
    </memory>

    <local_graph_data>
    {graph_data_text}
    </local_graph_data>

    <local_exploration_results>
    {local_exploration_text}
    </local_exploration_results>
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

    system_prompt = f"""You are exploring remote knowledge sources to gather relevant data from external sources.

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

Your available tools for remote exploration are:
{tools_text}
"""

    user_prompt = f'Explore remote knowledge sources to answer: "{query_text}"'
    
    # Include local graph data in the assistant prompt
    graph_data_text = ""
    if local_graph_data:
        graph_data_text = f"\n\nCurrent local graph data:\n{local_graph_data}"
    else:
        graph_data_text = "\n\nCurrent local graph data is empty."
    
    assistant_prompt = f"""HI! I AM THE ASSISTANT! I PROVIDE USEFUL DATA!
    PREVIOUS THOUGHT FROM THE {last_stage_name} STAGE:
    <thoughts>
    {last_llm_response or "Let's evaluate the available local data."}
    </thoughts>

    RECENT MEMORY:
    <memory>
    {memory_context}
    </memory>

    <local_graph_data>
    {graph_data_text}
    </local_graph_data>
"""
    
    return PromptStructure(
        system=system_prompt,
        user=user_prompt,
        assistant=assistant_prompt
    )

def create_remote_evaluation_prompt(query_text: str, memory_context: str, remote_data: list, local_graph_data: Optional[str] = None, last_llm_response: Optional[str] = None, next_step_tools: Optional[List] = None, last_stage_name: str = "Unknown") -> PromptStructure:
    """Create prompt for remote data evaluation."""
    
    tools_text = ""
    if next_step_tools:
        tools_text = "\n".join([f"\t\t- TOOL {tool['function']['name']}: {tool['function']['description']}" for tool in next_step_tools])

    system_prompt = f"""You are an expert agent evaluating remote data to decide if it's relevant for a user's query.
Your main goal is to incrementally build a knowledge graph. You must determine if the gathered remote data, even if incomplete, is related to the query and can contribute to the graph.

EVALUATION CRITERIA:
- The data does NOT need to directly answer the query but should contribute to building a relevant knowledge graph that might help answer the query in the future.
- The data IS RELEVANT if it contains entities, relationships, or facts related to the query topic.
- Your task is to collect pieces of information that, step-by-step, will build a graph to answer the query.
- SELECT ONLY RELEVANT DATA, DISCARD THE REST

USING LOCAL GRAPH DATA:
- Compare remote data with the existing local graph to see what's new.
- The remote data should complement or expand the local graph.
- If the graph is too big, only the entities and relationships relevant to the query will be shown.

DECISION CRITERIA:
Based on the relevance to the query topic, you must decide:
A) The data is relevant and on-topic -> ANSWER "LOCAL_GRAPH_UPDATE"
B) The data is completely irrelevant and off-topic -> ANSWER "EVAL_LOCAL_DATA"
C) The data is not sufficient, and you need to explore more remote data -> "REMOTE_GRAPH_EXPLORATION"

You MUST end your response with exactly one of:
- "LOCAL_GRAPH_UPDATE" - Data is relevant and should be added to the graph.
- "EVAL_LOCAL_DATA" - Data is not relevant and should be discarded.
- "REMOTE_GRAPH_EXPLORATION" - Not enough data, need more remote exploration.

IF YOU HAVE CHOSEN "LOCAL_GRAPH_UPDATE" or "REMOTE_GRAPH_EXPLORATION", you should provide a clear explanation of why you think the remote data is relevant, including any changes to the query or tools used.
- Evaluation: a summary or evaluation of the current situation and/or the tools executed (tool_name, tool_parameters, why the tool was executed)
- Your instructions of what to do next as if they were an order "You should ....", explicitly stating that these instructions should be based on the available tools, which should be listed below as:
- tool_name: "tool_description", "explanation of why this tool is needed" USING THE FOLLOWING LOCAL_GRAPH_UPDATE TOOLS:
{tools_text}
- A final invitation to use more tools at the same time if necessary
- The final choice token"""

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

    assistant_prompt = f"""HI! I AM THE ASSISTANT! I PROVIDE USEFUL DATA!
    PREVIOUS THOUGHT FROM THE {last_stage_name} STAGE:
    <thoughts>
    {last_llm_response or "Let's evaluate the relevance of this remote data."}
    </thoughts>

    <local_graph_data>
    {graph_data_text}
    </local_graph_data>

    <remote_graph_data>
    {remote_graph_data_text}
    </remote_graph_data>

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

def create_graph_update_prompt(query_text: str, memory_context: str, remote_data: Optional[List], local_graph_data: Optional[str] = None, last_llm_response: Optional[str] = None, next_step_tools: Optional[List] = None, last_stage_name: str = "Unknown") -> PromptStructure:
    """Create prompt for graph update."""
    
    tools_text = ""
    if next_step_tools:
        tools_text = "\n".join([f"\t\t- TOOL {tool['function']['name']}: {tool['function']['description']}" for tool in next_step_tools])

    system_prompt = f"""Update the local knowledge graph with relevant remote data.

CRITICAL INSTRUCTIONS FOR GRAPH CONSTRUCTION:

1. BUILD A CONNECTED GRAPH: Every new entity must be connected to at least one other entity through relationships.
   Do NOT add isolated nodes unless absolutely necessary.

2. RELATIONSHIP-FIRST APPROACH: You should always try to establish new relationships in the local data!!!
    When you find entities that are related in the remote data:
   - First add the entities using add_node
   - Immediately add relationships using add_edge to connect them

3. SELECT ONLY RELEVANT DATA:
   - The remote data should be directly related to the query topic
   - Focus on entities, relationships, and facts that contribute to building a relevant knowledge graph
   - Avoid adding irrelevant or off-topic data that does not contribute to the graph's purpose or that is redundant

4. PRESERVE DATA STRUCTURE: If the remote data shows relationships between existing entities in the local graph translate them directly into graph relationships. 
The local graph should mirror the remote structure as closely as possible. Each local relationship MUST correspond to a remote relationship.

5. INTEGRATE WITH EXISTING DATA: 
   - Review the current local graph data to understand existing structure
   - Connect new entities to existing entities when relationships exist
   - Avoid duplicating existing nodes and relationships
   - Maintain consistency with the existing graph schema

6. AVOID HALLUCINATION: Only create relationships that are explicitly present in the remote data.
   Do not invent relationships that aren't supported by the data.

7. WORK SYSTEMATICALLY:
   a) First, identify useful data in the remote data
   b) Check if any already exist in the local graph
   c) Add new entities as nodes with their properties
   d) Then systematically add all relationships found in the data

Your available tools for graph update are:
{tools_text}
"""

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

    assistant_prompt = f"""HI! I AM THE ASSISTANT! I PROVIDE USEFUL DATA!
    PREVIOUS THOUGHT FROM THE {last_stage_name} STAGE:
    <thoughts>
    {last_llm_response or "I'll systematically update the local knowledge graph with this remote data."}
    </thoughts>

    <local_graph_data>
    {graph_data_text}
    </local_graph_data>

    <remote_graph_data>
    {remote_graph_data_text}
    </remote_graph_data>

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
   PROOF SET PS_001: [Specific graph elements (nodes and relationships) that support sentence 1, specified by triples ID] -> EXAMPLE: (subject_id, relationship_id, object_id)

   SENTENCE 2: [Your second sentence] 
   PROOF SET PS_002: [Specific graph elements (nodes and relationships) that support sentence 2]

   ... continue for all sentences ...
   
   FINAL ANSWER: [Complete answer combining all sentences]
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

    <local_graph_data>
    {graph_data_text}
    </local_graph_data>

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

    system_prompt = f"""You are an expert query decomposition agent. Your task is to analyze complex questions and break them down into a sequence of simpler, more focused sub-questions that can be answered individually to build up to a comprehensive answer.

DECOMPOSITION STRATEGY:

1. ANALYZE THE ORIGINAL QUESTION:
   - Identify the main concepts, entities, and relationships involved

2. BREAK DOWN INTO SUB-QUESTIONS:
   - Create up to 5 focused sub-questions that together cover all aspects of the original question
   - Each sub-question should be:
     * Self-contained and answerable independently
     * Specific and focused on one particular aspect
     * Logically ordered to build understanding progressively
     * Designed to gather information that contributes to the final answer

3. SUB-QUESTION CHARACTERISTICS:
   - Start with foundational/definitional questions if needed
   - Progress from simple facts to more complex analysis
   - Include context-gathering questions before analysis questions
   - Ensure each question can be answered using the available tools and knowledge sources


YOUR AVAILABLE TOOLS AND CAPABILITIES:
You have access to these tools that can be used to answer sub-questions:
{tools_text}

Consider these capabilities when designing sub-questions to ensure they can be effectively answered.

OUTPUT FORMAT:
If decomposition is helpful, provide your sub-questions in this format:
1. [First sub-question]
2. [Second sub-question]
3. [Third sub-question]
...

"""

    user_prompt = f'Please analyze this question and determine if it should be decomposed into sub-questions: "{query_text}"'
    
    assistant_prompt = f"""HI! I AM THE ASSISTANT! I PROVIDE USEFUL DATA!

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
