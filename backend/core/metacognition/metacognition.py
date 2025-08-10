"""
Metacognition module for strategic assessment and meta-observation.

This module provides architecture-agnostic metacognitive capabilities that can observe,
evaluate, and suggest strategies in natural language without directly accessing internal
data structures like knowledge graphs.

The metacognition system consists of two main phases:
1. Strategic Assessment Module: Detects and makes explicit the strategy followed by an agent
2. Meta-Observation Module: Compares detected strategy with previous metacognitive observations
"""

import asyncio
import sys
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from ..logging import log_debug


@dataclass
class MetacognitiveObservation:
    """Represents a metacognitive observation about strategy and performance."""
    strategy_description: str
    task_context: str
    outcome_assessment: str
    suggested_improvements: Optional[str] = None
    timestamp: Optional[str] = None


class Metacognition:
    """
    Architecture-agnostic metacognitive system that observes and evaluates agent strategies.
    
    This system works by:
    1. Analyzing sequences of actions in natural language
    2. Inferring implicit strategies
    3. Comparing with previous observations
    4. Suggesting strategic corrections when needed
    """
    
    def __init__(self, 
                 agent: Any,
                 system_description: str,
                 agent_description: str,
                 task_description: str,
                 llm_settings: Optional[Any] = None):
        """
        Initialize the metacognition module.
        
        Args:
            agent: The LLM agent to use for metacognitive reasoning
            system_description: Description of the overall system
            agent_description: Description of the agent being observed
            task_description: Description of the current task
            llm_settings: Optional LLM settings for model selection
        """
        self.agent = agent
        self.system_description = system_description
        self.agent_description = agent_description
        self.task_description = task_description
        self.llm_settings = llm_settings
        self.previous_observation: Optional[MetacognitiveObservation] = None
        
    async def assess_strategy(self, action_sequence: str, task_outcome: str, query: str, current_turn: int = 0, max_turns: int = -1, statistics_data: Optional[Dict[str, Any]] = None, available_tools: Optional[Dict[str, List[Dict]]] = None) -> str:
        """
        Strategic Assessment Module: Detect and make explicit the strategy from action sequence.
        
        Args:
            action_sequence: Natural language description of actions performed
            task_outcome: Current outcome/results of the task in textual form
            query: The original user query being processed
            current_turn: Current turn number (0-based)
            max_turns: Maximum number of turns allowed (-1 for unlimited)
            statistics_data: Optional comprehensive statistics data from orchestrator
            available_tools: Optional dict of available tools by category (local_exploration, remote_exploration, graph_update, evaluation)
            
        Returns:
            Detected strategy description or "no recognizable strategy"
        """
        # Build enhanced context with statistics if available
        statistics_context = ""
        if statistics_data:
            # Extract key metrics for context
            state_stats = statistics_data.get("state_statistics", {})
            tool_stats = statistics_data.get("tool_statistics", {})
            graph_stats = statistics_data.get("graph_statistics", {})
            
            statistics_context = f"""

EXECUTION STATISTICS:
State Performance:
{self._format_state_statistics(state_stats)}

Tool Usage Patterns:
{self._format_tool_statistics(tool_stats)}

Knowledge Graph Evolution:
{self._format_graph_statistics(graph_stats)}
"""

        # Build tool descriptions context
        tools_context = ""
        if available_tools:
            tools_context = "\n\nAVAILABLE TOOLS BY CATEGORY:\n"
            for category, tools in available_tools.items():
                tools_context += f"\n{category.upper().replace('_', ' ')} TOOLS:\n"
                for tool in tools:
                    tool_name = tool.get('function', {}).get('name', 'Unknown')
                    tool_desc = tool.get('function', {}).get('description', 'No description')
                    tool_params = tool.get('function', {}).get('parameters', {}).get('properties', {})
                    
                    tools_context += f"  - {tool_name}: {tool_desc}\n"
                    if tool_params:
                        param_names = list(tool_params.keys())[:3]  # Show first 3 parameters
                        tools_context += f"    Parameters: {', '.join(param_names)}\n"
        
        system_prompt = f"""System: {self.system_description}
Agent: {self.agent_description}
Task: {self.task_description}

Analyze sequences of actions and identify the underlying strategic approach being followed. Focus on:

1. High-level strategy: What is the overall approach or methodology being used to accomplish the task?
2. Tool usage preferences: Which types of tools are being prioritized and in what order?
3. Data retrieval patterns: What approach is being taken to gather and validate information?
4. Knowledge graph construction strategy: How is the agent building and organizing the local graph?
5. Relationship exploration patterns: Is the agent exploring both forward (subject->object) and reverse (object<-subject) relationships effectively?
6. Performance indicators: Based on timing and success rates, what patterns emerge?

Provide a detailed strategic assessment that captures both the explicit actions and the implicit strategic decisions behind them. Consider whether the agent is effectively using bidirectional relationship exploration (both forward relationships from subjects and reverse relationships to find entities that reference key concepts). If no coherent strategy is detectable, state "no recognizable strategy".

Required output format:
Strategy detected: [detailed description of the high-level strategy, tool usage preferences, data retrieval approach, graph construction patterns, relationship exploration effectiveness, and performance characteristics in natural language]"""

        # Build turns context
        turns_context = ""
        if max_turns > 0:
            remaining_turns = max_turns - current_turn
            turns_context = f"""

TURN INFORMATION:
Current turn: {current_turn + 1}/{max_turns}
Remaining turns: {remaining_turns}
Turn urgency: {"HIGH - Graph consistency should be prioritized!" if remaining_turns <= 6 else "Medium" if remaining_turns <= 10 else "Low"}

IMPORTANT: When turns are running out (≤6 remaining), prioritize graph consistency and connectivity. Focus on fetching necessary relationships to avoid isolated nodes, as the current graph will be used for the final answer."""
        else:
            turns_context = f"""

TURN INFORMATION:
Current turn: {current_turn + 1}
Maximum turns: Unlimited"""

        user_prompt = f"""Original user query:
{query}

Sequence of actions performed:
{action_sequence}

Current task outcome assessment:
{task_outcome}{statistics_context}{tools_context}{turns_context}"""

        try:
            # Use metacognition model if available
            model = None
            if self.llm_settings:
                model = self.llm_settings.get_model_for_operation("metacognition")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": user_prompt}
            ]
            
            response = await self.agent.query_llm(messages, model=model, temperature=0.8, max_tokens=1500)
            strategy_text = response.content.strip()
            
            log_debug(f"Strategic assessment result: {strategy_text}", "METACOGNITION_STRATEGY")
            return strategy_text
            
        except Exception as e:
            log_debug(f"Error in strategic assessment: {e}", "METACOGNITION_ERROR")
            return "no recognizable strategy"
    
    async def meta_observe(self, 
                          detected_strategy: str, 
                          task_outcome: str,
                          query: str,
                          current_turn: int = 0,
                          max_turns: int = -1,
                          available_tools: Optional[Dict[str, List[Dict]]] = None) -> Optional[str]:
        """
        Meta-Observation Module: Compare detected strategy with previous observations
        and suggest corrections if needed.
        
        Args:
            detected_strategy: Strategy detected by the Strategic Assessment Module
            task_outcome: Current outcome/results of the task in textual form
            query: The original user query being processed
            current_turn: Current turn number (0-based)
            max_turns: Maximum number of turns allowed (-1 for unlimited)
            available_tools: Optional dict of available tools by category
            
        Returns:
            Strategic suggestion or None if no action needed
        """
        previous_obs_text = ""
        if self.previous_observation:
            previous_obs_text = f"""Strategy: {self.previous_observation.strategy_description}
Outcome: {self.previous_observation.outcome_assessment}
Previous suggestions: {self.previous_observation.suggested_improvements or 'None'}"""
        else:
            previous_obs_text = "NULL"

        # Build tool descriptions context
        tools_context = ""
        if available_tools:
            tools_context = "\n\nAVAILABLE TOOLS BY CATEGORY:\n"
            for category, tools in available_tools.items():
                tools_context += f"\n{category.upper().replace('_', ' ')} TOOLS:\n"
                for tool in tools:
                    tool_name = tool.get('function', {}).get('name', 'Unknown')
                    tool_desc = tool.get('function', {}).get('description', 'No description')
                    tool_params = tool.get('function', {}).get('parameters', {}).get('properties', {})
                    
                    tools_context += f"  - {tool_name}: {tool_desc}\n"
                    if tool_params:
                        param_names = list(tool_params.keys())[:3]  # Show first 3 parameters
                        tools_context += f"    Parameters: {', '.join(param_names)}\n"
        
        system_prompt = f"""System: {self.system_description}
Agent: {self.agent_description}
Task: {self.task_description}

Evaluate the current strategy against the task outcome and any previous observations. Consider:

1. Strategy coherence: Is the detected strategy internally consistent and well-suited for the current query? How does the current strategy compare? Is there improvement, regression, or consistency?
2. TOOL USE STRATEGY: ARE THERE SPECIFIC IMPROVEMENTS IN TOOL SELECTION, DATA RETRIEVAL APPROACH, OR GRAPH CONSTRUCTION THAT COULD ENHANCE PERFORMANCE?
3. BIDIRECTIONAL RELATIONSHIP EXPLORATION: Is the agent effectively using both forward relationships (subject->object) and reverse relationships (finding all entities that reference a target entity)? Consider suggesting fetch_relationship_to_node for discovering hidden connections.
4. TOOL PERFORMANCE: IF SOME TOOLS ARE CONSISTENTLY UNDERPERFORMING, SUGGEST ALTERNATIVES. 
5. LOCAL KNOWLEDGE GRAPH TOPOLOGY: THE LOCAL KNOWLEDGE GRAPH SHOULD BE CONNECTED. MINIMIZE ISOLATED NODES AND ENSURE A COHERENT TOPOLOGY. MAKE SURE THAT THE GRAPH IS NOT JUST A COLLECTION OF ISOLATED NODES BUT A COHERENT STRUCTURE THAT SUPPORTS QUERYING AND NAVIGATION.

Required output format:
Response: 
[Next strategy to follow including]
- Identify graph consistency issues: Relevant nodes missing? Relationships missing? Isolated nodes? 
- Consider bidirectional exploration: Are there reverse relationships that could reveal missing entities or connections?
- Next mandatory tool calls and alternative strategies in case of failure"""

        # Build turns context
        turns_context = ""
        if max_turns > 0:
            remaining_turns = max_turns - current_turn
            turns_context = f"""

TURN INFORMATION:
Current turn: {current_turn + 1}/{max_turns}
Remaining turns: {remaining_turns}
Turn urgency: {"HIGH - Graph consistency should be prioritized!" if remaining_turns <= 6 else "Medium" if remaining_turns <= 10 else "Low"}

CRITICAL GUIDANCE: When turns are running out (≤3 remaining), the current graph will be used for the final answer. IMMEDIATELY prioritize:
1. Connecting isolated nodes by fetching missing relationships
2. Ensuring the graph has a coherent, connected topology
3. Adding essential relationships that support the query requirements
4. Consider reverse relationship discovery to find entities that reference key concepts
5. Avoiding further exploration that doesn't improve graph connectivity"""
        else:
            turns_context = f"""

TURN INFORMATION:
Current turn: {current_turn + 1}
Maximum turns: Unlimited
Focus: Balanced exploration and graph building"""

        user_prompt = f"""Original user query:
{query}

Current detected strategy:
{detected_strategy}

Previous metacognitive observation (if available):
{previous_obs_text}

Current task outcome assessment:
{task_outcome}{tools_context}{turns_context}"""

        try:
            # Use metacognition model if available
            model = None
            if self.llm_settings:
                model = self.llm_settings.get_model_for_operation("metacognition")
                
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": user_prompt}
            ]
            
            response = await self.agent.query_llm(messages, model=model, temperature=0.7, max_tokens=2000)
            suggestion = response.content.strip()
            
            # Update previous observation with current state
            self.previous_observation = MetacognitiveObservation(
                strategy_description=detected_strategy,
                task_context=self.task_description,
                outcome_assessment=task_outcome,
                suggested_improvements=suggestion if suggestion and not suggestion.lower().startswith("no action") else None
            )
            
            log_debug(f"Meta-observation result: {suggestion}", "METACOGNITION_META")
            
            # Return suggestion only if it's actionable
            if suggestion and not any(phrase in suggestion.lower() for phrase in ["do nothing", "no action", "no change needed"]):
                return suggestion
            else:
                return None
                
        except Exception as e:
            log_debug(f"Error in meta-observation: {e}", "METACOGNITION_ERROR")
            return None
    
    async def process_metacognitive_cycle(self, 
                                        action_sequence: str, 
                                        task_outcome: str,
                                        query: str,
                                        current_turn: int = 0,
                                        max_turns: int = -1,
                                        statistics_data: Optional[Dict[str, Any]] = None,
                                        available_tools: Optional[Dict[str, List[Dict]]] = None) -> Optional[str]:
        """
        Run the complete metacognitive cycle: strategic assessment + meta-observation.
        
        Args:
            action_sequence: Natural language description of recent actions
            task_outcome: Current state/outcome of the task
            query: The original user query being processed
            current_turn: Current turn number (0-based)
            max_turns: Maximum number of turns allowed (-1 for unlimited)
            statistics_data: Optional comprehensive statistics data from orchestrator
            available_tools: Optional dict of available tools by category
            
        Returns:
            Strategic suggestion if any corrections are needed, None otherwise
        """
        log_debug("Starting metacognitive cycle", "METACOGNITION")
        
        # Phase 1: Strategic Assessment
        detected_strategy = await self.assess_strategy(action_sequence, task_outcome, query, current_turn, max_turns, statistics_data, available_tools)
        
        # Phase 2: Meta-Observation
        suggestion = await self.meta_observe(detected_strategy, task_outcome, query, current_turn, max_turns, available_tools)
        
        if suggestion:
            log_debug(f"Metacognitive suggestion generated: {suggestion}", "METACOGNITION")
        else:
            log_debug("No metacognitive intervention needed", "METACOGNITION")
            
        return suggestion
    
    def get_previous_observation(self) -> Optional[MetacognitiveObservation]:
        """Get the previous metacognitive observation."""
        return self.previous_observation
    
    def reset_observations(self):
        """Reset all previous observations."""
        self.previous_observation = None
        log_debug("Metacognitive observations reset", "METACOGNITION")
        
    def _format_state_statistics(self, state_stats: Dict[str, Any]) -> str:
        """Format state statistics for inclusion in prompts."""
        if not state_stats:
            return "No state statistics available"
            
        lines = []
        for state, stats in state_stats.items():
            avg_time = stats.get("average_time", 0)
            tool_calls = stats.get("tool_calls", 0)
            transitions = stats.get("transitions", 0)
            lines.append(f"  {state}: {transitions} transitions, avg {avg_time:.2f}s, {tool_calls} tool calls")
        
        return "\n".join(lines) if lines else "No state data"
    
    def _format_tool_statistics(self, tool_stats: Dict[str, Any]) -> str:
        """Format tool statistics for inclusion in prompts."""
        if not tool_stats:
            return "No tool statistics available"
            
        lines = []
        for tool_name, stats in tool_stats.items():
            count = stats.get("call_count", 0)
            success_rate = stats.get("success_rate", 0)
            avg_duration = stats.get("average_duration", 0)
            lines.append(f"  {tool_name}: {count} calls, {success_rate:.1%} success, avg {avg_duration:.2f}s")
        
        return "\n".join(lines) if lines else "No tool data"
    
    def _format_graph_statistics(self, graph_stats: Dict[str, Any]) -> str:
        """Format graph statistics for inclusion in prompts."""
        if not graph_stats:
            return "No graph statistics available"
            
        initial_nodes = graph_stats.get("initial_nodes", 0)
        final_nodes = graph_stats.get("final_nodes", 0)
        initial_edges = graph_stats.get("initial_edges", 0) 
        final_edges = graph_stats.get("final_edges", 0)
        nodes_added = graph_stats.get("nodes_added", 0)
        edges_added = graph_stats.get("edges_added", 0)
        
        result = f"  Growth: {initial_nodes}→{final_nodes} nodes (+{nodes_added}), {initial_edges}→{final_edges} edges (+{edges_added})"
        
        per_state = graph_stats.get("per_state_stats", {})
        if per_state:
            result += "\n  Per-state averages:"
            for state, stats in per_state.items():
                avg_nodes = stats.get("avg_nodes", 0)
                avg_edges = stats.get("avg_edges", 0)
                result += f"\n    {state}: {avg_nodes:.1f} nodes, {avg_edges:.1f} edges"
        
        return result


if __name__ == "__main__":
    """
    Demo of the metacognition module capabilities.
    """
    import sys
    from pathlib import Path
    
    # Add backend directory to path
    backend_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(backend_dir))
    
    from ..agents.openai import OpenAIAgent
    
    async def demo():
        """Demonstrate metacognition capabilities."""
        print("🧠 Metacognition Module Demo")
        print("=" * 50)
        
        # Create an agent for metacognitive reasoning
        agent = OpenAIAgent(model="gpt-4o")
        
        # Initialize metacognition module
        metacognition = Metacognition(
            agent=agent,
            system_description="RoboData - An AI-powered ontology exploration framework where a metacognitive LLM agent monitors and corrects the execution history of a multi-stage LLM-based orchestrator",
            agent_description="Multi-stage orchestrator with six distinct execution phases for local graph exploration, evaluation, remote Wikidata exploration, and knowledge graph construction using specialized toolboxes",
            task_description="Systematically explore the Wikidata ontology to populate a local Neo4j knowledge graph that can answer natural language queries with comprehensive supporting evidence"
        )
        
        # Example 1: First metacognitive cycle
        print("\n🔍 Example 1: Initial strategy assessment")
        action_sequence_1 = """
        1. Started local graph exploration for query about climate change
        2. Used search_local_entities tool to find related concepts
        3. Found no relevant data in local graph
        4. Initiated remote exploration using Wikipedia search
        5. Retrieved climate change definition and causes
        6. Updated local graph with new entities and relationships
        """
        
        task_outcome_1 = "Successfully retrieved basic climate change information and populated empty graph"
        
        # Mock statistics data for demonstration
        statistics_data_1 = {
            "state_statistics": {
                "EVAL_LOCAL_DATA": {"average_time": 2.1, "tool_calls": 3, "transitions": 1},
                "REMOTE_GRAPH_EXPLORATION": {"average_time": 5.2, "tool_calls": 8, "transitions": 1},
                "LOCAL_GRAPH_UPDATE": {"average_time": 1.8, "tool_calls": 4, "transitions": 1}
            },
            "tool_statistics": {
                "search_local_entities": {"call_count": 3, "success_rate": 0.33, "average_duration": 0.8},
                "wikipedia_search": {"call_count": 2, "success_rate": 1.0, "average_duration": 2.1},
                "add_entity": {"call_count": 5, "success_rate": 1.0, "average_duration": 0.3}
            },
            "graph_statistics": {
                "initial_nodes": 0, "final_nodes": 12, "nodes_added": 12,
                "initial_edges": 0, "final_edges": 18, "edges_added": 18,
                "per_state_stats": {
                    "EVAL_LOCAL_DATA": {"avg_nodes": 0.0, "avg_edges": 0.0},
                    "REMOTE_GRAPH_EXPLORATION": {"avg_nodes": 6.0, "avg_edges": 8.0},
                    "LOCAL_GRAPH_UPDATE": {"avg_nodes": 12.0, "avg_edges": 18.0}
                }
            }
        }
        
        suggestion_1 = await metacognition.process_metacognitive_cycle(
            action_sequence_1, task_outcome_1, "climate change impacts", 0, 10, statistics_data_1, None  # Demo with turn 0/10
        )
        print(f"Suggestion: {suggestion_1 or 'None - strategy appears effective'}")
        
        # Example 2: Second cycle with potential issues
        print("\n🔍 Example 2: Strategy with issues")
        action_sequence_2 = """
        1. Started local graph exploration for query about Albert Einstein
        2. Used search_local_entities repeatedly with same search terms
        3. All local searches returned empty results
        4. Attempted remote exploration with very generic search
        5. Retrieved irrelevant biographical data
        6. Failed to update graph due to data quality issues
        """
        
        task_outcome_2 = "Failed to find relevant information, graph remains mostly empty, user query not answered"
        
        # Mock statistics showing poor performance
        statistics_data_2 = {
            "state_statistics": {
                "EVAL_LOCAL_DATA": {"average_time": 4.5, "tool_calls": 8, "transitions": 3},
                "REMOTE_GRAPH_EXPLORATION": {"average_time": 8.1, "tool_calls": 12, "transitions": 2},
                "LOCAL_GRAPH_UPDATE": {"average_time": 0.2, "tool_calls": 1, "transitions": 1}
            },
            "tool_statistics": {
                "search_local_entities": {"call_count": 8, "success_rate": 0.0, "average_duration": 1.2},
                "wikipedia_search": {"call_count": 5, "success_rate": 0.4, "average_duration": 3.2},
                "add_entity": {"call_count": 1, "success_rate": 0.0, "average_duration": 0.2}
            },
            "graph_statistics": {
                "initial_nodes": 0, "final_nodes": 1, "nodes_added": 1,
                "initial_edges": 0, "final_edges": 0, "edges_added": 0,
                "per_state_stats": {
                    "EVAL_LOCAL_DATA": {"avg_nodes": 0.0, "avg_edges": 0.0},
                    "REMOTE_GRAPH_EXPLORATION": {"avg_nodes": 0.5, "avg_edges": 0.0},
                    "LOCAL_GRAPH_UPDATE": {"avg_nodes": 1.0, "avg_edges": 0.0}
                }
            }
        }
        
        suggestion_2 = await metacognition.process_metacognitive_cycle(
            action_sequence_2, task_outcome_2, "Albert Einstein biography", 8, 10, statistics_data_2, None  # Demo with turn 8/10 (critical)
        )
        print(f"Suggestion: {suggestion_2 or 'None - no issues detected'}")
        
        # Example 3: Third cycle showing improvement
        print("\n🔍 Example 3: Improved strategy")
        action_sequence_3 = """
        1. Started with specific entity search for 'Albert Einstein'
        2. Used diverse search terms and synonyms
        3. When local search failed, immediately moved to targeted remote search
        4. Used multiple authoritative sources (Wikipedia, Wikidata)
        5. Carefully validated and structured retrieved data
        6. Successfully updated graph with verified information
        """
        
        task_outcome_3 = "Successfully found comprehensive Einstein information, graph well-populated, query answered accurately"
        
        suggestion_3 = await metacognition.process_metacognitive_cycle(action_sequence_3, task_outcome_3, "Albert Einstein detailed information", 5, -1, None, None)  # Demo with unlimited turns
        print(f"Suggestion: {suggestion_3 or 'None - strategy appears effective'}")
        
        print("\n📊 Previous observation summary:")
        prev_obs = metacognition.get_previous_observation()
        if prev_obs:
            print(f"Strategy: {prev_obs.strategy_description}")
            print(f"Outcome: {prev_obs.outcome_assessment}")
            if prev_obs.suggested_improvements:
                print(f"Suggestions: {prev_obs.suggested_improvements}")
        
        print("\n✅ Demo completed")
    
    asyncio.run(demo())
