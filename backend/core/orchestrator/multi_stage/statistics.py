"""
Statistics collection module for multi-stage orchestrator.

This module provides comprehensive statistics collection including:
- State transition history with timing
- Tool execution statistics with arguments and outcomes
- Inference timing from LLM calls
- Metacognition statistics (when enabled)
- Aggregated performance metrics
"""

import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


@dataclass
class AttemptHistory:
    """Track previous attempts and their outcomes."""
    remote_explorations: int = 0
    local_explorations: int = 0
    failures: List[str] = field(default_factory=list)


@dataclass 
class StateTransition:
    """Record a state transition with timing information."""
    from_state: str
    to_state: str
    timestamp: float
    inference_time: Optional[float] = None
    inference_tokens: Optional[Dict[str, int]] = None  # prompt_tokens, completion_tokens, total_tokens
    metacognition_time: Optional[float] = None
    metacognition_tokens: Optional[Dict[str, int]] = None


@dataclass 
class SubstateTransition:
    """Record a substate transition with timing information."""
    from_substate: Optional[str]
    to_substate: str
    parent_state: str
    timestamp: float
    inference_time: Optional[float] = None
    inference_tokens: Optional[Dict[str, int]] = None
    metacognition_time: Optional[float] = None
    metacognition_tokens: Optional[Dict[str, int]] = None


@dataclass
class ExecutionLogEntry:
    """Record a full execution log entry for replay capability."""
    timestamp: float
    entry_type: str  # "state_transition", "substate_transition", "tool_execution", "inference", "metacognition"
    state: str
    substate: Optional[str] = None
    from_state: Optional[str] = None
    from_substate: Optional[str] = None
    to_state: Optional[str] = None
    to_substate: Optional[str] = None
    total_time: Optional[float] = None
    inference_time: Optional[float] = None
    inference_tokens: Optional[Dict[str, int]] = None
    metacognition_time: Optional[float] = None
    metacognition_tokens: Optional[Dict[str, int]] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    tool_duration: Optional[float] = None
    tool_outcome: Optional[str] = None
    tool_context: Optional[str] = None
    tool_success: Optional[bool] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class ToolExecution:
    """Record a tool execution with timing and outcome information."""
    tool_name: str
    arguments: Dict[str, Any]
    start_time: float
    end_time: float
    duration: float
    outcome: Any
    context: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class InferenceEvent:
    """Record an LLM inference event."""
    timestamp: float
    state: str
    duration: float
    substate: Optional[str] = None
    tokens: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    context_type: str = "orchestrator"  # "orchestrator" or "metacognition"


@dataclass
class MetacognitionEvent:
    """Record a metacognition cycle."""
    timestamp: float
    state: str
    duration: float
    suggestion: Optional[str] = None
    tokens: Optional[Dict[str, int]] = None


@dataclass
class GraphStatistics:
    """Record knowledge graph statistics at a specific point in time."""
    timestamp: float
    state: str
    substate: Optional[str] = None
    node_count: int = 0
    edge_count: int = 0
    relationship_count: int = 0  # alias for edge_count for Neo4j compatibility


class OrchestratorStatistics:
    """Comprehensive statistics collector for multi-stage orchestrator."""
    
    def __init__(self, experiment_id: str, query: str, enable_metacognition: bool = False):
        self.experiment_id = experiment_id
        self.query = query
        self.enable_metacognition = enable_metacognition
        self.start_time = time.time()
        
        # Core tracking data
        self.state_transitions: List[StateTransition] = []
        self.substate_transitions: List[SubstateTransition] = []
        self.tool_executions: List[ToolExecution] = []
        self.inference_events: List[InferenceEvent] = []
        self.metacognition_events: List[MetacognitionEvent] = []
        self.graph_statistics: List[GraphStatistics] = []
        self.execution_log: List[ExecutionLogEntry] = []
        
        # Current state tracking
        self.current_state = "READY"
        self.current_substate: Optional[str] = None
        self.state_start_time = self.start_time
        self.substate_start_time = self.start_time
        self.current_state_inference_time = 0.0
        self.current_state_tool_time = 0.0
        self.current_substate_inference_time = 0.0
        self.current_substate_tool_time = 0.0
        
        # Attempt history
        self.attempt_history = AttemptHistory()
        
        # Final results
        self.final_answer: Optional[str] = None
        self.end_time: Optional[float] = None
        self.success: bool = False
        
    def record_state_transition(self, from_state: str, to_state: str, 
                               inference_time: Optional[float] = None,
                               inference_tokens: Optional[Dict[str, int]] = None,
                               metacognition_time: Optional[float] = None,
                               metacognition_tokens: Optional[Dict[str, int]] = None):
        """Record a state transition."""
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            timestamp=time.time(),
            inference_time=inference_time,
            inference_tokens=inference_tokens,
            metacognition_time=metacognition_time,
            metacognition_tokens=metacognition_tokens
        )
        self.state_transitions.append(transition)
        
        # Add to execution log
        total_time = time.time() - self.state_start_time if from_state != "unknown" else 0
        log_entry = ExecutionLogEntry(
            timestamp=time.time(),
            entry_type="state_transition",
            state=to_state,
            from_state=from_state,
            to_state=to_state,
            total_time=total_time,
            inference_time=inference_time,
            inference_tokens=inference_tokens,
            metacognition_time=metacognition_time,
            metacognition_tokens=metacognition_tokens
        )
        self.execution_log.append(log_entry)
        
        # Update current state tracking
        self.current_state = to_state
        self.state_start_time = time.time()
        self.current_state_inference_time = 0.0
        self.current_state_tool_time = 0.0
        # Reset substate tracking when state changes
        self.current_substate = None
        self.substate_start_time = time.time()
        self.current_substate_inference_time = 0.0
        self.current_substate_tool_time = 0.0
        
    def record_substate_transition(self, from_substate: Optional[str], to_substate: str,
                                  inference_time: Optional[float] = None,
                                  inference_tokens: Optional[Dict[str, int]] = None,
                                  metacognition_time: Optional[float] = None,
                                  metacognition_tokens: Optional[Dict[str, int]] = None):
        """Record a substate transition within the current state."""
        transition = SubstateTransition(
            from_substate=from_substate,
            to_substate=to_substate,
            parent_state=self.current_state,
            timestamp=time.time(),
            inference_time=inference_time,
            inference_tokens=inference_tokens,
            metacognition_time=metacognition_time,
            metacognition_tokens=metacognition_tokens
        )
        self.substate_transitions.append(transition)
        
        # Add to execution log
        total_time = time.time() - self.substate_start_time if from_substate is not None else 0
        log_entry = ExecutionLogEntry(
            timestamp=time.time(),
            entry_type="substate_transition",
            state=self.current_state,
            substate=to_substate,
            from_substate=from_substate,
            to_substate=to_substate,
            total_time=total_time,
            inference_time=inference_time,
            inference_tokens=inference_tokens,
            metacognition_time=metacognition_time,
            metacognition_tokens=metacognition_tokens
        )
        self.execution_log.append(log_entry)
        
        # Update current substate tracking
        self.current_substate = to_substate
        self.substate_start_time = time.time()
        self.current_substate_inference_time = 0.0
        self.current_substate_tool_time = 0.0
        
    def start_tool_execution(self, tool_name: str, arguments: Dict[str, Any], context: str) -> float:
        """Start timing a tool execution. Returns the start time for later use."""
        return time.time()
        
    def record_tool_execution(self, tool_name: str, arguments: Dict[str, Any], 
                             start_time: float, outcome: Any, context: str,
                             success: bool = True, error_message: Optional[str] = None):
        """Record the completion of a tool execution."""
        end_time = time.time()
        duration = end_time - start_time
        
        execution = ToolExecution(
            tool_name=tool_name,
            arguments=arguments,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            outcome=outcome,
            context=context,
            success=success,
            error_message=error_message
        )
        self.tool_executions.append(execution)
        
        # Add to execution log
        log_entry = ExecutionLogEntry(
            timestamp=end_time,
            entry_type="tool_execution",
            state=self.current_state,
            substate=self.current_substate,
            tool_name=tool_name,
            tool_arguments=arguments,
            tool_duration=duration,
            tool_outcome=str(outcome)[:500],  # Truncate long outcomes
            tool_context=context,
            tool_success=success,
            additional_data={"error_message": error_message} if error_message else None
        )
        self.execution_log.append(log_entry)
        
        # Update current state and substate tool time
        self.current_state_tool_time += duration
        self.current_substate_tool_time += duration
        
    def record_inference_event(self, state: str, duration: float, 
                              tokens: Optional[Dict[str, int]] = None,
                              model: Optional[str] = None,
                              substate: Optional[str] = None,
                              context_type: str = "orchestrator"):
        """Record an LLM inference event."""
        event = InferenceEvent(
            timestamp=time.time(),
            state=state,
            substate=substate,
            duration=duration,
            tokens=tokens,
            model=model,
            context_type=context_type
        )
        self.inference_events.append(event)
        
        # Add to execution log
        log_entry = ExecutionLogEntry(
            timestamp=time.time(),
            entry_type="inference",
            state=state,
            substate=substate,
            inference_time=duration,
            inference_tokens=tokens,
            additional_data={"model": model, "context_type": context_type}
        )
        self.execution_log.append(log_entry)
        
        # Update current state and substate inference time
        if context_type == "orchestrator":
            self.current_state_inference_time += duration
            self.current_substate_inference_time += duration
            
    def record_metacognition_event(self, state: str, duration: float,
                                  suggestion: Optional[str] = None,
                                  tokens: Optional[Dict[str, int]] = None):
        """Record a metacognition cycle."""
        if not self.enable_metacognition:
            return
            
        event = MetacognitionEvent(
            timestamp=time.time(),
            state=state,
            duration=duration,
            suggestion=suggestion,
            tokens=tokens
        )
        self.metacognition_events.append(event)
        
        # Add to execution log
        log_entry = ExecutionLogEntry(
            timestamp=time.time(),
            entry_type="metacognition",
            state=state,
            substate=self.current_substate,
            metacognition_time=duration,
            metacognition_tokens=tokens,
            additional_data={"suggestion": suggestion}
        )
        self.execution_log.append(log_entry)
        
    def record_graph_statistics(self, node_count: int, edge_count: int, 
                               state: Optional[str] = None, substate: Optional[str] = None):
        """Record knowledge graph statistics at current point in time."""
        graph_stats = GraphStatistics(
            timestamp=time.time(),
            state=state or self.current_state,
            substate=substate or self.current_substate,
            node_count=node_count,
            edge_count=edge_count,
            relationship_count=edge_count  # Neo4j uses 'relationship' terminology
        )
        self.graph_statistics.append(graph_stats)
        
    async def record_current_graph_statistics(self, knowledge_graph):
        """Automatically record current graph statistics from knowledge graph instance."""
        try:
            if knowledge_graph and await knowledge_graph.is_connected():
                stats = await knowledge_graph.get_graph_statistics()
                node_count = stats.get('node_count', 0)
                edge_count = stats.get('relationship_count', 0)  # Neo4j uses 'relationship_count'
                self.record_graph_statistics(node_count, edge_count)
        except Exception as e:
            # Don't fail the whole process if graph stats collection fails
            pass
        
    def finalize(self, final_answer: str, success: bool = True):
        """Finalize the statistics collection."""
        self.end_time = time.time()
        self.final_answer = final_answer
        self.success = success
        
    def get_state_statistics(self) -> Dict[str, Any]:
        """Calculate state-level statistics."""
        state_times = {}
        state_counts = {}
        state_tool_calls = {}
        state_inference_times = {}
        
        # Group by state
        for transition in self.state_transitions:
            state = transition.to_state
            if state not in state_times:
                state_times[state] = []
                state_counts[state] = 0
                state_tool_calls[state] = 0
                state_inference_times[state] = []
                
            state_counts[state] += 1
            if transition.inference_time:
                state_inference_times[state].append(transition.inference_time)
        
        # Calculate time spent in each state from transitions
        for i, transition in enumerate(self.state_transitions):
            if i < len(self.state_transitions) - 1:
                next_transition = self.state_transitions[i + 1]
                time_in_state = next_transition.timestamp - transition.timestamp
                state_times[transition.to_state].append(time_in_state)
        
        # Add final state time if available
        if self.state_transitions and self.end_time:
            last_transition = self.state_transitions[-1]
            final_time = self.end_time - last_transition.timestamp
            state_times[last_transition.to_state].append(final_time)
            
        # Count tool calls per state
        for tool in self.tool_executions:
            # Map context to likely state (this is approximate)
            if tool.context == "local":
                state = "LOCAL_GRAPH_EXPLORATION"
            elif tool.context == "remote":
                state = "REMOTE_GRAPH_EXPLORATION"
            elif tool.context == "evaluation":
                state = "EVAL_LOCAL_DATA"  # or EVAL_REMOTE_DATA
            elif tool.context == "update":
                state = "LOCAL_GRAPH_UPDATE"
            else:
                state = "UNKNOWN"
                
            if state in state_tool_calls:
                state_tool_calls[state] += 1
                
        # Calculate statistics
        stats = {}
        for state in state_times:
            times = state_times[state]
            inference_times = state_inference_times[state]
            
            stats[state] = {
                "visits": state_counts[state],
                "total_time": sum(times) if times else 0,
                "average_time": sum(times) / len(times) if times else 0,
                "tool_calls": state_tool_calls.get(state, 0),
                "total_inference_time": sum(inference_times) if inference_times else 0,
                "average_inference_time": sum(inference_times) / len(inference_times) if inference_times else 0
            }
            
        return stats
        
    def get_substate_statistics(self) -> Dict[str, Any]:
        """Calculate substate-level statistics."""
        substate_times = {}
        substate_counts = {}
        substate_tool_calls = {}
        substate_inference_times = {}
        substate_metacognition_times = {}
        
        # Group by substate
        for transition in self.substate_transitions:
            substate_key = f"{transition.parent_state}.{transition.to_substate}"
            if substate_key not in substate_times:
                substate_times[substate_key] = []
                substate_counts[substate_key] = 0
                substate_tool_calls[substate_key] = 0
                substate_inference_times[substate_key] = []
                substate_metacognition_times[substate_key] = []
                
            substate_counts[substate_key] += 1
            if transition.inference_time:
                substate_inference_times[substate_key].append(transition.inference_time)
            if transition.metacognition_time:
                substate_metacognition_times[substate_key].append(transition.metacognition_time)
        
        # Calculate time spent in each substate from transitions
        for i, transition in enumerate(self.substate_transitions):
            substate_key = f"{transition.parent_state}.{transition.to_substate}"
            if i < len(self.substate_transitions) - 1:
                next_transition = self.substate_transitions[i + 1]
                time_in_substate = next_transition.timestamp - transition.timestamp
                substate_times[substate_key].append(time_in_substate)
        
        # Add final substate time if available
        if self.substate_transitions and self.end_time:
            last_transition = self.substate_transitions[-1]
            substate_key = f"{last_transition.parent_state}.{last_transition.to_substate}"
            final_time = self.end_time - last_transition.timestamp
            substate_times[substate_key].append(final_time)
            
        # Count tool calls per substate from inference events
        for event in self.inference_events:
            if event.substate:
                substate_key = f"{event.state}.{event.substate}"
                # Count tool executions that happened during this substate
                for tool in self.tool_executions:
                    if (tool.start_time <= event.timestamp <= tool.end_time + event.duration or
                        event.timestamp <= tool.start_time <= event.timestamp + event.duration):
                        if substate_key in substate_tool_calls:
                            substate_tool_calls[substate_key] += 1
                            
        # Alternative approach: count tools by matching timestamps with substate log entries
        for log_entry in self.execution_log:
            if log_entry.entry_type == "tool_execution" and log_entry.substate:
                substate_key = f"{log_entry.state}.{log_entry.substate}"
                if substate_key in substate_tool_calls:
                    substate_tool_calls[substate_key] += 1
                elif substate_key not in substate_tool_calls:
                    # Initialize if not already present
                    substate_tool_calls[substate_key] = 1
                    substate_times.setdefault(substate_key, [])
                    substate_counts.setdefault(substate_key, 0)
                    substate_inference_times.setdefault(substate_key, [])
                    substate_metacognition_times.setdefault(substate_key, [])
                
        # Calculate statistics
        stats = {}
        for substate_key in substate_times:
            times = substate_times[substate_key]
            inference_times = substate_inference_times[substate_key]
            metacognition_times = substate_metacognition_times[substate_key]
            
            stats[substate_key] = {
                "visits": substate_counts[substate_key],
                "total_time": sum(times) if times else 0,
                "average_time": sum(times) / len(times) if times else 0,
                "tool_calls": substate_tool_calls.get(substate_key, 0),
                "total_inference_time": sum(inference_times) if inference_times else 0,
                "average_inference_time": sum(inference_times) / len(inference_times) if inference_times else 0,
                "total_metacognition_time": sum(metacognition_times) if metacognition_times else 0,
                "average_metacognition_time": sum(metacognition_times) / len(metacognition_times) if metacognition_times else 0
            }
            
        return stats
        
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Calculate tool-level statistics."""
        tool_stats = {}
        
        for tool in self.tool_executions:
            name = tool.tool_name
            if name not in tool_stats:
                tool_stats[name] = {
                    "call_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "total_time": 0,
                    "times": [],
                    "contexts": set()
                }
                
            stats = tool_stats[name]
            stats["call_count"] += 1
            stats["total_time"] += tool.duration
            stats["times"].append(tool.duration)
            stats["contexts"].add(tool.context)
            
            if tool.success:
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1
                
        # Calculate averages and convert sets to lists
        for name, stats in tool_stats.items():
            stats["average_time"] = stats["total_time"] / stats["call_count"]
            stats["success_rate"] = stats["success_count"] / stats["call_count"]
            stats["contexts"] = list(stats["contexts"])
            
        return tool_stats
        
    def get_inference_statistics(self) -> Dict[str, Any]:
        """Calculate inference-level statistics."""
        orchestrator_events = [e for e in self.inference_events if e.context_type == "orchestrator"]
        metacognition_events = [e for e in self.inference_events if e.context_type == "metacognition"]
        
        def calc_stats(events):
            if not events:
                return {
                    "count": 0,
                    "total_time": 0,
                    "average_time": 0,
                    "total_tokens": 0,
                    "average_tokens": 0
                }
                
            total_time = sum(e.duration for e in events)
            total_tokens = sum(e.tokens.get("total_tokens", 0) for e in events if e.tokens)
            
            return {
                "count": len(events),
                "total_time": total_time,
                "average_time": total_time / len(events),
                "total_tokens": total_tokens,
                "average_tokens": total_tokens / len(events) if events else 0
            }
            
        return {
            "orchestrator": calc_stats(orchestrator_events),
            "metacognition": calc_stats(metacognition_events) if self.enable_metacognition else None
        }
        
    def get_metacognition_statistics(self) -> Optional[Dict[str, Any]]:
        """Calculate metacognition-specific statistics."""
        if not self.enable_metacognition or not self.metacognition_events:
            return None
            
        total_time = sum(e.duration for e in self.metacognition_events)
        total_tokens = sum(e.tokens.get("total_tokens", 0) for e in self.metacognition_events if e.tokens)
        suggestions = [e.suggestion for e in self.metacognition_events if e.suggestion]
        
        return {
            "cycle_count": len(self.metacognition_events),
            "total_time": total_time,
            "average_time": total_time / len(self.metacognition_events),
            "total_tokens": total_tokens,
            "average_tokens": total_tokens / len(self.metacognition_events),
            "suggestion_count": len(suggestions),
            "suggestion_rate": len(suggestions) / len(self.metacognition_events)
        }
        
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Calculate knowledge graph statistics over time."""
        if not self.graph_statistics:
            return {
                "snapshots": 0,
                "initial_nodes": 0,
                "initial_edges": 0,
                "final_nodes": 0,
                "final_edges": 0,
                "nodes_added": 0,
                "edges_added": 0,
                "max_nodes": 0,
                "max_edges": 0,
                "per_state_stats": {}
            }
            
        # Sort by timestamp
        sorted_stats = sorted(self.graph_statistics, key=lambda x: x.timestamp)
        
        initial = sorted_stats[0]
        final = sorted_stats[-1]
        
        # Calculate per-state statistics
        per_state_stats = {}
        for stat in sorted_stats:
            state = stat.state
            if state not in per_state_stats:
                per_state_stats[state] = {
                    "snapshots": 0,
                    "avg_nodes": 0,
                    "avg_edges": 0,
                    "max_nodes": 0,
                    "max_edges": 0,
                    "min_nodes": float('inf'),
                    "min_edges": float('inf'),
                    "node_counts": [],
                    "edge_counts": []
                }
            
            state_stats = per_state_stats[state]
            state_stats["snapshots"] += 1
            state_stats["max_nodes"] = max(state_stats["max_nodes"], stat.node_count)
            state_stats["max_edges"] = max(state_stats["max_edges"], stat.edge_count)
            state_stats["min_nodes"] = min(state_stats["min_nodes"], stat.node_count)
            state_stats["min_edges"] = min(state_stats["min_edges"], stat.edge_count)
            state_stats["node_counts"].append(stat.node_count)
            state_stats["edge_counts"].append(stat.edge_count)
        
        # Calculate averages
        for state_stats in per_state_stats.values():
            if state_stats["node_counts"]:
                state_stats["avg_nodes"] = sum(state_stats["node_counts"]) / len(state_stats["node_counts"])
                state_stats["avg_edges"] = sum(state_stats["edge_counts"]) / len(state_stats["edge_counts"])
            
            # Clean up temporary lists
            del state_stats["node_counts"]
            del state_stats["edge_counts"]
            
            # Handle infinite values
            if state_stats["min_nodes"] == float('inf'):
                state_stats["min_nodes"] = 0
            if state_stats["min_edges"] == float('inf'):
                state_stats["min_edges"] = 0
        
        return {
            "snapshots": len(sorted_stats),
            "initial_nodes": initial.node_count,
            "initial_edges": initial.edge_count,
            "final_nodes": final.node_count,
            "final_edges": final.edge_count,
            "nodes_added": final.node_count - initial.node_count,
            "edges_added": final.edge_count - initial.edge_count,
            "max_nodes": max(s.node_count for s in sorted_stats),
            "max_edges": max(s.edge_count for s in sorted_stats),
            "per_state_stats": per_state_stats
        }
        
    def export_statistics(self) -> Dict[str, Any]:
        """Export all statistics as a dictionary."""
        total_time = (self.end_time or time.time()) - self.start_time
        
        return {
            # Basic info
            "experiment_id": self.experiment_id,
            "query": self.query,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_time": total_time,
            "success": self.success,
            "final_answer": self.final_answer,
            "enable_metacognition": self.enable_metacognition,
            
            # Attempt history
            "attempt_history": {
                "remote_explorations": self.attempt_history.remote_explorations,
                "local_explorations": self.attempt_history.local_explorations,
                "failures": self.attempt_history.failures
            },
            
            # State statistics
            "state_statistics": self.get_state_statistics(),
            
            # Substate statistics  
            "substate_statistics": self.get_substate_statistics(),
            
            # Tool statistics
            "tool_statistics": self.get_tool_statistics(),
            
            # Inference statistics
            "inference_statistics": self.get_inference_statistics(),
            
            # Metacognition statistics
            "metacognition_statistics": self.get_metacognition_statistics(),
            
            # Graph statistics 
            "graph_statistics": self.get_graph_statistics(),
            
            # Raw data for detailed analysis
            "raw_data": {
                "state_transitions": [
                    {
                        "from_state": t.from_state,
                        "to_state": t.to_state,
                        "timestamp": t.timestamp,
                        "inference_time": t.inference_time,
                        "inference_tokens": t.inference_tokens,
                        "metacognition_time": t.metacognition_time,
                        "metacognition_tokens": t.metacognition_tokens
                    }
                    for t in self.state_transitions
                ],
                "substate_transitions": [
                    {
                        "from_substate": t.from_substate,
                        "to_substate": t.to_substate,
                        "parent_state": t.parent_state,
                        "timestamp": t.timestamp,
                        "inference_time": t.inference_time,
                        "inference_tokens": t.inference_tokens,
                        "metacognition_time": t.metacognition_time,
                        "metacognition_tokens": t.metacognition_tokens
                    }
                    for t in self.substate_transitions
                ],
                "execution_log": [
                    {
                        "timestamp": e.timestamp,
                        "entry_type": e.entry_type,
                        "state": e.state,
                        "substate": e.substate,
                        "from_state": e.from_state,
                        "from_substate": e.from_substate,
                        "to_state": e.to_state,
                        "to_substate": e.to_substate,
                        "total_time": e.total_time,
                        "inference_time": e.inference_time,
                        "inference_tokens": e.inference_tokens,
                        "metacognition_time": e.metacognition_time,
                        "metacognition_tokens": e.metacognition_tokens,
                        "tool_name": e.tool_name,
                        "tool_arguments": e.tool_arguments,
                        "tool_duration": e.tool_duration,
                        "tool_outcome": e.tool_outcome,
                        "tool_context": e.tool_context,
                        "tool_success": e.tool_success,
                        "additional_data": e.additional_data
                    }
                    for e in self.execution_log
                ],
                "tool_executions": [
                    {
                        "tool_name": t.tool_name,
                        "arguments": t.arguments,
                        "start_time": t.start_time,
                        "end_time": t.end_time,
                        "duration": t.duration,
                        "outcome": str(t.outcome)[:500],  # Truncate long outcomes
                        "context": t.context,
                        "success": t.success,
                        "error_message": t.error_message
                    }
                    for t in self.tool_executions
                ],
                "inference_events": [
                    {
                        "timestamp": e.timestamp,
                        "state": e.state,
                        "substate": e.substate,
                        "duration": e.duration,
                        "tokens": e.tokens,
                        "model": e.model,
                        "context_type": e.context_type
                    }
                    for e in self.inference_events
                ],
                "metacognition_events": [
                    {
                        "timestamp": e.timestamp,
                        "state": e.state,
                        "duration": e.duration,
                        "suggestion": e.suggestion,
                        "tokens": e.tokens
                    }
                    for e in self.metacognition_events
                ] if self.enable_metacognition else [],
                "graph_statistics": [
                    {
                        "timestamp": g.timestamp,
                        "state": g.state,
                        "substate": g.substate,
                        "node_count": g.node_count,
                        "edge_count": g.edge_count,
                        "relationship_count": g.relationship_count
                    }
                    for g in self.graph_statistics
                ]
            }
        }
        
    def save_to_file(self, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Save statistics to a JSON file."""
        if output_dir is None:
            output_dir = Path("experiments") / self.experiment_id
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"statistics_{timestamp}.json"
        filepath = output_dir / filename
        
        # Export and save
        stats = self.export_statistics()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
            
        return filepath


def create_statistics_collector(experiment_id: str, query: str, 
                              enable_metacognition: bool = False) -> OrchestratorStatistics:
    """Factory function to create a statistics collector."""
    return OrchestratorStatistics(experiment_id, query, enable_metacognition)


if __name__ == "__main__":
    # Example usage for testing
    stats = create_statistics_collector("test_experiment", "What is climate change?")
    
    # Simulate some activity
    stats.record_state_transition("READY", "LOCAL_GRAPH_EXPLORATION", inference_time=1.5)
    
    start_time = stats.start_tool_execution("cypher_query", {"query": "MATCH (n) RETURN n"}, "local")
    time.sleep(0.1)  # Simulate tool execution
    stats.record_tool_execution("cypher_query", {"query": "MATCH (n) RETURN n"}, 
                                start_time, {"results": []}, "local", success=True)
    
    stats.record_inference_event("LOCAL_GRAPH_EXPLORATION", 2.0, 
                                 tokens={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
    
    stats.finalize("Climate change is...", success=True)
    
    # Print statistics
    print(json.dumps(stats.export_statistics(), indent=2, default=str))
