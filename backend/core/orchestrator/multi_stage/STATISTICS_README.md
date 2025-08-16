# RoboData Multi-Stage Orchestrator Statistics System

## Overview

The RoboData multi-stage orchestrator now includes comprehensive statistics collection that provides detailed insights into experimental performance and behavior. This system replaces the simple `AttemptHistory` class with a sophisticated `OrchestratorStatistics` class that tracks:

## Core Statistics Collected

### 1. State Transition History
- **State transitions**: Complete trace of state changes with timestamps
- **Time spent in each state**: Total and average time per state
- **State visit counts**: How many times each state was visited
- **Inference time per state**: LLM inference timing within each state

### 2. Tool Execution Statistics
- **Tool call frequency**: Number of calls per tool across all contexts
- **Tool execution timing**: Start time, end time, and duration for each call
- **Tool success/failure rates**: Success rate and error tracking per tool
- **Tool arguments and outcomes**: Complete record of inputs and outputs
- **Context-aware tracking**: Tools tracked separately by context (local, remote, evaluation, update)

### 3. LLM Inference Tracking
- **Inference timing**: Duration of each LLM call
- **Token usage**: Prompt tokens, completion tokens, total tokens per call
- **Model information**: Which model was used for each call
- **Context separation**: Separate tracking for orchestrator vs metacognition calls

### 4. Metacognition Statistics (when enabled)
- **Metacognitive cycles**: Number and timing of metacognitive assessments
- **Suggestion generation**: Rate of actionable suggestions produced
- **Token usage**: Separate token tracking for metacognitive processes
- **Integration timing**: Time spent in metacognitive evaluation

### 5. Performance Metrics
- **Total execution time**: End-to-end query processing time
- **Turn efficiency**: Average time per orchestrator turn
- **Exploration efficiency**: Ratio of successful to failed explorations
- **Answer quality indicators**: Success/failure classification

## File Structure

### `statistics.py`
- **`AttemptHistory`**: Moved from prompts.py, tracks exploration attempts and failures
- **`OrchestratorStatistics`**: Main statistics collection class
- **Data classes**: Structured representations for transitions, tool executions, and inference events
- **Export functionality**: JSON serialization with comprehensive data export
- **File management**: Automatic saving with timestamped filenames

### `aggregate_statistics.py`
- **Multi-experiment aggregation**: Combines statistics from multiple runs
- **Statistical analysis**: Mean, median, standard deviation calculations
- **Comparative metrics**: Success rates, efficiency comparisons
- **Report generation**: Comprehensive experimental evaluation reports

### Integration with `multi_stage_orchestrator.py`
- **Seamless integration**: Minimal changes to existing orchestrator logic
- **Automatic collection**: Statistics collected transparently during execution
- **Performance optimization**: Low-overhead data collection
- **Error resilience**: Statistics collection continues even if individual components fail

## Usage Examples

### Single Query Analysis
```python
# Statistics are automatically collected during query processing
orchestrator = MultiStageOrchestrator(...)
result = await orchestrator.process_user_query("What is climate change?")

# Statistics file path is included in the result
stats_file = result["statistics_file"]
```

### Multi-Query Experimental Evaluation
```python
# Run multiple queries
queries = ["Query 1", "Query 2", "Query 3"]
for query in queries:
    result = await orchestrator.process_user_query(query)
    # Each query gets its own statistics file

# Aggregate results
from backend.core.orchestrator.multi_stage.aggregate_statistics import main
# Run: python aggregate_statistics.py -p "statistics_*.json" -o experiment_results.json
```

### Programmatic Analysis
```python
from backend.core.orchestrator.multi_stage.aggregate_statistics import StatisticsAggregator

aggregator = StatisticsAggregator()
aggregator.load_statistics_files("statistics_*.json", "experiments/")
report = aggregator.generate_aggregated_report()

print(f"Success rate: {report['basic_metrics']['success_rate']:.2%}")
print(f"Average execution time: {report['basic_metrics']['total_time']['mean']:.2f}s")
```

## Statistics File Format

Each query generates a JSON file with the following structure:

```json
{
  "experiment_id": "query_20250731_143022",
  "query": "What is climate change?",
  "total_time": 45.23,
  "success": true,
  "attempt_history": {
    "remote_explorations": 2,
    "local_explorations": 3,
    "failures": []
  },
  "state_statistics": {
    "LOCAL_GRAPH_EXPLORATION": {
      "visits": 3,
      "total_time": 12.45,
      "tool_calls": 8,
      "total_inference_time": 3.21
    }
  },
  "tool_statistics": {
    "cypher_query": {
      "call_count": 15,
      "success_rate": 0.93,
      "average_time": 0.45,
      "contexts": ["local", "remote"]
    }
  },
  "inference_statistics": {
    "orchestrator": {
      "count": 12,
      "total_time": 8.34,
      "total_tokens": 2450
    }
  },
  "raw_data": {
    // Complete detailed data for advanced analysis
  }
}
```

## Experimental Evaluation Benefits

### 1. Performance Analysis
- **Bottleneck identification**: Which states/tools consume the most time
- **Efficiency trends**: How performance changes across different query types
- **Resource utilization**: Token usage patterns and costs

### 2. Behavior Analysis
- **Exploration patterns**: How the orchestrator navigates the solution space
- **Tool usage patterns**: Which tools are most/least effective
- **Failure analysis**: Common failure modes and their contexts

### 3. Comparative Studies
- **Configuration comparison**: Impact of different settings (metacognition, decomposition)
- **Model comparison**: Performance differences between different LLMs
- **Scalability analysis**: How performance changes with complexity

### 4. Research Applications
- **Reproducible experiments**: Complete state tracking for research reproducibility
- **Hypothesis testing**: Statistical validation of orchestrator improvements
- **Publication-ready metrics**: Comprehensive data for academic papers

## Additional Statistics Suggestions for Experimental Evaluation

### Advanced Metrics
1. **Query Complexity Indicators**
   - Query length and linguistic complexity
   - Number of entities/concepts mentioned
   - Semantic similarity to training data

2. **Knowledge Graph Evolution**
   - Graph size before/after processing
   - New entities/relationships added
   - Graph density changes

3. **Memory System Analysis**
   - Memory utilization patterns
   - Summary generation frequency (if using SummaryMemory)
   - Information retention effectiveness

4. **Tool Effectiveness Metrics**
   - Information gain per tool call
   - Tool call redundancy detection
   - Context-specific tool performance

5. **Error Recovery Analysis**
   - Recovery time from failures
   - Success rate after errors
   - Error propagation patterns

6. **User Experience Metrics**
   - Time to first meaningful result
   - Answer completeness scores
   - Response relevance ratings

### Integration Recommendations
- **Real-time dashboards**: Live monitoring during experiments
- **Automated alerts**: Notification of unusual patterns or failures
- **A/B testing framework**: Built-in support for comparative experiments
- **Visualization tools**: Graphs and charts for pattern recognition

This comprehensive statistics system provides researchers and developers with unprecedented visibility into the multi-stage orchestrator's behavior, enabling data-driven optimization and thorough experimental validation.
