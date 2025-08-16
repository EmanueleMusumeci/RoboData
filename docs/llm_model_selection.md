# LLM Model Selection Configuration

This document explains how to configure separate LLM models for different operations in the RoboData system.

## Overview

The RoboData system now supports selecting different LLM models for different types of operations:

- **Metacognition Model**: Used for strategic assessment and meta-observation
- **Evaluation Model**: Used for data evaluation and answer production  
- **Exploration Model**: Used for local and remote graph exploration
- **Update Model**: Used for knowledge graph updates
- **Default Model**: Fallback model for other operations

## Configuration

### In Configuration Files

Add an `llm` section to your configuration YAML file:

```yaml
llm:
  provider: "openai"
  model: "gpt-4o"  # Default/fallback model
  temperature: 0.7
  max_tokens: 4096
  # Specific models for different operations
  metacognition_model: "gpt-5"  # Strategic assessment and meta-observation
  evaluation_model: "gpt-5"     # Data evaluation and answer production
  exploration_model: "gpt-4o"   # Local and remote graph exploration
  update_model: "gpt-4o"        # Knowledge graph updates
```

### Environment Variables

You can also override these settings using environment variables:

```bash
export LLM_MODEL="gpt-4o"                    # Default model
export LLM_METACOGNITION_MODEL="gpt-5"      # Metacognition model
export LLM_EVALUATION_MODEL="gpt-5"         # Evaluation model
export LLM_EXPLORATION_MODEL="gpt-4o"       # Exploration model
export LLM_UPDATE_MODEL="gpt-4o"            # Update model
```

## Default Values

If not specified, the system uses these defaults:

- **Default Model**: `"openai-pro"`
- **Metacognition Model**: `"gpt-5"`
- **Evaluation Model**: `"gpt-5"`
- **Exploration Model**: `"gpt-4o"`
- **Update Model**: `"gpt-4o"`

## Operation Types

### Metacognition
- **Purpose**: Strategic assessment and meta-observation
- **Default Model**: GPT-5 (for better reasoning about strategies)
- **Usage**: Analyzing agent behavior and suggesting improvements

### Evaluation
- **Purpose**: Data evaluation and answer production
- **Default Model**: GPT-5 (for better evaluation and reasoning)
- **Usage**: Assessing local/remote data quality and producing final answers

### Exploration
- **Purpose**: Local and remote graph exploration
- **Default Model**: GPT-4o (cost-effective for tool usage)
- **Usage**: Querying knowledge graphs and exploring data sources

### Update
- **Purpose**: Knowledge graph updates
- **Default Model**: GPT-4o (cost-effective for structured data operations)
- **Usage**: Adding/updating nodes and relationships in the knowledge graph

## Example Configurations

### High-Performance Setup
Use GPT-5 for all critical reasoning tasks:

```yaml
llm:
  model: "gpt-4o"
  metacognition_model: "gpt-5"
  evaluation_model: "gpt-5" 
  exploration_model: "gpt-5"
  update_model: "gpt-4o"
```

### Cost-Optimized Setup
Use GPT-4o for most operations:

```yaml
llm:
  model: "gpt-4o"
  metacognition_model: "gpt-4o"
  evaluation_model: "gpt-4o"
  exploration_model: "gpt-4o"
  update_model: "gpt-4o"
```

### Balanced Setup (Recommended)
Use GPT-5 for reasoning, GPT-4o for tool usage:

```yaml
llm:
  model: "gpt-4o"
  metacognition_model: "gpt-5"
  evaluation_model: "gpt-5"
  exploration_model: "gpt-4o" 
  update_model: "gpt-4o"
```

## Integration

The model selection is automatically integrated into:

1. **MultiStageOrchestrator**: Uses appropriate models for each orchestration phase
2. **Metacognition Module**: Uses metacognition model for strategic reasoning
3. **Agent Classes**: Support model override parameter in query_llm method

## Verification

You can test your configuration using the included test script:

```bash
python test_model_selection.py
```

This will verify that your LLM settings are loaded correctly and show which model will be used for each operation type.
