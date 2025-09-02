# RoboData - Ontology Explorer

![RoboData Visualization](images/architecture.png)

## About This Project

An interactive ontology explorer that uses LLM agents to navigate and query Wikidata entities, with a graphical visualization interface and comprehensive testing infrastructure.

## Project Overview

RoboData provides an intelligent interface for exploring Wikidata ontologies through natural language queries. The system combines LLM agents with specialized tools to enable intuitive navigation of knowledge graphs, entity relationships, and semantic hierarchies.

## Project Structure

```
RoboData/
├── backend/
│   ├── core/
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py          # Abstract base agent
│   │   │   └── gemini.py         # Gemini implementation with tool calling
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   └── memory.py         # Conversation and context memory
│   │   ├── orchestrator/
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py   # Main query orchestrator
│   │   │   └── multi_stage/
│   │   │       └── __init__.py   # Multi-stage query planning
│   │   ├── toolbox/
│   │   │   ├── __init__.py
│   │   │   ├── toolbox.py        # Dynamic tool management system
│   │   │   ├── graph/
│   │   │   │   ├── __init__.py
│   │   │   │   └── graph_tools.py # Neo4j graph database tools
│   │   │   └── wikidata/
│   │   │       ├── __init__.py
│   │   │       ├── base.py           # High-level Wikidata functions
│   │   │       ├── datamodel.py      # Pydantic data models
│   │   │       ├── wikidata_api.py   # REST API wrapper
│   │   │       ├── wikidata_kif_api.py # KIF API implementation
│   │   │       ├── queries.py        # SPARQL query tools
│   │   │       └── exploration.py    # Graph exploration tools
│   │   └── knowledge_base/
│   │       ├── __init__.py
│   │       ├── graph.py          # Neo4j database abstraction
│   │       ├── schema.py         # Local graph data models (Node, Edge, Graph)
│   │       └── interfaces/
│   │           ├── __init__.py
│   │           └── neo4j_interface.py # Neo4j connection interface
│   ├── test/
│   │   ├── __init__.py
│   │   ├── test_tools.py         # Comprehensive tool tests
│   │   ├── test_datamodel.py     # Data model validation tests
│   │   ├── test_base_tools.py    # Base function integration tests
│   │   ├── test_api.py           # REST API tests
│   │   ├── test_api_kif.py       # KIF API tests
│   │   └── run_tests.py          # Test runner with real API calls
│   ├── settings.py               # Configuration management
│   └── main.py                   # Interactive terminal application
├── config.yaml                   # Configuration file
├── requirements.txt               # Python dependencies
├── frontend/                     # React frontend (future)
└── README.md
```

## 🚀 Features

### Core Capabilities
- **Natural Language Queries**: Ask questions about Wikidata entities in plain English
- **Dynamic Tool System**: Extensible architecture with automatic tool registration
- **Multiple API Backends**: Support for both REST API and KIF (Knowledge Integration Framework)
- **Graph Database Integration**: Neo4j support for local knowledge graph storage
- **Comprehensive Testing**: Real API integration tests with validation

### Query Types
- Entity exploration and detailed information retrieval
- Subclass/superclass hierarchy navigation
- Instance finding and classification
- Path finding between entities
- Local graph construction around entities
- Custom SPARQL query execution

### Data Models
- Strongly typed Pydantic models for all Wikidata entities
- Automatic conversion between API formats and internal models
- Entity reference detection and validation
- Support for multilingual labels and descriptions

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RoboData
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   # OR create a .env file with:
   # OPENAI_API_KEY=your-openai-api-key
   ```

4. **Set up Neo4j** (for graph database features):
   
   **Install Neo4j on your system:**
   Following the [official install guide](https://neo4j.com/docs/operations-manual/current/installation/linux/)
   
   **Verify Installation:**
   Check if Neo4j is running
   ```
   sudo systemctl status neo4j
   ```

   Access Neo4j Browser (optional)
   Open http://localhost:7474 in your browser
   Login with username: neo4j, password: neo4j
   
   You will be prompted to change the password

   Test connection with cypher-shell
   ```
   cypher-shell -u neo4j -p robodata123
   ```
   
   **Update config.yaml with your Neo4j credentials:**
   ```yaml
   neo4j:
     uri: "bolt://localhost:7687"
     username: "neo4j"
     password: "robodata123"
     database: "neo4j"
   ```

5. **Configure settings** (optional):
   Edit `default_config.yaml` to customize behavior. Key configuration sections include:

   **Orchestrator Settings:**
   - `orchestrator.type`: Orchestrator type (currently "multi_stage")
   - `orchestrator.context_length`: Maximum context window for LLM (default: 16000)
   - `orchestrator.max_turns`: Maximum conversation turns (default: 30)
   - `orchestrator.enable_question_decomposition`: Break complex queries into sub-questions
   - `orchestrator.enable_metacognition`: Enable strategic assessment and meta-observation
   - `orchestrator.memory.use_summary_memory`: Enable conversation memory summarization
   - `orchestrator.memory.max_memory_slots`: Maximum memory slots to retain (default: 30)

   **LLM Configuration:**
   - `llm.provider`: LLM provider ("openai")
   - `llm.model`: Default model ("gpt-4o")
   - `llm.temperature`: Creativity level (0.0-1.0, default: 0.7)
   - `llm.max_tokens`: Maximum response tokens (default: 4096)
   - `llm.metacognition_model`: Model for strategic assessment
   - `llm.evaluation_model`: Model for data evaluation
   - `llm.exploration_model`: Model for graph exploration
   - `llm.update_model`: Model for knowledge graph updates

   **Output Settings:**
   - `output.save_results`: Save experiment results to files
   - `output.export_formats`: Export formats ["json", "cypher"]
   - `output.create_visualizations`: Generate graph visualizations
   - `output.results_directory`: Directory for saving results (default: "experiments")

   **Query Configuration:**
   - `queries`: List of queries to process sequentially
   - `experiment_id`: Unique identifier for the experiment
   - `dataset.path`: Path to dataset file for batch processing
   - `dataset.type`: Dataset format ("auto", "json", "jsonl", "lcquad") 

## 🎯 Usage

To run the ALL experiments in the paper, run the `run_experiments.sh` script.



Otherwise, to run inference with a custom query, use the following command-line commands:

### Basic Query Execution

**Simple query without metacognition:**
```bash
. .venv/bin/activate && python -m backend.main -q "Who was Albert Einstein?"
```

**Query with metacognition (strategic assessment and meta-observation):**
```bash
. .venv/bin/activate && python -m backend.main -q "What are the connections between Tesla and Edison?" --enable-metacognition
```

### Configuration-Based Execution

**Using a custom configuration file:**
```bash
. .venv/bin/activate && python -m backend.main -c experiment_configs/QA01_single_hop_hitchhikers_guide.yaml
```

**Override query in configuration file:**
```bash
. .venv/bin/activate && python -m backend.main -c default_config.yaml -q "What is quantum mechanics?"
```

### Additional Options

**Skip saving results to files:**
```bash
. .venv/bin/activate && python -m backend.main -q "What is machine learning?" --no-save
```

### Interactive Terminal

**Start interactive mode (simple terminal interface):**
```bash
. .venv/bin/activate && python -m backend.main --interactive
```

**Interactive mode with question decomposition:**
```bash
. .venv/bin/activate && python -m backend.main --interactive --enable-question-decomposition
```

**Interactive mode with metacognition:**
```bash
. .venv/bin/activate && python -m backend.main --interactive --enable-metacognition
```

### Command-Line Reference

| Argument | Description |
|----------|-------------|
| `-q, --query` | Natural language query to process |
| `-c, --config` | Path to experiment configuration YAML file |
| `--no-save` | Skip saving results to files |
| `--interactive` | Run in interactive mode |
| `--enable-metacognition` | Enable strategic assessment and meta-observation |
