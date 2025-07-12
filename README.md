# RoboData - Ontology Explorer

![RoboData Visualization](images/RoboData_light.png)

## About This Project

This project was developed as part of the [WikiProject Ontology Course](https://www.wikidata.org/w/index.php?title=Wikidata:WikiProject_Ontology/Ontology_Course) on Wikidata. It represents our work for the [RoboData project](https://www.wikidata.org/w/index.php?title=Wikidata:WikiProject_Ontology/Ontology_Course/Projects/RoboData), exploring ontology visualization and interaction using Wikidata.

---

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
│   │       └── graph.py          # Neo4j database abstraction
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
   export GEMINI_API_KEY="your-gemini-api-key"
   # OR create a .env file with:
   # GEMINI_API_KEY=your-gemini-api-key
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
   Edit `config.yaml` to customize behavior

## 🎯 Usage

### Interactive Terminal

Start the interactive terminal session:

```bash
cd backend
python main.py
```

### Example Queries

```
🎤 You: What are the subclasses of Q35120?
🎤 You: Explore entity Q5
🎤 You: Find instances of Q16521
🎤 You: Build a local graph around Q1
🎤 You: Find path between Q5 and Q35120
```

### Available Commands

- `help` - Show help information and query examples
- `tools` - List all available tools with descriptions
- `settings` - Show current configuration
- `clear` - Clear conversation history
- `exit`/`quit` - Exit the application

## 🔧 Tools

The system includes comprehensive tool sets for different domains:

### Wikidata Query Tools
- **SPARQLQueryTool**: Execute custom SPARQL queries with result formatting
- **SubclassQueryTool**: Find subclasses of an entity with hierarchy depth
- **SuperclassQueryTool**: Find superclasses and parent classifications
- **InstanceQueryTool**: Find instances of a class with filtering options

### Exploration Tools
- **NeighborsExplorationTool**: Deep dive into entity properties and relationships
- **LocalGraphTool**: Build neighborhood graphs with configurable depth

### Graph Database Tools (Optional)
- **Add/RemoveNodeTool**: Add nodes to local graph database
- **Add/RemoveEdgeTool**: Create relationships between nodes
- **QueryGraphTool**: Execute Cypher queries on local graph
- **GetNeighborsTool**: Find neighboring nodes with relationship filtering


## ⚙️ Configuration

Configure the system via `config.yaml`:

```yaml
llm:
  provider: "gemini"
  model: "gemini-pro"
  temperature: 0.7
  api_key: null  # Set via environment variable

wikidata:
  timeout: 30
  max_results: 100
  default_language: "en"

toolbox:
  auto_register_wikidata_tools: true
  max_tool_execution_time: 60

interactive:
  show_tool_calls: true
  show_intermediate_steps: true
  max_history_length: 100
```

## 🏗️ Architecture

### Core Components

1. **BaseAgent**: Abstract interface for LLM agents with tool calling capabilities
2. **Toolbox**: Dynamic tool registration and management with validation
3. **Data Models**: Strongly typed Pydantic models for all Wikidata structures
4. **API Abstraction**: Multiple backend support (REST, KIF) with consistent interface
5. **Settings Management**: YAML configuration with environment variable overrides


### API Backends

- **REST API**: Using the `wikidata` Python library for standard operations
- **KIF API**: IBM's Knowledge Integration Framework for advanced queries
- **SPARQL**: Direct SPARQL endpoint access for custom queries

## 📊 Data Models

The system uses strongly typed data models for all Wikidata structures:

```python
# Example entity model
WikidataEntity(
    id="Q42",
    label="Douglas Adams",
    description="British author and humorist",
    aliases=["Douglas Noel Adams", "DNA"],
    statements={
        "P31": [WikidataStatement(...)],  # instance of
        "P106": [WikidataStatement(...)]  # occupation
    }
)
```

Models include:
- **WikidataEntity**: Complete entity representation
- **WikidataProperty**: Property definitions and constraints
- **WikidataStatement**: Individual claims with qualifiers
- **SearchResult**: Search result metadata

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests for new functionality
4. Ensure all tests pass with real API calls
5. Update documentation as needed
6. Submit a pull request

### Development Guidelines
- Follow the existing tool pattern for new tools
- Add both unit and integration tests
- Use type hints and Pydantic models
- Include docstrings with examples

## 📚 Documentation

- **Tool Development**: See `core/toolbox/toolbox.py` for the tool interface
- **Data Models**: Check `core/toolbox/wikidata/datamodel.py` for schemas
- **API Usage**: Examples in test files demonstrate real usage patterns
- **Configuration**: All settings are documented in `settings.py`

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **WikiProject Ontology**: For providing the educational framework
- **Wikidata Community**: For maintaining the knowledge base
- **IBM KIF**: For the Knowledge Integration Framework
