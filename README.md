# RoboData - Ontology Explorer

![RoboData Visualization](robo-data-visualization.png) <!-- Add your image here -->

## About This Project

This project was developed as part of the [WikiProject Ontology Course](https://www.wikidata.org/w/index.php?title=Wikidata:WikiProject_Ontology/Ontology_Course) on Wikidata. It represents our work for the [RoboData project](https://www.wikidata.org/w/index.php?title=Wikidata:WikiProject_Ontology/Ontology_Course/Projects/RoboData), exploring ontology visualization and interaction using Wikidata.

---

An interactive ontology explorer that uses LLM agents to navigate and query Wikidata entities, with a graphical visualization interface.

## Project Overview

An interactive ontology explorer that uses LLM agents to navigate and query Wikidata entities, with a graphical visualization interface.

## Project Structure

```
robo_data/
├── app/
│   ├── api/           # FastAPI backend routes
│   ├── core/          # Core modules (LLM Agent, Orchestrator)
│   └── frontend/      # React + d3.js frontend
├── requirements.txt   # Python dependencies
└── README.md         # Project documentation
```

## Features

- Interactive graphical ontology visualization
- LLM-powered entity exploration
- Natural language query interface
- Modular architecture for future robot integration
- Resizable panes interface (menu, chat, graph)
- Wikidata entity exploration capabilities
- Headless operation capability

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
```

3. Start the application:
```bash
# Start Hypercorn server
hypercorn app.main:app --bind 0.0.0.0:8000

# In another terminal, start the frontend
npm start
```

## Docker Setup

1. Build the Docker image:
```bash
docker build -t robo-data .
```

2. Run the Docker container:
```bash
docker run -d -p 8000:8000 -p 3000:3000 robo-data
```

## Components

### LLM Agent
- Processes natural language queries
- Calls appropriate tools for entity exploration
- Interfaces with Wikidata API

### Orchestrator
- Manages component communication
- Coordinates LLM agent and visualization
- Handles state management

### Frontend
- React + d3.js interface
- Three-pane layout (menu, chat, graph)
- Interactive graph visualization
- Tooltips with entity information
- Headless API endpoints for programmatic access

## Frontend Architecture

The frontend is built with:
- React for UI components
- d3.js for graph visualization
- Material-UI for layout and styling
- Recharts for additional visualization components
- Axios for API communication

The UI features:
- Resizable panes using CSS Grid
- Interactive graph with drag-and-drop support
- Real-time updates through WebSocket
- Tooltips with entity information
- Clean, modern Material-UI design
    