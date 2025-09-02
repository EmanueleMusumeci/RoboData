from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from ...knowledge_base.schema import Graph as VisGraph, Node as VisNode, Edge as VisEdge, LiteralNode

@dataclass
class DBpediaStatement:
    """Represents a DBpedia statement (property-value pair)."""
    property_id: str
    value: Any
    datatype: str
    display_value: Optional[str] = None
    qualifiers: Optional[Dict[str, Any]] = None

@dataclass
class DBpediaEntity:
    """Represents a DBpedia entity with structured data."""
    id: str
    uri: str
    label: str
    description: str
    aliases: List[str]
    statements: Dict[str, Dict[str, DBpediaStatement]]
    link: str

@dataclass
class DBpediaProperty:
    """Represents a DBpedia property."""
    id: str
    uri: str
    label: str
    description: str
    aliases: List[str]
    datatype: str
    link: str

@dataclass
class SearchResult:
    """Represents a search result from DBpedia."""
    id: str
    uri: str
    label: str
    description: str
    categories: List[str]
    classes: List[str]

@dataclass
class NeighborExplorationResult:
    """Result from neighbor exploration in DBpedia."""
    entity: DBpediaEntity
    neighbors: Dict[str, DBpediaEntity]
    property_names: Dict[str, str]
    total_neighbors: int
    displayed_neighbors: int
    errors: List[str]

@dataclass
class LocalGraphResult:
    """Result from local graph exploration in DBpedia."""
    entities: Dict[str, DBpediaEntity]
    graph: VisGraph
    property_names: Dict[str, str]
    total_nodes: int
    total_edges: int
    errors: List[str]

def convert_api_entity_to_model(api_data: Dict[str, Any]) -> DBpediaEntity:
    """Convert API entity data to DBpediaEntity model."""
    statements = {}
    for prop_id, prop_statements in api_data.get('statements', {}).items():
        statements[prop_id] = {}
        for value_key, statement_data in prop_statements.items():
            statements[prop_id][value_key] = DBpediaStatement(
                property_id=statement_data.get('property_id', prop_id),
                value=statement_data.get('datavalue', {}).get('value'),
                datatype=statement_data.get('datatype', 'string'),
                display_value=statement_data.get('display_value'),
                qualifiers=statement_data.get('qualifiers', {})
            )
    
    return DBpediaEntity(
        id=api_data['id'],
        uri=api_data.get('uri', f"http://dbpedia.org/resource/{api_data['id']}"),
        label=api_data.get('labels', {}).get('en', api_data['id']),
        description=api_data.get('descriptions', {}).get('en', ''),
        aliases=api_data.get('aliases', {}).get('en', []),
        statements=statements,
        link=api_data.get('link', f"http://dbpedia.org/resource/{api_data['id']}")
    )

def convert_api_property_to_model(api_data: Dict[str, Any]) -> DBpediaProperty:
    """Convert API property data to DBpediaProperty model."""
    return DBpediaProperty(
        id=api_data['id'],
        uri=api_data.get('uri', f"http://dbpedia.org/ontology/{api_data['id']}"),
        label=api_data.get('labels', {}).get('en', api_data['id']),
        description=api_data.get('descriptions', {}).get('en', ''),
        aliases=api_data.get('aliases', {}).get('en', []),
        datatype=api_data.get('datatype', 'unknown'),
        link=api_data.get('link', f"http://dbpedia.org/ontology/{api_data['id']}")
    )

def convert_api_search_to_model(api_data: List[Dict[str, Any]]) -> List[SearchResult]:
    """Convert API search results to SearchResult models."""
    results = []
    for item in api_data:
        results.append(SearchResult(
            id=item.get('id', ''),
            uri=item.get('uri', ''),
            label=item.get('label', ''),
            description=item.get('description', ''),
            categories=item.get('categories', []),
            classes=item.get('classes', [])
        ))
    return results
