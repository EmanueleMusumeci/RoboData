from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel

class WikidataStatement(BaseModel):
    """Simplified statement representation for LLM consumption."""
    property_id: str
    value: Union[str, int, float]  # The actual value
    datatype: str
    is_entity_ref: bool = False  # True if value is an entity/property ID, False for literals
    entity_type: Optional[str] = None  # "item", "property", or None for literals

class WikidataEntity(BaseModel):
    """High-level Wikidata entity representation optimized for LLM function calling."""
    id: str
    label: str  # Primary label (usually English)
    description: Optional[str] = None  # Primary description
    aliases: List[str] = []  # Flat list of aliases
    statements: Dict[str, List[WikidataStatement]] = {}  # property_id -> list of statements
    link: str

class WikidataProperty(BaseModel):
    """High-level Wikidata property representation optimized for LLM function calling."""
    id: str
    label: str  # Primary label (usually English)
    description: Optional[str] = None  # Primary description
    datatype: str
    link: str

class SearchResult(BaseModel):
    """Simplified search result for LLM consumption."""
    id: str
    label: str
    description: Optional[str] = None
    url: Optional[str] = None

class NeighborExplorationResult(BaseModel):
    """Result from exploring an entity's neighbors."""
    entity: WikidataEntity
    neighbors: Dict[str, WikidataEntity]  # entity_id -> WikidataEntity
    relationships: Dict[str, List[str]]  # property_id -> list of entity_ids
    property_names: Dict[str, str]  # property_id -> property_name
    total_properties: int
    neighbor_count: int
    limit: int
    order_by_degree: bool

class LocalGraphResult(BaseModel):
    """Result from building a local graph around an entity."""
    nodes: Dict[str, WikidataEntity]  # entity_id -> WikidataEntity
    edges: List[Dict[str, str]]  # list of edge dictionaries
    center: str  # center entity ID
    depth: int
    properties: List[str]  # properties that were followed
    property_names: Dict[str, str]  # property_id -> property_name
    total_nodes: int
    total_edges: int
    limit: int
    order_by_degree: bool

def convert_api_entity_to_model(api_data: Dict[str, Any]) -> WikidataEntity:
    """Convert wikidata API response to WikidataEntity model."""
    # Get primary label (prefer English)
    labels = api_data.get("labels", {})
    label = labels.get("en", list(labels.values())[0] if labels else api_data["id"])
    
    # Get primary description (prefer English)
    descriptions = api_data.get("descriptions", {})
    description = descriptions.get("en", list(descriptions.values())[0] if descriptions else None)
    
    # Flatten aliases from all languages
    aliases = []
    for lang_aliases in api_data.get("aliases", {}).values():
        aliases.extend(lang_aliases)
    
    # Convert statements from API format to simplified format
    statements = {}
    api_statements = api_data.get("statements", {})
    
    for prop_id, prop_claims in api_statements.items():
        statements[prop_id] = []
        for claim_key, claim_data in prop_claims.items():
            datavalue = claim_data.get("datavalue")
            is_entity_ref = False
            entity_type = None
            value = claim_key  # Use the key as the value
            
            if datavalue and datavalue.get("type") == "wikibase-entityid":
                is_entity_ref = True
                if value.startswith("Q"):
                    entity_type = "item"
                elif value.startswith("P"):
                    entity_type = "property"
                else:
                    entity_type = "unknown_entity"
            
            statement = WikidataStatement(
                property_id=claim_data["property_id"],
                value=value,
                datatype=claim_data["datatype"],
                is_entity_ref=is_entity_ref,
                entity_type=entity_type
            )
            statements[prop_id].append(statement)
    
    return WikidataEntity(
        id=api_data["id"],
        label=label,
        description=description,
        aliases=aliases,
        statements=statements,
        link=api_data["link"]
    )

def convert_api_property_to_model(api_data: Dict[str, Any]) -> WikidataProperty:
    """Convert wikidata API property response to WikidataProperty model."""
    # Get primary label (prefer English)
    labels = api_data.get("labels", {})
    label = labels.get("en", list(labels.values())[0] if labels else api_data["id"])
    
    # Get primary description (prefer English)
    descriptions = api_data.get("descriptions", {})
    description = descriptions.get("en", list(descriptions.values())[0] if descriptions else None)
    
    return WikidataProperty(
        id=api_data["id"],
        label=label,
        description=description,
        datatype=api_data["datatype"],
        link=api_data["link"]
    )

def convert_api_search_to_model(api_data: List[Dict[str, Any]]) -> List[SearchResult]:
    """Convert wikidata API search response to SearchResult models."""
    results = []
    for item in api_data:
        results.append(SearchResult(
            id=item["id"],
            label=item.get("label", item["id"]),
            description=item.get("description"),
            url=item.get("url")
        ))
    return results
