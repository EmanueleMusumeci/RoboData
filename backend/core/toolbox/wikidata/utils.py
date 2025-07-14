from typing import List, Dict, Any

from .datamodel import WikidataStatement, LocalGraphResult

def get_property_numeric_id(prop_id: str) -> float:
    """Extracts the numeric part of a Wikidata property ID."""
    try:
        return int(prop_id[1:])
    except (ValueError, IndexError):
        return float('inf')

def order_properties_by_degree(statements: Dict[str, List[WikidataStatement]], enabled: bool = False) -> List[str]:
    """
    Orders property IDs based on degree (number of statements) and then by property ID numerically.
    If degree ordering is disabled, it sorts by property ID only.
    """
    if not statements:
        return []

    if enabled:
        def sort_key(prop_id: str):
            degree = len(statements.get(prop_id, []))
            numeric_id = get_property_numeric_id(prop_id)
            return -degree, numeric_id
        return sorted(statements.keys(), key=sort_key)
    else:
        # Sort by numeric property ID only
        return sorted(statements.keys(), key=get_property_numeric_id)

def order_local_graph_edges(result: LocalGraphResult, enabled: bool = False) -> List[Dict[str, Any]]:
    """
    Orders edges from a LocalGraphResult.
    If enabled, sorts by degree of the source node, property ID, and distance.
    If disabled, sorts by property ID and distance only.
    Edges with 'ID' in their label are moved to the bottom.
    """
    if not result or not result.edges:
        return []

    if enabled:
        # Calculate degree for each node for sorting
        degrees = {}
        for edge in result.edges:
            source_id = edge['from']
            degrees[source_id] = degrees.get(source_id, 0) + 1

        def sort_key(edge: Dict[str, Any]):
            source_id = edge['from']
            prop_id = edge['property']
            
            source_node = result.nodes.get(source_id)
            distance = getattr(source_node, 'depth', float('inf')) if source_node else float('inf')
            
            degree = degrees.get(source_id, 0)
            numeric_prop_id = get_property_numeric_id(prop_id)
            
            prop_label = result.property_names.get(prop_id, '')
            is_id_property = "ID" in prop_label

            return is_id_property, -degree, numeric_prop_id, distance
        
        return sorted(result.edges, key=sort_key)

    else:
        # Default sorting without degree
        def sort_key_no_degree(edge: Dict[str, Any]):
            source_id = edge['from']
            prop_id = edge['property']
            
            source_node = result.nodes.get(source_id)
            distance = getattr(source_node, 'depth', float('inf')) if source_node else float('inf')
            
            numeric_prop_id = get_property_numeric_id(prop_id)
            
            prop_label = result.property_names.get(prop_id, '')
            is_id_property = "ID" in prop_label

            return is_id_property, numeric_prop_id, distance

        return sorted(result.edges, key=sort_key_no_degree)
