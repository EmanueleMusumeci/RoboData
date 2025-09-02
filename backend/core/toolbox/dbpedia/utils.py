from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

def extract_name_from_uri(uri: str) -> str:
    """Extract the simple name from a DBpedia URI."""
    if uri.startswith("http://dbpedia.org/resource/"):
        return uri.replace("http://dbpedia.org/resource/", "")
    elif uri.startswith("http://dbpedia.org/property/"):
        return uri.replace("http://dbpedia.org/property/", "")
    elif uri.startswith("http://dbpedia.org/ontology/"):
        return uri.replace("http://dbpedia.org/ontology/", "")
    elif uri.startswith("dbr:"):
        return uri.replace("dbr:", "")
    elif uri.startswith("dbo:"):
        return uri.replace("dbo:", "")
    elif uri.startswith("dbp:"):
        return uri.replace("dbp:", "")
    return uri

def normalize_uri(uri: str, namespace: str = "resource") -> str:
    """Normalize a URI to full DBpedia format."""
    if uri.startswith("http://dbpedia.org/"):
        return uri
    elif uri.startswith("dbr:"):
        return uri.replace("dbr:", "http://dbpedia.org/resource/")
    elif uri.startswith("dbo:"):
        return uri.replace("dbo:", "http://dbpedia.org/ontology/")
    elif uri.startswith("dbp:"):
        return uri.replace("dbp:", "http://dbpedia.org/property/")
    else:
        # Default to resource namespace unless specified otherwise
        if namespace == "ontology":
            return f"http://dbpedia.org/ontology/{uri}"
        elif namespace == "property":
            return f"http://dbpedia.org/property/{uri}"
        else:
            return f"http://dbpedia.org/resource/{uri}"

def order_properties_by_degree(statements: Dict[str, Dict[str, Any]], 
                               property_details: Dict[str, Dict[str, Any]],
                               limit: int = 20) -> List[str]:
    """Order properties by their degree (number of statements)."""
    # Count statements per property
    property_counts = {prop_id: len(statements) for prop_id, statements in statements.items()}
    
    # Sort by count (descending) and take the top properties
    sorted_properties = sorted(property_counts.items(), key=lambda x: x[1], reverse=True)
    
    return [prop_id for prop_id, _ in sorted_properties[:limit]]

def order_local_graph_edges(edges: List[Any], limit: int = 100) -> List[Any]:
    """Order edges in a local graph by importance."""
    # For now, just return the first 'limit' edges
    # Could be enhanced to order by property frequency or other metrics
    return edges[:limit]

def format_dbpedia_result_display(result: Dict[str, Any], max_items: int = 10) -> str:
    """Format DBpedia results for better display."""
    if not result:
        return "No data available"
    
    output_lines = []
    
    # Basic info
    if 'id' in result:
        output_lines.append(f"ID: {result['id']}")
    if 'label' in result:
        output_lines.append(f"Label: {result['label']}")
    if 'description' in result:
        output_lines.append(f"Description: {result['description']}")
    
    # Statements
    if 'statements' in result and result['statements']:
        output_lines.append(f"\nStatements ({len(result['statements'])} properties):")
        count = 0
        for prop_id, statements in result['statements'].items():
            if count >= max_items:
                output_lines.append(f"... and {len(result['statements']) - max_items} more properties")
                break
            
            statement_count = len(statements)
            output_lines.append(f"  {prop_id}: {statement_count} value(s)")
            count += 1
    
    return "\n".join(output_lines)

def clean_sparql_results(raw_results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Clean and normalize SPARQL query results."""
    if not raw_results or 'results' not in raw_results or 'bindings' not in raw_results['results']:
        return []
    
    cleaned_results = []
    for binding in raw_results['results']['bindings']:
        cleaned_binding = {}
        for var, value_info in binding.items():
            cleaned_binding[var] = value_info.get('value', '')
        cleaned_results.append(cleaned_binding)
    
    return cleaned_results

def extract_entity_references_from_statements(statements: Dict[str, Dict[str, Any]]) -> List[str]:
    """Extract entity references from statements for neighbor exploration."""
    entity_refs = []
    
    for prop_id, prop_statements in statements.items():
        for value_key, statement in prop_statements.items():
            value = statement.get('datavalue', {}).get('value')
            if value and isinstance(value, str):
                # Check if the value looks like a DBpedia URI or entity reference
                if (value.startswith('http://dbpedia.org/resource/') or 
                    value.startswith('dbr:') or
                    # Simple heuristic: if it doesn't contain spaces and is capitalized, might be an entity
                    (not ' ' in value and value[0].isupper() and len(value) > 1)):
                    entity_refs.append(value)
    
    return entity_refs

def build_property_degree_map(all_statements: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, int]:
    """Build a map of property usage frequency across all entities."""
    property_counts = Counter()
    
    for entity_statements in all_statements.values():
        for prop_id in entity_statements.keys():
            property_counts[prop_id] += 1
    
    return dict(property_counts)
