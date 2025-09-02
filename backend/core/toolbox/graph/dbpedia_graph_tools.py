"""
DBpedia-specific graph tools that fetch from DBpedia instead of Wikidata.
These tools are knowledge-source aware alternatives to the generic graph tools.
"""

from typing import Dict, Any, List, Optional

# Conditional imports for testing vs normal operation
if __name__ == "__main__":
    import sys
    import os
    # Add the project root to the Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    sys.path.insert(0, project_root)
    
    from backend.core.toolbox.toolbox import Tool, ToolDefinition, ToolParameter
    from backend.core.knowledge_base.graph import get_knowledge_graph
    from backend.core.knowledge_base.schema import Node, Edge
    from backend.core.toolbox.dbpedia.dbpedia_api import dbpedia_api
    from backend.core.toolbox.dbpedia.datamodel import convert_api_entity_to_model, convert_api_property_to_model
else:
    from ..toolbox import Tool, ToolDefinition, ToolParameter
    from ...knowledge_base.graph import get_knowledge_graph
    from ...knowledge_base.schema import Node, Edge
    from ..dbpedia.dbpedia_api import dbpedia_api
    from ..dbpedia.datamodel import convert_api_entity_to_model, convert_api_property_to_model


class DBpediaFetchNodeTool(Tool):
    """Tool for fetching a DBpedia entity and adding it to the local graph."""
    
    def __init__(self):
        super().__init__(
            name="fetch_node",
            description="Fetch a DBpedia entity by ID and add it to the local graph database. " \
                       "USEFUL FOR: fetching entities from DBpedia and adding them to the local graph for further exploration. " \
                       "Minimizes the chance of hallucinations when adding an entity to the local graph."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_id",
                    type="string",
                    description="DBpedia entity ID (e.g., Douglas_Adams, London)"
                )
            ],
            return_type="string",
            return_description="ID of the created entity in the local graph. Side-effect: adds the entity to the local graph."
        )
    
    async def execute(self, **kwargs) -> str:
        """Fetch a DBpedia entity and add it to the local graph."""
        entity_id = kwargs.get("entity_id")
        if not entity_id:
            raise ValueError("entity_id is required")
        
        # Connect to local graph
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        # Check if entity already exists
        existing_entity = await graph.get_entity(entity_id)
        if existing_entity:
            return f"Entity {entity_id} already exists in the local graph"
        
        # Fetch from DBpedia
        api_data = await dbpedia_api.get_entity(entity_id)
        entity = convert_api_entity_to_model(api_data)
        
        # Convert to local graph schema
        node = Node(
            node_id=entity.id,
            node_type="DBpediaEntity",
            label=entity.label,
            description=entity.description or "",
            properties={
                "dbpedia_link": entity.link,
                "knowledge_source": "dbpedia"
            }
        )
        
        return await graph.add_entity(node)

    def format_result(self, result: str) -> str:
        """Format the result into a readable, concise string."""
        if "already exists" in result:
            return result
        return f"DBpedia entity successfully fetched and added to local graph. Node ID: {result}"


class DBpediaFetchRelationshipTool(Tool):
    """Tool for fetching all DBpedia statements for a specific property from a subject entity and adding them to the local graph."""
    
    def __init__(self):
        super().__init__(
            name="fetch_relationship_from_node",
            description="Fetch ALL DBpedia statements for a specific property from a subject entity and add them to the local graph database. " \
                       "This tool fetches the subject entity, retrieves all statements for the specified property, then fetches all object entities " \
                       "and creates relationships between the subject and each object in the local graph. " \
                       "USEFUL FOR: comprehensively fetching all relationships of a specific type from DBpedia and adding them to the local graph for exploration. " \
                       "Eliminates hallucinations by only adding relationships that actually exist in DBpedia."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="subject_id",
                    type="string",
                    description="DBpedia subject entity ID (e.g., Douglas_Adams)"
                ),
                ToolParameter(
                    name="property_id",
                    type="string",
                    description="DBpedia property ID (e.g., dbo:birthPlace)"
                )
            ],
            return_type="object",
            return_description="Result of the property fetch operation with details about created/merged entities and all relationships."
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Fetch all DBpedia statements for a property and add them to the local graph."""
        subject_id = kwargs.get("subject_id")
        property_id = kwargs.get("property_id")
        
        if not subject_id or not property_id:
            raise ValueError("Both subject_id and property_id are required")
        
        # Connect to local graph
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        result = {
            "subject_id": subject_id,
            "property_id": property_id,
            "operations": [],
            "relationships_created": [],
            "relationships_skipped": []
        }
        
        # First, fetch the subject entity to get its label
        try:
            api_data = await dbpedia_api.get_entity(subject_id)
            subject_entity = convert_api_entity_to_model(api_data)
            result["subject_label"] = subject_entity.label
            result["operations"].append(f"Subject {subject_id} ({subject_entity.label}) fetched from DBpedia")
        except Exception as e:
            result["operations"].append(f"Failed to fetch subject {subject_id}: {str(e)}")
            return result
        
        #print(api_data["statements"])
        #raise
        #print('creator' in api_data["statements"])

        # Get statements for the specified property directly from the entity data
        try:
            statements = api_data.get("statements", {})
            
            # Check if the property exists in the statements. DBpedia statements
            # keys are often the short property name (e.g., 'creator') rather
            # than a namespaced form ('dbo:creator'), so try several fallbacks.
            def _find_statement_key(statements_dict: Dict[str, Any], prop: str) -> Optional[str]:
                # Direct match
                if prop in statements_dict:
                    return prop
                # If namespaced like 'dbo:creator', try short name 'creator'
                if ':' in prop:
                    short = prop.split(':', 1)[1]
                    if short in statements_dict:
                        return short
                    # also try the property namespace variant (dbp: vs dbo:)
                    alt_ns = 'dbp:' + short if prop.startswith('dbo:') else 'dbo:' + short
                    if alt_ns in statements_dict:
                        return alt_ns
                # If given a full URI, try extracting the last segment
                if prop.startswith('http://') or prop.startswith('https://'):
                    short = prop.rstrip('/').split('/')[-1]
                    if short in statements_dict:
                        return short
                # As a last resort, try bare short name if prop itself has no prefix
                if prop not in statements_dict and prop in statements_dict:
                    return prop
                return None

            matched_key = _find_statement_key(statements, property_id)
            if not matched_key:
                result["operations"].append(f"Subject {subject_id} has no statements for property {property_id}")
                result["statements_found"] = False
                return result

            property_statements = statements[matched_key]
            if matched_key != property_id:
                result["operations"].append(f"Used fallback property key '{matched_key}' for requested property '{property_id}'")
            result["statements_found"] = True

            print(property_statements)
            
            # Extract object values from the statements
            object_values = list(property_statements.keys())
            result["object_count"] = len(object_values)
            result["operations"].append(f"Found {len(object_values)} object values for property {property_id}")
            
        except Exception as e:
            result["operations"].append(f"Failed to process statements for property {property_id}: {str(e)}")
            return result
        
        # Add/merge subject entity to local graph
        existing_subject = await graph.get_entity(subject_id)
        if existing_subject:
            result["operations"].append(f"Subject {subject_id} already exists in local graph")
        else:
            try:
                # Convert to local graph schema
                subject_node = Node(
                    node_id=subject_entity.id,
                    node_type="DBpediaEntity",
                    label=subject_entity.label,
                    description=subject_entity.description or "",
                    properties={
                        "dbpedia_link": subject_entity.link,
                        "knowledge_source": "dbpedia"
                    }
                )
                await graph.add_entity(subject_node)
                result["operations"].append(f"Added subject {subject_id} to local graph")
            except Exception as e:
                result["operations"].append(f"Failed to add subject {subject_id} to local graph: {str(e)}")
        
        # Fetch property information
        try:
            property_data = await dbpedia_api.get_property(property_id)
            result["property_label"] = property_data.get('labels', {}).get('en', property_id)
        except Exception as e:
            result["operations"].append(f"Failed to fetch property {property_id}: {str(e)}")
            result["property_label"] = property_id
        
        # Process each object value in the statements
        for object_value in object_values:
            try:
                statement_data = property_statements[object_value]
                display_value = statement_data.get("display_value", object_value)
                
                # For entity references, try to fetch the entity
                if object_value.startswith("http://dbpedia.org/resource/"):
                    object_id = object_value.replace("http://dbpedia.org/resource/", "")
                    
                    try:
                        # Fetch object entity
                        object_api_data = await dbpedia_api.get_entity(object_id)
                        object_entity = convert_api_entity_to_model(object_api_data)
                        
                        # Add/merge object entity to local graph
                        existing_object = await graph.get_entity(object_id)
                        if not existing_object:
                            object_node = Node(
                                node_id=object_entity.id,
                                node_type="DBpediaEntity",
                                label=object_entity.label,
                                description=object_entity.description or "",
                                properties={
                                    "dbpedia_link": object_entity.link,
                                    "knowledge_source": "dbpedia"
                                }
                            )
                            await graph.add_entity(object_node)
                            result["operations"].append(f"Added object {object_id} ({object_entity.label}) to local graph")
                        else:
                            result["operations"].append(f"Object {object_id} already exists in local graph")
                        
                        # Create relationship
                        relationship_id = f"{subject_id}_{property_id}_{object_id}"
                        existing_relationship = await graph.get_relationship(relationship_id)
                        
                        if not existing_relationship:
                            edge = Edge(
                                source_id=subject_id,
                                target_id=object_id,
                                relationship_type=property_id,
                                label=result.get("property_label", property_id),
                                description=f"{subject_entity.label} {result.get('property_label', property_id)} {object_entity.label}",
                                properties={
                                    "id": relationship_id,
                                    "knowledge_source": "dbpedia"
                                }
                            )
                            await graph.add_relationship(edge)
                            result["relationships_created"].append({
                                "subject": subject_id,
                                "property": property_id,
                                "object": object_id,
                                "relationship_id": relationship_id
                            })
                            result["operations"].append(f"Created relationship: {subject_id} -[{property_id}]-> {object_id}")
                        else:
                            result["relationships_skipped"].append({
                                "subject": subject_id,
                                "property": property_id,
                                "object": object_id,
                                "reason": "relationship already exists"
                            })
                            result["operations"].append(f"Skipped relationship {relationship_id} (already exists)")
                    
                    except Exception as e:
                        result["operations"].append(f"Failed to process object entity {object_id}: {str(e)}")
                        # Fall back to creating a literal node
                        literal_id = f"literal_{abs(hash(object_value))}"
                        
                        try:
                            literal_node = Node(
                                node_id=literal_id,
                                node_type="Literal",
                                label=display_value,
                                description=f"Literal value: {display_value}",
                                properties={
                                    "literal_value": object_value,
                                    "knowledge_source": "dbpedia"
                                }
                            )
                            await graph.add_entity(literal_node)
                            
                            relationship_id = f"{subject_id}_{property_id}_{literal_id}"
                            edge = Edge(
                                source_id=subject_id,
                                target_id=literal_id,
                                relationship_type=property_id,
                                label=result.get("property_label", property_id),
                                description=f"{subject_entity.label} {result.get('property_label', property_id)} {display_value}",
                                properties={
                                    "id": relationship_id,
                                    "knowledge_source": "dbpedia"
                                }
                            )
                            await graph.add_relationship(edge)
                            result["relationships_created"].append({
                                "subject": subject_id,
                                "property": property_id,
                                "object": literal_id,
                                "relationship_id": relationship_id
                            })
                            result["operations"].append(f"Created literal relationship: {subject_id} -[{property_id}]-> {display_value}")
                        except Exception as e2:
                            result["operations"].append(f"Failed to create literal for {object_value}: {str(e2)}")
                
                else:
                    # Create literal node for non-entity values
                    literal_id = f"literal_{abs(hash(object_value))}"
                    
                    try:
                        literal_node = Node(
                            node_id=literal_id,
                            node_type="Literal",
                            label=display_value,
                            description=f"Literal value: {display_value}",
                            properties={
                                "literal_value": object_value,
                                "knowledge_source": "dbpedia"
                            }
                        )
                        await graph.add_entity(literal_node)
                        
                        relationship_id = f"{subject_id}_{property_id}_{literal_id}"
                        edge = Edge(
                            source_id=subject_id,
                            target_id=literal_id,
                            relationship_type=property_id,
                            label=result.get("property_label", property_id),
                            description=f"{subject_entity.label} {result.get('property_label', property_id)} {display_value}",
                            properties={
                                "id": relationship_id,
                                "knowledge_source": "dbpedia"
                            }
                        )
                        await graph.add_relationship(edge)
                        result["relationships_created"].append({
                            "subject": subject_id,
                            "property": property_id,
                            "object": literal_id,
                            "relationship_id": relationship_id
                        })
                        result["operations"].append(f"Created literal relationship: {subject_id} -[{property_id}]-> {display_value}")
                    except Exception as e:
                        result["operations"].append(f"Failed to create literal for {object_value}: {str(e)}")
                
            except Exception as e:
                result["operations"].append(f"Failed to process object {object_value}: {str(e)}")
        
        return result

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No result returned"
        
        # Check if statements were found
        if result.get("statements_found") == False:
            property_id = result.get("property_id")
            subject_id = result.get("subject_id")
            subject_label = result.get("subject_label", subject_id)
            return f"No statements found for property {property_id} on subject {subject_id} ({subject_label})"
        
        subject_id = result.get("subject_id")
        property_id = result.get("property_id")
        subject_label = result.get("subject_label", subject_id)
        property_label = result.get("property_label", property_id)
        relationships_created = result.get("relationships_created", [])
        relationships_skipped = result.get("relationships_skipped", [])
        operations = result.get("operations", [])
        
        subject_str = f"{subject_id} ({subject_label})" if subject_label != subject_id else subject_id
        
        output = f"Fetched property {property_id} ({property_label}) for {subject_str}\n"
        output += f"Created {len(relationships_created)} relationships, skipped {len(relationships_skipped)}\n"
        
        if relationships_created:
            output += f"Created relationships:\n"
            for rel in relationships_created[:5]:  # Show first 5
                output += f"  {rel['subject']} -[{rel['property']}]-> {rel['object']}\n"
            if len(relationships_created) > 5:
                output += f"  ... and {len(relationships_created) - 5} more\n"
        
        if relationships_skipped:
            output += f"Skipped {len(relationships_skipped)} existing relationships\n"
        
        # Show summary of operations if not too verbose
        if len(operations) <= 10:
            output += f"Operations: {'; '.join(operations)}\n"
        else:
            output += f"Performed {len(operations)} operations total\n"
        
        return output.strip()


class DBpediaFetchReverseRelationshipTool(Tool):
    """Tool for fetching all entities that have a specific relationship pointing to a given object entity."""
    
    def __init__(self):
        super().__init__(
            name="fetch_relationship_to_node",
            description="Find and fetch all DBpedia entities that have a specific relationship pointing to a given object entity. " \
                       "This tool uses SPARQL queries to find all subjects that have the specified property pointing to the given object, " \
                       "then fetches those entities and creates the relationships in the local graph. " \
                       "USEFUL FOR: discovering all entities that reference a specific entity through a particular property (e.g., all people born in a city, all works created by an author). " \
                       "Eliminates hallucinations by only fetching relationships that actually exist in DBpedia."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="object_id",
                    type="string",
                    description="DBpedia object entity ID that should be the target of relationships (e.g., London)"
                ),
                ToolParameter(
                    name="property_id",
                    type="string",
                    description="DBpedia property ID to search for (e.g., dbo:birthPlace)"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of subject entities to fetch (default: 50, max: 500)",
                    required=False,
                    default=50
                )
            ],
            return_type="object",
            return_description="Result of the reverse relationship fetch operation with details about all discovered subject entities and created relationships."
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Find all entities that reference the given object through the specified property."""
        object_id = kwargs.get("object_id")
        property_id = kwargs.get("property_id")
        limit = min(kwargs.get("limit", 50), 500)  # Cap at 500 to prevent excessive queries
        
        if not object_id or not property_id:
            raise ValueError("Both object_id and property_id are required")
        
        # Connect to local graph
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        result = {
            "object_id": object_id,
            "property_id": property_id,
            "limit": limit,
            "operations": [],
            "subjects_found": [],
            "relationships_created": [],
            "relationships_skipped": []
        }
        
        # First, ensure the object entity exists in our local graph
        try:
            object_api_data = await dbpedia_api.get_entity(object_id)
            object_entity = convert_api_entity_to_model(object_api_data)
            result["object_label"] = object_entity.label
            result["operations"].append(f"Object {object_id} ({object_entity.label}) fetched from DBpedia")
        except Exception as e:
            result["operations"].append(f"Failed to fetch object {object_id}: {str(e)}")
            return result
        
        # Add/merge object entity to local graph
        existing_object = await graph.get_entity(object_id)
        if existing_object:
            result["operations"].append(f"Object {object_id} already exists in local graph")
        else:
            try:
                object_node = Node(
                    node_id=object_entity.id,
                    node_type="DBpediaEntity",
                    label=object_entity.label,
                    description=object_entity.description or "",
                    properties={
                        "dbpedia_link": object_entity.link,
                        "knowledge_source": "dbpedia"
                    }
                )
                await graph.add_entity(object_node)
                result["operations"].append(f"Added object {object_id} to local graph")
            except Exception as e:
                result["operations"].append(f"Failed to add object {object_id} to local graph: {str(e)}")
        
        # Fetch property information
        try:
            property_data = await dbpedia_api.get_property(property_id)
            property_entity = convert_api_property_to_model(property_data)
            result["property_label"] = property_entity.label
        except Exception as e:
            result["operations"].append(f"Failed to fetch property {property_id}: {str(e)}")
            result["property_label"] = property_id
        
        # Use SPARQL to find all subjects that have this property pointing to our object
        # Try multiple candidate property URIs (ontology, property, and short name fallbacks)
        candidates = []
        if property_id.startswith('dbo:'):
            short = property_id.replace('dbo:', '')
            candidates = [
                f"http://dbpedia.org/ontology/{short}",
                f"http://dbpedia.org/property/{short}",
                short
            ]
        elif property_id.startswith('dbp:'):
            short = property_id.replace('dbp:', '')
            candidates = [
                f"http://dbpedia.org/property/{short}",
                f"http://dbpedia.org/ontology/{short}",
                short
            ]
        elif property_id.startswith('http://') or property_id.startswith('https://'):
            candidates = [property_id, property_id.rstrip('/').split('/')[-1]]
        else:
            # bare property name
            candidates = [f"http://dbpedia.org/property/{property_id}", f"http://dbpedia.org/ontology/{property_id}", property_id]

        # Use full URI in angle brackets to avoid escaping issues with apostrophes
        object_uri = f"http://dbpedia.org/resource/{object_id}"

        # Import here to avoid the relative import issue during testing
        try:
            try:
                from ..dbpedia.queries import SPARQLQueryTool
            except ImportError:
                from backend.core.toolbox.dbpedia.queries import SPARQLQueryTool
        except Exception as e:
            result["operations"].append(f"Failed to import SPARQLQueryTool: {e}")
            return result

        sparql_tool = SPARQLQueryTool()
        subject_ids = []
        sparql_success = False

        for candidate in candidates:
            # Build property URI or short form accordingly
            if candidate.startswith('http://') or candidate.startswith('https://'):
                prop_uri = candidate
            else:
                # short form: use as-is in triple (it may match statement keys but SPARQL needs full URI)
                # try property namespace as dbp first
                prop_uri = f"http://dbpedia.org/property/{candidate}"

            sparql_query = f"""
            SELECT DISTINCT ?subject WHERE {{
                ?subject <{prop_uri}> <{object_uri}> .
            }}
            LIMIT {limit}
            """

            try:
                sparql_result = await sparql_tool.execute(query=sparql_query)
                if sparql_result.get("success") and sparql_result.get("results"):
                    for binding in sparql_result["results"].get("bindings", []):
                        if "subject" in binding:
                            subject_uri = binding["subject"]["value"]
                            subject_id = subject_uri.split("/")[-1]
                            subject_ids.append(subject_id)
                    result["subjects_found"] = subject_ids
                    result["operations"].append(f"SPARQL query with property candidate {candidate} found {len(subject_ids)} subject entities")
                    sparql_success = True
                    break
                else:
                    result["operations"].append(f"SPARQL query with property candidate {candidate} returned no results")
            except Exception as e:
                result["operations"].append(f"SPARQL query failed for candidate {candidate}: {str(e)}")

        if not sparql_success:
            result["operations"].append("SPARQL query returned no results for any candidate property URI")
            return result
        
        # Process each subject entity found
        for subject_id in subject_ids:
            try:
                # Fetch subject entity
                subject_api_data = await dbpedia_api.get_entity(subject_id)
                subject_entity = convert_api_entity_to_model(subject_api_data)
                
                # Add/merge subject entity to local graph
                existing_subject = await graph.get_entity(subject_id)
                if not existing_subject:
                    subject_node = Node(
                        node_id=subject_entity.id,
                        node_type="DBpediaEntity",
                        label=subject_entity.label,
                        description=subject_entity.description or "",
                        properties={
                            "dbpedia_link": subject_entity.link,
                            "knowledge_source": "dbpedia"
                        }
                    )
                    await graph.add_entity(subject_node)
                    result["operations"].append(f"Added subject {subject_id} ({subject_entity.label}) to local graph")
                else:
                    result["operations"].append(f"Subject {subject_id} already exists in local graph")
                
                # Create relationship
                relationship_id = f"{subject_id}_{property_id}_{object_id}"
                existing_relationship = await graph.get_relationship(relationship_id)
                
                if not existing_relationship:
                    edge = Edge(
                        source_id=subject_id,
                        target_id=object_id,
                        relationship_type=property_id,
                        label=result.get("property_label", property_id),
                        description=f"{subject_entity.label} {result.get('property_label', property_id)} {object_entity.label}",
                        properties={
                            "id": relationship_id,
                            "knowledge_source": "dbpedia"
                        }
                    )
                    await graph.add_relationship(edge)
                    result["relationships_created"].append({
                        "subject": subject_id,
                        "property": property_id,
                        "object": object_id,
                        "relationship_id": relationship_id
                    })
                    result["operations"].append(f"Created relationship: {subject_id} -[{property_id}]-> {object_id}")
                else:
                    result["relationships_skipped"].append({
                        "subject": subject_id,
                        "property": property_id,
                        "object": object_id,
                        "reason": "relationship already exists"
                    })
                    result["operations"].append(f"Skipped relationship {relationship_id} (already exists)")
                
            except Exception as e:
                result["operations"].append(f"Failed to process subject {subject_id}: {str(e)}")
        
        return result

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No result returned"
        
        object_id = result.get("object_id")
        property_id = result.get("property_id")
        object_label = result.get("object_label", object_id)
        property_label = result.get("property_label", property_id)
        subjects_found = result.get("subjects_found", [])
        relationships_created = result.get("relationships_created", [])
        relationships_skipped = result.get("relationships_skipped", [])
        operations = result.get("operations", [])
        
        object_str = f"{object_id} ({object_label})" if object_label != object_id else object_id
        
        # Check if any relationships were actually found
        if not subjects_found:
            output = f"No reverse relationships found for property {property_id} ({property_label}) pointing to {object_str}\n"
        else:
            output = f"Fetched reverse relationships for property {property_id} ({property_label}) pointing to {object_str}\n"
        
        output += f"Found {len(subjects_found)} subject entities, created {len(relationships_created)} relationships, skipped {len(relationships_skipped)}\n"
        
        if relationships_created:
            output += f"Created relationships:\n"
            for rel in relationships_created[:5]:  # Show first 5
                output += f"  {rel['subject']} -[{rel['property']}]-> {rel['object']}\n"
            if len(relationships_created) > 5:
                output += f"  ... and {len(relationships_created) - 5} more\n"
        
        if relationships_skipped:
            output += f"Skipped {len(relationships_skipped)} existing relationships\n"
        
        # Show summary of operations if not too verbose
        if len(operations) <= 10:
            output += f"Operations: {'; '.join(operations)}\n"
        else:
            output += f"Performed {len(operations)} operations total\n"
        
        return output.strip()


# Export all DBpedia-specific tools
all_dbpedia_tools = [
    DBpediaFetchNodeTool,
    DBpediaFetchRelationshipTool,
    DBpediaFetchReverseRelationshipTool
]



if __name__ == "__main__":
    import asyncio
    import sys
    import os
    
    # Add the project root to the Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    sys.path.insert(0, project_root)
    
    # Import modules with absolute paths
    from backend.core.toolbox.dbpedia.dbpedia_api import dbpedia_api
    from backend.core.toolbox.dbpedia.datamodel import convert_api_entity_to_model, convert_api_property_to_model
    from backend.core.toolbox.toolbox import Tool, ToolDefinition, ToolParameter
    from backend.core.knowledge_base.graph import get_knowledge_graph
    from backend.core.knowledge_base.schema import Node, Edge
    
    async def test_dbpedia_tools():
        """Test all DBpedia graph tools to debug relationship fetching issues."""
        print("=== Testing DBpedia Graph Tools ===\n")
        
        # Test 1: Test fetch_node tool directly
        print("1. Testing DBpediaFetchNodeTool...")
        try:
            fetch_tool = DBpediaFetchNodeTool()
            result = await fetch_tool.execute(entity_id="Douglas_Adams")
            print(f"   Result: {result}")
            formatted = fetch_tool.format_result(result)
            print(f"   Formatted: {formatted}")
        except Exception as e:
            print(f"   Error testing fetch_node: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        # Test 2: Deep dive into Hitchhiker's Guide entity data
        print("2. Deep analysis of The_Hitchhiker's_Guide_to_the_Galaxy entity...")
        try:
            api_data = await dbpedia_api.get_entity("The_Hitchhiker's_Guide_to_the_Galaxy")
            print(f"   Entity ID: {api_data['id']}")
            print(f"   Entity Label: {api_data['labels'].get('en', 'N/A')}")
            
            statements = api_data.get('statements', {})
            print(f"   Total statements: {len(statements)}")
            print(f"   All property keys:")
            for prop in sorted(statements.keys()):
                print(f"     - {prop}")
            print()
            
            # Look for author-related properties
            author_props = [prop for prop in statements.keys() if 'author' in prop.lower()]
            print(f"   Author-related properties: {author_props}")
            
            # Look for writer-related properties
            writer_props = [prop for prop in statements.keys() if 'writer' in prop.lower()]
            print(f"   Writer-related properties: {writer_props}")
            
            # Look for creator-related properties
            creator_props = [prop for prop in statements.keys() if 'creator' in prop.lower()]
            print(f"   Creator-related properties: {creator_props}")
            
            # Check specific properties that might contain Douglas Adams
            test_properties = [
                'dbo:author', 'dbp:author', 'dbo:writer', 'dbp:writer', 
                'dbo:creator', 'dbp:creator', 'dbo:notableWork', 'dbp:notableWork'
            ]
            
            print(f"   Testing specific properties:")
            for prop in test_properties:
                if prop in statements:
                    values = statements[prop]
                    print(f"     {prop}: {len(values)} values")
                    for value, details in list(values.items())[:3]:
                        display = details.get('display_value', value)
                        print(f"       - {value} -> {display}")
                else:
                    print(f"     {prop}: NOT FOUND")
            
            # Look for Douglas Adams anywhere in the statements
            print(f"   Searching for Douglas Adams references:")
            douglas_found = False
            for prop, values in statements.items():
                for value, details in values.items():
                    display = details.get('display_value', value)
                    if 'Douglas' in str(value) or 'Adams' in str(value) or 'Douglas' in str(display) or 'Adams' in str(display):
                        print(f"     Found in {prop}: {value} -> {display}")
                        douglas_found = True
            
            if not douglas_found:
                print(f"     No Douglas Adams references found in any statements!")
                
        except Exception as e:
            print(f"   Error analyzing Hitchhiker's Guide: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        # Test 3: Test fetch_relationship_from_node tool directly
        print("3. Testing DBpediaFetchRelationshipTool...")
        try:
            rel_tool = DBpediaFetchRelationshipTool()
            
            # Test different property combinations, including the 'creator' property we found
            test_cases = [
                ("The_Hitchhiker's_Guide_to_the_Galaxy", "creator"),  # This should work!
                ("The_Hitchhiker's_Guide_to_the_Galaxy", "dbo:author"),
                ("The_Hitchhiker's_Guide_to_the_Galaxy", "dbp:author"), 
                ("The_Hitchhiker's_Guide_to_the_Galaxy", "dbo:writer"),
                ("The_Hitchhiker's_Guide_to_the_Galaxy", "dbp:writer"),
                ("Douglas_Adams", "dbo:notableWork"),
                ("Douglas_Adams", "dbp:notableWork")
            ]
            
            for subject_id, property_id in test_cases:
                try:
                    print(f"   Testing {subject_id} -> {property_id}")
                    result = await rel_tool.execute(subject_id=subject_id, property_id=property_id)
                    formatted = rel_tool.format_result(result)
                    print(f"     Result: {formatted}")
                    print()
                except Exception as e:
                    print(f"     Error: {e}")
                    print()
                    
        except Exception as e:
            print(f"   Error testing fetch_relationship_from_node: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        # Test 4: Test reverse relationship tool
        print("4. Testing DBpediaFetchReverseRelationshipTool...")
        try:
            rev_tool = DBpediaFetchReverseRelationshipTool()
            
            # Test finding all works by Douglas Adams, including the 'creator' property
            test_cases = [
                ("Douglas_Adams", "creator"),  # This should work!
                ("Douglas_Adams", "dbo:author"),
                ("Douglas_Adams", "dbp:author"), 
                ("Douglas_Adams", "dbo:writer"),
                ("Douglas_Adams", "dbp:writer")
            ]
            
            for object_id, property_id in test_cases:
                try:
                    print(f"   Testing reverse: ? -> {property_id} -> {object_id}")
                    result = await rev_tool.execute(object_id=object_id, property_id=property_id, limit=5)
                    formatted = rev_tool.format_result(result)
                    print(f"     Result: {formatted}")
                    print()
                except Exception as e:
                    print(f"     Error: {e}")
                    print()
                    
        except Exception as e:
            print(f"   Error testing fetch_relationship_to_node: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        # Test 5: Test SPARQL queries to understand the actual relationships
        print("5. Testing direct SPARQL queries for debugging...")
        try:
            # Query 1: Find all properties of Hitchhiker's Guide (using proper URI escaping)
            query1 = """
            SELECT DISTINCT ?prop ?value WHERE {
                <http://dbpedia.org/resource/The_Hitchhiker's_Guide_to_the_Galaxy> ?prop ?value .
            }
            LIMIT 50
            """
            
            print("   Query 1: All properties of Hitchhiker's Guide")
            result = await dbpedia_api._execute_sparql_query(query1)
            bindings = result.get('results', {}).get('bindings', [])
            print(f"   Found {len(bindings)} properties:")
            for binding in bindings:
                prop = binding.get('prop', {}).get('value', '')
                value = binding.get('value', {}).get('value', '')
                if 'Douglas' in value or 'Adams' in value or 'author' in prop or 'writer' in prop or 'creator' in prop:
                    print(f"     RELEVANT: {prop} -> {value}")
            print()
            
            # Query 2: Find all works authored by Douglas Adams (with proper URI format)
            query2 = """
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX dbp: <http://dbpedia.org/property/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT DISTINCT ?work ?workLabel ?prop WHERE {
                ?work ?prop <http://dbpedia.org/resource/Douglas_Adams> .
                OPTIONAL { ?work rdfs:label ?workLabel . FILTER(LANG(?workLabel) = "en") }
                FILTER(?prop = dbo:author || ?prop = dbp:author || ?prop = dbo:writer || ?prop = dbp:writer || 
                       ?prop = dbp:creator || CONTAINS(STR(?prop), "creator"))
            }
            LIMIT 20
            """
            
            print("   Query 2: Works authored/written by Douglas Adams")
            result = await dbpedia_api._execute_sparql_query(query2)
            bindings = result.get('results', {}).get('bindings', [])
            print(f"   Found {len(bindings)} works:")
            for binding in bindings:
                work = binding.get('work', {}).get('value', '')
                workLabel = binding.get('workLabel', {}).get('value', work.split('/')[-1])
                prop = binding.get('prop', {}).get('value', '')
                print(f"     {workLabel} (via {prop.split('/')[-1]})")
            print()
            
            # Query 3: Find the exact relationship between Hitchhiker's Guide and Douglas Adams (with full URIs)
            query3 = """
            SELECT DISTINCT ?prop WHERE {
                { <http://dbpedia.org/resource/The_Hitchhiker's_Guide_to_the_Galaxy> ?prop <http://dbpedia.org/resource/Douglas_Adams> . }
                UNION
                { <http://dbpedia.org/resource/Douglas_Adams> ?prop <http://dbpedia.org/resource/The_Hitchhiker's_Guide_to_the_Galaxy> . }
            }
            """
            
            print("   Query 3: Direct relationships between Hitchhiker's Guide and Douglas Adams")
            result = await dbpedia_api._execute_sparql_query(query3)
            bindings = result.get('results', {}).get('bindings', [])
            print(f"   Found {len(bindings)} direct relationships:")
            for binding in bindings:
                prop = binding.get('prop', {}).get('value', '')
                print(f"     {prop}")
            print()
            
        except Exception as e:
            print(f"   SPARQL Error: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        print("=== Diagnosis Complete ===")
        print("Based on this analysis, we should be able to identify:")
        print("1. Whether the entities exist in DBpedia")
        print("2. What properties actually connect them")
        print("3. Why our tools might be failing to fetch the relationships")
        print("4. What the correct property names should be")
    
        # Test 6: Test DBpedia search_entities (Lookup API)
        print("6. Testing DBpedia search_entities (Lookup API)...")
        try:
            # Try searching for 'Douglas Adams' and 'Hitchhiker's Guide'
            queries = ["Douglas Adams", "Hitchhiker's Guide", "London", "Python"]
            for query in queries:
                print(f"   Searching for: {query}")
                try:
                    results = await dbpedia_api.search_entities(query, limit=5)
                    print(f"     Found {len(results)} results:")
                    for i, result in enumerate(results):
                        label = result.get('label', result.get('title', 'N/A'))
                        uri = result.get('uri', 'N/A')
                        description = result.get('description', '')
                        print(f"       {i+1}. {label} ({uri})\n          {description[:80]}{'...' if len(description) > 80 else ''}")
                    if not results:
                        print("     No results found.")
                except Exception as e:
                    print(f"     Error searching for '{query}': {e}")
        except Exception as e:
            print(f"   Error testing search_entities: {e}")
            import traceback
            traceback.print_exc()
        print()
    # Run the test
    asyncio.run(test_dbpedia_tools())
