from typing import Dict, Any, List, Optional
from ..toolbox import Tool, ToolDefinition, ToolParameter
from ...knowledge_base.graph import get_knowledge_graph
from ...knowledge_base.schema import Node, Edge
# Add imports for Wikidata integration
from ..wikidata.wikidata_api import wikidata_api
from ..wikidata.datamodel import convert_api_entity_to_model, convert_api_property_to_model
import json
import time

class AddNodeTool(Tool):
    """Tool for adding nodes to the graph database."""
    
    def __init__(self):
        super().__init__(
            name="add_node",
            description="Add a new node to the graph database"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_id",
                    type="string",
                    description="Unique ID for the entity"
                ),
                ToolParameter(
                    name="entity_type",
                    type="string",
                    description="Type of the entity (e.g. 'WikidataEntity')"
                ),
                ToolParameter(
                    name="description",
                    type="string",
                    description="Description of the entity",
                    required=False,
                    default=""
                ),
                ToolParameter(
                    name="properties",
                    type="object",
                    description="Entity properties as JSON object"
                )
            ],
            return_type="string",
            return_description="ID of the created entity"
        )
    
    async def execute(self, **kwargs) -> str:
        """Add an entity to the knowledge graph."""
        entity_id = kwargs.get("entity_id")
        entity_type = kwargs.get("entity_type")
        description = kwargs.get("description", "")
        properties = kwargs.get("properties", {})

        if not entity_id or not entity_type:
            raise ValueError("entity_id and entity_type are required")
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
            
        node = Node(
            node_id=entity_id,
            node_type=entity_type,
            label=properties.get('label', ''),
            description=description,
            properties=properties
        )
        
        return await graph.add_entity(node)

    def format_result(self, result: str) -> str:
        """Format the result into a readable, concise string."""
        # This tool returns only the ID. A more detailed format would require another call to get_node.
        # For now, we keep it simple as the node was just added.
        return f"Node added. ID: {result}"

class AddEdgeTool(Tool):
    """Tool for adding edges to the graph database."""
    
    def __init__(self):
        super().__init__(
            name="add_edge",
            description="Add a new edge/relationship between two entities"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="from_node_id",
                    type="string",
                    description="ID of the source entity"
                ),
                ToolParameter(
                    name="to_node_id",
                    type="string",
                    description="ID of the target entity"
                ),
                ToolParameter(
                    name="relationship_type",
                    type="string",
                    description="Type of relationship"
                ),
                ToolParameter(
                    name="description",
                    type="string",
                    description="Description of the edge",
                    required=False,
                    default=""
                ),
                ToolParameter(
                    name="properties",
                    type="object",
                    description="Edge properties as JSON object",
                    required=False,
                    default={}
                )
            ],
            return_type="string",
            return_description="ID of the created edge"
        )
    
    async def execute(self, **kwargs) -> str:
        """Add a relationship to the knowledge graph."""
        from_node_id = kwargs.get("from_node_id")
        to_node_id = kwargs.get("to_node_id")
        relationship_type = kwargs.get("relationship_type")
        description = kwargs.get("description", "")
        properties = kwargs.get("properties", {})

        if not from_node_id or not to_node_id or not relationship_type:
            raise ValueError("from_node_id, to_node_id, and relationship_type are required")
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()

        # Check if nodes exist
        source_node = await graph.get_entity(from_node_id)
        target_node = await graph.get_entity(to_node_id)
        if not source_node:
            raise ValueError(f"Source node with ID '{from_node_id}' not found.")
        if not target_node:
            raise ValueError(f"Target node with ID '{to_node_id}' not found.")

        edge_id = f"{from_node_id}_{relationship_type}_{to_node_id}"
        
        edge = Edge(
            source_id=from_node_id,
            target_id=to_node_id,
            relationship_type=relationship_type,
            label=properties.get('label', relationship_type),
            description=description,
            properties=properties
        )
        edge.properties['id'] = edge_id
        
        await graph.add_relationship(edge)
        return edge_id

    def format_result(self, result: str) -> str:
        """Format the result into a readable, concise string."""
        # This tool returns only the ID. A more detailed format would require another call to get_edge.
        # For now, we keep it simple as the edge was just added.
        return f"Edge added. ID: {result}"

class GetNodeTool(Tool):
    """Tool for retrieving nodes from the graph database."""
    
    def __init__(self):
        super().__init__(
            name="get_node",
            description="Get a node from the local knowledge graph by ID" \
            "USEFUL FOR: retrieving node details to understand its properties and relationships in the local graph."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="node_id",
                    type="string",
                    description="ID of the node to retrieve"
                )
            ],
            return_type="object",
            return_description="Node data including properties and labels"
        )
    
    async def execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Get an entity and its relationships from the knowledge graph."""
        node_id = kwargs.get("node_id")
        if not node_id:
            return None
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        node = await graph.get_entity(node_id)
        if not node:
            return None
            
        relationships = await graph.get_entity_relationships(node_id)
        
        # Get neighbor details for rich formatting
        neighbor_ids = {rel.source_id for rel in relationships} | {rel.target_id for rel in relationships}
        neighbor_ids.discard(node_id)
        
        neighbors = {}
        for nid in neighbor_ids:
            neighbor_node = await graph.get_entity(nid)
            if neighbor_node:
                neighbors[nid] = neighbor_node.to_dict()

        node_data = node.to_dict()
        node_data['relationships'] = [rel.to_dict() for rel in relationships]
        node_data['neighbors'] = neighbors
        return node_data

    def format_result(self, result: Optional[Dict[str, Any]]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "Node not found."
        
        node_id = result.get('id')
        node_type = result.get('type')
        description = result.get('description')
        
        output = f"{node_id} ({node_type})\n"
        if description:
            output += f'"{description}"\n'

        relationships = result.get('relationships', [])
        neighbors = result.get('neighbors', {})
        
        if relationships:
            output += "Relationships:\n"
            for rel in relationships:
                source_id = rel.get('source_id')
                target_id = rel.get('target_id')
                rel_type = rel.get('type')
                
                source_type = node_type if source_id == node_id else neighbors.get(source_id, {}).get('type', 'Unknown')
                target_type = node_type if target_id == node_id else neighbors.get(target_id, {}).get('type', 'Unknown')

                source_str = f"{source_id} [{source_type}]"
                target_str = f"{target_id} [{target_type}]"

                if source_id == node_id:
                    output += f"  - ({source_str})-[{rel_type}]->({target_str})\n"
                else:
                    output += f"  - ({source_str})<-[{rel_type}]-({target_str})\n"
        else:
            output += "Relationships: None"
            
        return output.strip()

class GetEdgeTool(Tool):
    """Tool for retrieving edges from the graph database."""
    
    def __init__(self):
        super().__init__(
            name="get_edge",
            description="Get an edge from the local knowledge graph by ID." \
            "USEFUL FOR: retrieving edges to understand relationships between entities in the local graph. Navigate the local knowledge graph effectively."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="edge_id",
                    type="string",
                    description="ID of the edge to retrieve"
                )
            ],
            return_type="object",
            return_description="If the edge is present, a structured representation of the retrieved edge, including source and target nodes, relationship type, and properties."
        )
    
    async def execute(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Get a relationship from the knowledge graph."""
        edge_id = kwargs.get("edge_id")
        if not edge_id:
            return None
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        edge = await graph.get_relationship(edge_id)
        if not edge:
            return None

        edge_data = edge.to_dict()
        source_node = await graph.get_entity(edge.source_id)
        target_node = await graph.get_entity(edge.target_id)
        
        edge_data['source_type'] = source_node.type if source_node else 'Unknown'
        edge_data['target_type'] = target_node.type if target_node else 'Unknown'
        
        return edge_data

    def format_result(self, result: Optional[Dict[str, Any]]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "Edge not found."
        
        source_id = result.get('source_id')
        target_id = result.get('target_id')
        source_type = result.get('source_type')
        target_type = result.get('target_type')
        rel_type = result.get('type')
        description = result.get('description', '')
        
        source_str = f"{source_id} [{source_type}]"
        target_str = f"{target_id} [{target_type}]"

        output = f"({source_str})-[{rel_type}]->({target_str})\n"
        if description:
            output += f'"{description}"\n'
            
        properties = result.get('properties', {})
        if properties:
            output += "Properties:\n"
            for key, value in properties.items():
                if key not in ['id', 'description', 'source_id', 'target_id', 'type', 'label']: # Don't repeat core fields
                    output += f"  - {key}: {value}\n"
                
        return output.strip()

class FetchNodeTool(Tool):
    """Tool for fetching a Wikidata entity and adding it to the local graph."""
    
    def __init__(self):
        super().__init__(
            name="fetch_node",
            description="Fetch a Wikidata entity by ID and add it to the local graph database." \
            "USEFUL FOR: fetching entities from Wikidata and adding them to the local graph for further exploration. Minimizes the chance of hallucinations when adding an entity to the local graph."

        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_id",
                    type="string",
                    description="Wikidata entity ID (e.g., Q42 for Douglas Adams)"
                )
            ],
            return_type="string",
            return_description="ID of the created entity in the local graph. Side-effect: adds the entity to the local graph."
        )
    
    async def execute(self, **kwargs) -> str:
        """Fetch a Wikidata entity and add it to the local graph."""
        entity_id = kwargs.get("entity_id")
        if not entity_id:
            raise ValueError("entity_id is required")
        
        # Fetch from Wikidata
        api_data = await wikidata_api.get_entity(entity_id)
        entity = convert_api_entity_to_model(api_data)
        
        # Connect to local graph
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        # Check if entity already exists
        existing_entity = await graph.get_entity(entity_id)
        if existing_entity:
            return f"Entity {entity_id} already exists in the local graph"
        
        # Convert to local graph schema
        node = Node(
            node_id=entity.id,
            node_type="WikidataEntity",
            label=entity.label,
            description=entity.description or "",
            properties={
                "wikidata_link": entity.link
            }
        )
        
        return await graph.add_entity(node)

    def format_result(self, result: str) -> str:
        """Format the result into a readable, concise string."""
        if "already exists" in result:
            return result
        return f"Wikidata entity successfully fetched and added to local graph. Node ID: {result}"

class FetchRelationshipTool(Tool):
    """Tool for fetching all Wikidata statements for a specific property from a subject entity and adding them to the local graph."""
    
    def __init__(self):
        super().__init__(
            name="fetch_relationship",
            description="Fetch ALL Wikidata statements for a specific property from a subject entity and add them to the local graph database. " \
            "This tool fetches the subject entity, retrieves all statements for the specified property, then fetches all object entities " \
            "and creates relationships between the subject and each object in the local graph. " \
            "USEFUL FOR: comprehensively fetching all relationships of a specific type from Wikidata and adding them to the local graph for exploration. " \
            "Eliminates hallucinations by only adding relationships that actually exist in Wikidata."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="subject_id",
                    type="string",
                    description="Wikidata subject entity ID (e.g., Q42 for Douglas Adams)"
                ),
                ToolParameter(
                    name="property_id",
                    type="string",
                    description="Wikidata property ID (e.g., P31 for 'instance of')"
                )
            ],
            return_type="object",
            return_description="Result of the property fetch operation with details about created/merged entities and all relationships."
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Fetch all Wikidata statements for a property and add them to the local graph."""
        subject_id = kwargs.get("subject_id")
        property_id = kwargs.get("property_id")
        
        if not subject_id or not property_id:
            raise ValueError("subject_id and property_id are required")
        
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
            api_data = await wikidata_api.get_entity(subject_id)
            subject_entity = convert_api_entity_to_model(api_data)
            result["subject_label"] = subject_entity.label
            result["operations"].append(f"Subject {subject_id} ({subject_entity.label}) fetched from Wikidata")
        except Exception as e:
            result["operations"].append(f"Failed to fetch subject {subject_id}: {str(e)}")
            return result
        
        # Get all statements for the specified property
        try:
            statements = await wikidata_api.get_entity_statements(subject_id, property_id)
            
            # Check if the property exists in the statements
            if property_id not in statements:
                result["operations"].append(f"Subject {subject_id} has no statements for property {property_id}")
                result["statements_found"] = False
                return result
            
            property_statements = statements[property_id]
            object_ids = list(property_statements.keys())
            result["operations"].append(f"Found {len(object_ids)} statements for property {property_id}: {object_ids}")
            result["statements_found"] = True
            result["object_ids"] = object_ids
            
        except Exception as e:
            result["operations"].append(f"Failed to get statements for property {property_id}: {str(e)}")
            return result
        
        # Add/merge subject entity to local graph
        existing_subject = await graph.get_entity(subject_id)
        if existing_subject:
            result["operations"].append(f"Subject {subject_id} already exists in local graph, skipping")
        else:
            try:
                # Add to local graph
                subject_node = Node(
                    node_id=subject_entity.id,
                    node_type="WikidataEntity",
                    label=subject_entity.label,
                    description=subject_entity.description or "",
                    properties={
                        "wikidata_link": subject_entity.link,
                    }
                )
                
                await graph.add_entity(subject_node)
                result["operations"].append(f"Subject {subject_id} ({subject_entity.label}) added to local graph")
            except Exception as e:
                result["operations"].append(f"Failed to add subject to local graph: {str(e)}")
                return result
        
        # Fetch property information
        try:
            api_data = await wikidata_api.get_property(property_id)
            prop = convert_api_property_to_model(api_data)
            result["property_label"] = prop.label
            result["property_description"] = prop.description
        except Exception as e:
            result["operations"].append(f"Failed to fetch property {property_id}: {str(e)}")
            result["property_label"] = property_id
            result["property_description"] = ""
        
        # Process each object in the statements
        for object_key in object_ids:
            try:
                # Get the statement data for this object
                statement_data = property_statements[object_key]
                datavalue = statement_data.get("datavalue")
                datatype = statement_data.get("datatype")
                
                # Determine if this is an entity reference or a literal value
                is_entity = datavalue and datavalue.get("type") == "wikibase-entityid"
                
                if is_entity:
                    # This is an entity reference - treat as before
                    object_id = object_key  # object_key is the entity ID
                    object_label = object_id  # Default label
                    
                    # Check if relationship already exists in local graph
                    existing_relationships = await graph.get_entity_relationships(subject_id)
                    relationship_exists = False
                    for rel in existing_relationships:
                        if (rel.source_id == subject_id and rel.target_id == object_id and 
                            rel.type == property_id):
                            relationship_exists = True
                            break
                    
                    if relationship_exists:
                        result["relationships_skipped"].append({
                            "object_id": object_id,
                            "reason": "already exists in local graph"
                        })
                        continue
                    
                    # Fetch and add/merge object entity
                    existing_object = await graph.get_entity(object_id)
                    if existing_object:
                        object_label = existing_object.label
                        result["operations"].append(f"Object {object_id} already exists in local graph, skipping")
                    else:
                        try:
                            # Fetch object from Wikidata
                            api_data = await wikidata_api.get_entity(object_id)
                            object_entity = convert_api_entity_to_model(api_data)
                            object_label = object_entity.label
                            
                            # Add to local graph
                            object_node = Node(
                                node_id=object_entity.id,
                                node_type="WikidataEntity",
                                label=object_entity.label,
                                description=object_entity.description or "",
                                properties={
                                    "wikidata_link": object_entity.link,
                                }
                            )
                            
                            await graph.add_entity(object_node)
                            result["operations"].append(f"Object {object_id} ({object_entity.label}) added to local graph")
                        except Exception as e:
                            result["operations"].append(f"Failed to fetch/add object {object_id}: {str(e)}")
                            result["relationships_skipped"].append({
                                "object_id": object_id,
                                "reason": f"failed to fetch object: {str(e)}"
                            })
                            continue
                    
                    # Create the relationship to entity
                    edge = Edge(
                        source_id=subject_id,
                        target_id=object_id,
                        relationship_type=property_id,
                        label=result["property_label"],
                        description=result["property_description"] or "",
                        properties={
                            "wikidata_link": f"https://www.wikidata.org/wiki/Property:{property_id}"
                        }
                    )
                    
                    edge_id = await graph.add_relationship(edge)
                    result["relationships_created"].append({
                        "object_id": object_id,
                        "object_label": object_label,
                        "edge_id": edge_id,
                        "value_type": "entity"
                    })
                    result["operations"].append(f"Relationship {subject_id} -[{property_id}]-> {object_id} created with ID: {edge_id}")
                    
                else:
                    # This is a literal value (date, string, number, etc.)
                    literal_value = object_key
                    literal_node_id = f"{subject_id}_{property_id}_{hash(literal_value)}"
                    
                    # Check if this literal relationship already exists
                    existing_relationships = await graph.get_entity_relationships(subject_id)
                    relationship_exists = False
                    for rel in existing_relationships:
                        if (rel.source_id == subject_id and rel.target_id == literal_node_id and 
                            rel.type == property_id):
                            relationship_exists = True
                            break
                    
                    if relationship_exists:
                        result["relationships_skipped"].append({
                            "object_id": literal_node_id,
                            "reason": "already exists in local graph"
                        })
                        continue
                    
                    # Create or get literal value node
                    existing_literal = await graph.get_entity(literal_node_id)
                    if not existing_literal:
                        # Create a literal value node
                        literal_node = Node(
                            node_id=literal_node_id,
                            node_type="LiteralValue",
                            label=str(literal_value),
                            description=f"{datatype} value: {literal_value}",
                            properties={
                                "value": literal_value,
                                "datatype": datatype,
                                "wikidata_property": property_id
                            }
                        )
                        
                        await graph.add_entity(literal_node)
                        result["operations"].append(f"Literal value node {literal_node_id} created for value: {literal_value}")
                    
                    # Create the relationship to literal value
                    edge = Edge(
                        source_id=subject_id,
                        target_id=literal_node_id,
                        relationship_type=property_id,
                        label=result["property_label"],
                        description=result["property_description"] or "",
                        properties={
                            "wikidata_link": f"https://www.wikidata.org/wiki/Property:{property_id}",
                            "literal_value": literal_value,
                            "datatype": datatype
                        }
                    )
                    
                    edge_id = await graph.add_relationship(edge)
                    result["relationships_created"].append({
                        "object_id": literal_node_id,
                        "object_label": str(literal_value),
                        "edge_id": edge_id,
                        "value_type": "literal",
                        "datatype": datatype
                    })
                    result["operations"].append(f"Relationship {subject_id} -[{property_id}]-> {literal_node_id} (literal: {literal_value}) created with ID: {edge_id}")
                
            except Exception as e:
                result["operations"].append(f"Failed to process object {object_key}: {str(e)}")
                result["relationships_skipped"].append({
                    "object_id": object_key,
                    "reason": f"processing error: {str(e)}"
                })
        
        return result

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "Failed to fetch property statements."
        
        # Check if statements were found
        if result.get("statements_found") == False:
            subject_id = result.get("subject_id")
            property_id = result.get("property_id")
            operations = result.get("operations", [])
            
            output = f"No statements found for property {property_id} on subject {subject_id}\n"
            output += "Operations performed:\n"
            for op in operations:
                output += f"  - {op}\n"
            
            return output.strip()
        
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
            output += "\nRelationships created:\n"
            for rel in relationships_created:
                object_id = rel.get("object_id")
                object_label = rel.get("object_label", object_id)
                value_type = rel.get("value_type", "unknown")
                datatype = rel.get("datatype")
                
                if value_type == "entity":
                    object_str = f"{object_id} ({object_label})" if object_label != object_id else object_id
                    output += f"  - {subject_str} -[{property_id}]-> {object_str}\n"
                elif value_type == "literal":
                    datatype_str = f" ({datatype})" if datatype else ""
                    output += f"  - {subject_str} -[{property_id}]-> {object_label}{datatype_str} [literal]\n"
                else:
                    object_str = f"{object_id} ({object_label})" if object_label != object_id else object_id
                    output += f"  - {subject_str} -[{property_id}]-> {object_str}\n"
        
        if relationships_skipped:
            output += "\nRelationships skipped:\n"
            for rel in relationships_skipped:
                object_id = rel.get("object_id")
                reason = rel.get("reason")
                output += f"  - {object_id}: {reason}\n"
        
        # Show summary of operations if not too verbose
        if len(operations) <= 10:
            output += "\nOperations performed:\n"
            for op in operations:
                output += f"  - {op}\n"
        else:
            output += f"\n{len(operations)} operations performed (detailed log available)"
        
        return output.strip()

class FetchTripleTool(Tool):
    """Tool for fetching a Wikidata triple (subject-property-object) and adding it to the local graph."""
    
    def __init__(self):
        super().__init__(
            name="fetch_triple",
            description="Fetch a complete Wikidata triple (subject-property-object) and add it to the local graph database ONLY IF IT EXISTS in Wikidata. " \
            "This tool first fetches the subject entity, then verifies that the specified relationship to the object actually exists in Wikidata by checking the subject's statements. " \
            "If verified, it will fetch both entities and create the relationship between them, or merge if they already exist. " \
            "USEFUL FOR: safely fetching a verified triple (subject-property-object) relationship from Wikidata and adding it to the local graph for further exploration. " \
            "Eliminates hallucinations by ensuring the triple actually exists in Wikidata before adding it to the local graph."
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="subject_id",
                    type="string",
                    description="Wikidata subject entity ID (e.g., Q42 for Douglas Adams)"
                ),
                ToolParameter(
                    name="property_id",
                    type="string",
                    description="Wikidata property ID (e.g., P31 for 'instance of')"
                ),
                ToolParameter(
                    name="object_id",
                    type="string",
                    description="Wikidata object entity ID (e.g., Q5 for 'human')"
                )
            ],
            return_type="object",
            return_description="Result of the triple fetch operation with details about created/merged entities and relationship."
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Fetch a Wikidata triple and add it to the local graph."""
        subject_id = kwargs.get("subject_id")
        property_id = kwargs.get("property_id")
        object_id = kwargs.get("object_id")
        
        if not subject_id or not property_id or not object_id:
            raise ValueError("subject_id, property_id, and object_id are required")
        
        # Connect to local graph
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        result = {
            "subject_id": subject_id,
            "property_id": property_id,
            "object_id": object_id,
            "operations": []
        }
        
        # First, fetch the subject entity to get its label
        try:
            api_data = await wikidata_api.get_entity(subject_id)
            subject_entity = convert_api_entity_to_model(api_data)
            result["subject_label"] = subject_entity.label
            result["operations"].append(f"Subject {subject_id} ({subject_entity.label}) fetched from Wikidata")
        except Exception as e:
            result["operations"].append(f"Failed to fetch subject {subject_id}: {str(e)}")
            return result
        
        # Verify the relationship exists in Wikidata by checking the subject's statements
        try:
            statements = await wikidata_api.get_entity_statements(subject_id, property_id)
            
            # Check if the property exists in the statements
            if property_id not in statements:
                result["operations"].append(f"Subject {subject_id} has no statements for property {property_id}")
                result["relationship_verified"] = False
                return result
            
            # Check if the object_id is among the values for this property
            property_statements = statements[property_id]
            if object_id not in property_statements:
                result["operations"].append(f"Subject {subject_id} property {property_id} does not reference object {object_id}")
                result["relationship_verified"] = False
                return result
            
            result["operations"].append(f"Verified: {subject_id} -[{property_id}]-> {object_id} exists in Wikidata")
            result["relationship_verified"] = True
            
        except Exception as e:
            result["operations"].append(f"Failed to verify relationship in Wikidata: {str(e)}")
            return result
        
        # Add/merge subject entity to local graph
        existing_subject = await graph.get_entity(subject_id)
        if existing_subject:
            result["operations"].append(f"Subject {subject_id} already exists in local graph, skipping")
        else:
            try:
                # Add to local graph
                subject_node = Node(
                    node_id=subject_entity.id,
                    node_type="WikidataEntity",
                    label=subject_entity.label,
                    description=subject_entity.description or "",
                    properties={
                        "wikidata_link": subject_entity.link,
                    }
                )
                
                await graph.add_entity(subject_node)
                result["operations"].append(f"Subject {subject_id} ({subject_entity.label}) added to local graph")
            except Exception as e:
                result["operations"].append(f"Failed to add subject to local graph: {str(e)}")
                return result
        
        # Fetch and add/merge object entity
        existing_object = await graph.get_entity(object_id)
        if existing_object:
            result["operations"].append(f"Object {object_id} already exists in local graph, skipping")
            result["object_label"] = existing_object.label
        else:
            try:
                # Fetch object from Wikidata
                api_data = await wikidata_api.get_entity(object_id)
                object_entity = convert_api_entity_to_model(api_data)
                result["object_label"] = object_entity.label
                
                # Add to local graph
                object_node = Node(
                    node_id=object_entity.id,
                    node_type="WikidataEntity",
                    label=object_entity.label,
                    description=object_entity.description or "",
                    properties={
                        "wikidata_link": object_entity.link,
                    }
                )
                
                await graph.add_entity(object_node)
                result["operations"].append(f"Object {object_id} ({object_entity.label}) added to local graph")
            except Exception as e:
                result["operations"].append(f"Failed to fetch/add object {object_id}: {str(e)}")
                return result
        
        # Fetch property information
        try:
            api_data = await wikidata_api.get_property(property_id)
            prop = convert_api_property_to_model(api_data)
            result["property_label"] = prop.label
            result["property_description"] = prop.description
        except Exception as e:
            result["operations"].append(f"Failed to fetch property {property_id}: {str(e)}")
            result["property_label"] = property_id
            result["property_description"] = ""
        
        # Check if relationship already exists in local graph
        existing_relationships = await graph.get_entity_relationships(subject_id)
        relationship_exists = False
        for rel in existing_relationships:
            if (rel.source_id == subject_id and rel.target_id == object_id and 
                rel.type == property_id):
                relationship_exists = True
                break
        
        if relationship_exists:
            result["operations"].append(f"Relationship {subject_id} -[{property_id}]-> {object_id} already exists in local graph")
        else:
            # Create the relationship since it's verified to exist in Wikidata
            edge = Edge(
                source_id=subject_id,
                target_id=object_id,
                relationship_type=property_id,
                label=result["property_label"],
                description=result["property_description"] or "",
                properties={
                    "wikidata_link": f"https://www.wikidata.org/wiki/Property:{property_id}"
                }
            )
            
            edge_id = await graph.add_relationship(edge)
            result["operations"].append(f"Relationship {subject_id} -[{property_id}]-> {object_id} created with ID: {edge_id}")
            result["edge_id"] = edge_id
        
        return result

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "Failed to fetch relationship triple."
        
        # Check if relationship verification failed
        if result.get("relationship_verified") == False:
            subject_id = result.get("subject_id")
            property_id = result.get("property_id")
            object_id = result.get("object_id")
            operations = result.get("operations", [])
            
            output = f"Relationship verification failed: {subject_id} -[{property_id}]-> {object_id} does not exist in Wikidata\n"
            output += "Operations performed:\n"
            for op in operations:
                output += f"  - {op}\n"
            
            return output.strip()
        
        subject_id = result.get("subject_id")
        property_id = result.get("property_id")
        object_id = result.get("object_id")
        subject_label = result.get("subject_label", subject_id)
        object_label = result.get("object_label", object_id)
        property_label = result.get("property_label", property_id)
        operations = result.get("operations", [])
        
        subject_str = f"{subject_id} ({subject_label})" if subject_label != subject_id else subject_id
        object_str = f"{object_id} ({object_label})" if object_label != object_id else object_id
        
        output = f"Fetched triple: {subject_str} -[{property_id}:{property_label}]-> {object_str}\n"
        output += "Operations performed:\n"
        for op in operations:
            output += f"  - {op}\n"
        
        return output.strip()

class RemoveNodeTool(Tool):
    """Tool for removing nodes from the graph database."""
    
    def __init__(self):
        super().__init__(
            name="remove_node",
            description="Remove a node from the graph database by ID"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="node_id",
                    type="string",
                    description="ID of the node to remove"
                )
            ],
            return_type="boolean",
            return_description="True if the node was removed, False otherwise"
        )
    
    async def execute(self, **kwargs) -> bool:
        """Remove an entity from the knowledge graph."""
        node_id = kwargs.get("node_id")
        if not node_id:
            return False
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        return await graph.remove_entity(node_id)

    def format_result(self, result: bool) -> str:
        """Format the result into a readable, concise string."""
        if result:
            return "Node removed ."
        return "Node removal failed. The node may not have existed."

class RemoveEdgeTool(Tool):
    """Tool for removing edges from the graph database."""
    
    def __init__(self):
        super().__init__(
            name="remove_edge",
            description="Remove an edge from the graph database by ID"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="edge_id",
                    type="string",
                    description="ID of the edge to remove"
                )
            ],
            return_type="boolean",
            return_description="True if the edge was removed, False otherwise"
        )
    
    async def execute(self, **kwargs) -> bool:
        """Remove a relationship from the knowledge graph."""
        edge_id = kwargs.get("edge_id")
        if not edge_id:
            return False
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        return await graph.remove_relationship(edge_id)

    def format_result(self, result: bool) -> str:
        """Format the result into a readable, concise string."""
        if result:
            return "Edge removed ."
        return "Edge removal failed. The edge may not have existed."

class FindNodesTool(Tool):
    """Tool for finding nodes in the graph database."""
    
    def __init__(self):
        super().__init__(
            name="find_nodes",
            description="Find all nodes of a given type in the graph database"
        )
        self.entity_limits = {}  # Track limits for each entity type
        self.initial_limit = 100
        self.increment = 100
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="entity_type",
                    type="string",
                    description="Type of the entity to find"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of nodes to return",
                    required=False,
                    default=100
                ),
            ],
            return_type="list",
            return_description="List of nodes matching the criteria"
        )
    
    async def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """Find entities in the knowledge graph."""
        entity_type = kwargs.get("entity_type")
        properties = kwargs.get("properties")
        increase_limit = kwargs.get("increase_limit", False)

        # Use a combination of entity_type and a hash of properties for unique tracking
        key = f"{entity_type or ''}-{hash(json.dumps(properties, sort_keys=True)) if properties else ''}"
        
        limit = self.entity_limits.get(key, self.initial_limit)
        if increase_limit:
            limit += self.increment
        self.entity_limits[key] = limit

        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        nodes = await graph.find_entities(entity_type, properties, limit=limit)
        return [node.to_dict() for node in nodes]

    def format_result(self, result: List[Dict[str, Any]]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No nodes found matching the criteria."
        
        lines = []
        for node in result:
            node_id = node.get('id')
            node_type = node.get('type')
            description = node.get('description')
            desc_str = f' "{description}"' if description else ""
            lines.append(f'{node_id} ({node_type}){desc_str}')
            
        return "\n".join(lines)

class FindEdgesTool(Tool):
    """Tool for finding edges in the graph database."""
    
    def __init__(self):
        super().__init__(
            name="find_edges",
            description="Find all edges of a given type in the graph database"
        )
        self.entity_limits = {}  # Track limits for each relationship type
        self.initial_limit = 100
        self.increment = 100
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="relationship_type",
                    type="string",
                    description="Type of the relationship to find"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of edges to return",
                    required=False,
                    default=100
                ),
            ],
            return_type="list",
            return_description="List of edges matching the criteria"
        )
    
    async def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """Find relationships in the knowledge graph."""
        relationship_type = kwargs.get("relationship_type")
        increase_limit = kwargs.get("increase_limit", False)

        key = relationship_type or ""
        
        limit = self.entity_limits.get(key, self.initial_limit)
        if increase_limit:
            limit += self.increment
        self.entity_limits[key] = limit

        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        edges = await graph.find_relationships(relationship_type, limit=limit)
        return [edge.to_dict() for edge in edges]

    def format_result(self, result: List[Dict[str, Any]]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No edges found matching the criteria."
        
        lines = []
        for edge in result:
            source_id = edge.get('source_id')
            target_id = edge.get('target_id')
            rel_type = edge.get('type')
            lines.append(f"({source_id})-[{rel_type}]->({target_id})")
            
        return "\n".join(lines)

class GetNeighborsTool(Tool):
    """Tool for retrieving neighboring nodes of a given node."""
    
    def __init__(self):
        super().__init__(
            name="get_neighbors",
            description="Get neighboring nodes of a node by ID"
        )
        self.entity_limits = {}  # Track limits for each entity
        self.initial_limit = 100
        self.increment = 100
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="node_id",
                    type="string",
                    description="ID of the node to find neighbors for"
                ),
                ToolParameter(
                    name="relationship_type",
                    type="string",
                    description="Type of relationship to follow",
                    required=False
                ),
                ToolParameter(
                    name="direction",
                    type="string",
                    description="Direction of the relationship (inbound/outbound)",
                    required=False,
                    default="outbound"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of neighbors to return",
                    required=False,
                    default=100
                )
            ],
            return_type="list",
            return_description="List of neighboring nodes"
        )
    
    async def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """Get neighbors of an entity in the knowledge graph."""
        node_id = kwargs.get("node_id")
        relationship_type = kwargs.get("relationship_type")
        increase_limit = kwargs.get("increase_limit", False)

        if not node_id:
            return []
        
        limit = self.entity_limits.get(node_id, self.initial_limit)
        if increase_limit:
            limit += self.increment
        self.entity_limits[node_id] = limit

        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        # This needs to be more complex to get the edge info as well.
        # For now, we get neighbors and then their relationships to the source node.
        neighbors = await graph.get_neighbors(node_id, relationship_type, limit=limit)
        
        results = []
        for neighbor in neighbors:
            # This could be slow if there are many neighbors.
            # A dedicated DB query would be better.
            rels = await graph.get_entity_relationships(neighbor.id)
            
            # Find the specific edge connecting to the source node_id
            connecting_edge = None
            for r in rels:
                if (r.source_id == node_id and r.target_id == neighbor.id) or \
                   (r.source_id == neighbor.id and r.target_id == node_id):
                   connecting_edge = r
                   break

            results.append({
                "neighbor": neighbor.to_dict(),
                "edge": connecting_edge.to_dict() if connecting_edge else None
            })

        return results

    def format_result(self, result: List[Dict[str, Any]]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "No neighbors found."
            
        lines = []
        for item in result:
            neighbor = item.get('neighbor', {})
            edge = item.get('edge')

            node_id = neighbor.get('id')
            node_type = neighbor.get('type')
            description = neighbor.get('description')
            desc_str = f' "{description}"' if description else ""
            lines.append(f'{node_id} ({node_type}){desc_str}')

            if edge:
                source = edge.get('source_id')
                target = edge.get('target_id')
                rel_type = edge.get('type')
                lines.append(f"  - Edge: ({source})-[{rel_type}]->({target})")
            else:
                lines.append(f"  - Edge: (No direct edge info found)")

        return "\n".join(lines)

class GetSubgraphTool(Tool):
    """Tool for retrieving a subgraph around a given node."""
    
    def __init__(self):
        super().__init__(
            name="get_subgraph",
            description="Get a subgraph from the graph database"
        )
        self.max_relationships_for_full_view = 20
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="node_ids",
                    type="array",
                    description="List of node IDs to include in the subgraph"
                ),
                ToolParameter(
                    name="relationship_types",
                    type="array",
                    description="List of relationship types to include",
                    required=False
                ),
                ToolParameter(
                    name="max_depth",
                    type="integer",
                    description="Maximum depth of relationships to follow",
                    required=False,
                    default=1
                )
            ],
            return_type="object",
            return_description="Subgraph data including nodes and relationships"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Get a subgraph from the knowledge graph."""
        node_ids = kwargs.get("node_ids", [])
        max_depth = kwargs.get("max_depth", 1)
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        subgraph = await graph.get_subgraph(node_ids, max_depth)
        return subgraph.to_json()

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the result into a readable, concise string."""
        nodes = result.get('nodes', [])
        edges = result.get('edges', [])
        node_count = len(nodes)
        edge_count = len(edges)

        if node_count == 0 and edge_count == 0:
            return "Subgraph is empty."

        if edge_count < self.max_relationships_for_full_view:
            # Show full triples
            output = f"Subgraph with {node_count} nodes and {edge_count} edges:\n"
            node_types = {node['id']: node['type'] for node in nodes}
            
            for edge in edges:
                source_id = edge.get('source_id')
                target_id = edge.get('target_id')
                rel_type = edge.get('type')
                
                source_type = node_types.get(source_id, 'Unknown')
                target_type = node_types.get(target_id, 'Unknown')
                
                source_str = f"{source_id} [{source_type}]"
                target_str = f"{target_id} [{target_type}]"
                
                output += f"- ({source_str})-[{rel_type}]->({target_str})\n"
            return output.strip()
        else:
            # Show summary
            output = f"Subgraph with {node_count} nodes and {edge_count} edges (too large to show all relationships):\n"
            output += "\nNodes:\n"
            for node in nodes[:10]: # Preview some nodes
                node_id = node.get('id')
                node_type = node.get('type')
                description = node.get('description')
                desc_str = f' "{description}"' if description else ""
                output += f"- {node_id} ({node_type}){desc_str}\n"
            if node_count > 10:
                output += f"- ... and {node_count - 10} more nodes\n"

            edge_types = sorted(list(set(edge.get('type') for edge in edges)))
            output += "\nEdge Types:\n"
            for edge_type in edge_types:
                output += f"- {edge_type}\n"
            return output.strip()

class GetGraphStatsTool(Tool):
    """Tool for retrieving statistics about the graph."""
    
    def __init__(self):
        super().__init__(
            name="get_graph_stats",
            description="Get statistics of the graph database"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[],
            return_type="object",
            return_description="Graph statistics including node count, edge count, and label information"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Get statistics from the knowledge graph."""
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        return await graph.get_graph_statistics()

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the result into a readable, concise string."""
        if not result:
            return "Could not retrieve graph stats."
        return f"Graph Stats: Nodes={result.get('node_count', 0)}, Edges={result.get('relationship_count', 0)}, Labels={len(result.get('labels', []))}, RelationshipTypes={len(result.get('relationship_types', []))}."

class CypherQueryTool(Tool):
    """Tool for executing Cypher queries on the graph database."""
    
    def __init__(self):
        super().__init__(
            name="cypher_query",
            description="Execute a Cypher query against the graph database"
        )
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="cypher_query",
                    type="string",
                    description="The Cypher query to execute"
                ),
                ToolParameter(
                    name="parameters",
                    type="object",
                    description="Parameters for the Cypher query",
                    required=False,
                    default={}
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout for the query in seconds",
                    required=False
                )
            ],
            return_type="object",
            return_description="Result of the Cypher query execution"
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute a Cypher query with enhanced error handling and result formatting."""
        
        cypher_query = kwargs.get("cypher_query")
        parameters = kwargs.get("parameters", {})
        timeout = kwargs.get("timeout", 30)
        
        if not cypher_query:
            return {
                "success": False,
                "error": "cypher_query is required",
                "results": [],
                "metadata": {
                    "record_count": 0,
                    "execution_time_seconds": 0,
                    "query": "",
                    "parameters": {}
                }
            }
        
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        start_time = time.time()
        
        try:
            # Execute the query
            results = await graph.execute_custom_query(cypher_query, parameters)
            
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "results": results,
                "metadata": {
                    "record_count": len(results),
                    "execution_time_seconds": round(execution_time, 3),
                    "query": cypher_query,
                    "parameters": parameters
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "metadata": {
                    "record_count": 0,
                    "execution_time_seconds": round(execution_time, 3),
                    "query": cypher_query,
                    "parameters": parameters
                }
            }

    def format_result(self, result: Dict[str, Any]) -> str:
        """Format the result into a readable, concise string."""
        if not result.get("success"):
            return f"Cypher query failed: {result.get('error', 'Unknown error')}"
            
        metadata = result.get('metadata', {})
        record_count = metadata.get('record_count', 0)
        execution_time = metadata.get('execution_time_seconds', 0)
        
        if record_count == 0:
            return f"Cypher query successful, returned no records. Time: {execution_time}s"
        
        # Convert results to JSON-serializable format for preview
        results_preview = []
        for record in result.get('results', [])[:2]:
            serializable_record = {}
            for key, value in record.items():
                if hasattr(value, 'to_dict'):
                    # Handle Node/Edge objects that have to_dict method
                    serializable_record[key] = value.to_dict()
                elif isinstance(value, (str, int, float, bool, type(None))):
                    # Handle primitive types
                    serializable_record[key] = value
                else:
                    # Convert other objects to string representation
                    serializable_record[key] = str(value)
            results_preview.append(serializable_record)
            
        return f"Cypher query successful. Records: {record_count}, Time: {execution_time}s. Preview: {json.dumps(results_preview)}"

# Export all tools for registration
all_tools = [
    AddNodeTool,
    AddEdgeTool,
    GetNodeTool,
    GetEdgeTool,
    RemoveNodeTool,
    RemoveEdgeTool,
    FindNodesTool,
    FindEdgesTool,
    GetNeighborsTool,
    GetSubgraphTool,
    GetGraphStatsTool,
    CypherQueryTool,
    FetchNodeTool,
    FetchRelationshipTool
]

def register_tools():
    for tool in all_tools:
        tool_instance = tool()
        # Register each tool instance with the framework
        # This is a placeholder for the actual registration logic
        print(f"Registering tool: {tool_instance.name}")

if __name__ == '__main__':
    import asyncio
    import time

    async def run_tests():
        # Mock graph data for testing
        graph = get_knowledge_graph()
        if not await graph.is_connected():
            await graph.connect()
        
        # Clear graph for clean test run
        print("--- Clearing Graph ---")
        await graph.execute_custom_query("MATCH (n) DETACH DELETE n")
        print("Graph cleared.")

        # Test GetGraphStatsTool on empty graph
        print("\n--- Testing GetGraphStatsTool (Empty) ---")
        stats_tool = GetGraphStatsTool()
        result = await stats_tool.execute()
        print(f"Initial stats: {stats_tool.format_result(result)}")

        # Test AddNodeTool
        print("\n--- Testing AddNodeTool ---")
        add_node_tool = AddNodeTool()
        p1_id = await add_node_tool.execute(entity_id='person1', entity_type='Person', description='A person named Alice', properties={'label': 'Alice', 'age': 30})
        print(f"Add person1: {add_node_tool.format_result(p1_id)}")
        p2_id = await add_node_tool.execute(entity_id='person2', entity_type='Person', description='A person named Bob', properties={'label': 'Bob', 'age': 25})
        print(f"Add person2: {add_node_tool.format_result(p2_id)}")
        c1_id = await add_node_tool.execute(entity_id='city1', entity_type='City', description='The Big Apple', properties={'label': 'New York', 'population': 8400000})
        print(f"Add city1: {add_node_tool.format_result(c1_id)}")

        # Test AddEdgeTool
        print("\n--- Testing AddEdgeTool ---")
        add_edge_tool = AddEdgeTool()
        edge1_id = await add_edge_tool.execute(from_node_id='person1', to_node_id='city1', relationship_type='LIVES_IN', description="Alice lives in New York", properties={'label': 'lives in'})
        print(f"Add edge p1-c1: {add_edge_tool.format_result(edge1_id)}")
        edge2_id = await add_edge_tool.execute(from_node_id='person2', to_node_id='city1', relationship_type='LIVES_IN', properties={'label': 'lives in'})
        print(f"Add edge p2-c1: {add_edge_tool.format_result(edge2_id)}")

        try:
            edge3_id = await add_edge_tool.execute(from_node_id='person3', to_node_id='city1', relationship_type='LIVES_IN', properties={'label': 'lives in'})
            print(f"Add edge p3-c1: {add_edge_tool.format_result(edge3_id)}")
        except ValueError as e:
            print(f"Expected error when adding edge with non-existent node: {e}")

        # Test GetNodeTool
        print("\n--- Testing GetNodeTool ---")
        get_node_tool = GetNodeTool()
        result = await get_node_tool.execute(node_id='person1')
        print(f"Get person1: {get_node_tool.format_result(result)}")
        result = await get_node_tool.execute(node_id='nonexistent')
        print(f"Get nonexistent: {get_node_tool.format_result(result)}")

        # Test GetEdgeTool
        print("\n--- Testing GetEdgeTool ---")
        get_edge_tool = GetEdgeTool()   
        result = await get_edge_tool.execute(edge_id=edge1_id)
        print(f"Get edge p1-c1: {get_edge_tool.format_result(result)}")

        # Test FindNodesTool
        print("\n--- Testing FindNodesTool ---")
        find_nodes_tool = FindNodesTool()
        
        # Initial call
        result = await find_nodes_tool.execute(entity_type='Person')
        print(f"Initial find:\n{find_nodes_tool.format_result(result)}")
        
        # Increase limit
        result = await find_nodes_tool.execute(entity_type='Person', increase_limit=True)
        print(f"Increased limit find:\n{find_nodes_tool.format_result(result)}")

        # Test FindEdgesTool
        print("\n--- Testing FindEdgesTool ---")
        find_edges_tool = FindEdgesTool()
        result = await find_edges_tool.execute(relationship_type='LIVES_IN')
        print(f"Find 'LIVES_IN' edges:\n{find_edges_tool.format_result(result)}")

        # Test GetNeighborsTool
        print("\n--- Testing GetNeighborsTool ---")
        get_neighbors_tool = GetNeighborsTool()
        
        # Add more neighbors for sorting test
        for i in range(5):
            node_id = f'friend{i}'
            await graph.add_entity(Node(node_id=node_id, node_type='Person', label=f'Friend {i}'))
            await graph.add_relationship(Edge(source_id='person1', target_id=node_id, relationship_type='FRIEND_OF'))
        
        # Initial call
        result = await get_neighbors_tool.execute(node_id='person1')
        print(f"Initial neighbors:\n{get_neighbors_tool.format_result(result)}")
        
        # Increase limit
        result = await get_neighbors_tool.execute(node_id='person1', increase_limit=True)
        print(f"Increased limit neighbors:\n{get_neighbors_tool.format_result(result)}")

        # Test CypherQueryTool
        print("\n--- Testing CypherQueryTool ---")
        cypher_tool = CypherQueryTool()
        query = "MATCH (p:Person)-[:LIVES_IN]->(c:City) WHERE c.label = $city_name RETURN p.label as name"
        params = {"city_name": "New York"}
        result = await cypher_tool.execute(cypher_query=query, parameters=params)
        print(f"Cypher query for people in NY: {cypher_tool.format_result(result)}")
        result = await cypher_tool.execute(cypher_query="MATCH (n) RETURN n.name LIMIT 1", timeout=5)
        print(f"Cypher query (success): {cypher_tool.format_result(result)}")
        result = await cypher_tool.execute(cypher_query="MATCH (n) RETUN n.name") # Intentional typo
        print(f"Cypher query (fail): {cypher_tool.format_result(result)}")

        # Test GetSubgraphTool
        print("\n--- Testing GetSubgraphTool ---")
        subgraph_tool = GetSubgraphTool()
        result = await subgraph_tool.execute(node_ids=['person1', 'city1'], max_depth=1)
        print(f"Get subgraph for p1, c1: {subgraph_tool.format_result(result)}")

        # Test RemoveEdgeTool
        print("\n--- Testing RemoveEdgeTool ---")
        remove_edge_tool = RemoveEdgeTool()
        result = await remove_edge_tool.execute(edge_id=edge2_id)
        print(f"Remove edge p2-c1: {remove_edge_tool.format_result(result)}")

        # Test RemoveNodeTool
        print("\n--- Testing RemoveNodeTool ---")
        remove_node_tool = RemoveNodeTool()
        result = await remove_node_tool.execute(node_id='person2')
        print(f"Remove person2: {remove_node_tool.format_result(result)}")
        result = await remove_node_tool.execute(node_id='person2')
        print(f"Remove person2 (again): {remove_node_tool.format_result(result)}")

        # Test GetGraphStatsTool at the end
        print("\n--- Testing GetGraphStatsTool (End) ---")
        result = await stats_tool.execute()
        print(f"Final stats: {stats_tool.format_result(result)}")

        # Test FetchNodeTool
        print("\n--- Testing FetchNodeTool ---")
        fetch_node_tool = FetchNodeTool()
        
        # Test fetching a real Wikidata entity: Albert Einstein (Q937)
        try:
            result = await fetch_node_tool.execute(entity_id='Q937')
            print(f"Fetch Q937: {fetch_node_tool.format_result(result)}")
        except Exception as e:
            print(f"Error fetching Q937: {e}")
        
        # Test fetching an already existing entity
        try:
            result = await fetch_node_tool.execute(entity_id='Q42')
            print(f"Fetch Q42 (duplicate): {fetch_node_tool.format_result(result)}")
        except Exception as e:
            print(f"Error fetching Q42 (duplicate): {e}")

        # Test FetchRelationshipTool
        print("\n--- Testing FetchRelationshipTool ---")
        fetch_relationship_tool = FetchRelationshipTool()
        
        # Test fetching a real Wikidata triple: Douglas Adams (Q42) - instance of (P31) - human (Q5)
        try:
            result = await fetch_relationship_tool.execute(
                subject_id='Q42',
                property_id='P31',
                object_id='Q5'
            )
            print(f"Fetch Q42-P31-Q5: {fetch_relationship_tool.format_result(result)}")
        except Exception as e:
            print(f"Error fetching Q42-P31-Q5: {e}")
        
        # Test fetching another triple with already existing subject
        try:
            result = await fetch_relationship_tool.execute(
                subject_id='Q42',
                property_id='P106',
                object_id='Q36180'
            )
            print(f"Fetch Q42-P106-Q36180: {fetch_relationship_tool.format_result(result)}")
        except Exception as e:
            print(f"Error fetching Q42-P106-Q36180: {e}")
        
        # Test with existing relationship
        try:
            result = await fetch_relationship_tool.execute(
                subject_id='Q42',
                property_id='P31',
                object_id='Q5'
            )
            print(f"Fetch Q42-P31-Q5 (duplicate): {fetch_relationship_tool.format_result(result)}")
        except Exception as e:
            print(f"Error fetching Q42-P31-Q5 (duplicate): {e}")
        
        # Test with non-existent relationship to verify verification works
        try:
            result = await fetch_relationship_tool.execute(
                subject_id='Q42',
                property_id='P31',
                object_id='Q6256'  # country - Douglas Adams is not a country
            )
            print(f"Fetch Q42-P31-Q6256 (non-existent): {fetch_relationship_tool.format_result(result)}")
        except Exception as e:
            print(f"Error fetching Q42-P31-Q6256 (non-existent): {e}")

        # Test RemoveEdgeTool
        print("\n--- Testing RemoveEdgeTool ---")
        remove_edge_tool = RemoveEdgeTool()
        result = await remove_edge_tool.execute(edge_id=edge2_id)
        print(f"Remove edge p2-c1: {remove_edge_tool.format_result(result)}")

        # Test RemoveNodeTool
        print("\n--- Testing RemoveNodeTool ---")
        remove_node_tool = RemoveNodeTool()
        result = await remove_node_tool.execute(node_id='person2')
        print(f"Remove person2: {remove_node_tool.format_result(result)}")
        result = await remove_node_tool.execute(node_id='person2')
        print(f"Remove person2 (again): {remove_node_tool.format_result(result)}")

        # Test GetGraphStatsTool at the end
        print("\n--- Testing GetGraphStatsTool (End) ---")
        result = await stats_tool.execute()
        print(f"Final stats: {stats_tool.format_result(result)}")

        await graph.close()

    asyncio.run(run_tests())
