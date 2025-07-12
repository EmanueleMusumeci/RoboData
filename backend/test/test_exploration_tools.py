import pytest
import asyncio
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.toolbox.wikidata.exploration import (
    NeighborsExplorationTool, LocalGraphTool
)
from core.toolbox.wikidata.datamodel import (
    NeighborExplorationResult, LocalGraphResult, WikidataEntity
)

class TestNeighborsExplorationTool:
    """Test the NeighborsExplorationTool."""
    
    def test_neighbors_tool_definition(self):
        """Test neighbors exploration tool definition."""
        tool = NeighborsExplorationTool()
        definition = tool.get_definition()
        
        assert definition.name == "explore_entity_neighbors"
        assert "neighbor" in definition.description.lower()
        assert any(p.name == "entity_id" for p in definition.parameters)
        assert any(p.name == "include_properties" for p in definition.parameters)
        assert any(p.name == "max_values_per_property" for p in definition.parameters)
    
    @pytest.mark.asyncio
    async def test_neighbors_exploration_douglas_adams(self):
        """Test neighbors exploration with Douglas Adams (Q42)."""
        tool = NeighborsExplorationTool()
        
        result = await tool.execute("Q42", max_values_per_property=5)
        
        assert isinstance(result, NeighborExplorationResult)
        assert result.entity.id == "Q42"
        assert result.entity.label == "Douglas Adams"
        assert result.total_properties > 0
        assert result.neighbor_count >= 0
        assert isinstance(result.neighbors, dict)
        assert isinstance(result.relationships, dict)
        assert isinstance(result.property_names, dict)
    
    @pytest.mark.asyncio
    async def test_neighbors_exploration_with_specific_properties(self):
        """Test neighbors exploration with specific properties."""
        tool = NeighborsExplorationTool()
        
        # Focus on instance of (P31) and occupation (P106)
        result = await tool.execute("Q42", include_properties=["P31", "P106"])
        
        assert isinstance(result, NeighborExplorationResult)
        assert result.entity.id == "Q42"
        
        # Check that only specified properties are included
        for prop_id in result.relationships.keys():
            assert prop_id in ["P31", "P106"]
        
        # Verify property names were fetched
        if "P31" in result.property_names:
            assert "instance of" in result.property_names["P31"].lower()
    
    @pytest.mark.asyncio
    async def test_neighbors_exploration_albert_einstein(self):
        """Test neighbors exploration with Albert Einstein (Q937)."""
        tool = NeighborsExplorationTool()
        
        result = await tool.execute("Q937", max_values_per_property=5)
        
        assert isinstance(result, NeighborExplorationResult)
        assert result.entity.id == "Q937"
        assert "Einstein" in result.entity.label
        # Based on real data: Albert Einstein has 411 properties
        assert result.total_properties > 300
        # Based on real data: should find substantial neighbors (112 found in test)
        assert result.neighbor_count >= 50
        
        # Check that entity references are properly detected
        entity_refs_found = False
        for prop_id, values in result.relationships.items():
            for value in values:
                if value in result.neighbors:
                    entity_refs_found = True
                    assert isinstance(result.neighbors[value], WikidataEntity)
                    break
            if entity_refs_found:
                break
        
        # Einstein should have many entity references
        assert entity_refs_found
        
        # Verify common properties for Einstein are present
        common_props = ["P31", "P106", "P569", "P570", "P19", "P20"]  # instance of, occupation, birth date, death date, birth place, death place
        found_props = [prop for prop in common_props if prop in result.relationships]
        assert len(found_props) >= 3, f"Should find at least 3 common properties, found: {found_props}"

class TestLocalGraphTool:
    """Test the LocalGraphTool."""
    
    def test_local_graph_tool_definition(self):
        """Test local graph tool definition."""
        tool = LocalGraphTool()
        definition = tool.get_definition()
        
        assert definition.name == "build_local_graph"
        assert "graph" in definition.description.lower()
        assert any(p.name == "center_entity" for p in definition.parameters)
        assert any(p.name == "depth" for p in definition.parameters)
        assert any(p.name == "properties" for p in definition.parameters)
        assert any(p.name == "max_nodes" for p in definition.parameters)
    
    @pytest.mark.asyncio
    async def test_local_graph_basic_douglas_adams(self):
        """Test basic local graph building with Douglas Adams (Q42)."""
        tool = LocalGraphTool()
        
        result = await tool.execute("Q42", depth=2, max_nodes=20)
        
        assert isinstance(result, LocalGraphResult)
        assert result.center == "Q42"
        assert result.depth == 2
        assert result.total_nodes > 0
        assert result.total_edges >= 0
        assert isinstance(result.nodes, dict)
        assert isinstance(result.edges, list)
        assert isinstance(result.property_names, dict)
        
        # Center entity should be in nodes
        assert "Q42" in result.nodes
        assert result.nodes["Q42"].label == "Douglas Adams"
    
    @pytest.mark.asyncio
    async def test_local_graph_basic_albert_einstein(self):
        """Test basic local graph building with Albert Einstein (Q937)."""
        tool = LocalGraphTool()
        
        result = await tool.execute("Q937", depth=2, max_nodes=20)
        
        assert isinstance(result, LocalGraphResult)
        assert result.center == "Q937"
        assert result.depth == 2
        # Based on real data: with default properties P279, P31, should get 2+ nodes
        assert result.total_nodes >= 2
        # Based on real data: should get multiple edges (7 found in test)
        assert result.total_edges >= 5
        assert isinstance(result.nodes, dict)
        assert isinstance(result.edges, list)
        assert isinstance(result.property_names, dict)
        
        # Center entity should be in nodes
        assert "Q937" in result.nodes
        assert "Einstein" in result.nodes["Q937"].label
        
        # Should find human (Q5) as a connected entity via P31
        if result.total_nodes > 1:
            human_found = any("human" in node.label.lower() for node in result.nodes.values())
            assert human_found, "Should find 'human' entity connected via P31"
    
    @pytest.mark.asyncio
    async def test_local_graph_with_custom_properties(self):
        """Test local graph with custom properties."""
        tool = LocalGraphTool()
        
        # Use instance of (P31) and occupation (P106)
        properties = ["P31", "P106"]
        result = await tool.execute("Q42", depth=2, properties=properties, max_nodes=15)
        
        assert isinstance(result, LocalGraphResult)
        assert result.properties == properties
        assert result.center == "Q42"
        
        # Check that property names were fetched
        for prop in properties:
            if prop in result.property_names:
                assert len(result.property_names[prop]) > 0
        
        # Check that edges use the specified properties
        if result.edges:
            edge_properties = set(edge["property"] for edge in result.edges)
            for prop in edge_properties:
                assert prop in properties
    
    @pytest.mark.asyncio
    async def test_local_graph_with_einstein_family_properties(self):
        """Test local graph with Einstein's family-related properties."""
        tool = LocalGraphTool()
        
        # Use family-related properties that Einstein definitely has
        properties = ["P22", "P40", "P26"]  # father, child, spouse
        result = await tool.execute("Q937", depth=2, properties=properties, max_nodes=15)
        
        assert isinstance(result, LocalGraphResult)
        assert result.properties == properties
        assert result.center == "Q937"
        
        # Should find family members
        # Based on real data: Einstein has father, children
        if result.total_nodes > 1:
            family_members = [node.label for node in result.nodes.values() if node.id != "Q937"]
            # Look for Einstein family members
            has_family = any("Einstein" in name for name in family_members)
            assert has_family, f"Should find Einstein family members, found: {family_members}"
        
        # Check that property names were fetched
        expected_props = ["P22", "P40"]  # father, child (more likely to be present)
        for prop in expected_props:
            if prop in result.property_names:
                assert len(result.property_names[prop]) > 0
    
    @pytest.mark.asyncio
    async def test_local_graph_depth_limits(self):
        """Test local graph with different depth limits."""
        tool = LocalGraphTool()
        
        # Test depth 1
        result_depth1 = await tool.execute("Q42", depth=1, max_nodes=10)
        assert result_depth1.depth == 1
        
        # Test depth 3 (if nodes allow)
        result_depth3 = await tool.execute("Q42", depth=3, max_nodes=25)
        assert result_depth3.depth == 3
        
        # Depth 3 should generally have more or equal nodes than depth 1
        # (unless hitting max_nodes limit)
        if result_depth1.total_nodes < 10 and result_depth3.total_nodes < 25:
            assert result_depth3.total_nodes >= result_depth1.total_nodes
    
    @pytest.mark.asyncio
    async def test_local_graph_depth_limits_realistic(self):
        """Test local graph with different depth limits using realistic expectations."""
        tool = LocalGraphTool()
        
        # Test depth 1 with smaller limit
        result_depth1 = await tool.execute("Q937", depth=1, max_nodes=5)
        assert result_depth1.depth == 1
        assert result_depth1.total_nodes <= 5
        
        # Test depth 2 with larger limit
        result_depth2 = await tool.execute("Q937", depth=2, max_nodes=15)
        assert result_depth2.depth == 2
        
        # Depth 2 should generally have more or equal nodes than depth 1
        # (unless hitting max_nodes limit for depth 1)
        if result_depth1.total_nodes < 5:
            assert result_depth2.total_nodes >= result_depth1.total_nodes
    
    @pytest.mark.asyncio
    async def test_local_graph_max_nodes_limit(self):
        """Test that max_nodes limit is respected."""
        tool = LocalGraphTool()
        
        result = await tool.execute("Q42", depth=3, max_nodes=5)
        
        assert result.total_nodes <= 5
        assert len(result.nodes) <= 5

class TestExplorationToolsIntegration:
    """Integration tests for exploration tools."""
    
    @pytest.mark.asyncio
    async def test_exploration_workflow(self):
        """Test complete exploration workflow."""
        # 1. Start with neighbors exploration
        neighbors_tool = NeighborsExplorationTool()
        neighbors_result = await neighbors_tool.execute("Q937", max_values_per_property=3)
        
        assert neighbors_result.entity.id == "Q937"
        assert neighbors_result.neighbor_count >= 0
        
        # 2. Build local graph around the same entity
        graph_tool = LocalGraphTool()
        graph_result = await graph_tool.execute("Q937", depth=2, max_nodes=15)
        
        assert graph_result.center == "Q937"
        assert graph_result.total_nodes > 0
        
        # 3. Verify consistency - center entity should be the same
        assert neighbors_result.entity.id == graph_result.center
        assert neighbors_result.entity.label == graph_result.nodes["Q937"].label
    
    @pytest.mark.asyncio
    async def test_exploration_workflow_einstein(self):
        """Test complete exploration workflow with Einstein."""
        # 1. Start with neighbors exploration
        neighbors_tool = NeighborsExplorationTool()
        neighbors_result = await neighbors_tool.execute("Q937", max_values_per_property=3)
        
        assert neighbors_result.entity.id == "Q937"
        # Based on real data: Einstein should have many neighbors
        assert neighbors_result.neighbor_count >= 20
        assert neighbors_result.total_properties >= 300
        
        # 2. Build local graph around the same entity
        graph_tool = LocalGraphTool()
        graph_result = await graph_tool.execute("Q937", depth=2, max_nodes=15)
        
        assert graph_result.center == "Q937"
        assert graph_result.total_nodes >= 2
        
        # 3. Verify consistency - center entity should be the same
        assert neighbors_result.entity.id == graph_result.center
        assert neighbors_result.entity.label == graph_result.nodes["Q937"].label
    
    @pytest.mark.asyncio
    async def test_exploration_tools_with_different_entities(self):
        """Test exploration tools with different entity types."""
        entities_to_test = [
            ("Q42", "Douglas Adams"),  # Person
            ("Q5", "human"),           # Class/Concept
        ]
        
        neighbors_tool = NeighborsExplorationTool()
        graph_tool = LocalGraphTool()
        
        for entity_id, expected_label in entities_to_test:
            # Test neighbors exploration
            neighbors_result = await neighbors_tool.execute(entity_id, max_values_per_property=2)
            assert neighbors_result.entity.id == entity_id
            assert expected_label.lower() in neighbors_result.entity.label.lower()
            
            # Test local graph building
            graph_result = await graph_tool.execute(entity_id, depth=2, max_nodes=10)
            assert graph_result.center == entity_id
            assert expected_label.lower() in graph_result.nodes[entity_id].label.lower()
    
    @pytest.mark.asyncio
    async def test_exploration_tools_with_different_complexity_entities(self):
        """Test exploration tools with entities of different complexity."""
        entities_to_test = [
            ("Q42", "Douglas Adams", 50, 5),      # Person - moderate complexity
            ("Q937", "Einstein", 300, 50),        # Person - high complexity
            ("Q5", "human", 100, 10),             # Class/Concept - variable complexity
        ]
        
        neighbors_tool = NeighborsExplorationTool()
        graph_tool = LocalGraphTool()
        
        for entity_id, expected_label_part, min_properties, min_neighbors in entities_to_test:
            # Test neighbors exploration
            neighbors_result = await neighbors_tool.execute(entity_id, max_values_per_property=5)
            assert neighbors_result.entity.id == entity_id
            assert expected_label_part.lower() in neighbors_result.entity.label.lower()
            assert neighbors_result.total_properties >= min_properties
            assert neighbors_result.neighbor_count >= min_neighbors
            
            # Test local graph building
            graph_result = await graph_tool.execute(entity_id, depth=2, max_nodes=10)
            assert graph_result.center == entity_id
            assert expected_label_part.lower() in graph_result.nodes[entity_id].label.lower()
            assert graph_result.total_nodes >= 1

class TestExplorationToolsErrorHandling:
    """Test error handling in exploration tools."""
    
    @pytest.mark.asyncio
    async def test_neighbors_tool_invalid_entity(self):
        """Test neighbors tool with invalid entity ID."""
        tool = NeighborsExplorationTool()
        
        with pytest.raises(Exception):  # Should raise some kind of error
            await tool.execute("Q999999999999")
    
    @pytest.mark.asyncio
    async def test_local_graph_tool_invalid_entity(self):
        """Test local graph tool with invalid entity ID."""
        tool = LocalGraphTool()
        
        with pytest.raises(Exception):  # Should raise some kind of error
            await tool.execute("Q999999999999")
    
    @pytest.mark.asyncio
    async def test_exploration_tools_empty_properties(self):
        """Test exploration tools with empty property lists."""
        neighbors_tool = NeighborsExplorationTool()
        graph_tool = LocalGraphTool()
        
        # Test with empty include_properties (should include all)
        neighbors_result = await neighbors_tool.execute("Q42", include_properties=[])
        assert neighbors_result.total_properties > 0
        
        # Test with empty properties list (should use defaults)
        graph_result = await graph_tool.execute("Q42", properties=[], max_nodes=5)
        assert graph_result.total_nodes > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
