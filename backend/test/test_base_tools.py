import pytest
import asyncio
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.toolbox.wikidata.base import (
    get_entity_info, get_entity_info_async,
    get_property_info, get_property_info_async,
    search_entities, search_entities_async
)
from core.toolbox.wikidata.datamodel import WikidataEntity, WikidataProperty, SearchResult

class TestBaseFunctionsIntegration:
    """Integration tests using real API calls through the complete pipeline."""
    
    @pytest.mark.asyncio
    async def test_get_entity_info_async_douglas_adams(self):
        """Test async entity info retrieval with real API."""
        result = await get_entity_info_async("Q42")
        
        # Verify it's the right entity
        assert isinstance(result, WikidataEntity)
        assert result.id == "Q42"
        assert result.label == "Douglas Adams"
        assert result.description is not None
        assert len(result.aliases) > 0
        assert len(result.statements) > 0
        
        # Check for expected properties
        if "P31" in result.statements:  # instance of
            assert len(result.statements["P31"]) > 0
            # Should be human (Q5)
            human_found = any(stmt.value == "Q5" for stmt in result.statements["P31"])
            assert human_found
            
            # Check entity reference detection
            p31_stmt = result.statements["P31"][0]
            assert p31_stmt.is_entity_ref == True
            assert p31_stmt.entity_type == "item"
    
    @pytest.mark.asyncio
    async def test_get_property_info_async_instance_of(self):
        """Test async property info retrieval with real API."""
        result = await get_property_info_async("P31")
        
        # Verify it's the right property
        assert isinstance(result, WikidataProperty)
        assert result.id == "P31"
        assert result.label == "instance of"
        assert result.description is not None
        assert result.datatype == "wikibase-item"
        assert result.link == "https://www.wikidata.org/wiki/P31"
    
    @pytest.mark.asyncio
    async def test_search_entities_async_douglas_adams(self):
        """Test async entity search with real API."""
        result = await search_entities_async("Douglas Adams", limit=5)
        
        # Verify search results
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= 5
        
        # First result should be Douglas Adams
        first_result = result[0]
        assert isinstance(first_result, SearchResult)
        assert first_result.id == "Q42"
        assert first_result.label == "Douglas Adams"
        assert first_result.description is not None
    
    def test_get_entity_info_sync_wrapper(self):
        """Test sync wrapper for entity info."""
        result = get_entity_info("Q42")
        
        assert isinstance(result, WikidataEntity)
        assert result.id == "Q42"
        assert result.label == "Douglas Adams"
    
    def test_get_property_info_sync_wrapper(self):
        """Test sync wrapper for property info."""
        result = get_property_info("P31")
        
        assert isinstance(result, WikidataProperty)
        assert result.id == "P31"
        assert result.label == "instance of"
    
    def test_search_entities_sync_wrapper(self):
        """Test sync wrapper for search."""
        result = search_entities("Douglas Adams", limit=3)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= 3
        assert result[0].id == "Q42"

class TestEntityStatementConversion:
    """Test entity statement conversion with real data."""
    
    @pytest.mark.asyncio
    async def test_entity_with_different_statement_types(self):
        """Test entity with various statement types."""
        result = await get_entity_info_async("Q42")
        
        # Check for different types of statements
        statement_types = {}
        for prop_id, statements in result.statements.items():
            for stmt in statements:
                if stmt.is_entity_ref:
                    statement_types[f"{prop_id}_entity_{stmt.entity_type}"] = True
                else:
                    statement_types[f"{prop_id}_literal"] = True
        
        # Should have at least some entity references
        has_entity_refs = any("_entity_" in key for key in statement_types.keys())
        assert has_entity_refs, "Should have entity references"
    
    @pytest.mark.asyncio
    async def test_property_references_in_statements(self):
        """Test detection of property references in statements."""
        result = await get_entity_info_async("Q42")
        
        # Look for property references (entities starting with P)
        property_refs_found = False
        for prop_id, statements in result.statements.items():
            for stmt in statements:
                if stmt.is_entity_ref and stmt.entity_type == "property":
                    property_refs_found = True
                    assert stmt.value.startswith("P")
                    break
            if property_refs_found:
                break
        
        # Property references are not common, so this test might not always find them
        print(f"Property references found: {property_refs_found}")

class TestErrorHandlingIntegration:
    """Test error handling with real API calls."""
    
    @pytest.mark.asyncio
    async def test_get_entity_info_invalid_id(self):
        """Test handling of invalid entity ID."""
        with pytest.raises(ValueError, match="Failed to get entity"):
            await get_entity_info_async("Q999999999999")
    
    @pytest.mark.asyncio
    async def test_get_property_info_invalid_id(self):
        """Test handling of invalid property ID."""
        with pytest.raises(ValueError, match="Failed to get property"):
            await get_property_info_async("P999999999")
    
    @pytest.mark.asyncio
    async def test_search_entities_empty_results(self):
        """Test search with query that returns no results."""
        result = await search_entities_async("xyzabcnonexistentquery123456789", limit=1)
        
        assert isinstance(result, list)
        assert len(result) == 0

class TestComprehensiveWorkflow:
    """Test complete workflow from search to entity exploration."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test a complete workflow: search -> get entity -> get property."""
        
        # 1. Search for Douglas Adams
        search_results = await search_entities_async("Douglas Adams", limit=3)
        assert len(search_results) > 0
        
        entity_id = search_results[0].id
        assert entity_id == "Q42"
        
        # 2. Get detailed entity information
        entity = await get_entity_info_async(entity_id)
        assert entity.id == entity_id
        assert entity.label == "Douglas Adams"
        
        # 3. Get information about a property used in the entity
        if "P31" in entity.statements:
            prop_info = await get_property_info_async("P31")
            assert prop_info.id == "P31"
            assert prop_info.label == "instance of"
        
        # 4. Verify statement structure
        assert len(entity.statements) > 0
        
        # Check that entity references are properly detected
        entity_refs_found = 0
        for prop_id, statements in entity.statements.items():
            for stmt in statements:
                if stmt.is_entity_ref:
                    entity_refs_found += 1
                    # Entity references should start with Q or P
                    assert stmt.value.startswith(("Q", "P"))
                    assert stmt.entity_type in ["item", "property"]
        
        assert entity_refs_found > 0, "Should find entity references in statements"

class TestDatamodelConsistency:
    """Test consistency between API data and datamodel."""
    
    @pytest.mark.asyncio
    async def test_conversion_consistency(self):
        """Test that conversion produces consistent results."""
        from core.toolbox.wikidata.wikidata_api import wikidata_api
        from core.toolbox.wikidata.datamodel import convert_api_entity_to_model
        
        # Get raw API data
        api_data = await wikidata_api.get_entity("Q42")
        
        # Convert using our function
        entity = convert_api_entity_to_model(api_data)
        
        # Get via base functions
        entity_via_base = await get_entity_info_async("Q42")
        
        # Should be equivalent
        assert entity.id == entity_via_base.id
        assert entity.label == entity_via_base.label
        assert entity.description == entity_via_base.description
        assert entity.aliases == entity_via_base.aliases
        assert len(entity.statements) == len(entity_via_base.statements)

class TestMultipleEntities:
    """Test with multiple different entities to verify robustness."""
    
    @pytest.mark.asyncio
    async def test_different_entity_types(self):
        """Test with different types of entities."""
        entities_to_test = [
            ("Q42", "Douglas Adams"),  # Person
            ("Q5", "human"),           # Class
            ("Q1", "Universe"),        # Concept
        ]
        
        for entity_id, expected_label in entities_to_test:
            result = await get_entity_info_async(entity_id)
            assert result.id == entity_id
            assert result.label == expected_label
            assert len(result.statements) > 0
    
    @pytest.mark.asyncio
    async def test_different_properties(self):
        """Test with different types of properties."""
        properties_to_test = [
            ("P31", "instance of", "wikibase-item"),
            ("P279", "subclass of", "wikibase-item"),
            ("P569", "date of birth", "time"),
        ]
        
        for prop_id, expected_label, expected_datatype in properties_to_test:
            result = await get_property_info_async(prop_id)
            assert result.id == prop_id
            assert result.label == expected_label
            assert result.datatype == expected_datatype

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
