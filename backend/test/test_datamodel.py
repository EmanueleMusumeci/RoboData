import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.toolbox.wikidata.datamodel import (
    WikidataStatement, WikidataEntity, WikidataProperty, SearchResult,
    convert_api_entity_to_model, convert_api_property_to_model, convert_api_search_to_model
)

class TestWikidataStatement:
    """Test WikidataStatement model."""
    
    def test_statement_creation_literal(self):
        """Test creating a statement with literal value."""
        stmt = WikidataStatement(
            property_id="P569",
            value="1952-03-11T00:00:00Z",
            datatype="time"
        )
        
        assert stmt.property_id == "P569"
        assert stmt.value == "1952-03-11T00:00:00Z"
        assert stmt.datatype == "time"
        assert stmt.is_entity_ref == False
        assert stmt.entity_type is None
    
    def test_statement_creation_entity_ref(self):
        """Test creating a statement with entity reference."""
        stmt = WikidataStatement(
            property_id="P31",
            value="Q5",
            datatype="wikibase-item",
            is_entity_ref=True,
            entity_type="item"
        )
        
        assert stmt.property_id == "P31"
        assert stmt.value == "Q5"
        assert stmt.datatype == "wikibase-item"
        assert stmt.is_entity_ref == True
        assert stmt.entity_type == "item"
    
    def test_statement_creation_property_ref(self):
        """Test creating a statement with property reference."""
        stmt = WikidataStatement(
            property_id="P1659",
            value="P106",
            datatype="wikibase-property",
            is_entity_ref=True,
            entity_type="property"
        )
        
        assert stmt.property_id == "P1659"
        assert stmt.value == "P106"
        assert stmt.is_entity_ref == True
        assert stmt.entity_type == "property"

class TestWikidataEntity:
    """Test WikidataEntity model."""
    
    def test_entity_creation_minimal(self):
        """Test creating entity with minimal data."""
        entity = WikidataEntity(
            id="Q42",
            label="Douglas Adams",
            link="https://www.wikidata.org/wiki/Q42"
        )
        
        assert entity.id == "Q42"
        assert entity.label == "Douglas Adams"
        assert entity.description is None
        assert entity.aliases == []
        assert entity.statements == {}
        assert entity.link == "https://www.wikidata.org/wiki/Q42"
    
    def test_entity_creation_full(self):
        """Test creating entity with full data."""
        statements = {
            "P31": [
                WikidataStatement(
                    property_id="P31",
                    value="Q5",
                    datatype="wikibase-item",
                    is_entity_ref=True,
                    entity_type="item"
                )
            ]
        }
        
        entity = WikidataEntity(
            id="Q42",
            label="Douglas Adams",
            description="British author and humorist",
            aliases=["Douglas Noel Adams", "DNA"],
            statements=statements,
            link="https://www.wikidata.org/wiki/Q42"
        )
        
        assert entity.id == "Q42"
        assert entity.label == "Douglas Adams"
        assert entity.description == "British author and humorist"
        assert "Douglas Noel Adams" in entity.aliases
        assert "DNA" in entity.aliases
        assert "P31" in entity.statements
        assert len(entity.statements["P31"]) == 1
        assert entity.statements["P31"][0].value == "Q5"

class TestWikidataProperty:
    """Test WikidataProperty model."""
    
    def test_property_creation(self):
        """Test creating property."""
        prop = WikidataProperty(
            id="P31",
            label="instance of",
            description="that class of which this subject is a particular example and member",
            datatype="wikibase-item",
            link="https://www.wikidata.org/wiki/P31"
        )
        
        assert prop.id == "P31"
        assert prop.label == "instance of"
        assert "class of which" in prop.description
        assert prop.datatype == "wikibase-item"
        assert prop.link == "https://www.wikidata.org/wiki/P31"

class TestSearchResult:
    """Test SearchResult model."""
    
    def test_search_result_creation(self):
        """Test creating search result."""
        result = SearchResult(
            id="Q42",
            label="Douglas Adams",
            description="British author",
            url="https://www.wikidata.org/wiki/Q42"
        )
        
        assert result.id == "Q42"
        assert result.label == "Douglas Adams"
        assert result.description == "British author"
        assert result.url == "https://www.wikidata.org/wiki/Q42"

class TestConversionFunctionsIntegration:
    """Test API conversion functions with real API data structures."""
    
    @pytest.mark.asyncio
    async def test_convert_real_entity_data(self):
        """Test converting real API entity data to model."""
        from core.toolbox.wikidata.wikidata_api import wikidata_api
        
        # Get real API data
        api_data = await wikidata_api.get_entity('Q42')
        
        # Convert to model
        entity = convert_api_entity_to_model(api_data)
        
        # Verify conversion
        assert entity.id == "Q42"
        assert entity.label == "Douglas Adams"
        assert entity.description is not None
        assert len(entity.aliases) > 0
        assert len(entity.statements) > 0
        assert entity.link == "https://www.wikidata.org/wiki/Q42"
        
        # Check that P31 (instance of) exists and has entity references
        if "P31" in entity.statements:
            p31_stmt = entity.statements["P31"][0]
            assert p31_stmt.is_entity_ref == True
            assert p31_stmt.entity_type == "item"
            assert p31_stmt.value.startswith("Q")
    
    @pytest.mark.asyncio
    async def test_convert_real_property_data(self):
        """Test converting real API property data to model."""
        from core.toolbox.wikidata.wikidata_api import wikidata_api
        
        # Get real API data
        api_data = await wikidata_api.get_property('P31')
        
        # Convert to model
        prop = convert_api_property_to_model(api_data)
        
        # Verify conversion
        assert prop.id == "P31"
        assert prop.label == "instance of"
        assert prop.description is not None
        assert prop.datatype == "wikibase-item"
        assert prop.link == "https://www.wikidata.org/wiki/P31"
    
    @pytest.mark.asyncio
    async def test_convert_real_search_data(self):
        """Test converting real API search data to model."""
        from core.toolbox.wikidata.wikidata_api import wikidata_api
        
        # Get real API data
        api_data = await wikidata_api.search_entities('Douglas Adams', limit=5)
        
        # Convert to model
        results = convert_api_search_to_model(api_data)
        
        # Verify conversion
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check first result
        first_result = results[0]
        assert first_result.id == "Q42"
        assert first_result.label == "Douglas Adams"
        assert first_result.description is not None

class TestModelValidation:
    """Test model validation and edge cases."""
    
    def test_entity_with_numeric_values(self):
        """Test entity with numeric statement values."""
        statements = {
            "P1082": [
                WikidataStatement(
                    property_id="P1082",
                    value=66650000,
                    datatype="quantity"
                )
            ]
        }
        
        entity = WikidataEntity(
            id="Q145",
            label="United Kingdom",
            statements=statements,
            link="https://www.wikidata.org/wiki/Q145"
        )
        
        assert entity.statements["P1082"][0].value == 66650000
        assert isinstance(entity.statements["P1082"][0].value, int)
    
    def test_entity_serialization(self):
        """Test entity can be serialized to dict."""
        entity = WikidataEntity(
            id="Q42",
            label="Douglas Adams",
            description="British author",
            aliases=["DNA"],
            link="https://www.wikidata.org/wiki/Q42"
        )
        
        data = entity.dict()
        
        assert data["id"] == "Q42"
        assert data["label"] == "Douglas Adams"
        assert data["description"] == "British author"
        assert data["aliases"] == ["DNA"]
        assert data["statements"] == {}
        assert data["link"] == "https://www.wikidata.org/wiki/Q42"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
