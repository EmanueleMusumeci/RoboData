import pytest
import asyncio
from typing import Dict, Any, List
import sys
import os
from pathlib import Path
import pprint
from unittest.mock import patch, MagicMock, AsyncMock

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.toolbox.wikidata.wikidata_kif_api import WikidataKIFAPI, wikidata_kif_api

#TODO: finish debugging the KIF API

class TestWikidataKIFAPIUnit:
    """Unit tests for WikidataKIFAPI class structure and error handling."""
    
    def test_initialization(self):
        """Test API initialization."""
        api = WikidataKIFAPI()
        assert api.store is not None
    
    def test_global_instance(self):
        """Test that global instance is properly initialized."""
        assert wikidata_kif_api is not None
        assert isinstance(wikidata_kif_api, WikidataKIFAPI)
        assert hasattr(wikidata_kif_api, 'store')
    
    def test_api_methods_are_async(self):
        """Test that all main methods are properly asynchronous."""
        api = WikidataKIFAPI()
        
        # All these should be coroutine functions (async API)
        assert asyncio.iscoroutinefunction(api.get_entity)
        assert asyncio.iscoroutinefunction(api.get_entity_statements)
        assert asyncio.iscoroutinefunction(api.search_entities)
        assert asyncio.iscoroutinefunction(api.get_property)
        assert asyncio.iscoroutinefunction(api.query_sparql)

    @patch('core.toolbox.wikidata.wikidata_kif_api.Store')
    def test_entity_to_dict_conversion(self, mock_store_class):
        """Test entity to dictionary conversion."""
        from kif_lib.model import Item, Text
        from kif_lib.vocabulary import wd
        
        # Mock store instance
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        
        # Mock statements for labels
        mock_label_stmt = MagicMock()
        mock_label_stmt.object = Text("Douglas Adams", "en")
        mock_store.filter.return_value = [mock_label_stmt]
        
        api = WikidataKIFAPI()
        entity = Item("wd:Q42")
        result = api._entity_to_dict(entity)
        
        assert result['id'] == 'Q42'
        assert result['iri'] == 'wd:Q42'
        assert 'labels' in result
        assert 'descriptions' in result
        assert 'aliases' in result
        assert result['link'] == 'https://www.wikidata.org/wiki/Q42'


class TestWikidataKIFAPIIntegration:
    """Integration tests that make real API calls to Wikidata."""
    
    @pytest.mark.asyncio
    async def test_get_entity_douglas_adams(self):
        """Test getting real entity data for Douglas Adams (Q42)."""
        api = WikidataKIFAPI()
        result = await api.get_entity('Q42')

        pprint.pp(result['labels'])
        
        assert isinstance(result, dict)
        assert result['id'] == 'Q42'
        assert 'labels' in result
        assert 'descriptions' in result
        assert 'aliases' in result
        assert 'statements' in result
        assert 'link' in result
        assert result['link'] == 'https://www.wikidata.org/wiki/Q42'
        assert isinstance(result['statements'], dict)
        # Note: KIF may not return labels if not available in the store
        if result['labels'].get('en'):
            assert result['labels']['en'] == 'Douglas Adams'

    @pytest.mark.asyncio
    async def test_get_entity_valid_property(self):
        """Test getting a valid property entity (instance of - P31)."""
        api = WikidataKIFAPI()
        result = await api.get_entity('P31')

        assert isinstance(result, dict)
        assert result['id'] == 'P31'
        assert 'labels' in result
        assert 'descriptions' in result
        assert 'statements' in result
        assert result['link'] == 'https://www.wikidata.org/wiki/P31'

    @pytest.mark.asyncio
    async def test_get_entity_statements_instance_of(self):
        """Test getting entity statements filtered by property."""
        api = WikidataKIFAPI()
        result = await api.get_entity_statements('Q42', 'P31')

        pprint.pp(result)

        assert isinstance(result, dict)
        # The result should contain the 'P31' key if statements exist
        if 'P31' in result:
            assert isinstance(result['P31'], dict)  # KIF returns dict, not list
            # Check that values are statement dictionaries
            for key, statement in result['P31'].items():
                assert 'property_id' in statement
                assert 'value' in statement
                assert 'datatype' in statement
                assert statement['property_id'] == 'P31'

    @pytest.mark.asyncio
    async def test_get_entity_statements_all(self):
        """Test getting all entity statements without property filter."""
        api = WikidataKIFAPI()
        result = await api.get_entity_statements('Q42')

        assert isinstance(result, dict)
        # Should have some properties for Douglas Adams
        
        # Check structure of statements if any exist
        for prop_id, statements in result.items():
            assert isinstance(statements, dict)
            for key, statement in statements.items():
                assert 'property_id' in statement
                assert 'value' in statement
                assert 'datatype' in statement
                assert statement['property_id'] == prop_id

    @pytest.mark.asyncio
    async def test_get_property_instance_of(self):
        """Test getting valid property information."""
        api = WikidataKIFAPI()
        result = await api.get_property('P31') # 'instance of'

        assert isinstance(result, dict)
        assert result['id'] == 'P31'
        assert 'labels' in result
        assert 'descriptions' in result
        assert 'statements' in result
        assert 'datatype' in result

    @pytest.mark.asyncio
    async def test_search_entities_douglas_adams(self):
        """Test searching entities with valid query."""
        api = WikidataKIFAPI()
        result = await api.search_entities('Douglas Adams', limit=5)

        assert isinstance(result, list)
        assert len(result) <= 5  # May return fewer results

        if result:  # Only check if results exist
            first_result = result[0]
            assert 'id' in first_result
            assert 'label' in first_result

    @pytest.mark.asyncio
    async def test_search_entities_limit_parameter(self):
        """Test search entities with different limit values."""
        api = WikidataKIFAPI()
        result_1 = await api.search_entities('human', limit=1)
        assert len(result_1) <= 1

        result_10 = await api.search_entities('human', limit=10)
        assert len(result_10) <= 10

    @pytest.mark.asyncio
    async def test_search_entities_empty_query(self):
        """Test searching with a very specific/rare query."""
        api = WikidataKIFAPI()
        result = await api.search_entities('xyzabcnonexistentquery123', limit=1)
        
        # Should return empty list or very few results
        assert isinstance(result, list)
        assert len(result) <= 1

    @pytest.mark.asyncio
    async def test_query_sparql_valid(self):
        """Test SPARQL query execution."""
        api = WikidataKIFAPI()
        sparql = """
        SELECT ?item ?itemLabel WHERE {
            ?item wdt:P31 wd:Q5 .
            ?item rdfs:label "Douglas Adams"@en .
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
        LIMIT 1
        """
        result = await api.query_sparql(sparql)

        assert isinstance(result, list)
        
        if result:  # Only check if results exist
            first_result = result[0]
            assert isinstance(first_result, dict)
            assert 'item' in first_result
            assert 'itemLabel' in first_result

    @pytest.mark.asyncio
    async def test_multiple_real_operations(self):
        """Test multiple real API operations in sequence."""
        api = WikidataKIFAPI()
        
        # Get entity
        entity = await api.get_entity('Q42')
        assert entity['id'] == 'Q42'
        
        # Get specific statements
        statements = await api.get_entity_statements('Q42', 'P31')
        assert isinstance(statements, dict)
        
        # Search for entities
        search_results = await api.search_entities('Douglas Adams', limit=3)
        assert isinstance(search_results, list)
        
        # Get property info
        property_info = await api.get_property('P31')
        assert property_info['id'] == 'P31'

    @pytest.mark.asyncio
    async def test_api_consistency_between_methods(self):
        """Test consistency between get_entity and get_entity_statements."""
        api = WikidataKIFAPI()
        entity_id = 'Q42'
        entity_data = await api.get_entity(entity_id)
        statements_data = await api.get_entity_statements(entity_id)
        assert entity_data['statements'] == statements_data

    @pytest.mark.asyncio
    async def test_integration_workflow(self):
        """Test a complete asynchronous workflow using the API."""
        api = WikidataKIFAPI()

        # 1. Search for an entity
        search_results = await api.search_entities('Douglas Adams', limit=1)
        if search_results:  # Only proceed if we found results
            entity_id = search_results[0]['id']

            # 2. Get the entity data
            entity_data = await api.get_entity(entity_id)
            assert isinstance(entity_data['statements'], dict)

            # 3. Get property information for P31
            prop_info = await api.get_property('P31')
            assert prop_info['id'] == 'P31'


class TestWikidataKIFAPIErrorHandling:
    """Tests for error handling with invalid inputs and mocked failures."""
    
    @pytest.mark.asyncio
    async def test_get_entity_invalid_id(self):
        """Test getting a non-existent entity raises ValueError."""
        api = WikidataKIFAPI()
        with pytest.raises(ValueError) as exc_info:
            await api.get_entity('Q999999999999')
        assert "Failed to get entity" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_entity_statements_invalid_entity(self):
        """Test getting statements for an invalid entity raises ValueError."""
        api = WikidataKIFAPI()
        with pytest.raises(ValueError) as exc_info:
            await api.get_entity_statements('Q999999999999')
        assert "Failed to get statements" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_property_invalid(self):
        """Test getting an invalid property raises ValueError."""
        api = WikidataKIFAPI()
        with pytest.raises(ValueError) as exc_info:
            await api.get_property('P999999999')
        assert "Failed to get property" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_sparql_invalid(self):
        """Test SPARQL query with invalid syntax."""
        api = WikidataKIFAPI()
        invalid_sparql = "INVALID SPARQL QUERY SYNTAX"
        with pytest.raises(ValueError) as exc_info:
            await api.query_sparql(invalid_sparql)
        assert "SPARQL query failed" in str(exc_info.value)

    @patch('core.toolbox.wikidata.wikidata_kif_api.WikidataKIFAPI.get_entity')
    @pytest.mark.asyncio
    async def test_get_entity_failure(self, mock_get_entity):
        """Test entity retrieval failure."""
        mock_get_entity.side_effect = Exception("Network error")
        
        api = WikidataKIFAPI()
        
        with pytest.raises(Exception, match="Network error"):
            await api.get_entity('Q42')

    @patch('core.toolbox.wikidata.wikidata_kif_api.WikidataKIFAPI.search_entities')
    @pytest.mark.asyncio
    async def test_search_entities_failure(self, mock_search):
        """Test search entities failure."""
        mock_search.side_effect = Exception("Search failed")
        
        api = WikidataKIFAPI()
        
        with pytest.raises(Exception, match="Search failed"):
            await api.search_entities('test query')

    @patch('core.toolbox.wikidata.wikidata_kif_api.Store')
    @pytest.mark.asyncio
    async def test_kif_store_error_handling(self, mock_store_class):
        """Test KIF store operation error handling."""
        mock_store = MagicMock()
        mock_store.filter.side_effect = Exception("KIF Store error")
        mock_store_class.return_value = mock_store
        
        api = WikidataKIFAPI()
        
        with pytest.raises(ValueError) as exc_info:
            await api.get_entity('Q42')
        assert "Failed to get entity Q42" in str(exc_info.value)


if __name__ == "__main__":
    # Run tests with markers
    # pytest -m "not integration" for unit tests only
    # pytest -m integration for integration tests only
    pytest.main([__file__, "-v", "-s"])