import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from pathlib import Path
import pprint

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.toolbox.wikidata.wikidata_api import WikidataRestAPI, wikidata_api


class TestWikidataRestAPIUnit:
    """Unit tests for WikidataRestAPI class structure and error handling."""
    
    def test_initialization(self):
        """Test API initialization."""
        api = WikidataRestAPI()
        assert api.client is not None
    
    def test_global_instance(self):
        """Test that global instance is properly initialized."""
        assert wikidata_api is not None
        assert isinstance(wikidata_api, WikidataRestAPI)
        assert hasattr(wikidata_api, 'client')
    
    @pytest.mark.asyncio
    async def test_api_methods_are_async(self):
        """Test that all main methods are properly async."""
        api = WikidataRestAPI()
        
        # All these should be coroutine functions
        assert asyncio.iscoroutinefunction(api.get_entity)
        assert asyncio.iscoroutinefunction(api.get_entity_statements)
        assert asyncio.iscoroutinefunction(api.search_entities)
        assert asyncio.iscoroutinefunction(api.get_property)

class TestWikidataRestAPIIntegration:
    """Integration tests that make real API calls to Wikidata."""
    
    @pytest.mark.asyncio
    async def test_get_entity_douglas_adams(self):
        """Test getting real entity data for Douglas Adams (Q42)."""
        api = WikidataRestAPI()
        result = await api.get_entity('Q42')

        pprint.pp(result["labels"])
        
        # Verify structure and content
        assert isinstance(result, dict)
        assert result['id'] == 'Q42'
        assert 'labels' in result
        assert 'en' in result['labels']
        assert result['labels']['en'] == 'Douglas Adams'
        assert 'descriptions' in result
        assert 'statements' in result
    
    @pytest.mark.asyncio
    async def test_get_entity_statements_instance_of(self):
        """Test getting real statements for Douglas Adams (Q42) with P31 (instance of)."""
        api = WikidataRestAPI()
        result = await api.get_entity_statements('Q42', 'P31')
        
        # Verify structure
        assert isinstance(result, dict)
        assert 'P31' in result
        assert isinstance(result['P31'], dict)
        assert len(result['P31']) > 0
        
        pprint.pp(result['P31'])

        # Verify that Douglas Adams is an instance of human (Q5)
        assert 'Q5' in result['P31']  # Q5 = human should be a key
        assert result['P31']['Q5']['datavalue']['value']['id'] == 'Q5'
    
    @pytest.mark.asyncio
    async def test_get_entity_statements_all(self):
        """Test getting all statements for Douglas Adams (Q42)."""
        api = WikidataRestAPI()
        result = await api.get_entity_statements('Q42')
        
        # Verify structure
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Should contain common properties for a person
        assert 'P31' in result  # instance of
        assert 'P106' in result  # occupation
        assert 'P569' in result  # date of birth
    
    @pytest.mark.asyncio
    async def test_search_entities_douglas_adams(self):
        """Test searching for entities with 'Douglas Adams'."""
        api = WikidataRestAPI()
        result = await api.search_entities('Douglas Adams', limit=5)
        
        # Verify structure
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= 5
        
        # First result should be Douglas Adams himself
        first_result = result[0]
        assert first_result['id'] == 'Q42'
        assert 'Douglas Adams' in first_result['label']
        assert 'description' in first_result
    
    @pytest.mark.asyncio
    async def test_search_entities_empty_query(self):
        """Test searching with a very specific/rare query."""
        api = WikidataRestAPI()
        result = await api.search_entities('xyzabcnonexistentquery123', limit=1)
        
        # Should return empty list or very few results
        assert isinstance(result, list)
        assert len(result) == 0 or len(result) < 2
    
    @pytest.mark.asyncio
    async def test_get_property_instance_of(self):
        """Test getting real property data for P31 (instance of)."""
        api = WikidataRestAPI()
        result = await api.get_property('P31')
        
        # Verify structure and content
        assert isinstance(result, dict)
        assert result['id'] == 'P31'
        assert 'labels' in result
        assert 'en' in result['labels']
        assert 'instance of' in result['labels']['en']
        assert 'datatype' in result
        assert result['datatype'] == 'wikibase-item'
    
    @pytest.mark.asyncio
    async def test_multiple_real_operations(self):
        """Test multiple real API operations in sequence."""
        api = WikidataRestAPI()
        
        # Get entity
        entity = await api.get_entity('Q42')
        assert entity['id'] == 'Q42'
        
        # Get specific statements
        statements = await api.get_entity_statements('Q42', 'P31')
        assert 'P31' in statements
        
        # Search for entities
        search_results = await api.search_entities('Douglas Adams', limit=3)
        assert len(search_results) > 0
        assert search_results[0]['id'] == 'Q42'
        
        # Get property info
        property_info = await api.get_property('P31')
        assert property_info['id'] == 'P31'

class TestWikidataRestAPIErrorHandling:
    """Tests for error handling with mocked failures."""
    
    @pytest.mark.asyncio
    @patch('asyncio.get_event_loop')
    async def test_get_entity_failure(self, mock_get_loop):
        """Test entity retrieval failure."""
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        
        # Mock exception
        future = asyncio.Future()
        future.set_exception(Exception("Entity not found"))
        mock_loop.run_in_executor.return_value = future
        
        api = WikidataRestAPI()
        
        with pytest.raises(ValueError, match="Failed to get entity Q999999"):
            await api.get_entity('Q999999')
    
    @pytest.mark.asyncio
    @patch('asyncio.get_event_loop')
    async def test_get_property_failure(self, mock_get_loop):
        """Test property retrieval failure."""
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        
        future = asyncio.Future()
        future.set_exception(Exception("Property not found"))
        mock_loop.run_in_executor.return_value = future
        
        api = WikidataRestAPI()
        
        with pytest.raises(ValueError, match="Failed to get property P999999"):
            await api.get_property('P999999')
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_search_entities_http_error(self, mock_get):
        """Test search entities with HTTP error."""
        # Mock HTTP error
        mock_get.side_effect = Exception("HTTP Error")
        
        api = WikidataRestAPI()
        
        with pytest.raises(ValueError, match="Failed to search entities"):
            await api.search_entities('test query')

if __name__ == "__main__":
    # Run tests with markers
    # pytest -m "not integration" for unit tests only
    # pytest -m integration for integration tests only
    pytest.main([__file__, "-v", "-s"])