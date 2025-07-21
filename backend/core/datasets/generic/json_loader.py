"""
JSON dataset loader implementation.
"""

from typing import Dict, Any, List, Union
from pathlib import Path
import json
import logging

from ..dataloader import DataLoader

logger = logging.getLogger(__name__)


class JSONDataLoader(DataLoader):
    """Generic JSON data loader for structured datasets."""
    
    def __init__(self, data_path: Union[str, Path], **kwargs):
        super().__init__(data_path, **kwargs)
        self._data: List[Dict[str, Any]] = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Load JSON data from file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(raw_data, dict):
                # If it's a dict, look for common list keys
                if 'data' in raw_data:
                    self._data = raw_data['data']
                elif 'items' in raw_data:
                    self._data = raw_data['items']
                elif 'questions' in raw_data:
                    self._data = raw_data['questions']
                else:
                    # Convert dict to list of single item
                    self._data = [raw_data]
            elif isinstance(raw_data, list):
                self._data = raw_data
            else:
                raise ValueError(f"Expected list or dict with data, got {type(raw_data)}")
            
            if not isinstance(self._data, list):
                raise ValueError(f"Expected list data, got {type(self._data)}")
            
            self.metadata = {
                'total_items': len(self._data),
                'data_path': str(self.data_path),
                'format': 'json'
            }
            
            logger.info(f"Loaded {len(self._data)} items from {self.data_path}")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading JSON data: {e}")
    
    def load(self) -> List[Dict[str, Any]]:
        """Load the complete dataset."""
        return self._data.copy()
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self._data)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a specific item by index."""
        if index < 0 or index >= len(self._data):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self._data)}")
        return self._data[index].copy()
