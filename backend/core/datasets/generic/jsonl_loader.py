"""
JSONL (JSON Lines) dataset loader implementation.
"""

from typing import Dict, Any, List, Union
from pathlib import Path
import json
import logging

from ..dataloader import DataLoader

logger = logging.getLogger(__name__)


class JSONLDataLoader(DataLoader):
    """Data loader for JSON Lines format datasets."""
    
    def __init__(self, data_path: Union[str, Path], **kwargs):
        super().__init__(data_path, **kwargs)
        self._data: List[Dict[str, Any]] = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Load JSONL data from file."""
        try:
            self._data = []
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            item = json.loads(line)
                            self._data.append(item)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
            
            self.metadata = {
                'total_items': len(self._data),
                'data_path': str(self.data_path),
                'format': 'jsonl'
            }
            
            logger.info(f"Loaded {len(self._data)} items from {self.data_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading JSONL data: {e}")
    
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
