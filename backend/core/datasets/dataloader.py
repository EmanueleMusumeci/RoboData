"""
Abstract base class for all dataset loaders.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Iterator, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Abstract base class for all data loaders."""
    
    def __init__(self, data_path: Union[str, Path], **kwargs):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the dataset file or directory
            **kwargs: Additional loader-specific parameters
        """
        self.data_path = Path(data_path)
        self.metadata = {}
        self._validate_path()
    
    def _validate_path(self) -> None:
        """Validate that the data path exists."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.data_path}")
    
    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """
        Load the complete dataset.
        
        Returns:
            List of dataset items as dictionaries
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a specific item by index."""
        pass
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all items in the dataset."""
        for i in range(len(self)):
            yield self[i]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata information."""
        return self.metadata.copy()
    
    def save_processed(self, output_path: Union[str, Path], format: str = "json") -> None:
        """
        Save the processed dataset to a file.
        
        Args:
            output_path: Path to save the dataset
            format: Output format ('json', 'jsonl')
        """
        import json
        
        output_path = Path(output_path)
        data = self.load()
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(data)} items to {output_path}")
