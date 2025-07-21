"""
Dataset utilities and factory functions for RoboData.
"""

from typing import Dict, Any, List, Optional, Union, Type
from pathlib import Path
import logging

from .dataloader import DataLoader
from .generic import JSONDataLoader, JSONLDataLoader
from .question_answering import LCQuadDataLoader

logger = logging.getLogger(__name__)


# Registry of available data loaders
LOADER_REGISTRY: Dict[str, Type[DataLoader]] = {
    'json': JSONDataLoader,
    'jsonl': JSONLDataLoader,
    'lcquad': LCQuadDataLoader,
    'lc-quad': LCQuadDataLoader,  # Alias
}


def create_data_loader(
    dataset_type: str, 
    data_path: Union[str, Path], 
    **kwargs
) -> DataLoader:
    """
    Factory function to create a data loader based on dataset type.
    
    Args:
        dataset_type: Type of dataset ('json', 'jsonl', 'lcquad', etc.)
        data_path: Path to the dataset file
        **kwargs: Additional arguments for the specific loader
        
    Returns:
        Initialized DataLoader instance
        
    Raises:
        ValueError: If dataset_type is not supported
    """
    dataset_type = dataset_type.lower()
    
    if dataset_type not in LOADER_REGISTRY:
        available_types = ', '.join(LOADER_REGISTRY.keys())
        raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                        f"Available types: {available_types}")
    
    loader_class = LOADER_REGISTRY[dataset_type]
    return loader_class(data_path, **kwargs)


def auto_detect_dataset_type(data_path: Union[str, Path]) -> str:
    """
    Automatically detect the dataset type based on file extension and content.
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        Detected dataset type
    """
    data_path = Path(data_path)
    
    # Check file extension first
    if data_path.suffix == '.jsonl':
        return 'jsonl'
    elif data_path.suffix == '.json':
        # Could be regular JSON or LC-QUAD
        # Try to detect LC-QUAD by looking for specific fields
        try:
            import json
            with open(data_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
            
            # Check if it looks like LC-QUAD format
            if isinstance(sample_data, list) and sample_data:
                first_item = sample_data[0]
                if isinstance(first_item, dict):
                    # LC-QUAD typically has 'question' and 'sparql_query' fields
                    if 'question' in first_item and 'sparql_query' in first_item:
                        return 'lcquad'
            
            return 'json'
            
        except Exception as e:
            logger.warning(f"Could not analyze file content: {e}")
            return 'json'
    
    # Default to JSON for unknown extensions
    return 'json'


def load_dataset(
    data_path: Union[str, Path], 
    dataset_type: Optional[str] = None,
    **kwargs
) -> DataLoader:
    """
    Load a dataset with automatic type detection if not specified.
    
    Args:
        data_path: Path to the dataset file
        dataset_type: Type of dataset (auto-detected if None)
        **kwargs: Additional arguments for the specific loader
        
    Returns:
        Initialized DataLoader instance
    """
    if dataset_type is None:
        dataset_type = auto_detect_dataset_type(data_path)
        logger.info(f"Auto-detected dataset type: {dataset_type}")
    
    return create_data_loader(dataset_type, data_path, **kwargs)


def get_supported_formats() -> List[str]:
    """Get list of supported dataset formats."""
    return list(LOADER_REGISTRY.keys())


def validate_dataset(loader: DataLoader, sample_size: int = 5) -> Dict[str, Any]:
    """
    Validate a dataset by checking a sample of items.
    
    Args:
        loader: DataLoader instance to validate
        sample_size: Number of items to sample for validation
        
    Returns:
        Validation report
    """
    validation_report = {
        'total_items': len(loader),
        'sample_size': min(sample_size, len(loader)),
        'errors': [],
        'warnings': [],
        'metadata': loader.get_metadata()
    }
    
    # Sample items for validation
    sample_indices = range(min(sample_size, len(loader)))
    
    for i in sample_indices:
        try:
            item = loader[i]
            
            # Basic validation
            if not isinstance(item, dict):
                validation_report['errors'].append(f"Item {i}: Expected dict, got {type(item)}")
                continue
            
            # Check for common required fields based on loader type
            if isinstance(loader, LCQuadDataLoader):
                if 'question' not in item:
                    validation_report['warnings'].append(f"Item {i}: Missing 'question' field")
                if 'sparql_query' not in item:
                    validation_report['warnings'].append(f"Item {i}: Missing 'sparql_query' field")
            
        except Exception as e:
            validation_report['errors'].append(f"Item {i}: Error loading - {e}")
    
    validation_report['is_valid'] = len(validation_report['errors']) == 0
    
    return validation_report
