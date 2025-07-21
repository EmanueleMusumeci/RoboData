"""
Generic dataset loaders for common formats (JSON, JSONL).
"""

from .json_loader import JSONDataLoader
from .jsonl_loader import JSONLDataLoader

__all__ = ['JSONDataLoader', 'JSONLDataLoader']
