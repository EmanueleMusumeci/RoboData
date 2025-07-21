"""
Question answering dataset loaders (LC-QUAD, etc.).
"""

from .base_qa import BaseQADataLoader
from .lcquad_loader import LCQuadDataLoader, create_lcquad_loader

__all__ = ['BaseQADataLoader', 'LCQuadDataLoader', 'create_lcquad_loader']
