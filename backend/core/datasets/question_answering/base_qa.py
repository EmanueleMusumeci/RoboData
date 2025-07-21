"""
Base abstract class for question answering datasets.
"""

from abc import abstractmethod
from typing import Dict, Any, List, Union, Optional
from pathlib import Path

from ..dataloader import DataLoader


class BaseQADataLoader(DataLoader):
    """Abstract base class for question answering dataset loaders."""
    
    @abstractmethod
    def get_question(self, index: int) -> str:
        """Get the natural language question at the given index."""
        pass
    
    @abstractmethod
    def get_answer(self, index: int) -> str:
        """Get the expected answer at the given index."""
        pass
    
    def get_question_id(self, index: int) -> str:
        """Get the question ID at the given index."""
        item = self[index]
        return str(item.get('id', index))
    
    def filter_by_length(self, min_length: int = 0, max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Filter questions by length.
        
        Args:
            min_length: Minimum question length in words
            max_length: Maximum question length in words (None for no limit)
            
        Returns:
            List of questions matching the length criteria
        """
        filtered = []
        for i in range(len(self)):
            question = self.get_question(i)
            question_length = len(question.split())
            
            if question_length >= min_length:
                if max_length is None or question_length <= max_length:
                    filtered.append(self[i])
        
        return filtered
