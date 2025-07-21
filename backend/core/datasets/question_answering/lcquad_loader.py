"""
LC-QUAD Dataset Loader for RoboData.

LC-QUAD is a dataset for complex question answering over knowledge graphs.
It contains natural language questions with corresponding SPARQL queries.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
import re

from .base_qa import BaseQADataLoader
from ..generic.json_loader import JSONDataLoader

logger = logging.getLogger(__name__)


class LCQuadDataLoader(BaseQADataLoader, JSONDataLoader):
    """
    Data loader for the LC-QUAD dataset.
    
    LC-QUAD is a dataset for question answering over knowledge graphs, 
    containing natural language questions paired with SPARQL queries.
    
    Expected JSON structure:
    [
        {
            "question": "Natural language question",
            "sparql_query": "SPARQL query string",
            "uri": "Question URI/ID",
            "id": "Question ID",
            "entities": [...],  # Optional
            "relations": [...], # Optional
            "template": "Template ID", # Optional
            ...
        },
        ...
    ]
    """
    
    def __init__(self, data_path: Union[str, Path], **kwargs):
        """
        Initialize LC-QUAD data loader.
        
        Args:
            data_path: Path to the LC-QUAD JSON file
            **kwargs: Additional parameters
        """
        JSONDataLoader.__init__(self, data_path, **kwargs)
        self._analyze_dataset()
    
    def _analyze_dataset(self) -> None:
        """Analyze the loaded dataset and extract metadata."""
        if not self._data:
            return
        
        # Count questions by various criteria
        question_lengths = []
        sparql_lengths = []
        entity_counts = []
        relation_counts = []
        template_counts = {}
        
        for item in self._data:
            # Question analysis
            question = item.get('question', '')
            question_lengths.append(len(question.split()))
            
            # SPARQL analysis
            sparql = item.get('sparql_query', '')
            sparql_lengths.append(len(sparql.split()))
            
            # Entity analysis
            entities = item.get('entities', [])
            if isinstance(entities, list):
                entity_counts.append(len(entities))
            else:
                entity_counts.append(0)
            
            # Relation analysis
            relations = item.get('relations', [])
            if isinstance(relations, list):
                relation_counts.append(len(relations))
            else:
                relation_counts.append(0)
            
            # Template analysis
            template = item.get('template', 'unknown')
            template_counts[template] = template_counts.get(template, 0) + 1
        
        # Update metadata with analysis
        self.metadata.update({
            'dataset_type': 'LC-QUAD',
            'avg_question_length': sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            'avg_sparql_length': sum(sparql_lengths) / len(sparql_lengths) if sparql_lengths else 0,
            'avg_entities_per_question': sum(entity_counts) / len(entity_counts) if entity_counts else 0,
            'avg_relations_per_question': sum(relation_counts) / len(relation_counts) if relation_counts else 0,
            'template_distribution': template_counts,
            'total_templates': len(template_counts)
        })
        
        logger.info(f"LC-QUAD dataset analysis complete: {len(self._data)} questions, "
                   f"{len(template_counts)} templates")
    
    def get_question(self, index: int) -> str:
        """Get the natural language question at the given index."""
        item = self[index]
        return item.get('question', '')
    
    def get_answer(self, index: int) -> str:
        """Get the expected answer (SPARQL query) at the given index."""
        return self.get_sparql_query(index)
    
    def get_sparql_query(self, index: int) -> str:
        """Get the SPARQL query at the given index."""
        item = self[index]
        return item.get('sparql_query', '')
    
    def get_entities(self, index: int) -> List[str]:
        """Get the entities mentioned in the question at the given index."""
        item = self[index]
        entities = item.get('entities', [])
        return entities if isinstance(entities, list) else []
    
    def get_relations(self, index: int) -> List[str]:
        """Get the relations used in the question at the given index."""
        item = self[index]
        relations = item.get('relations', [])
        return relations if isinstance(relations, list) else []
    
    def get_template(self, index: int) -> Optional[str]:
        """Get the template ID for the question at the given index."""
        item = self[index]
        return item.get('template')
    
    def filter_by_template(self, template: str) -> List[Dict[str, Any]]:
        """
        Filter questions by template ID.
        
        Args:
            template: Template ID to filter by
            
        Returns:
            List of questions matching the template
        """
        return [item for item in self._data if item.get('template') == template]
    
    def filter_by_entity_count(self, min_entities: int = 0, max_entities: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Filter questions by number of entities.
        
        Args:
            min_entities: Minimum number of entities
            max_entities: Maximum number of entities (None for no limit)
            
        Returns:
            List of questions matching the entity count criteria
        """
        filtered = []
        for item in self._data:
            entities = item.get('entities', [])
            entity_count = len(entities) if isinstance(entities, list) else 0
            
            if entity_count >= min_entities:
                if max_entities is None or entity_count <= max_entities:
                    filtered.append(item)
        
        return filtered
    
    def get_sample(self, n: int = 10, template: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a sample of questions from the dataset.
        
        Args:
            n: Number of samples to return
            template: Optional template to filter by
            
        Returns:
            List of sample questions
        """
        if template:
            data = self.filter_by_template(template)
        else:
            data = self._data
        
        # Return first n items (or all if fewer than n)
        return data[:min(n, len(data))]
    
    def extract_wikidata_entities(self, sparql_query: str) -> List[str]:
        """
        Extract Wikidata entity IDs (Q-numbers) from a SPARQL query.
        
        Args:
            sparql_query: The SPARQL query string
            
        Returns:
            List of Wikidata entity IDs found in the query
        """
        # Pattern to match Wikidata entity IDs (Q followed by digits)
        pattern = r'wd:Q\d+'
        matches = re.findall(pattern, sparql_query)
        
        # Remove the 'wd:' prefix to get clean Q-numbers
        entities = [match.replace('wd:', '') for match in matches]
        
        return list(set(entities))  # Remove duplicates
    
    def extract_wikidata_properties(self, sparql_query: str) -> List[str]:
        """
        Extract Wikidata property IDs (P-numbers) from a SPARQL query.
        
        Args:
            sparql_query: The SPARQL query string
            
        Returns:
            List of Wikidata property IDs found in the query
        """
        # Pattern to match Wikidata property IDs (P followed by digits)
        pattern = r'wdt?:P\d+'
        matches = re.findall(pattern, sparql_query)
        
        # Remove prefixes to get clean P-numbers
        properties = []
        for match in matches:
            if match.startswith('wdt:'):
                properties.append(match.replace('wdt:', ''))
            elif match.startswith('wd:'):
                properties.append(match.replace('wd:', ''))
        
        return list(set(properties))  # Remove duplicates
    
    def get_wikidata_analysis(self, index: int) -> Dict[str, Any]:
        """
        Get Wikidata-specific analysis for a question.
        
        Args:
            index: Index of the question
            
        Returns:
            Dictionary with Wikidata entity and property analysis
        """
        item = self[index]
        sparql = item.get('sparql_query', '')
        
        entities = self.extract_wikidata_entities(sparql)
        properties = self.extract_wikidata_properties(sparql)
        
        return {
            'question': item.get('question', ''),
            'sparql_query': sparql,
            'wikidata_entities': entities,
            'wikidata_properties': properties,
            'entity_count': len(entities),
            'property_count': len(properties),
            'template': item.get('template'),
            'original_entities': item.get('entities', []),
            'original_relations': item.get('relations', [])
        }
    
    def export_for_evaluation(self, output_path: Union[str, Path], include_wikidata_analysis: bool = True) -> None:
        """
        Export the dataset in a format suitable for evaluation.
        
        Args:
            output_path: Path to save the exported dataset
            include_wikidata_analysis: Whether to include Wikidata entity/property analysis
        """
        import json
        
        output_path = Path(output_path)
        
        export_data = []
        for i in range(len(self._data)):
            item = self[i].copy()
            
            if include_wikidata_analysis:
                wikidata_info = self.get_wikidata_analysis(i)
                item.update({
                    'wikidata_entities': wikidata_info['wikidata_entities'],
                    'wikidata_properties': wikidata_info['wikidata_properties'],
                    'wikidata_entity_count': wikidata_info['entity_count'],
                    'wikidata_property_count': wikidata_info['property_count']
                })
            
            export_data.append(item)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(export_data)} LC-QUAD questions to {output_path}")


# Factory function for creating LC-QUAD loader
def create_lcquad_loader(data_path: Union[str, Path]) -> LCQuadDataLoader:
    """
    Factory function to create an LC-QUAD data loader.
    
    Args:
        data_path: Path to the LC-QUAD dataset file
        
    Returns:
        Initialized LCQuadDataLoader instance
    """
    return LCQuadDataLoader(data_path)
