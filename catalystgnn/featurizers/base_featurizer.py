"""
Base featurizer class for graph neural network models.

This module provides the abstract base class for all featurizers in CatalystGNN.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
import torch
import numpy as np


class BaseFeaturizer(ABC):
    """
    Abstract base class for all graph featurizers.
    
    This class defines the interface that all featurizers must implement
    to convert chemical structures into graph representations suitable
    for GNN models.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the base featurizer.
        
        Args:
            **kwargs: Additional configuration parameters
        """
        self.config = kwargs
        self._setup_featurizer()
    
    @abstractmethod
    def _setup_featurizer(self):
        """Setup featurizer-specific configurations."""
        pass
    
    @abstractmethod
    def featurize(self, structure: Any) -> Dict[str, torch.Tensor]:
        """
        Convert a chemical structure to graph representation.
        
        Args:
            structure: Chemical structure (molecule, crystal, etc.)
            
        Returns:
            Dictionary containing graph data with tensors for:
            - node_features: Node feature matrix
            - edge_features: Edge feature matrix  
            - edge_indices: Edge connectivity
            - global_features: Global/graph-level features (optional)
        """
        pass
    
    @abstractmethod
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get the dimensions of different feature types.
        
        Returns:
            Dictionary with feature dimensions
        """
        pass
    
    def featurize_batch(self, structures: List[Any]) -> List[Dict[str, torch.Tensor]]:
        """
        Featurize a batch of structures.
        
        Args:
            structures: List of chemical structures
            
        Returns:
            List of graph data dictionaries
        """
        return [self.featurize(structure) for structure in structures]
    
    def _validate_graph_data(self, graph_data: Dict[str, torch.Tensor]) -> bool:
        """
        Validate that graph data has the required format.
        
        Args:
            graph_data: Graph data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['node_features', 'edge_indices']
        
        for key in required_keys:
            if key not in graph_data:
                return False
            if not isinstance(graph_data[key], torch.Tensor):
                return False
        
        # Check dimensions
        num_nodes = graph_data['node_features'].shape[0]
        edge_indices = graph_data['edge_indices']
        
        if edge_indices.shape[0] != 2:
            return False
        
        if edge_indices.max() >= num_nodes:
            return False
        
        return True
    
    def _normalize_features(self, features: torch.Tensor, method: str = 'standard') -> torch.Tensor:
        """
        Normalize feature matrix.
        
        Args:
            features: Feature tensor
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            Normalized features
        """
        if method == 'standard':
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True)
            return (features - mean) / (std + 1e-8)
        elif method == 'minmax':
            min_val = features.min(dim=0, keepdim=True)[0]
            max_val = features.max(dim=0, keepdim=True)[0]
            return (features - min_val) / (max_val - min_val + 1e-8)
        elif method == 'robust':
            median = features.median(dim=0, keepdim=True)[0]
            mad = torch.median(torch.abs(features - median), dim=0, keepdim=True)[0]
            return (features - median) / (mad + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get featurizer configuration."""
        return self.config.copy()
    
    def __repr__(self) -> str:
        """String representation of the featurizer."""
        return f"{self.__class__.__name__}({self.config})"