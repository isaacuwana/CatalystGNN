"""
Base model class for Graph Neural Networks.

This module provides the abstract base class for all GNN models in CatalystGNN.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseGNNModel(nn.Module, ABC):
    """
    Abstract base class for all GNN models.
    
    This class defines the interface that all GNN models must implement
    for catalyst property prediction.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        **kwargs
    ):
        """
        Initialize the base GNN model.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            global_feature_dim: Dimension of global features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
            activation: Activation function name
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.global_feature_dim = global_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Set activation function
        self.activation = self._get_activation(activation)
        
        # Initialize model components
        self._build_model()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    @abstractmethod
    def _build_model(self):
        """Build the model architecture. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            graph_data: Dictionary containing graph tensors
            
        Returns:
            Predicted property values
        """
        pass
    
    def _init_weights(self, module: nn.Module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': self.__class__.__name__,
            'node_feature_dim': self.node_feature_dim,
            'edge_feature_dim': self.edge_feature_dim,
            'global_feature_dim': self.global_feature_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def reset_parameters(self):
        """Reset all model parameters."""
        self.apply(self._init_weights)
    
    def freeze_layers(self, num_layers: int):
        """Freeze the first num_layers layers."""
        layer_count = 0
        for name, param in self.named_parameters():
            if 'gnn_layers' in name:
                layer_idx = int(name.split('.')[1])
                if layer_idx < num_layers:
                    param.requires_grad = False
                    layer_count += 1
        
        logger.info(f"Frozen {layer_count} parameters in first {num_layers} layers")
    
    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
        
        logger.info("Unfrozen all model parameters")
    
    def get_embeddings(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get node embeddings from the model.
        
        Args:
            graph_data: Dictionary containing graph tensors
            
        Returns:
            Node embeddings
        """
        # This should be implemented by subclasses if needed
        raise NotImplementedError("get_embeddings not implemented for this model")
    
    def predict_with_uncertainty(
        self, 
        graph_data: Dict[str, torch.Tensor], 
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using Monte Carlo dropout.
        
        Args:
            graph_data: Dictionary containing graph tensors
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        self.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(graph_data)
                predictions.append(pred)
        
        self.eval()  # Disable dropout
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty
    
    def compute_attention_weights(self, graph_data: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Compute attention weights if the model supports it.
        
        Args:
            graph_data: Dictionary containing graph tensors
            
        Returns:
            Attention weights or None if not supported
        """
        # This should be implemented by attention-based models
        return None
    
    def save_checkpoint(self, filepath: str, optimizer: Optional[Any] = None, epoch: int = 0):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            optimizer: Optimizer state (optional)
            epoch: Current epoch
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.get_model_info(),
            'epoch': epoch
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = 'cpu'):
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['model_config']
        
        # Create model instance
        model = cls(
            node_feature_dim=config['node_feature_dim'],
            edge_feature_dim=config['edge_feature_dim'],
            global_feature_dim=config['global_feature_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Loaded model from {filepath}")
        return model, checkpoint.get('epoch', 0)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters by category."""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        param_counts = {}
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
            else:
                frozen_params += num_params
            
            # Categorize by layer type
            if 'gnn_layers' in name:
                category = 'gnn_layers'
            elif 'node_embedding' in name:
                category = 'node_embedding'
            elif 'edge_embedding' in name:
                category = 'edge_embedding'
            elif 'predictor' in name:
                category = 'predictor'
            else:
                category = 'other'
            
            param_counts[category] = param_counts.get(category, 0) + num_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'by_category': param_counts
        }