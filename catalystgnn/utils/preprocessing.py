"""
Data preprocessing utilities for CatalystGNN.

This module provides functions for preprocessing graph data,
handling missing values, and normalizing features.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


def normalize_features(
    features: torch.Tensor,
    method: str = 'standard',
    dim: int = 0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalize feature tensor.
    
    Args:
        features: Input feature tensor
        method: Normalization method ('standard', 'minmax', 'robust', 'unit')
        dim: Dimension along which to normalize
        eps: Small value to avoid division by zero
        
    Returns:
        Normalized feature tensor
    """
    if method == 'standard':
        mean = features.mean(dim=dim, keepdim=True)
        std = features.std(dim=dim, keepdim=True)
        return (features - mean) / (std + eps)
    
    elif method == 'minmax':
        min_val = features.min(dim=dim, keepdim=True)[0]
        max_val = features.max(dim=dim, keepdim=True)[0]
        return (features - min_val) / (max_val - min_val + eps)
    
    elif method == 'robust':
        median = features.median(dim=dim, keepdim=True)[0]
        mad = torch.median(torch.abs(features - median), dim=dim, keepdim=True)[0]
        return (features - median) / (mad + eps)
    
    elif method == 'unit':
        norm = torch.norm(features, dim=dim, keepdim=True)
        return features / (norm + eps)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def handle_missing_values(
    features: torch.Tensor,
    method: str = 'mean',
    fill_value: float = 0.0
) -> torch.Tensor:
    """
    Handle missing values in feature tensor.
    
    Args:
        features: Input feature tensor
        method: Method to handle missing values ('mean', 'median', 'zero', 'constant')
        fill_value: Value to use for 'constant' method
        
    Returns:
        Feature tensor with missing values handled
    """
    # Check for NaN values
    nan_mask = torch.isnan(features)
    
    if not nan_mask.any():
        return features  # No missing values
    
    features_filled = features.clone()
    
    if method == 'mean':
        # Fill with column means
        for col in range(features.shape[1]):
            col_mask = nan_mask[:, col]
            if col_mask.any():
                col_mean = features[~col_mask, col].mean()
                features_filled[col_mask, col] = col_mean
    
    elif method == 'median':
        # Fill with column medians
        for col in range(features.shape[1]):
            col_mask = nan_mask[:, col]
            if col_mask.any():
                col_median = features[~col_mask, col].median()
                features_filled[col_mask, col] = col_median
    
    elif method == 'zero':
        features_filled[nan_mask] = 0.0
    
    elif method == 'constant':
        features_filled[nan_mask] = fill_value
    
    else:
        raise ValueError(f"Unknown missing value method: {method}")
    
    return features_filled


def preprocess_graph_batch(
    graph_batch: List[Dict[str, torch.Tensor]],
    normalize_nodes: bool = True,
    normalize_edges: bool = True,
    handle_missing: bool = True,
    node_norm_method: str = 'standard',
    edge_norm_method: str = 'standard'
) -> List[Dict[str, torch.Tensor]]:
    """
    Preprocess a batch of graph data.
    
    Args:
        graph_batch: List of graph data dictionaries
        normalize_nodes: Whether to normalize node features
        normalize_edges: Whether to normalize edge features
        handle_missing: Whether to handle missing values
        node_norm_method: Normalization method for node features
        edge_norm_method: Normalization method for edge features
        
    Returns:
        Preprocessed graph batch
    """
    if not graph_batch:
        return graph_batch
    
    processed_batch = []
    
    # Collect all features for batch normalization
    if normalize_nodes or normalize_edges:
        all_node_features = []
        all_edge_features = []
        
        for graph_data in graph_batch:
            if 'node_features' in graph_data:
                all_node_features.append(graph_data['node_features'])
            if 'edge_features' in graph_data:
                all_edge_features.append(graph_data['edge_features'])
        
        # Concatenate features
        if all_node_features:
            batch_node_features = torch.cat(all_node_features, dim=0)
        if all_edge_features:
            batch_edge_features = torch.cat(all_edge_features, dim=0)
    
    # Process each graph
    node_start_idx = 0
    edge_start_idx = 0
    
    for graph_data in graph_batch:
        processed_graph = graph_data.copy()
        
        # Handle missing values
        if handle_missing:
            if 'node_features' in processed_graph:
                processed_graph['node_features'] = handle_missing_values(
                    processed_graph['node_features']
                )
            if 'edge_features' in processed_graph:
                processed_graph['edge_features'] = handle_missing_values(
                    processed_graph['edge_features']
                )
        
        # Normalize features
        if normalize_nodes and 'node_features' in processed_graph:
            num_nodes = processed_graph['node_features'].shape[0]
            node_end_idx = node_start_idx + num_nodes
            
            # Use batch statistics for normalization
            if all_node_features:
                normalized_features = normalize_features(
                    batch_node_features, method=node_norm_method, dim=0
                )
                processed_graph['node_features'] = normalized_features[node_start_idx:node_end_idx]
                node_start_idx = node_end_idx
        
        if normalize_edges and 'edge_features' in processed_graph:
            num_edges = processed_graph['edge_features'].shape[0]
            edge_end_idx = edge_start_idx + num_edges
            
            # Use batch statistics for normalization
            if all_edge_features:
                normalized_features = normalize_features(
                    batch_edge_features, method=edge_norm_method, dim=0
                )
                processed_graph['edge_features'] = normalized_features[edge_start_idx:edge_end_idx]
                edge_start_idx = edge_end_idx
        
        processed_batch.append(processed_graph)
    
    return processed_batch


def create_feature_scaler(
    features: torch.Tensor,
    method: str = 'standard'
) -> Any:
    """
    Create a feature scaler from training data.
    
    Args:
        features: Training feature tensor
        method: Scaling method ('standard', 'minmax', 'robust')
        
    Returns:
        Fitted scaler object
    """
    features_np = features.detach().cpu().numpy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    scaler.fit(features_np)
    return scaler


def apply_feature_scaler(
    features: torch.Tensor,
    scaler: Any
) -> torch.Tensor:
    """
    Apply fitted scaler to features.
    
    Args:
        features: Input feature tensor
        scaler: Fitted scaler object
        
    Returns:
        Scaled feature tensor
    """
    features_np = features.detach().cpu().numpy()
    scaled_features = scaler.transform(features_np)
    return torch.tensor(scaled_features, dtype=features.dtype, device=features.device)


def pad_sequences(
    sequences: List[torch.Tensor],
    max_length: Optional[int] = None,
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    Pad sequences to same length.
    
    Args:
        sequences: List of tensors with different lengths
        max_length: Maximum length (uses longest sequence if None)
        pad_value: Value to use for padding
        
    Returns:
        Padded tensor of shape (num_sequences, max_length, feature_dim)
    """
    if not sequences:
        return torch.empty(0)
    
    if max_length is None:
        max_length = max(seq.shape[0] for seq in sequences)
    
    feature_dim = sequences[0].shape[1] if sequences[0].dim() > 1 else 1
    padded = torch.full(
        (len(sequences), max_length, feature_dim),
        pad_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device
    )
    
    for i, seq in enumerate(sequences):
        seq_len = min(seq.shape[0], max_length)
        if seq.dim() == 1:
            padded[i, :seq_len, 0] = seq[:seq_len]
        else:
            padded[i, :seq_len] = seq[:seq_len]
    
    return padded


def create_attention_mask(
    lengths: List[int],
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Create attention mask for padded sequences.
    
    Args:
        lengths: List of sequence lengths
        max_length: Maximum length (uses max of lengths if None)
        
    Returns:
        Boolean mask tensor of shape (num_sequences, max_length)
    """
    if max_length is None:
        max_length = max(lengths)
    
    mask = torch.zeros(len(lengths), max_length, dtype=torch.bool)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return mask


def augment_graph_data(
    graph_data: Dict[str, torch.Tensor],
    augmentation_type: str = 'noise',
    noise_std: float = 0.1,
    dropout_prob: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Apply data augmentation to graph data.
    
    Args:
        graph_data: Input graph data
        augmentation_type: Type of augmentation ('noise', 'dropout', 'permute')
        noise_std: Standard deviation for noise augmentation
        dropout_prob: Probability for dropout augmentation
        
    Returns:
        Augmented graph data
    """
    augmented_data = {k: v.clone() for k, v in graph_data.items()}
    
    if augmentation_type == 'noise':
        # Add Gaussian noise to node features
        if 'node_features' in augmented_data:
            noise = torch.randn_like(augmented_data['node_features']) * noise_std
            augmented_data['node_features'] += noise
        
        # Add noise to edge features
        if 'edge_features' in augmented_data:
            noise = torch.randn_like(augmented_data['edge_features']) * noise_std
            augmented_data['edge_features'] += noise
    
    elif augmentation_type == 'dropout':
        # Randomly set some features to zero
        if 'node_features' in augmented_data:
            dropout_mask = torch.rand_like(augmented_data['node_features']) > dropout_prob
            augmented_data['node_features'] *= dropout_mask.float()
        
        if 'edge_features' in augmented_data:
            dropout_mask = torch.rand_like(augmented_data['edge_features']) > dropout_prob
            augmented_data['edge_features'] *= dropout_mask.float()
    
    elif augmentation_type == 'permute':
        # Randomly permute node order (and corresponding edges)
        if 'node_features' in augmented_data and 'edge_indices' in augmented_data:
            num_nodes = augmented_data['node_features'].shape[0]
            perm = torch.randperm(num_nodes)
            
            # Permute node features
            augmented_data['node_features'] = augmented_data['node_features'][perm]
            
            # Update edge indices
            edge_indices = augmented_data['edge_indices']
            inv_perm = torch.argsort(perm)
            augmented_data['edge_indices'] = inv_perm[edge_indices]
    
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    return augmented_data


def compute_graph_statistics(
    graph_data_list: List[Dict[str, torch.Tensor]]
) -> Dict[str, Any]:
    """
    Compute statistics for a list of graph data.
    
    Args:
        graph_data_list: List of graph data dictionaries
        
    Returns:
        Dictionary with computed statistics
    """
    if not graph_data_list:
        return {}
    
    stats = {}
    
    # Node statistics
    if 'node_features' in graph_data_list[0]:
        all_node_features = torch.cat([g['node_features'] for g in graph_data_list], dim=0)
        num_nodes_list = [g['node_features'].shape[0] for g in graph_data_list]
        
        stats['node_features'] = {
            'mean': all_node_features.mean(dim=0).tolist(),
            'std': all_node_features.std(dim=0).tolist(),
            'min': all_node_features.min(dim=0)[0].tolist(),
            'max': all_node_features.max(dim=0)[0].tolist(),
            'feature_dim': all_node_features.shape[1]
        }
        
        stats['num_nodes'] = {
            'mean': np.mean(num_nodes_list),
            'std': np.std(num_nodes_list),
            'min': min(num_nodes_list),
            'max': max(num_nodes_list),
            'median': np.median(num_nodes_list)
        }
    
    # Edge statistics
    if 'edge_features' in graph_data_list[0]:
        all_edge_features = torch.cat([g['edge_features'] for g in graph_data_list], dim=0)
        num_edges_list = [g['edge_features'].shape[0] for g in graph_data_list]
        
        stats['edge_features'] = {
            'mean': all_edge_features.mean(dim=0).tolist(),
            'std': all_edge_features.std(dim=0).tolist(),
            'min': all_edge_features.min(dim=0)[0].tolist(),
            'max': all_edge_features.max(dim=0)[0].tolist(),
            'feature_dim': all_edge_features.shape[1]
        }
        
        stats['num_edges'] = {
            'mean': np.mean(num_edges_list),
            'std': np.std(num_edges_list),
            'min': min(num_edges_list),
            'max': max(num_edges_list),
            'median': np.median(num_edges_list)
        }
    
    # Global features statistics
    if 'global_features' in graph_data_list[0]:
        all_global_features = torch.stack([g['global_features'] for g in graph_data_list], dim=0)
        
        stats['global_features'] = {
            'mean': all_global_features.mean(dim=0).tolist(),
            'std': all_global_features.std(dim=0).tolist(),
            'min': all_global_features.min(dim=0)[0].tolist(),
            'max': all_global_features.max(dim=0)[0].tolist(),
            'feature_dim': all_global_features.shape[1]
        }
    
    # Graph connectivity statistics
    if 'edge_indices' in graph_data_list[0]:
        degrees = []
        for graph_data in graph_data_list:
            edge_indices = graph_data['edge_indices']
            num_nodes = graph_data['node_features'].shape[0]
            
            # Compute node degrees
            node_degrees = torch.zeros(num_nodes)
            for i in range(num_nodes):
                node_degrees[i] = (edge_indices[0] == i).sum() + (edge_indices[1] == i).sum()
            
            degrees.extend(node_degrees.tolist())
        
        stats['node_degrees'] = {
            'mean': np.mean(degrees),
            'std': np.std(degrees),
            'min': min(degrees),
            'max': max(degrees),
            'median': np.median(degrees)
        }
    
    stats['num_graphs'] = len(graph_data_list)
    
    return stats


def validate_graph_data(
    graph_data: Dict[str, torch.Tensor],
    check_connectivity: bool = True,
    check_features: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate graph data for consistency and correctness.
    
    Args:
        graph_data: Graph data dictionary
        check_connectivity: Whether to check graph connectivity
        check_features: Whether to check feature validity
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required keys
    required_keys = ['node_features', 'edge_indices']
    for key in required_keys:
        if key not in graph_data:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        return False, errors
    
    # Check tensor types and shapes
    node_features = graph_data['node_features']
    edge_indices = graph_data['edge_indices']
    
    if not isinstance(node_features, torch.Tensor):
        errors.append("node_features must be a torch.Tensor")
    
    if not isinstance(edge_indices, torch.Tensor):
        errors.append("edge_indices must be a torch.Tensor")
    
    if node_features.dim() != 2:
        errors.append(f"node_features must be 2D, got {node_features.dim()}D")
    
    if edge_indices.dim() != 2 or edge_indices.shape[0] != 2:
        errors.append(f"edge_indices must be 2xN, got shape {edge_indices.shape}")
    
    if errors:
        return False, errors
    
    num_nodes = node_features.shape[0]
    
    # Check connectivity
    if check_connectivity:
        if edge_indices.numel() > 0:
            max_node_idx = edge_indices.max().item()
            min_node_idx = edge_indices.min().item()
            
            if max_node_idx >= num_nodes:
                errors.append(f"Edge index {max_node_idx} >= num_nodes {num_nodes}")
            
            if min_node_idx < 0:
                errors.append(f"Negative edge index: {min_node_idx}")
    
    # Check features
    if check_features:
        if torch.isnan(node_features).any():
            errors.append("node_features contains NaN values")
        
        if torch.isinf(node_features).any():
            errors.append("node_features contains infinite values")
        
        if 'edge_features' in graph_data:
            edge_features = graph_data['edge_features']
            
            if edge_features.shape[0] != edge_indices.shape[1]:
                errors.append(
                    f"Mismatch: {edge_features.shape[0]} edge features "
                    f"but {edge_indices.shape[1]} edges"
                )
            
            if torch.isnan(edge_features).any():
                errors.append("edge_features contains NaN values")
            
            if torch.isinf(edge_features).any():
                errors.append("edge_features contains infinite values")
    
    return len(errors) == 0, errors