"""
Graph Neural Network model implementations.

This module contains implementations of various GNN architectures
optimized for catalyst property prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import logging

from .base_model import BaseGNNModel

logger = logging.getLogger(__name__)


class CGCNNModel(BaseGNNModel):
    """
    Crystal Graph Convolutional Neural Network.
    
    Implementation of CGCNN for predicting properties of crystalline materials.
    Particularly effective for MOFs, zeolites, and other porous materials.
    
    Reference: Xie & Grossman, "Crystal Graph Convolutional Neural Networks 
    for an Accurate and Interpretable Prediction of Material Properties"
    """
    
    def __init__(
        self,
        node_feature_dim: int = 92,
        edge_feature_dim: int = 41,
        global_feature_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: str = 'relu',
        pool_method: str = 'mean',
        **kwargs
    ):
        """
        Initialize CGCNN model.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            global_feature_dim: Dimension of global features
            hidden_dim: Hidden layer dimension
            num_layers: Number of CGCNN layers
            dropout: Dropout probability
            activation: Activation function
            pool_method: Graph pooling method ('mean', 'max', 'sum', 'attention')
        """
        self.pool_method = pool_method
        super().__init__(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            **kwargs
        )
    
    def _build_model(self):
        """Build CGCNN architecture."""
        # Node embedding layer
        self.node_embedding = nn.Linear(self.node_feature_dim, self.hidden_dim)
        
        # Edge embedding layer
        self.edge_embedding = nn.Linear(self.edge_feature_dim, self.hidden_dim)
        
        # CGCNN layers
        self.gnn_layers = nn.ModuleList([
            CGCNNLayer(self.hidden_dim, self.activation, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Global feature processing
        if self.global_feature_dim > 0:
            self.global_embedding = nn.Linear(self.global_feature_dim, self.hidden_dim // 2)
        
        # Attention pooling if specified
        if self.pool_method == 'attention':
            self.attention_pool = AttentionPooling(self.hidden_dim)
        
        # Prediction head
        predictor_input_dim = self.hidden_dim
        if self.global_feature_dim > 0:
            predictor_input_dim += self.hidden_dim // 2
        
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
    
    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of CGCNN.
        
        Args:
            graph_data: Dictionary containing graph tensors
            
        Returns:
            Predicted property values
        """
        # Extract graph components
        node_features = graph_data['node_features']
        edge_indices = graph_data['edge_indices']
        edge_features = graph_data['edge_features']
        
        # Embed initial features
        h = self.node_embedding(node_features)
        edge_attr = self.edge_embedding(edge_features)
        
        # Apply CGCNN layers
        for i, (gnn_layer, batch_norm) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            h_new = gnn_layer(h, edge_indices, edge_attr)
            h_new = batch_norm(h_new)
            h = h + h_new  # Residual connection
        
        # Graph pooling
        if self.pool_method == 'mean':
            graph_repr = torch.mean(h, dim=0, keepdim=True)
        elif self.pool_method == 'max':
            graph_repr = torch.max(h, dim=0, keepdim=True)[0]
        elif self.pool_method == 'sum':
            graph_repr = torch.sum(h, dim=0, keepdim=True)
        elif self.pool_method == 'attention':
            graph_repr = self.attention_pool(h)
        else:
            graph_repr = torch.mean(h, dim=0, keepdim=True)
        
        # Include global features if available
        if self.global_feature_dim > 0 and 'global_features' in graph_data:
            global_features = graph_data['global_features']
            global_repr = self.global_embedding(global_features.unsqueeze(0))
            graph_repr = torch.cat([graph_repr, global_repr], dim=-1)
        
        # Predict property
        prediction = self.predictor(graph_repr)
        
        return prediction.squeeze()


class MPNNModel(BaseGNNModel):
    """
    Message Passing Neural Network.
    
    Implementation of MPNN for molecular property prediction.
    Particularly effective for organic molecules and molecular catalysts.
    
    Reference: Gilmer et al., "Neural Message Passing for Quantum Chemistry"
    """
    
    def __init__(
        self,
        node_feature_dim: int = 133,
        edge_feature_dim: int = 23,
        global_feature_dim: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        message_function: str = 'edge_network',
        update_function: str = 'gru',
        **kwargs
    ):
        """
        Initialize MPNN model.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            global_feature_dim: Dimension of global features
            hidden_dim: Hidden layer dimension
            num_layers: Number of message passing steps
            dropout: Dropout probability
            activation: Activation function
            message_function: Type of message function ('edge_network', 'simple')
            update_function: Type of update function ('gru', 'lstm', 'mlp')
        """
        self.message_function = message_function
        self.update_function = update_function
        super().__init__(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            **kwargs
        )
    
    def _build_model(self):
        """Build MPNN architecture."""
        # Node embedding
        self.node_embedding = nn.Linear(self.node_feature_dim, self.hidden_dim)
        
        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            MPNNLayer(
                hidden_dim=self.hidden_dim,
                edge_feature_dim=self.edge_feature_dim,
                message_function=self.message_function,
                update_function=self.update_function,
                activation=self.activation,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Global feature processing
        if self.global_feature_dim > 0:
            self.global_embedding = nn.Linear(self.global_feature_dim, self.hidden_dim // 2)
        
        # Set2Set pooling for better molecular representation
        self.set2set = Set2SetPooling(self.hidden_dim, num_layers=2)
        
        # Prediction head
        predictor_input_dim = 2 * self.hidden_dim  # Set2Set doubles the dimension
        if self.global_feature_dim > 0:
            predictor_input_dim += self.hidden_dim // 2
        
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
    
    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of MPNN.
        
        Args:
            graph_data: Dictionary containing graph tensors
            
        Returns:
            Predicted property values
        """
        # Extract graph components
        node_features = graph_data['node_features']
        edge_indices = graph_data['edge_indices']
        edge_features = graph_data['edge_features']
        
        # Embed initial node features
        h = self.node_embedding(node_features)
        
        # Message passing
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_indices, edge_features)
        
        # Set2Set pooling
        graph_repr = self.set2set(h)
        
        # Include global features if available
        if self.global_feature_dim > 0 and 'global_features' in graph_data:
            global_features = graph_data['global_features']
            global_repr = self.global_embedding(global_features.unsqueeze(0))
            graph_repr = torch.cat([graph_repr, global_repr], dim=-1)
        
        # Predict property
        prediction = self.predictor(graph_repr)
        
        return prediction.squeeze()


class GATModel(BaseGNNModel):
    """
    Graph Attention Network.
    
    Implementation of GAT with multi-head attention for catalyst property prediction.
    Provides interpretable attention weights showing important atomic interactions.
    
    Reference: Veličković et al., "Graph Attention Networks"
    """
    
    def __init__(
        self,
        node_feature_dim: int = 92,
        edge_feature_dim: int = 41,
        global_feature_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = 'elu',
        attention_dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize GAT model.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            global_feature_dim: Dimension of global features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            activation: Activation function
            attention_dropout: Attention dropout probability
        """
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        super().__init__(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            **kwargs
        )
    
    def _build_model(self):
        """Build GAT architecture."""
        # Node embedding
        self.node_embedding = nn.Linear(self.node_feature_dim, self.hidden_dim)
        
        # GAT layers
        self.gnn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                # First layer
                layer = GATLayer(
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim // self.num_heads,
                    num_heads=self.num_heads,
                    edge_feature_dim=self.edge_feature_dim,
                    dropout=self.dropout,
                    attention_dropout=self.attention_dropout,
                    activation=self.activation,
                    concat=True
                )
            elif i == self.num_layers - 1:
                # Last layer
                layer = GATLayer(
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim,
                    num_heads=1,
                    edge_feature_dim=self.edge_feature_dim,
                    dropout=self.dropout,
                    attention_dropout=self.attention_dropout,
                    activation=self.activation,
                    concat=False
                )
            else:
                # Middle layers
                layer = GATLayer(
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim // self.num_heads,
                    num_heads=self.num_heads,
                    edge_feature_dim=self.edge_feature_dim,
                    dropout=self.dropout,
                    attention_dropout=self.attention_dropout,
                    activation=self.activation,
                    concat=True
                )
            self.gnn_layers.append(layer)
        
        # Global feature processing
        if self.global_feature_dim > 0:
            self.global_embedding = nn.Linear(self.global_feature_dim, self.hidden_dim // 2)
        
        # Attention-based pooling
        self.attention_pool = AttentionPooling(self.hidden_dim)
        
        # Prediction head
        predictor_input_dim = self.hidden_dim
        if self.global_feature_dim > 0:
            predictor_input_dim += self.hidden_dim // 2
        
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Store attention weights for interpretability
        self.attention_weights = []
    
    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of GAT.
        
        Args:
            graph_data: Dictionary containing graph tensors
            
        Returns:
            Predicted property values
        """
        # Extract graph components
        node_features = graph_data['node_features']
        edge_indices = graph_data['edge_indices']
        edge_features = graph_data['edge_features']
        
        # Embed initial node features
        h = self.node_embedding(node_features)
        
        # Clear previous attention weights
        self.attention_weights = []
        
        # Apply GAT layers
        for gnn_layer in self.gnn_layers:
            h, attention = gnn_layer(h, edge_indices, edge_features)
            self.attention_weights.append(attention)
        
        # Attention-based pooling
        graph_repr = self.attention_pool(h)
        
        # Include global features if available
        if self.global_feature_dim > 0 and 'global_features' in graph_data:
            global_features = graph_data['global_features']
            global_repr = self.global_embedding(global_features.unsqueeze(0))
            graph_repr = torch.cat([graph_repr, global_repr], dim=-1)
        
        # Predict property
        prediction = self.predictor(graph_repr)
        
        return prediction.squeeze()
    
    def compute_attention_weights(self, graph_data: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute and return attention weights.
        
        Args:
            graph_data: Dictionary containing graph tensors
            
        Returns:
            List of attention weight tensors for each layer
        """
        _ = self.forward(graph_data)  # Run forward pass to compute attention
        return self.attention_weights


# Helper layers and modules

class CGCNNLayer(nn.Module):
    """Single CGCNN layer."""
    
    def __init__(self, hidden_dim: int, activation: nn.Module, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        
        # Convolution weights
        self.weight = nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim)
        self.gate = nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim)
    
    def forward(self, h: torch.Tensor, edge_indices: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass of CGCNN layer."""
        row, col = edge_indices
        
        # Gather node features for edges
        h_i = h[row]  # Source nodes
        h_j = h[col]  # Target nodes
        
        # Concatenate node and edge features
        edge_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        
        # Compute messages
        messages = self.activation(self.weight(edge_input))
        gates = torch.sigmoid(self.gate(edge_input))
        messages = messages * gates
        
        # Aggregate messages
        h_new = torch.zeros_like(h)
        h_new = h_new.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.hidden_dim), messages)
        
        return self.dropout(h_new)


class MPNNLayer(nn.Module):
    """Single MPNN layer."""
    
    def __init__(
        self,
        hidden_dim: int,
        edge_feature_dim: int,
        message_function: str,
        update_function: str,
        activation: nn.Module,
        dropout: float
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_function = message_function
        self.update_function = update_function
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        
        # Message function
        if message_function == 'edge_network':
            self.message_net = nn.Sequential(
                nn.Linear(2 * hidden_dim + edge_feature_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.message_net = nn.Linear(hidden_dim + edge_feature_dim, hidden_dim)
        
        # Update function
        if update_function == 'gru':
            self.update_net = nn.GRUCell(hidden_dim, hidden_dim)
        elif update_function == 'lstm':
            self.update_net = nn.LSTMCell(hidden_dim, hidden_dim)
        else:
            self.update_net = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim)
            )
    
    def forward(self, h: torch.Tensor, edge_indices: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass of MPNN layer."""
        row, col = edge_indices
        
        # Compute messages
        if self.message_function == 'edge_network':
            edge_input = torch.cat([h[row], h[col], edge_attr], dim=-1)
        else:
            edge_input = torch.cat([h[col], edge_attr], dim=-1)
        
        messages = self.message_net(edge_input)
        
        # Aggregate messages
        aggregated = torch.zeros_like(h)
        aggregated = aggregated.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.hidden_dim), messages)
        
        # Update node states
        if self.update_function == 'gru':
            h_new = self.update_net(aggregated, h)
        elif self.update_function == 'lstm':
            h_new, _ = self.update_net(aggregated, h)  # LSTM returns (h, c)
        else:
            h_new = self.update_net(torch.cat([h, aggregated], dim=-1))
        
        return self.dropout(h_new)


class GATLayer(nn.Module):
    """Single GAT layer with multi-head attention."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        edge_feature_dim: int,
        dropout: float,
        attention_dropout: float,
        activation: nn.Module,
        concat: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.activation = activation
        
        # Linear transformations
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * out_features + edge_feature_dim))
        
        # Edge feature processing
        self.edge_transform = nn.Linear(edge_feature_dim, num_heads * out_features)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, h: torch.Tensor, edge_indices: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of GAT layer."""
        N = h.size(0)
        
        # Linear transformation
        h_transformed = self.W(h).view(N, self.num_heads, self.out_features)
        
        # Process edge features
        edge_transformed = self.edge_transform(edge_attr).view(-1, self.num_heads, self.out_features)
        
        # Compute attention
        row, col = edge_indices
        h_i = h_transformed[row]  # Source nodes
        h_j = h_transformed[col]  # Target nodes
        
        # Attention mechanism
        attention_input = torch.cat([h_i, h_j, edge_transformed], dim=-1)
        e = torch.sum(self.a * attention_input, dim=-1)
        e = F.leaky_relu(e, negative_slope=0.2)
        
        # Compute attention weights using segment-wise softmax
        # Use a simpler approach that's compatible with older PyTorch versions
        
        # Compute exp of attention scores
        e_exp = torch.exp(e - e.max())  # Subtract max for numerical stability
        
        # Compute sum for each target node
        attention_sum = torch.zeros(N, self.num_heads, device=h.device)
        attention_sum = attention_sum.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.num_heads), e_exp)
        
        # Normalize to get attention weights
        attention_weights = e_exp / (attention_sum[row] + 1e-8)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to node features
        h_out = torch.zeros(N, self.num_heads, self.out_features, device=h.device)
        weighted_features = attention_weights.unsqueeze(-1) * h_j
        h_out = h_out.scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.out_features), weighted_features)
        
        # Store attention weights for interpretability
        attention = attention_weights
        
        # Concatenate or average heads
        if self.concat:
            h_out = h_out.view(N, self.num_heads * self.out_features)
        else:
            h_out = h_out.mean(dim=1)
        
        h_out = self.dropout(self.activation(h_out))
        
        return h_out, attention


class AttentionPooling(nn.Module):
    """Attention-based graph pooling."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply attention pooling."""
        attention_weights = F.softmax(self.attention(h), dim=0)
        pooled = torch.sum(attention_weights * h, dim=0, keepdim=True)
        return pooled


class Set2SetPooling(nn.Module):
    """Set2Set pooling for molecular graphs."""
    
    def __init__(self, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply Set2Set pooling."""
        batch_size = 1
        num_nodes = h.size(0)
        
        # Initialize LSTM state
        h_lstm = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=h.device)
        c_lstm = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=h.device)
        
        # Set2Set iterations
        q_star = torch.zeros(batch_size, 2 * self.hidden_dim, device=h.device)
        
        for _ in range(num_nodes):
            # LSTM step
            q, (h_lstm, c_lstm) = self.lstm(q_star.unsqueeze(1), (h_lstm, c_lstm))
            q = q.squeeze(1)
            
            # Attention
            e = torch.sum(self.attention(q).unsqueeze(0) * h, dim=-1)
            a = F.softmax(e, dim=0)
            
            # Read
            r = torch.sum(a.unsqueeze(-1) * h, dim=0, keepdim=True)
            
            # Update
            q_star = torch.cat([q, r], dim=-1)
        
        return q_star