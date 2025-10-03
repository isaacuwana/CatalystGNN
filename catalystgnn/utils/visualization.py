"""
Visualization utilities for CatalystGNN.

This module provides functions for visualizing predictions, structures,
and model interpretability results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


def plot_predictions(
    y_true: Union[List[float], np.ndarray, torch.Tensor],
    y_pred: Union[List[float], np.ndarray, torch.Tensor],
    uncertainties: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None,
    title: str = "Prediction Results",
    xlabel: str = "True Values",
    ylabel: str = "Predicted Values",
    save_path: Optional[str] = None,
    show_metrics: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot predicted vs true values with optional uncertainty.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        uncertainties: Prediction uncertainties (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the plot
        show_metrics: Whether to show performance metrics
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if uncertainties is not None and isinstance(uncertainties, torch.Tensor):
        uncertainties = uncertainties.detach().cpu().numpy()
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot
    ax1 = axes[0]
    
    if uncertainties is not None:
        scatter = ax1.scatter(y_true, y_pred, c=uncertainties, cmap='viridis', alpha=0.6)
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Uncertainty')
    else:
        ax1.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(f'{title} - Scatter Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    ax2 = axes[1]
    residuals = y_pred - y_true
    
    if uncertainties is not None:
        scatter = ax2.scatter(y_pred, residuals, c=uncertainties, cmap='viridis', alpha=0.6)
    else:
        ax2.scatter(y_pred, residuals, alpha=0.6)
    
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel(ylabel)
    ax2.set_ylabel('Residuals (Predicted - True)')
    ax2.set_title(f'{title} - Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    # Add performance metrics
    if show_metrics:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}'
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction plot to {save_path}")
    
    return fig


def plot_structure(
    structure: Any,
    title: str = "Chemical Structure",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    show_bonds: bool = True,
    show_labels: bool = True
) -> plt.Figure:
    """
    Plot chemical structure (2D or 3D).
    
    Args:
        structure: Chemical structure object
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        show_bonds: Whether to show bonds
        show_labels: Whether to show atom labels
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    
    try:
        # Try to extract positions and atomic information
        positions = _extract_positions(structure)
        symbols = _extract_symbols(structure)
        bonds = _extract_bonds(structure) if show_bonds else None
        
        if positions.shape[1] == 3:
            # 3D plot
            ax = fig.add_subplot(111, projection='3d')
            _plot_3d_structure(ax, positions, symbols, bonds, show_labels)
        else:
            # 2D plot
            ax = fig.add_subplot(111)
            _plot_2d_structure(ax, positions, symbols, bonds, show_labels)
        
        ax.set_title(title)
        
    except Exception as e:
        logger.warning(f"Could not plot structure: {e}")
        # Create a simple placeholder plot
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"Structure visualization not available\n{str(structure)[:100]}...", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved structure plot to {save_path}")
    
    return fig


def _extract_positions(structure: Any) -> np.ndarray:
    """Extract atomic positions from structure."""
    if hasattr(structure, 'positions'):
        return np.array(structure.positions)
    elif hasattr(structure, 'cart_coords'):
        return np.array(structure.cart_coords)
    elif hasattr(structure, 'coords'):
        return np.array(structure.coords)
    elif isinstance(structure, dict) and 'positions' in structure:
        return np.array(structure['positions'])
    else:
        # Create dummy positions
        num_atoms = _get_num_atoms(structure)
        return np.random.rand(num_atoms, 3) * 10


def _extract_symbols(structure: Any) -> List[str]:
    """Extract atomic symbols from structure."""
    if hasattr(structure, 'symbols'):
        return list(structure.symbols)
    elif hasattr(structure, 'species'):
        return [str(specie) for specie in structure.species]
    elif isinstance(structure, dict) and 'symbols' in structure:
        return structure['symbols']
    else:
        # Create dummy symbols
        num_atoms = _get_num_atoms(structure)
        return ['C'] * num_atoms


def _extract_bonds(structure: Any) -> Optional[List[Tuple[int, int]]]:
    """Extract bond information from structure."""
    try:
        if hasattr(structure, 'GetBonds'):
            # RDKit molecule
            bonds = []
            for bond in structure.GetBonds():
                bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            return bonds
        elif hasattr(structure, 'connectivity'):
            return structure.connectivity
        else:
            return None
    except:
        return None


def _get_num_atoms(structure: Any) -> int:
    """Get number of atoms in structure."""
    if hasattr(structure, '__len__'):
        return len(structure)
    elif hasattr(structure, 'GetNumAtoms'):
        return structure.GetNumAtoms()
    elif isinstance(structure, dict) and 'num_atoms' in structure:
        return structure['num_atoms']
    else:
        return 10  # Default


def _plot_3d_structure(
    ax: plt.Axes,
    positions: np.ndarray,
    symbols: List[str],
    bonds: Optional[List[Tuple[int, int]]],
    show_labels: bool
):
    """Plot 3D structure."""
    # Color map for elements
    element_colors = {
        'H': 'white', 'C': 'black', 'N': 'blue', 'O': 'red',
        'F': 'green', 'P': 'orange', 'S': 'yellow', 'Cl': 'green',
        'Br': 'brown', 'I': 'purple', 'Zn': 'gray', 'Cu': 'brown'
    }
    
    # Plot atoms
    for i, (pos, symbol) in enumerate(zip(positions, symbols)):
        color = element_colors.get(symbol, 'gray')
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=100, alpha=0.8)
        
        if show_labels:
            ax.text(pos[0], pos[1], pos[2], f'{symbol}{i}', fontsize=8)
    
    # Plot bonds
    if bonds:
        for bond in bonds:
            i, j = bond
            if i < len(positions) and j < len(positions):
                pos1, pos2 = positions[i], positions[j]
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                       'k-', alpha=0.6)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')


def _plot_2d_structure(
    ax: plt.Axes,
    positions: np.ndarray,
    symbols: List[str],
    bonds: Optional[List[Tuple[int, int]]],
    show_labels: bool
):
    """Plot 2D structure."""
    # Use only first 2 dimensions
    positions_2d = positions[:, :2]
    
    # Color map for elements
    element_colors = {
        'H': 'white', 'C': 'black', 'N': 'blue', 'O': 'red',
        'F': 'green', 'P': 'orange', 'S': 'yellow', 'Cl': 'green',
        'Br': 'brown', 'I': 'purple', 'Zn': 'gray', 'Cu': 'brown'
    }
    
    # Plot bonds first (so they appear behind atoms)
    if bonds:
        for bond in bonds:
            i, j = bond
            if i < len(positions_2d) and j < len(positions_2d):
                pos1, pos2 = positions_2d[i], positions_2d[j]
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', alpha=0.6)
    
    # Plot atoms
    for i, (pos, symbol) in enumerate(zip(positions_2d, symbols)):
        color = element_colors.get(symbol, 'gray')
        ax.scatter(pos[0], pos[1], c=color, s=200, alpha=0.8, edgecolors='black')
        
        if show_labels:
            ax.text(pos[0], pos[1], symbol, ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_aspect('equal')


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot training history (loss, metrics over epochs).
    
    Args:
        history: Dictionary with training history
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss', marker='s')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot metrics
    ax2 = axes[1]
    metric_plotted = False
    
    for key, values in history.items():
        if key not in ['train_loss', 'val_loss'] and 'loss' not in key.lower():
            ax2.plot(values, label=key.replace('_', ' ').title(), marker='o')
            metric_plotted = True
    
    if metric_plotted:
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Training Metrics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No metrics to display', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Metrics')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: Union[List[float], np.ndarray, torch.Tensor],
    title: str = "Feature Importance",
    top_k: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance scores.
    
    Args:
        feature_names: List of feature names
        importance_scores: Feature importance scores
        title: Plot title
        top_k: Number of top features to show
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy array
    if isinstance(importance_scores, torch.Tensor):
        importance_scores = importance_scores.detach().cpu().numpy()
    importance_scores = np.array(importance_scores)
    
    # Sort by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    top_indices = sorted_indices[:top_k]
    
    top_names = [feature_names[i] for i in top_indices]
    top_scores = importance_scores[top_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(range(len(top_names)), top_scores)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    
    # Color bars by importance
    colors = plt.cm.viridis(top_scores / top_scores.max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        ax.text(bar.get_width() + 0.01 * top_scores.max(), bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    return fig


def plot_attention_weights(
    attention_weights: torch.Tensor,
    node_labels: Optional[List[str]] = None,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weight matrix
        node_labels: Labels for nodes (optional)
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')
    
    # Set labels
    if node_labels:
        ax.set_xticks(range(len(node_labels)))
        ax.set_yticks(range(len(node_labels)))
        ax.set_xticklabels(node_labels, rotation=45, ha='right')
        ax.set_yticklabels(node_labels)
    
    ax.set_xlabel('Target Node')
    ax.set_ylabel('Source Node')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention weights plot to {save_path}")
    
    return fig


def plot_property_distribution(
    properties: Union[List[float], np.ndarray, torch.Tensor],
    property_name: str = "Property",
    bins: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot distribution of property values.
    
    Args:
        properties: Property values
        property_name: Name of the property
        bins: Number of histogram bins
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy array
    if isinstance(properties, torch.Tensor):
        properties = properties.detach().cpu().numpy()
    properties = np.array(properties)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(properties, bins=bins, alpha=0.7, edgecolor='black')
    ax1.set_xlabel(property_name)
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{property_name} Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    ax2.boxplot(properties, vert=True)
    ax2.set_ylabel(property_name)
    ax2.set_title(f'{property_name} Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Mean: {np.mean(properties):.3f}\n'
    stats_text += f'Std: {np.std(properties):.3f}\n'
    stats_text += f'Min: {np.min(properties):.3f}\n'
    stats_text += f'Max: {np.max(properties):.3f}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved property distribution plot to {save_path}")
    
    return fig


def create_model_summary_plot(
    model_info: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a summary visualization of model architecture and parameters.
    
    Args:
        model_info: Dictionary with model information
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Model architecture info
    ax1 = axes[0, 0]
    arch_info = [
        f"Model Type: {model_info.get('model_type', 'Unknown')}",
        f"Hidden Dim: {model_info.get('hidden_dim', 'N/A')}",
        f"Num Layers: {model_info.get('num_layers', 'N/A')}",
        f"Dropout: {model_info.get('dropout', 'N/A')}",
        f"Total Parameters: {model_info.get('num_parameters', 'N/A'):,}",
        f"Trainable Parameters: {model_info.get('trainable_parameters', 'N/A'):,}"
    ]
    
    ax1.text(0.1, 0.9, '\n'.join(arch_info), transform=ax1.transAxes, 
             verticalalignment='top', fontsize=12, fontfamily='monospace')
    ax1.set_title('Model Architecture')
    ax1.axis('off')
    
    # Parameter distribution (if available)
    ax2 = axes[0, 1]
    if 'by_category' in model_info:
        categories = list(model_info['by_category'].keys())
        param_counts = list(model_info['by_category'].values())
        
        ax2.pie(param_counts, labels=categories, autopct='%1.1f%%')
        ax2.set_title('Parameter Distribution')
    else:
        ax2.text(0.5, 0.5, 'Parameter distribution\nnot available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Parameter Distribution')
    
    # Feature dimensions
    ax3 = axes[1, 0]
    feature_dims = [
        f"Node Features: {model_info.get('node_feature_dim', 'N/A')}",
        f"Edge Features: {model_info.get('edge_feature_dim', 'N/A')}",
        f"Global Features: {model_info.get('global_feature_dim', 'N/A')}"
    ]
    
    ax3.text(0.1, 0.7, '\n'.join(feature_dims), transform=ax3.transAxes, 
             verticalalignment='top', fontsize=12, fontfamily='monospace')
    ax3.set_title('Feature Dimensions')
    ax3.axis('off')
    
    # Model complexity visualization
    ax4 = axes[1, 1]
    total_params = model_info.get('num_parameters', 0)
    trainable_params = model_info.get('trainable_parameters', 0)
    frozen_params = total_params - trainable_params
    
    if total_params > 0:
        sizes = [trainable_params, frozen_params] if frozen_params > 0 else [trainable_params]
        labels = ['Trainable', 'Frozen'] if frozen_params > 0 else ['Trainable']
        colors = ['lightblue', 'lightcoral'] if frozen_params > 0 else ['lightblue']
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax4.set_title('Parameter Status')
    else:
        ax4.text(0.5, 0.5, 'Parameter information\nnot available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Parameter Status')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model summary plot to {save_path}")
    
    return fig