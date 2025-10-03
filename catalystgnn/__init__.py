"""
CatalystGNN: Graph Neural Networks for Catalyst Property Prediction

A comprehensive package for predicting catalytic properties using graph neural networks.
Supports multiple input formats including CIF files and SMILES strings.

Author: Isaac U. Adeyeye
"""

__version__ = "0.1.0"
__author__ = "Isaac U. Adeyeye"
__email__ = "isaac.adeyeye@example.com"

from .core.predictor import CatalystPredictor
from .featurizers.crystal_featurizer import CrystalGraphFeaturizer
from .featurizers.molecular_featurizer import MolecularGraphFeaturizer
from .models.gnn_models import CGCNNModel, MPNNModel, GATModel
from .utils.data_loader import DataLoader
from .utils.visualization import plot_predictions, plot_structure

__all__ = [
    "CatalystPredictor",
    "CrystalGraphFeaturizer", 
    "MolecularGraphFeaturizer",
    "CGCNNModel",
    "MPNNModel", 
    "GATModel",
    "DataLoader",
    "plot_predictions",
    "plot_structure",
]

# Package metadata
PACKAGE_INFO = {
    "name": "CatalystGNN",
    "version": __version__,
    "description": "Graph Neural Networks for Catalyst Property Prediction",
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/isaacuwana/CatalystGNN",
    "license": "MIT",
}