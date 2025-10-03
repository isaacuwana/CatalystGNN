"""
Main predictor class for CatalystGNN.

This module provides the high-level interface for predicting catalytic properties
from molecular and crystal structures using pre-trained GNN models.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
import logging

from ..featurizers.crystal_featurizer import CrystalGraphFeaturizer
from ..featurizers.molecular_featurizer import MolecularGraphFeaturizer
from ..models.gnn_models import CGCNNModel, MPNNModel, GATModel
from ..utils.file_handlers import load_structure_from_file, parse_smiles
from ..utils.preprocessing import normalize_features, handle_missing_values

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatalystPredictor:
    """
    Main predictor class for catalyst property prediction using GNNs.
    
    This class provides a unified interface for predicting various catalytic
    properties from different input formats (CIF, SMILES, etc.).
    
    Attributes:
        model_type (str): Type of property to predict ('co2_adsorption', 'catalytic_activity')
        device (torch.device): Device for model inference
        model: Loaded GNN model
        featurizer: Graph featurizer appropriate for the input type
        scaler: Feature scaler for normalization
    """
    
    SUPPORTED_PROPERTIES = {
        'co2_adsorption': {
            'model_class': CGCNNModel,
            'featurizer_class': CrystalGraphFeaturizer,
            'model_file': 'co2_adsorption_cgcnn.pth',
            'scaler_file': 'co2_adsorption_scaler.pkl',
            'description': 'CO2 adsorption energy prediction for porous materials',
            'units': 'kJ/mol'
        },
        'catalytic_activity': {
            'model_class': MPNNModel,
            'featurizer_class': MolecularGraphFeaturizer,
            'model_file': 'catalytic_activity_mpnn.pth',
            'scaler_file': 'catalytic_activity_scaler.pkl',
            'description': 'Catalytic activity prediction for molecular catalysts',
            'units': 'log(turnover frequency)'
        },
        'selectivity': {
            'model_class': GATModel,
            'featurizer_class': CrystalGraphFeaturizer,
            'model_file': 'selectivity_gat.pth',
            'scaler_file': 'selectivity_scaler.pkl',
            'description': 'Selectivity prediction for separation processes',
            'units': 'selectivity ratio'
        }
    }
    
    def __init__(
        self, 
        model_type: str = 'co2_adsorption',
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the CatalystPredictor.
        
        Args:
            model_type: Type of property to predict
            device: Device for inference ('cpu', 'cuda', or None for auto-detection)
            model_path: Custom path to model files (optional)
            verbose: Whether to print initialization information
        """
        if model_type not in self.SUPPORTED_PROPERTIES:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(self.SUPPORTED_PROPERTIES.keys())}"
            )
        
        self.model_type = model_type
        self.verbose = verbose
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if self.verbose:
            logger.info(f"Initializing CatalystPredictor for {model_type}")
            logger.info(f"Using device: {self.device}")
        
        # Get model configuration
        self.config = self.SUPPORTED_PROPERTIES[model_type]
        
        # Set model path
        if model_path is None:
            self.model_path = Path(__file__).parent.parent / 'models'
        else:
            self.model_path = Path(model_path)
        
        # Initialize components
        self._load_model()
        self._initialize_featurizer()
        self._load_scaler()
        
        if self.verbose:
            logger.info(f"Successfully initialized predictor for {self.config['description']}")
    
    def _load_model(self):
        """Load the pre-trained GNN model."""
        model_file = self.model_path / self.config['model_file']
        
        if not model_file.exists():
            logger.warning(f"Model file not found: {model_file}")
            logger.info("Creating dummy model for demonstration purposes")
            self._create_dummy_model()
            return
        
        try:
            # Load model state dict
            state_dict = torch.load(model_file, map_location=self.device)
            
            # Initialize model architecture
            self.model = self.config['model_class']()
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            if self.verbose:
                logger.info(f"Loaded model from {model_file}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Creating dummy model for demonstration purposes")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for demonstration when pre-trained models are not available."""
        self.model = self.config['model_class']()
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize with random weights (for demonstration)
        for param in self.model.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)
    
    def _initialize_featurizer(self):
        """Initialize the appropriate graph featurizer."""
        self.featurizer = self.config['featurizer_class']()
        
        if self.verbose:
            logger.info(f"Initialized {self.config['featurizer_class'].__name__}")
    
    def _load_scaler(self):
        """Load the feature scaler."""
        scaler_file = self.model_path / self.config['scaler_file']
        
        if scaler_file.exists():
            try:
                import joblib
                self.scaler = joblib.load(scaler_file)
                if self.verbose:
                    logger.info(f"Loaded scaler from {scaler_file}")
            except Exception as e:
                logger.warning(f"Could not load scaler: {e}")
                self.scaler = None
        else:
            logger.warning(f"Scaler file not found: {scaler_file}")
            self.scaler = None
    
    def predict_from_file(
        self, 
        file_path: str, 
        return_uncertainty: bool = False,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Predict property from a structure file.
        
        Args:
            file_path: Path to structure file (.cif, .xyz, etc.)
            return_uncertainty: Whether to return prediction uncertainty
            return_features: Whether to return graph features
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Load structure from file
            structure = load_structure_from_file(file_path)
            
            # Featurize structure
            graph_data = self.featurizer.featurize(structure)
            
            # Make prediction
            result = self._predict_from_graph(
                graph_data, 
                return_uncertainty=return_uncertainty,
                return_features=return_features
            )
            
            # Add metadata
            result['input_file'] = file_path
            result['structure_formula'] = getattr(structure, 'composition', 'Unknown')
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting from file {file_path}: {e}")
            raise
    
    def predict_from_smiles(
        self, 
        smiles: str,
        return_uncertainty: bool = False,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Predict property from SMILES string.
        
        Args:
            smiles: SMILES string representation
            return_uncertainty: Whether to return prediction uncertainty
            return_features: Whether to return graph features
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Parse SMILES
            molecule = parse_smiles(smiles)
            
            # Featurize molecule
            graph_data = self.featurizer.featurize(molecule)
            
            # Make prediction
            result = self._predict_from_graph(
                graph_data,
                return_uncertainty=return_uncertainty,
                return_features=return_features
            )
            
            # Add metadata
            result['input_smiles'] = smiles
            result['molecular_formula'] = getattr(molecule, 'formula', 'Unknown')
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting from SMILES {smiles}: {e}")
            raise
    
    def predict_batch(
        self,
        inputs: List[Union[str, Dict]],
        batch_size: int = 32,
        return_uncertainty: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Predict properties for a batch of inputs.
        
        Args:
            inputs: List of file paths, SMILES strings, or input dictionaries
            batch_size: Batch size for processing
            return_uncertainty: Whether to return prediction uncertainties
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_results = []
            
            for inp in batch:
                try:
                    if isinstance(inp, str):
                        if inp.endswith(('.cif', '.xyz', '.poscar')):
                            result = self.predict_from_file(inp, return_uncertainty=return_uncertainty)
                        else:
                            # Assume SMILES string
                            result = self.predict_from_smiles(inp, return_uncertainty=return_uncertainty)
                    elif isinstance(inp, dict):
                        # Handle dictionary input with explicit type
                        if 'file_path' in inp:
                            result = self.predict_from_file(inp['file_path'], return_uncertainty=return_uncertainty)
                        elif 'smiles' in inp:
                            result = self.predict_from_smiles(inp['smiles'], return_uncertainty=return_uncertainty)
                        else:
                            raise ValueError("Dictionary input must contain 'file_path' or 'smiles' key")
                    else:
                        raise ValueError(f"Unsupported input type: {type(inp)}")
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing input {inp}: {e}")
                    batch_results.append({
                        'error': str(e),
                        'input': inp,
                        'prediction': None
                    })
            
            results.extend(batch_results)
            
            if self.verbose and len(inputs) > batch_size:
                logger.info(f"Processed {min(i + batch_size, len(inputs))}/{len(inputs)} inputs")
        
        return results
    
    def _predict_from_graph(
        self,
        graph_data: Dict[str, torch.Tensor],
        return_uncertainty: bool = False,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction from graph data.
        
        Args:
            graph_data: Featurized graph data
            return_uncertainty: Whether to return uncertainty estimate
            return_features: Whether to return graph features
            
        Returns:
            Dictionary containing prediction results
        """
        with torch.no_grad():
            # Move data to device
            for key, value in graph_data.items():
                if isinstance(value, torch.Tensor):
                    graph_data[key] = value.to(self.device)
            
            # Apply scaling if available
            if self.scaler is not None:
                graph_data = self._apply_scaling(graph_data)
            
            # Make prediction
            if return_uncertainty:
                # For uncertainty estimation, we could use dropout at inference
                # or ensemble methods. For now, we'll use a simple approach.
                predictions = []
                self.model.train()  # Enable dropout
                
                for _ in range(10):  # Monte Carlo dropout
                    pred = self.model(graph_data)
                    predictions.append(pred.cpu().numpy())
                
                self.model.eval()
                predictions = np.array(predictions)
                
                prediction = np.mean(predictions, axis=0)
                uncertainty = np.std(predictions, axis=0)
            else:
                prediction = self.model(graph_data).cpu().numpy()
                uncertainty = None
            
            # Prepare result
            result = {
                'prediction': float(prediction.item() if prediction.size == 1 else prediction),
                'property': self.model_type,
                'units': self.config['units'],
                'model_description': self.config['description']
            }
            
            if uncertainty is not None:
                result['uncertainty'] = float(uncertainty.item() if uncertainty.size == 1 else uncertainty)
            
            if return_features:
                result['graph_features'] = {
                    key: value.cpu().numpy().tolist() if isinstance(value, torch.Tensor) else value
                    for key, value in graph_data.items()
                }
            
            return result
    
    def _apply_scaling(self, graph_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply feature scaling to graph data."""
        # This is a simplified version - in practice, you'd need to handle
        # different types of features (node, edge, global) separately
        if 'node_features' in graph_data and hasattr(self.scaler, 'transform'):
            try:
                scaled_features = self.scaler.transform(graph_data['node_features'].cpu().numpy())
                graph_data['node_features'] = torch.tensor(scaled_features, dtype=torch.float32)
            except Exception as e:
                logger.warning(f"Could not apply scaling: {e}")
        
        return graph_data
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_type': self.model_type,
            'description': self.config['description'],
            'units': self.config['units'],
            'model_class': self.config['model_class'].__name__,
            'featurizer_class': self.config['featurizer_class'].__name__,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, str]]:
        """List all available pre-trained models."""
        return {
            model_type: {
                'description': config['description'],
                'units': config['units'],
                'model_class': config['model_class'].__name__,
                'featurizer_class': config['featurizer_class'].__name__
            }
            for model_type, config in cls.SUPPORTED_PROPERTIES.items()
        }