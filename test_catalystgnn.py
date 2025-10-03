"""
Comprehensive test script for CatalystGNN.

This script tests all major components of the CatalystGNN package
to ensure everything is working correctly.
"""

import sys
import traceback
from pathlib import Path
import torch
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    try:
        # Core imports
        from catalystgnn import CatalystPredictor
        from catalystgnn.featurizers import CrystalGraphFeaturizer, MolecularGraphFeaturizer
        from catalystgnn.models import CGCNNModel, MPNNModel, GATModel
        from catalystgnn.utils import DataLoader, plot_predictions
        
        print("✓ All core modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_featurizers():
    """Test featurizer functionality."""
    print("\n" + "="*60)
    print("TESTING FEATURIZERS")
    print("="*60)
    
    try:
        from catalystgnn.featurizers import MolecularGraphFeaturizer, CrystalGraphFeaturizer
        from catalystgnn.utils.file_handlers import parse_smiles
        
        # Test molecular featurizer
        mol_featurizer = MolecularGraphFeaturizer()
        molecule = parse_smiles('CCO')
        mol_graph = mol_featurizer.featurize(molecule)
        
        print(f"✓ Molecular featurizer: {mol_graph['node_features'].shape} nodes")
        
        # Test crystal featurizer
        crystal_featurizer = CrystalGraphFeaturizer()
        # Use dummy crystal structure
        from catalystgnn.utils.file_handlers import create_dummy_crystal_structure
        crystal = create_dummy_crystal_structure()
        crystal_graph = crystal_featurizer.featurize(crystal)
        
        print(f"✓ Crystal featurizer: {crystal_graph['node_features'].shape} nodes")
        
        return True
        
    except Exception as e:
        print(f"✗ Featurizer error: {e}")
        traceback.print_exc()
        return False


def test_models():
    """Test model architectures."""
    print("\n" + "="*60)
    print("TESTING MODELS")
    print("="*60)
    
    try:
        from catalystgnn.models import CGCNNModel, MPNNModel, GATModel
        
        # Test CGCNN
        cgcnn = CGCNNModel(node_feature_dim=10, edge_feature_dim=5, hidden_dim=32, num_layers=2, global_feature_dim=10)
        dummy_data = {
            'node_features': torch.randn(5, 10),
            'edge_indices': torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
            'edge_features': torch.randn(5, 5),
            'global_features': torch.randn(10)
        }
        
        output = cgcnn(dummy_data)
        print(f"✓ CGCNN output shape: {output.shape}")
        
        # Test MPNN
        mpnn = MPNNModel(node_feature_dim=10, edge_feature_dim=5, hidden_dim=32, num_layers=2, global_feature_dim=10)
        output = mpnn(dummy_data)
        print(f"✓ MPNN output shape: {output.shape}")
        
        # Test GAT
        gat = GATModel(node_feature_dim=10, edge_feature_dim=5, hidden_dim=32, num_layers=2, num_heads=4, global_feature_dim=10)
        output = gat(dummy_data)
        print(f"✓ GAT output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model error: {e}")
        traceback.print_exc()
        return False


def test_predictor():
    """Test the main predictor functionality."""
    print("\n" + "="*60)
    print("TESTING PREDICTOR")
    print("="*60)
    
    try:
        from catalystgnn import CatalystPredictor
        
        # Test different model types
        model_types = ['co2_adsorption', 'catalytic_activity', 'selectivity']
        
        for model_type in model_types:
            predictor = CatalystPredictor(model_type=model_type, verbose=False)
            
            # Test SMILES prediction
            result = predictor.predict_from_smiles('CCO')
            print(f"✓ {model_type}: {result['prediction']:.3f} {result['units']}")
            
            # Test model info
            info = predictor.get_model_info()
            print(f"  Model parameters: {info['num_parameters']:,}")
        
        # Test batch prediction
        predictor = CatalystPredictor(model_type='co2_adsorption', verbose=False)
        smiles_list = ['CCO', 'CC', 'C']
        results = predictor.predict_batch(smiles_list)
        print(f"✓ Batch prediction: {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"✗ Predictor error: {e}")
        traceback.print_exc()
        return False


def test_data_loader():
    """Test data loading functionality."""
    print("\n" + "="*60)
    print("TESTING DATA LOADER")
    print("="*60)
    
    try:
        from catalystgnn.utils.data_loader import DataLoader, create_example_dataset
        from catalystgnn.featurizers import MolecularGraphFeaturizer
        
        # Create example dataset
        dataset_path = "test_dataset.csv"
        create_example_dataset(dataset_path, num_samples=10, dataset_type='molecular')
        
        # Test data loader
        featurizer = MolecularGraphFeaturizer()
        data_loader = DataLoader(featurizer, batch_size=3)
        
        dataset = data_loader.load_dataset(dataset_path)
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Test data splitting
        train_loader, val_loader, test_loader = data_loader.create_data_loaders(dataset)
        print(f"✓ Data split: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")
        
        # Test batch processing
        sample_batch = data_loader.get_sample_batch(dataset, batch_size=3)
        graph_list, targets = sample_batch
        print(f"✓ Sample batch: {len(graph_list)} graphs, targets shape {targets.shape}")
        
        # Clean up
        Path(dataset_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader error: {e}")
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test preprocessing utilities."""
    print("\n" + "="*60)
    print("TESTING PREPROCESSING")
    print("="*60)
    
    try:
        from catalystgnn.utils.preprocessing import (
            normalize_features, handle_missing_values, 
            augment_graph_data, validate_graph_data
        )
        
        # Test feature normalization
        features = torch.randn(10, 5)
        normalized = normalize_features(features, method='standard')
        print(f"✓ Feature normalization: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
        
        # Test missing value handling
        features_with_nan = features.clone()
        features_with_nan[0, 0] = float('nan')
        filled = handle_missing_values(features_with_nan, method='mean')
        print(f"✓ Missing value handling: NaN count before={torch.isnan(features_with_nan).sum()}, after={torch.isnan(filled).sum()}")
        
        # Test graph validation
        valid_graph = {
            'node_features': torch.randn(5, 10),
            'edge_indices': torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
            'edge_features': torch.randn(5, 5)
        }
        
        is_valid, errors = validate_graph_data(valid_graph)
        print(f"✓ Graph validation: valid={is_valid}, errors={len(errors)}")
        
        # Test data augmentation
        augmented = augment_graph_data(valid_graph, augmentation_type='noise')
        print(f"✓ Data augmentation: original shape={valid_graph['node_features'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        traceback.print_exc()
        return False


def test_file_handlers():
    """Test file handling utilities."""
    print("\n" + "="*60)
    print("TESTING FILE HANDLERS")
    print("="*60)
    
    try:
        from catalystgnn.utils.file_handlers import (
            parse_smiles, create_dummy_molecular_structure, 
            create_dummy_crystal_structure, get_supported_formats
        )
        
        # Test SMILES parsing
        molecule = parse_smiles('CCO')
        print(f"✓ SMILES parsing: {type(molecule)}")
        
        # Test dummy structure creation
        mol_struct = create_dummy_molecular_structure()
        crystal_struct = create_dummy_crystal_structure()
        
        print(f"✓ Dummy molecular structure: {mol_struct['num_atoms']} atoms")
        print(f"✓ Dummy crystal structure: {crystal_struct['num_atoms']} atoms")
        
        # Test supported formats
        formats = get_supported_formats()
        print(f"✓ Supported formats: {formats}")
        
        return True
        
    except Exception as e:
        print(f"✗ File handler error: {e}")
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization utilities."""
    print("\n" + "="*60)
    print("TESTING VISUALIZATION")
    print("="*60)
    
    try:
        from catalystgnn.utils.visualization import (
            plot_predictions, plot_property_distribution, 
            plot_training_history
        )
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Test prediction plot
        y_true = np.random.randn(20)
        y_pred = y_true + np.random.randn(20) * 0.1
        
        fig = plot_predictions(y_true, y_pred, title="Test Predictions")
        print(f"✓ Prediction plot created: {type(fig)}")
        
        # Test property distribution plot
        properties = np.random.randn(100)
        fig = plot_property_distribution(properties, property_name="Test Property")
        print(f"✓ Property distribution plot created: {type(fig)}")
        
        # Test training history plot
        history = {
            'train_loss': [1.0, 0.8, 0.6, 0.4, 0.3],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.4],
            'r2_score': [0.5, 0.6, 0.7, 0.8, 0.85]
        }
        fig = plot_training_history(history, title="Test Training")
        print(f"✓ Training history plot created: {type(fig)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization error: {e}")
        traceback.print_exc()
        return False


def test_cli():
    """Test command-line interface."""
    print("\n" + "="*60)
    print("TESTING CLI")
    print("="*60)
    
    try:
        from catalystgnn.cli import create_parser
        
        # Test parser creation
        parser = create_parser()
        print(f"✓ CLI parser created: {type(parser)}")
        
        # Test argument parsing
        test_args = ['predict', '--input', 'CCO', '--property', 'co2_adsorption']
        args = parser.parse_args(test_args)
        
        print(f"✓ CLI argument parsing: command={args.command}, input={args.input}")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI error: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test end-to-end integration."""
    print("\n" + "="*60)
    print("TESTING INTEGRATION")
    print("="*60)
    
    try:
        from catalystgnn import CatalystPredictor
        from catalystgnn.utils.data_loader import create_example_dataset, DataLoader
        from catalystgnn.featurizers import MolecularGraphFeaturizer
        
        # Create a small dataset
        dataset_path = "integration_test.csv"
        create_example_dataset(dataset_path, num_samples=5, dataset_type='molecular')
        
        # Load and process dataset
        featurizer = MolecularGraphFeaturizer()
        data_loader = DataLoader(featurizer, batch_size=2)
        dataset = data_loader.load_dataset(dataset_path)
        
        # Get predictions for dataset
        predictor = CatalystPredictor(model_type='co2_adsorption', verbose=False)
        
        # Test on first few samples
        sample_smiles = ['CCO', 'CC', 'C']
        results = predictor.predict_batch(sample_smiles)
        
        print(f"✓ End-to-end test: processed {len(results)} samples")
        
        # Verify results structure
        for i, result in enumerate(results):
            if 'prediction' in result:
                print(f"  Sample {i+1}: {result['prediction']:.3f} {result['units']}")
            else:
                print(f"  Sample {i+1}: Error - {result.get('error', 'Unknown')}")
        
        # Clean up
        Path(dataset_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Integration error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("CatalystGNN Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Featurizers", test_featurizers),
        ("Models", test_models),
        ("Predictor", test_predictor),
        ("Data Loader", test_data_loader),
        ("Preprocessing", test_preprocessing),
        ("File Handlers", test_file_handlers),
        ("Visualization", test_visualization),
        ("CLI", test_cli),
        ("Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{test_name:<15}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n All tests passed! CatalystGNN is ready to use.")
        return 0
    else:
        print(f"\n  {total-passed} tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)