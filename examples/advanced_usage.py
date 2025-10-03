"""
Advanced usage examples for CatalystGNN.

This script demonstrates advanced features including custom featurizers,
model interpretability, and integration with other tools.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add the parent directory to the path so we can import catalystgnn
sys.path.insert(0, str(Path(__file__).parent.parent))

from catalystgnn import CatalystPredictor
from catalystgnn.featurizers import CrystalGraphFeaturizer, MolecularGraphFeaturizer
from catalystgnn.models import CGCNNModel, MPNNModel, GATModel
from catalystgnn.utils.preprocessing import normalize_features, preprocess_graph_batch
from catalystgnn.utils.data_loader import DataLoader, create_example_dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_custom_featurization():
    """Example 1: Custom featurization and preprocessing."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Custom Featurization")
    print("="*60)
    
    # Initialize different featurizers
    molecular_featurizer = MolecularGraphFeaturizer(
        use_chirality=True,
        use_formal_charge=True,
        normalize_features=True
    )
    
    crystal_featurizer = CrystalGraphFeaturizer(
        cutoff_radius=8.0,
        max_neighbors=12,
        normalize_features=True
    )
    
    print("Featurizer configurations:")
    print(f"Molecular featurizer: {molecular_featurizer.get_config()}")
    print(f"Crystal featurizer: {crystal_featurizer.get_config()}")
    
    # Get feature dimensions
    mol_dims = molecular_featurizer.get_feature_dimensions()
    crystal_dims = crystal_featurizer.get_feature_dimensions()
    
    print(f"\nFeature dimensions:")
    print(f"Molecular - Nodes: {mol_dims['node_features']}, Edges: {mol_dims['edge_features']}")
    print(f"Crystal - Nodes: {crystal_dims['node_features']}, Edges: {crystal_dims['edge_features']}")
    
    # Test featurization
    from catalystgnn.utils.file_handlers import parse_smiles
    
    try:
        molecule = parse_smiles('CCO')
        graph_data = molecular_featurizer.featurize(molecule)
        
        print(f"\nFeaturized ethanol:")
        print(f"  Nodes: {graph_data['node_features'].shape}")
        print(f"  Edges: {graph_data['edge_features'].shape}")
        print(f"  Edge indices: {graph_data['edge_indices'].shape}")
        
    except Exception as e:
        print(f"Featurization error: {e}")


def example_2_model_architectures():
    """Example 2: Different model architectures."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Model Architectures")
    print("="*60)
    
    # Initialize different models
    models = {
        'CGCNN': CGCNNModel(
            node_feature_dim=92,
            edge_feature_dim=41,
            hidden_dim=128,
            num_layers=4
        ),
        'MPNN': MPNNModel(
            node_feature_dim=133,
            edge_feature_dim=23,
            hidden_dim=128,
            num_layers=3
        ),
        'GAT': GATModel(
            node_feature_dim=92,
            edge_feature_dim=41,
            hidden_dim=128,
            num_layers=3,
            num_heads=8
        )
    }
    
    print("Model Information:")
    for name, model in models.items():
        info = model.get_model_info()
        print(f"\n{name}:")
        print(f"  Parameters: {info['num_parameters']:,}")
        print(f"  Hidden dim: {info['hidden_dim']}")
        print(f"  Layers: {info['num_layers']}")
        
        # Count parameters by category
        param_counts = model.count_parameters()
        print(f"  Parameter breakdown: {param_counts['by_category']}")


def example_3_batch_processing():
    """Example 3: Advanced batch processing and preprocessing."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing")
    print("="*60)
    
    # Create example dataset
    dataset_path = "temp_batch_dataset.csv"
    create_example_dataset(dataset_path, num_samples=10, dataset_type='molecular')
    
    try:
        # Initialize components
        featurizer = MolecularGraphFeaturizer()
        data_loader = DataLoader(featurizer, batch_size=5)
        
        # Load dataset
        dataset = data_loader.load_dataset(dataset_path)
        
        # Create data loaders
        train_loader, val_loader, test_loader = data_loader.create_data_loaders(
            dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        
        print(f"Created data loaders:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Process a batch with preprocessing
        for batch_graphs, batch_targets in train_loader:
            print(f"\nProcessing batch with {len(batch_graphs)} graphs")
            
            # Apply preprocessing
            processed_graphs = preprocess_graph_batch(
                batch_graphs,
                normalize_nodes=True,
                normalize_edges=True,
                handle_missing=True
            )
            
            print(f"  Original batch size: {len(batch_graphs)}")
            print(f"  Processed batch size: {len(processed_graphs)}")
            print(f"  Target shape: {batch_targets.shape}")
            
            # Show feature statistics
            if processed_graphs:
                node_features = processed_graphs[0]['node_features']
                print(f"  Node features shape: {node_features.shape}")
                print(f"  Node features mean: {node_features.mean():.3f}")
                print(f"  Node features std: {node_features.std():.3f}")
            
            break  # Just process first batch
    
    finally:
        # Clean up
        Path(dataset_path).unlink(missing_ok=True)


def example_4_uncertainty_estimation():
    """Example 4: Uncertainty estimation and model interpretability."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Uncertainty Estimation")
    print("="*60)
    
    # Initialize predictor
    predictor = CatalystPredictor(model_type='co2_adsorption')
    
    # Test molecules
    test_molecules = ['CCO', 'CC', 'C', 'CCCO']
    
    print("Predictions with uncertainty:")
    for smiles in test_molecules:
        try:
            result = predictor.predict_from_smiles(
                smiles, 
                return_uncertainty=True,
                return_features=False
            )
            
            pred = result['prediction']
            unc = result.get('uncertainty', 0.0)
            
            print(f"  {smiles}: {pred:.3f} ± {unc:.3f} {result['units']}")
            
        except Exception as e:
            print(f"  {smiles}: Error - {e}")
    
    # Demonstrate attention weights (for GAT model)
    try:
        # Create a GAT model for demonstration
        gat_model = GATModel(node_feature_dim=133, edge_feature_dim=23)
        
        # Create dummy graph data
        dummy_graph = {
            'node_features': torch.randn(5, 133),
            'edge_indices': torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
            'edge_features': torch.randn(5, 23)
        }
        
        # Get attention weights
        attention_weights = gat_model.compute_attention_weights(dummy_graph)
        
        print(f"\nAttention analysis:")
        print(f"  Number of attention layers: {len(attention_weights)}")
        if attention_weights:
            print(f"  Attention shape: {attention_weights[0].shape}")
            print(f"  Max attention: {attention_weights[0].max():.3f}")
            print(f"  Min attention: {attention_weights[0].min():.3f}")
    
    except Exception as e:
        print(f"Attention analysis error: {e}")


def example_5_data_augmentation():
    """Example 5: Data augmentation techniques."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Data Augmentation")
    print("="*60)
    
    from catalystgnn.utils.preprocessing import augment_graph_data
    from catalystgnn.utils.file_handlers import parse_smiles
    
    # Create sample graph data
    featurizer = MolecularGraphFeaturizer()
    
    try:
        molecule = parse_smiles('CCO')
        original_graph = featurizer.featurize(molecule)
        
        print("Original graph:")
        print(f"  Node features shape: {original_graph['node_features'].shape}")
        print(f"  Node features mean: {original_graph['node_features'].mean():.3f}")
        
        # Apply different augmentations
        augmentation_types = ['noise', 'dropout', 'permute']
        
        for aug_type in augmentation_types:
            try:
                augmented_graph = augment_graph_data(
                    original_graph,
                    augmentation_type=aug_type,
                    noise_std=0.1,
                    dropout_prob=0.1
                )
                
                print(f"\n{aug_type.title()} augmentation:")
                print(f"  Node features shape: {augmented_graph['node_features'].shape}")
                print(f"  Node features mean: {augmented_graph['node_features'].mean():.3f}")
                
                # Check if augmentation changed the data
                if not torch.equal(original_graph['node_features'], augmented_graph['node_features']):
                    print(f"  ✓ Features modified by {aug_type} augmentation")
                else:
                    print(f"  - Features unchanged by {aug_type} augmentation")
                    
            except Exception as e:
                print(f"  Error with {aug_type} augmentation: {e}")
    
    except Exception as e:
        print(f"Data augmentation example error: {e}")


def example_6_model_comparison():
    """Example 6: Compare different models on the same data."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Model Comparison")
    print("="*60)
    
    # Test molecules
    test_smiles = ['CCO', 'CC', 'C', 'CCCO', 'CC(C)O']
    
    # Available model types
    model_types = ['co2_adsorption', 'catalytic_activity']
    
    print("Model comparison on test molecules:")
    print(f"{'SMILES':<10} {'Model':<20} {'Prediction':<12} {'Units'}")
    print("-" * 60)
    
    for smiles in test_smiles:
        for model_type in model_types:
            try:
                predictor = CatalystPredictor(model_type=model_type, verbose=False)
                result = predictor.predict_from_smiles(smiles)
                
                pred = result['prediction']
                units = result['units']
                
                print(f"{smiles:<10} {model_type:<20} {pred:<12.3f} {units}")
                
            except Exception as e:
                print(f"{smiles:<10} {model_type:<20} Error: {str(e)[:20]}")


def example_7_feature_analysis():
    """Example 7: Feature importance and analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Feature Analysis")
    print("="*60)
    
    from catalystgnn.utils.preprocessing import compute_graph_statistics
    from catalystgnn.utils.file_handlers import parse_smiles
    
    # Create multiple graph samples
    smiles_list = ['CCO', 'CC', 'C', 'CCCO', 'CC(C)O', 'CCC', 'CCCCO']
    featurizer = MolecularGraphFeaturizer()
    
    graph_data_list = []
    
    for smiles in smiles_list:
        try:
            molecule = parse_smiles(smiles)
            graph_data = featurizer.featurize(molecule)
            graph_data_list.append(graph_data)
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
    
    if graph_data_list:
        # Compute statistics
        stats = compute_graph_statistics(graph_data_list)
        
        print("Dataset statistics:")
        print(f"  Number of graphs: {stats['num_graphs']}")
        
        if 'num_nodes' in stats:
            print(f"  Nodes per graph: {stats['num_nodes']['mean']:.1f} ± {stats['num_nodes']['std']:.1f}")
            print(f"  Node range: {stats['num_nodes']['min']} - {stats['num_nodes']['max']}")
        
        if 'num_edges' in stats:
            print(f"  Edges per graph: {stats['num_edges']['mean']:.1f} ± {stats['num_edges']['std']:.1f}")
        
        if 'node_degrees' in stats:
            print(f"  Average node degree: {stats['node_degrees']['mean']:.2f}")
        
        if 'node_features' in stats:
            node_stats = stats['node_features']
            print(f"  Node feature dim: {node_stats['feature_dim']}")
            print(f"  Node feature range: [{min(node_stats['min']):.3f}, {max(node_stats['max']):.3f}]")


def main():
    """Run all advanced examples."""
    print("CatalystGNN Advanced Usage Examples")
    print("=" * 60)
    
    try:
        example_1_custom_featurization()
        example_2_model_architectures()
        example_3_batch_processing()
        example_4_uncertainty_estimation()
        example_5_data_augmentation()
        example_6_model_comparison()
        example_7_feature_analysis()
        
        print("\n" + "="*60)
        print("All advanced examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error running advanced examples: {e}")
        raise


if __name__ == '__main__':
    main()