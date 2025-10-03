"""
Basic usage examples for CatalystGNN.

This script demonstrates how to use CatalystGNN for predicting
catalyst properties from different input formats.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the parent directory to the path so we can import catalystgnn
sys.path.insert(0, str(Path(__file__).parent.parent))

from catalystgnn import CatalystPredictor
from catalystgnn.utils.data_loader import create_example_dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_single_smiles() -> None:
    """Example 1: Predict from a single SMILES string."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single SMILES Prediction")
    print("="*60)

    # Initialize predictor for CO2 adsorption energy
    predictor = CatalystPredictor(model_type='co2_adsorption')

    # Predict from SMILES string (ethanol)
    smiles = 'CCO'
    result = predictor.predict_from_smiles(smiles, return_uncertainty=True)

    print(f"SMILES: {smiles}")
    print(f"Predicted CO2 adsorption energy: {result['prediction']:.3f} {result['units']}")
    print(f"Uncertainty: ±{result['uncertainty']:.3f}")
    print(f"Model: {result['model_description']}")


def example_2_batch_prediction() -> None:
    """Example 2: Batch prediction from multiple SMILES."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch SMILES Prediction")
    print("="*60)

    # Initialize predictor for catalytic activity
    predictor = CatalystPredictor(model_type='catalytic_activity')

    # List of SMILES strings
    smiles_list: List[str] = [
        'CCO',      # Ethanol
        'CC',       # Ethane
        'C',        # Methane
        'CCCO',     # Propanol
        'CC(C)O'    # Isopropanol
    ]

    # Batch prediction - cast to proper type for type checker
    results: List[Dict[str, Any]] = predictor.predict_batch(smiles_list, return_uncertainty=True)  # type: ignore[arg-type]

    print(f"Processed {len(results)} molecules:")
    for i, (smiles, result) in enumerate(zip(smiles_list, results), 1):
        if 'error' not in result:
            print(f"{i}. {smiles}: {result['prediction']:.3f} ± {result['uncertainty']:.3f} {result['units']}")
        else:
            print(f"{i}. {smiles}: Error - {result['error']}")


def example_3_model_information() -> None:
    """Example 3: Get model information."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Model Information")
    print("="*60)

    # List available models
    available_models = CatalystPredictor.list_available_models()

    print("Available Models:")
    for model_type, info in available_models.items():
        print(f"\n{model_type}:")
        print(f"  Description: {info['description']}")
        print(f"  Units: {info['units']}")
        print(f"  Model Class: {info['model_class']}")

    # Get detailed info for a specific model
    predictor = CatalystPredictor(model_type='co2_adsorption')
    model_info = predictor.get_model_info()

    print(f"\nDetailed info for CO2 adsorption model:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")


def example_4_create_and_use_dataset() -> None:
    """Example 4: Create example dataset and use data loader."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Dataset Creation and Loading")
    print("="*60)

    # Create example dataset
    dataset_path = "example_molecular_dataset.csv"
    create_example_dataset(
        output_path=dataset_path,
        num_samples=20,
        dataset_type='molecular'
    )
    print(f"Created example dataset: {dataset_path}")

    # Load and use dataset
    from catalystgnn.utils.data_loader import DataLoader
    from catalystgnn.featurizers import MolecularGraphFeaturizer

    # Initialize featurizer and data loader
    featurizer = MolecularGraphFeaturizer()
    data_loader = DataLoader(featurizer, batch_size=5)

    # Load dataset
    dataset = data_loader.load_dataset(dataset_path)

    # Get dataset statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Number of samples: {stats['num_samples']}")
    print(f"  Target mean: {stats['target_mean']:.3f}")
    print(f"  Target std: {stats['target_std']:.3f}")
    print(f"  Structure types: {stats['structure_types']}")

    # Get a sample batch with explicit type annotation
    sample_batch = data_loader.get_sample_batch(dataset, batch_size=3)
    graph_data_list: List[Dict[str, Any]]
    graph_data_list, targets = sample_batch  # type: ignore[assignment]

    print(f"\nSample batch:")
    print(f"  Batch size: {len(graph_data_list)}")
    print(f"  Targets: {targets.tolist()}")  # type: ignore[attr-defined]

    # Clean up
    Path(dataset_path).unlink()


def example_5_different_properties() -> None:
    """Example 5: Predict different properties."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Different Property Predictions")
    print("="*60)

    smiles = 'CCO'  # Ethanol

    # Predict different properties
    properties = ['co2_adsorption', 'catalytic_activity', 'selectivity']

    for prop in properties:
        try:
            predictor = CatalystPredictor(model_type=prop)
            result = predictor.predict_from_smiles(smiles)

            print(f"{prop.replace('_', ' ').title()}:")
            print(f"  Prediction: {result['prediction']:.3f} {result['units']}")
            print(f"  Description: {result['model_description']}")
        except Exception as e:
            print(f"{prop}: Error - {e}")


def example_6_visualization() -> None:
    """Example 6: Visualization capabilities."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Visualization")
    print("="*60)

    try:
        from catalystgnn.utils.visualization import plot_predictions, plot_property_distribution
        import numpy as np
        import matplotlib.pyplot as plt

        # Generate some example data
        np.random.seed(42)
        y_true = np.random.normal(0, 1, 50)
        y_pred = y_true + np.random.normal(0, 0.2, 50)  # Add some noise
        uncertainties = np.abs(np.random.normal(0, 0.1, 50))

        # Plot predictions and store figure (but don't need to access it)
        _ = plot_predictions(
            y_true, y_pred, uncertainties,
            title="Example Predictions",
            xlabel="True CO2 Adsorption Energy (kJ/mol)",
            ylabel="Predicted CO2 Adsorption Energy (kJ/mol)"
        )

        # Plot property distribution and store figure (but don't need to access it)
        _ = plot_property_distribution(
            y_true,
            property_name="CO2 Adsorption Energy (kJ/mol)"
        )

        print("Generated visualization plots (close the plot windows to continue)")
        plt.show()  # type: ignore[attr-defined]

    except ImportError as e:
        print(f"Visualization requires additional dependencies: {e}")
    except Exception as e:
        print(f"Visualization error: {e}")


def main() -> None:
    """Run all examples."""
    print("CatalystGNN Basic Usage Examples")
    print("=" * 60)

    try:
        example_1_single_smiles()
        example_2_batch_prediction()
        example_3_model_information()
        example_4_create_and_use_dataset()
        example_5_different_properties()
        example_6_visualization()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == '__main__':
    main()
