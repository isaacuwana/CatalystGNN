"""
Command-line interface for CatalystGNN.

This module provides a command-line interface for using CatalystGNN
to predict catalyst properties from structure files or SMILES strings.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .core.predictor import CatalystPredictor
from .utils.data_loader import create_example_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Execute command
    try:
        if args.command == 'predict':
            predict_command(args)
        elif args.command == 'list-models':
            list_models_command(args)
        elif args.command == 'create-dataset':
            create_dataset_command(args)
        elif args.command == 'visualize':
            visualize_command(args)
        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"Error executing command: {e}")
        if args.verbose:
            raise
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description='CatalystGNN: Graph Neural Networks for Catalyst Property Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict CO2 adsorption energy from a CIF file
  catalystgnn predict --input structure.cif --property co2_adsorption --output results.json

  # Predict catalytic activity from SMILES strings
  catalystgnn predict --input molecules.smi --property catalytic_activity --batch-size 32

  # List available models
  catalystgnn list-models

  # Create example dataset
  catalystgnn create-dataset --output example_data.csv --num-samples 100 --type molecular

  # Visualize structure
  catalystgnn visualize --input structure.cif --output structure.png
        """
    )

    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict catalyst properties')
    predict_parser.add_argument('--input', '-i', required=True, 
                               help='Input file (structure file, SMILES file, or single SMILES string)')
    predict_parser.add_argument('--property', '-p', default='co2_adsorption',
                               choices=['co2_adsorption', 'catalytic_activity', 'selectivity'],
                               help='Property to predict')
    predict_parser.add_argument('--output', '-o', help='Output file for results (JSON format)')
    predict_parser.add_argument('--batch-size', '-b', type=int, default=32, 
                               help='Batch size for processing multiple inputs')
    predict_parser.add_argument('--uncertainty', action='store_true', 
                               help='Include uncertainty estimates')
    predict_parser.add_argument('--device', choices=['cpu', 'cuda'], 
                               help='Device to use for inference')
    predict_parser.add_argument('--model-path', help='Path to custom model files')

    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available pre-trained models')
    list_parser.add_argument('--detailed', action='store_true', help='Show detailed model information')

    # Create dataset command
    dataset_parser = subparsers.add_parser('create-dataset', help='Create example dataset')
    dataset_parser.add_argument('--output', '-o', required=True, help='Output file path')
    dataset_parser.add_argument('--num-samples', '-n', type=int, default=100, 
                               help='Number of samples to generate')
    dataset_parser.add_argument('--type', choices=['molecular', 'crystal'], default='molecular',
                               help='Type of dataset to create')

    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize chemical structures')
    viz_parser.add_argument('--input', '-i', required=True, help='Input structure file or SMILES')
    viz_parser.add_argument('--output', '-o', help='Output image file')
    viz_parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                           help='Output image format')
    viz_parser.add_argument('--size', nargs=2, type=int, default=[10, 8], 
                           help='Figure size (width height)')

    return parser


def predict_command(args: argparse.Namespace) -> None:
    """Execute the predict command."""
    logger.info(f"Predicting {args.property} from {args.input}")

    # Initialize predictor
    predictor = CatalystPredictor(
        model_type=str(args.property),
        device=str(args.device) if args.device else None,
        model_path=str(args.model_path) if args.model_path else None
    )

    # Determine input type and process
    input_path = Path(args.input)

    if input_path.exists() and input_path.is_file():
        # Input is a file
        if input_path.suffix in ['.smi', '.smiles', '.txt']:
            # SMILES file
            results = process_smiles_file(predictor, input_path, args)
        else:
            # Structure file
            results = process_structure_file(predictor, input_path, args)
    else:
        # Assume single SMILES string
        results = process_single_smiles(predictor, str(args.input), args)

    # Output results
    if args.output:
        save_results(results, str(args.output))
        logger.info(f"Results saved to {args.output}")
    else:
        print_results(results)


def process_structure_file(predictor: CatalystPredictor, file_path: Path, 
                          args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Process a single structure file."""
    logger.info(f"Processing structure file: {file_path}")

    try:
        result = predictor.predict_from_file(
            str(file_path),
            return_uncertainty=bool(args.uncertainty)
        )
        return [result]

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return [{'error': str(e), 'input': str(file_path)}]


def process_smiles_file(predictor: CatalystPredictor, file_path: Path, 
                       args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Process a file containing SMILES strings."""
    logger.info(f"Processing SMILES file: {file_path}")

    try:
        # Read SMILES strings
        with open(file_path, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]

        logger.info(f"Found {len(smiles_list)} SMILES strings")

        # Process in batches
        results = predictor.predict_batch(
            smiles_list,  # type: ignore[arg-type]
            batch_size=int(args.batch_size),
            return_uncertainty=bool(args.uncertainty)
        )

        return results

    except Exception as e:
        logger.error(f"Error processing SMILES file {file_path}: {e}")
        return [{'error': str(e), 'input': str(file_path)}]


def process_single_smiles(predictor: CatalystPredictor, smiles: str, 
                         args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Process a single SMILES string."""
    logger.info(f"Processing SMILES: {smiles}")

    try:
        result = predictor.predict_from_smiles(
            smiles,
            return_uncertainty=bool(args.uncertainty)
        )
        return [result]

    except Exception as e:
        logger.error(f"Error processing SMILES {smiles}: {e}")
        return [{'error': str(e), 'input': smiles}]


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def print_results(results: List[Dict[str, Any]]) -> None:
    """Print results to console."""
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print("-" * 20)

        if 'error' in result:
            print(f"Error: {result['error']}")
            print(f"Input: {result['input']}")
        else:
            input_display = result.get('input_file', result.get('input_smiles', 'Unknown'))
            print(f"Input: {input_display}")
            print(f"Property: {result['property']}")
            print(f"Prediction: {result['prediction']:.4f} {result['units']}")

            if 'uncertainty' in result:
                print(f"Uncertainty: ±{result['uncertainty']:.4f}")

            if 'structure_formula' in result:
                print(f"Formula: {result['structure_formula']}")
            elif 'molecular_formula' in result:
                print(f"Formula: {result['molecular_formula']}")


def list_models_command(args: argparse.Namespace) -> None:
    """Execute the list-models command."""
    logger.info("Listing available models")

    models = CatalystPredictor.list_available_models()

    print("\n" + "="*60)
    print("AVAILABLE MODELS")
    print("="*60)

    for model_type, info in models.items():
        print(f"\nModel: {model_type}")
        print("-" * 30)
        print(f"Description: {info['description']}")
        print(f"Units: {info['units']}")

        if args.detailed:
            print(f"Model Class: {info.get('model_class', 'N/A')}")
            print(f"Featurizer: {info.get('featurizer_class', 'N/A')}")


def create_dataset_command(args: argparse.Namespace) -> None:
    """Execute the create-dataset command."""
    logger.info(f"Creating {args.type} dataset with {args.num_samples} samples")

    output_path = create_example_dataset(
        output_path=str(args.output),
        num_samples=int(args.num_samples),
        dataset_type=str(args.type)
    )

    print(f"\nExample dataset created: {output_path}")
    print(f"Type: {args.type}")
    print(f"Samples: {args.num_samples}")


def visualize_command(args: argparse.Namespace) -> None:
    """Execute the visualize command."""
    logger.info(f"Visualizing structure from {args.input}")

    try:
        from .utils.file_handlers import load_structure_from_file, parse_smiles
        from .utils.visualization import plot_structure

        # Load structure
        input_path = Path(args.input)
        if input_path.exists():
            structure = load_structure_from_file(str(input_path))
            title = f"Structure: {input_path.name}"
        else:
            # Assume SMILES string
            structure = parse_smiles(str(args.input))
            title = f"Molecule: {args.input}"

        # Create plot
        size_tuple = tuple(int(s) for s in args.size)
        fig = plot_structure(
            structure,
            title=title,
            figsize=size_tuple  # type: ignore[arg-type]
        )

        # Save or show
        if args.output:
            output_path = Path(args.output)
            if not output_path.suffix:
                output_path = output_path.with_suffix(f'.{args.format}')

            fig.savefig(str(output_path), format=str(args.format), dpi=300, bbox_inches='tight')
            logger.info(f"Structure visualization saved to {output_path}")
        else:
            import matplotlib.pyplot as plt
            plt.show()

    except Exception as e:
        logger.error(f"Error visualizing structure: {e}")
        raise


if __name__ == '__main__':
    main()
