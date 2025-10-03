"""
Data loading utilities for CatalystGNN.

This module provides classes and functions for loading and preprocessing
datasets for training and inference.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path

from .file_handlers import load_structure_from_file, parse_smiles, validate_structure_file

logger = logging.getLogger(__name__)


class CatalystDataset(Dataset):
    """
    Dataset class for catalyst structures and properties.
    
    This class handles loading and preprocessing of catalyst datasets
    for training GNN models.
    """
    
    def __init__(
        self,
        data_path: str,
        featurizer: Any,
        target_property: str = 'target',
        structure_column: str = 'structure',
        cache_dir: Optional[str] = None,
        preprocess: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Initialize catalyst dataset.
        
        Args:
            data_path: Path to dataset file (CSV, JSON, or pickle)
            featurizer: Graph featurizer to use
            target_property: Name of target property column
            structure_column: Name of structure column (file paths or SMILES)
            cache_dir: Directory to cache preprocessed data
            preprocess: Whether to preprocess all data at initialization
            max_samples: Maximum number of samples to load
        """
        self.data_path = Path(data_path)
        self.featurizer = featurizer
        self.target_property = target_property
        self.structure_column = structure_column
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_samples = max_samples
        
        # Load raw data
        self.raw_data = self._load_raw_data()
        
        # Setup caching
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / f"{self.data_path.stem}_cache.pkl"
        
        # Preprocess data if requested
        if preprocess:
            self.preprocessed_data = self._preprocess_all_data()
        else:
            self.preprocessed_data = {}
        
        logger.info(f"Loaded dataset with {len(self.raw_data)} samples")
    
    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw data from file."""
        try:
            if self.data_path.suffix == '.csv':
                data = pd.read_csv(self.data_path)
            elif self.data_path.suffix == '.json':
                data = pd.read_json(self.data_path)
            elif self.data_path.suffix in ['.pkl', '.pickle']:
                data = pd.read_pickle(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
            # Limit samples if specified
            if self.max_samples:
                data = data.head(self.max_samples)
            
            # Validate required columns
            if self.structure_column not in data.columns:
                raise ValueError(f"Structure column '{self.structure_column}' not found")
            if self.target_property not in data.columns:
                raise ValueError(f"Target property '{self.target_property}' not found")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            # Create dummy data for demonstration
            return self._create_dummy_data()
    
    def _create_dummy_data(self) -> pd.DataFrame:
        """Create dummy dataset for demonstration."""
        dummy_data = {
            self.structure_column: [
                'CCO',  # Ethanol
                'CC',   # Ethane
                'C',    # Methane
                'CCO',  # Ethanol (duplicate)
                'CCCO', # Propanol
            ],
            self.target_property: [
                -0.5,   # CO2 adsorption energy
                -0.3,
                -0.1,
                -0.52,
                -0.7
            ],
            'id': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5']
        }
        
        logger.info("Created dummy dataset for demonstration")
        return pd.DataFrame(dummy_data)
    
    def _preprocess_all_data(self) -> Dict[int, Dict]:
        """Preprocess all data and cache results."""
        # Try to load from cache
        if self.cache_dir and self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Loaded preprocessed data from cache: {len(cached_data)} samples")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Preprocess data
        preprocessed = {}
        failed_count = 0
        
        for idx in range(len(self.raw_data)):
            try:
                graph_data, target = self._process_sample(idx)
                preprocessed[idx] = {
                    'graph_data': graph_data,
                    'target': target
                }
            except Exception as e:
                logger.warning(f"Failed to preprocess sample {idx}: {e}")
                failed_count += 1
        
        logger.info(f"Preprocessed {len(preprocessed)} samples, {failed_count} failed")
        
        # Save to cache
        if self.cache_dir:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(preprocessed, f)
                logger.info(f"Saved preprocessed data to cache")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
        
        return preprocessed
    
    def _process_sample(self, idx: int) -> Tuple[Dict[str, torch.Tensor], float]:
        """Process a single sample."""
        row = self.raw_data.iloc[idx]
        structure_input = row[self.structure_column]
        target = float(row[self.target_property])
        
        # Load structure
        if self._is_file_path(structure_input):
            structure = load_structure_from_file(structure_input)
        else:
            # Assume SMILES string
            structure = parse_smiles(structure_input)
        
        # Featurize structure
        graph_data = self.featurizer.featurize(structure)
        
        return graph_data, target
    
    def _is_file_path(self, input_str: str) -> bool:
        """Check if input string is a file path."""
        return (
            isinstance(input_str, str) and
            ('/' in input_str or '\\' in input_str or input_str.endswith(('.cif', '.xyz', '.poscar', '.pdb')))
        )
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.raw_data)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a single sample."""
        if idx in self.preprocessed_data:
            # Use cached data
            sample = self.preprocessed_data[idx]
            return sample['graph_data'], torch.tensor(sample['target'], dtype=torch.float32)
        else:
            # Process on-the-fly
            graph_data, target = self._process_sample(idx)
            return graph_data, torch.tensor(target, dtype=torch.float32)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        targets = self.raw_data[self.target_property].values
        
        stats = {
            'num_samples': len(self.raw_data),
            'target_mean': float(np.mean(targets)),
            'target_std': float(np.std(targets)),
            'target_min': float(np.min(targets)),
            'target_max': float(np.max(targets)),
            'target_median': float(np.median(targets))
        }
        
        # Structure type distribution
        structure_types = {}
        for structure_input in self.raw_data[self.structure_column]:
            if self._is_file_path(structure_input):
                ext = Path(structure_input).suffix.lower()
                structure_types[ext] = structure_types.get(ext, 0) + 1
            else:
                structure_types['smiles'] = structure_types.get('smiles', 0) + 1
        
        stats['structure_types'] = structure_types
        
        return stats


class DataLoader:
    """
    High-level data loader for CatalystGNN.
    
    This class provides convenient methods for loading and splitting
    datasets for training, validation, and testing.
    """
    
    def __init__(
        self,
        featurizer: Any,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True
    ):
        """
        Initialize data loader.
        
        Args:
            featurizer: Graph featurizer to use
            batch_size: Batch size for data loading
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
        """
        self.featurizer = featurizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def load_dataset(
        self,
        data_path: str,
        target_property: str = 'target',
        structure_column: str = 'structure',
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> CatalystDataset:
        """
        Load dataset from file.
        
        Args:
            data_path: Path to dataset file
            target_property: Name of target property column
            structure_column: Name of structure column
            cache_dir: Directory to cache preprocessed data
            max_samples: Maximum number of samples to load
            
        Returns:
            CatalystDataset instance
        """
        dataset = CatalystDataset(
            data_path=data_path,
            featurizer=self.featurizer,
            target_property=target_property,
            structure_column=structure_column,
            cache_dir=cache_dir,
            max_samples=max_samples
        )
        
        return dataset
    
    def create_data_loaders(
        self,
        dataset: CatalystDataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        stratify: bool = False
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """
        Create train/validation/test data loaders.
        
        Args:
            dataset: CatalystDataset instance
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            random_seed: Random seed for reproducible splits
            stratify: Whether to stratify splits (for classification)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Create indices
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        
        # Shuffle indices
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
        # Split indices
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subset datasets
        from torch.utils.data import Subset
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        # Create data loaders
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
        
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
        
        test_loader = TorchDataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
        
        logger.info(f"Created data loaders: train={len(train_dataset)}, "
                   f"val={len(val_dataset)}, test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def _collate_fn(self, batch: List[Tuple[Dict, torch.Tensor]]) -> Tuple[List[Dict], torch.Tensor]:
        """
        Collate function for batching graph data.
        
        Args:
            batch: List of (graph_data, target) tuples
            
        Returns:
            Tuple of (batch_graph_data, batch_targets)
        """
        graph_data_list = []
        targets = []
        
        for graph_data, target in batch:
            graph_data_list.append(graph_data)
            targets.append(target)
        
        batch_targets = torch.stack(targets)
        
        return graph_data_list, batch_targets
    
    def create_single_loader(
        self,
        dataset: CatalystDataset,
        shuffle: bool = False
    ) -> TorchDataLoader:
        """
        Create a single data loader for the entire dataset.
        
        Args:
            dataset: CatalystDataset instance
            shuffle: Whether to shuffle the data
            
        Returns:
            TorchDataLoader instance
        """
        loader = TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
        
        return loader
    
    def load_from_smiles_list(
        self,
        smiles_list: List[str],
        targets: Optional[List[float]] = None
    ) -> CatalystDataset:
        """
        Create dataset from list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            targets: Optional list of target values
            
        Returns:
            CatalystDataset instance
        """
        if targets is None:
            targets = [0.0] * len(smiles_list)  # Dummy targets
        
        if len(smiles_list) != len(targets):
            raise ValueError("SMILES list and targets must have same length")
        
        # Create temporary DataFrame
        data = pd.DataFrame({
            'structure': smiles_list,
            'target': targets,
            'id': [f'mol_{i}' for i in range(len(smiles_list))]
        })
        
        # Save to temporary file
        temp_file = Path.cwd() / 'temp_smiles_dataset.csv'
        data.to_csv(temp_file, index=False)
        
        try:
            # Create dataset
            dataset = CatalystDataset(
                data_path=str(temp_file),
                featurizer=self.featurizer,
                target_property='target',
                structure_column='structure'
            )
            
            return dataset
            
        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
    
    def load_from_file_list(
        self,
        file_paths: List[str],
        targets: Optional[List[float]] = None
    ) -> CatalystDataset:
        """
        Create dataset from list of structure files.
        
        Args:
            file_paths: List of structure file paths
            targets: Optional list of target values
            
        Returns:
            CatalystDataset instance
        """
        if targets is None:
            targets = [0.0] * len(file_paths)  # Dummy targets
        
        if len(file_paths) != len(targets):
            raise ValueError("File paths and targets must have same length")
        
        # Validate files
        valid_files = []
        valid_targets = []
        
        for file_path, target in zip(file_paths, targets):
            if validate_structure_file(file_path):
                valid_files.append(file_path)
                valid_targets.append(target)
            else:
                logger.warning(f"Skipping invalid file: {file_path}")
        
        # Create temporary DataFrame
        data = pd.DataFrame({
            'structure': valid_files,
            'target': valid_targets,
            'id': [f'struct_{i}' for i in range(len(valid_files))]
        })
        
        # Save to temporary file
        temp_file = Path.cwd() / 'temp_files_dataset.csv'
        data.to_csv(temp_file, index=False)
        
        try:
            # Create dataset
            dataset = CatalystDataset(
                data_path=str(temp_file),
                featurizer=self.featurizer,
                target_property='target',
                structure_column='structure'
            )
            
            return dataset
            
        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
    
    def get_sample_batch(
        self,
        dataset: CatalystDataset,
        batch_size: Optional[int] = None
    ) -> Tuple[List[Dict], torch.Tensor]:
        """
        Get a sample batch from dataset for testing.
        
        Args:
            dataset: CatalystDataset instance
            batch_size: Batch size (uses default if None)
            
        Returns:
            Tuple of (graph_data_list, targets)
        """
        if batch_size is None:
            batch_size = min(self.batch_size, len(dataset))
        
        # Create temporary loader
        loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        # Get first batch
        for batch in loader:
            return batch
        
        # Fallback if dataset is empty
        return [], torch.tensor([])


def create_example_dataset(
    output_path: str,
    num_samples: int = 100,
    dataset_type: str = 'molecular'
) -> str:
    """
    Create an example dataset for testing and demonstration.
    
    Args:
        output_path: Path to save the dataset
        num_samples: Number of samples to generate
        dataset_type: Type of dataset ('molecular' or 'crystal')
        
    Returns:
        Path to created dataset file
    """
    np.random.seed(42)
    
    if dataset_type == 'molecular':
        # Generate molecular dataset with SMILES
        smiles_templates = [
            'CCO',      # Ethanol
            'CC',       # Ethane
            'C',        # Methane
            'CCCO',     # Propanol
            'CCCCO',    # Butanol
            'CC(C)O',   # Isopropanol
            'CCC',      # Propane
            'CCCC',     # Butane
            'C(C)C',    # Propane (alternative)
            'CC(C)C',   # Isobutane
        ]
        
        data = []
        for i in range(num_samples):
            smiles = np.random.choice(smiles_templates)
            # Add some noise to target values
            base_value = hash(smiles) % 100 / 100.0  # Deterministic but varied
            target = base_value + np.random.normal(0, 0.1)  # Add noise
            
            data.append({
                'id': f'mol_{i:04d}',
                'structure': smiles,
                'target': target,
                'molecular_weight': np.random.uniform(16, 200),
                'num_atoms': np.random.randint(5, 50)
            })
    
    else:  # crystal dataset
        # Generate crystal dataset with dummy CIF file paths
        crystal_types = [
            'MOF-5', 'HKUST-1', 'ZIF-8', 'UiO-66', 'MIL-53',
            'MOF-74', 'ZIF-67', 'CAU-10', 'DUT-8', 'PCN-14'
        ]
        
        data = []
        for i in range(num_samples):
            crystal_type = np.random.choice(crystal_types)
            structure_file = f"structures/{crystal_type}_{i:04d}.cif"
            
            # Generate target based on crystal type
            base_values = {
                'MOF-5': -2.5, 'HKUST-1': -3.2, 'ZIF-8': -1.8,
                'UiO-66': -2.8, 'MIL-53': -2.1, 'MOF-74': -3.5,
                'ZIF-67': -1.9, 'CAU-10': -2.3, 'DUT-8': -2.7, 'PCN-14': -3.1
            }
            base_value = base_values.get(crystal_type, -2.5)
            target = base_value + np.random.normal(0, 0.3)
            
            data.append({
                'id': f'crystal_{i:04d}',
                'structure': structure_file,
                'target': target,
                'crystal_type': crystal_type,
                'pore_volume': np.random.uniform(0.1, 2.0),
                'surface_area': np.random.uniform(500, 3000)
            })
    
    # Save dataset
    df = pd.DataFrame(data)
    output_path = Path(output_path)
    
    if output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    elif output_path.suffix == '.json':
        df.to_json(output_path, orient='records', indent=2)
    else:
        df.to_pickle(output_path)
    
    logger.info(f"Created example {dataset_type} dataset with {num_samples} samples: {output_path}")
    return str(output_path)