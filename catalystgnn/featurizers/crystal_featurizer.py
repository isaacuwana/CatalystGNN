"""
Crystal graph featurizer for periodic structures.

This module implements graph featurization for crystalline materials,
particularly useful for MOFs, zeolites, and other porous materials.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
import logging

from .base_featurizer import BaseFeaturizer

logger = logging.getLogger(__name__)


class CrystalGraphFeaturizer(BaseFeaturizer):
    """
    Graph featurizer for crystalline structures.
    
    This featurizer converts crystal structures into graph representations
    suitable for Crystal Graph Convolutional Networks (CGCNN) and similar
    architectures. It handles periodic boundary conditions and creates
    edges based on atomic distances.
    
    Attributes:
        cutoff_radius: Maximum distance for edge creation
        max_neighbors: Maximum number of neighbors per atom
        use_voronoi: Whether to use Voronoi tessellation for neighbor finding
        include_images: Whether to include periodic images
    """
    
    # Atomic properties for featurization
    ATOMIC_PROPERTIES = {
        'H': {'atomic_number': 1, 'period': 1, 'group': 1, 'electronegativity': 2.20, 'atomic_radius': 0.31, 'covalent_radius': 0.31},
        'He': {'atomic_number': 2, 'period': 1, 'group': 18, 'electronegativity': 0.0, 'atomic_radius': 0.28, 'covalent_radius': 0.28},
        'Li': {'atomic_number': 3, 'period': 2, 'group': 1, 'electronegativity': 0.98, 'atomic_radius': 1.28, 'covalent_radius': 1.28},
        'Be': {'atomic_number': 4, 'period': 2, 'group': 2, 'electronegativity': 1.57, 'atomic_radius': 0.96, 'covalent_radius': 0.96},
        'B': {'atomic_number': 5, 'period': 2, 'group': 13, 'electronegativity': 2.04, 'atomic_radius': 0.84, 'covalent_radius': 0.84},
        'C': {'atomic_number': 6, 'period': 2, 'group': 14, 'electronegativity': 2.55, 'atomic_radius': 0.76, 'covalent_radius': 0.76},
        'N': {'atomic_number': 7, 'period': 2, 'group': 15, 'electronegativity': 3.04, 'atomic_radius': 0.71, 'covalent_radius': 0.71},
        'O': {'atomic_number': 8, 'period': 2, 'group': 16, 'electronegativity': 3.44, 'atomic_radius': 0.66, 'covalent_radius': 0.66},
        'F': {'atomic_number': 9, 'period': 2, 'group': 17, 'electronegativity': 3.98, 'atomic_radius': 0.57, 'covalent_radius': 0.57},
        'Ne': {'atomic_number': 10, 'period': 2, 'group': 18, 'electronegativity': 0.0, 'atomic_radius': 0.58, 'covalent_radius': 0.58},
        'Na': {'atomic_number': 11, 'period': 3, 'group': 1, 'electronegativity': 0.93, 'atomic_radius': 1.66, 'covalent_radius': 1.66},
        'Mg': {'atomic_number': 12, 'period': 3, 'group': 2, 'electronegativity': 1.31, 'atomic_radius': 1.41, 'covalent_radius': 1.41},
        'Al': {'atomic_number': 13, 'period': 3, 'group': 13, 'electronegativity': 1.61, 'atomic_radius': 1.21, 'covalent_radius': 1.21},
        'Si': {'atomic_number': 14, 'period': 3, 'group': 14, 'electronegativity': 1.90, 'atomic_radius': 1.11, 'covalent_radius': 1.11},
        'P': {'atomic_number': 15, 'period': 3, 'group': 15, 'electronegativity': 2.19, 'atomic_radius': 1.07, 'covalent_radius': 1.07},
        'S': {'atomic_number': 16, 'period': 3, 'group': 16, 'electronegativity': 2.58, 'atomic_radius': 1.05, 'covalent_radius': 1.05},
        'Cl': {'atomic_number': 17, 'period': 3, 'group': 17, 'electronegativity': 3.16, 'atomic_radius': 1.02, 'covalent_radius': 1.02},
        'Ar': {'atomic_number': 18, 'period': 3, 'group': 18, 'electronegativity': 0.0, 'atomic_radius': 1.06, 'covalent_radius': 1.06},
        'K': {'atomic_number': 19, 'period': 4, 'group': 1, 'electronegativity': 0.82, 'atomic_radius': 2.03, 'covalent_radius': 2.03},
        'Ca': {'atomic_number': 20, 'period': 4, 'group': 2, 'electronegativity': 1.00, 'atomic_radius': 1.76, 'covalent_radius': 1.76},
        'Zn': {'atomic_number': 30, 'period': 4, 'group': 12, 'electronegativity': 1.65, 'atomic_radius': 1.34, 'covalent_radius': 1.34},
        'Zr': {'atomic_number': 40, 'period': 5, 'group': 4, 'electronegativity': 1.33, 'atomic_radius': 1.60, 'covalent_radius': 1.60},
        'Cu': {'atomic_number': 29, 'period': 4, 'group': 11, 'electronegativity': 1.90, 'atomic_radius': 1.28, 'covalent_radius': 1.28},
    }
    
    def __init__(
        self,
        cutoff_radius: float = 8.0,
        max_neighbors: int = 12,
        use_voronoi: bool = False,
        include_images: bool = True,
        normalize_features: bool = True,
        **kwargs
    ):
        """
        Initialize the crystal graph featurizer.
        
        Args:
            cutoff_radius: Maximum distance for creating edges (Angstroms)
            max_neighbors: Maximum number of neighbors per atom
            use_voronoi: Whether to use Voronoi tessellation for neighbors
            include_images: Whether to include periodic images
            normalize_features: Whether to normalize node features
        """
        self.cutoff_radius = cutoff_radius
        self.max_neighbors = max_neighbors
        self.use_voronoi = use_voronoi
        self.include_images = include_images
        self.normalize_features = normalize_features
        
        super().__init__(**kwargs)
    
    def _setup_featurizer(self):
        """Setup crystal-specific featurizer configurations."""
        # Define feature dimensions
        self.node_feature_dim = 92  # Comprehensive atomic features
        self.edge_feature_dim = 41  # Distance and angle features
        self.global_feature_dim = 10  # Crystal system, space group, etc.
        
        logger.info(f"Crystal featurizer initialized with cutoff={self.cutoff_radius}Å")
    
    def featurize(self, structure: Any) -> Dict[str, torch.Tensor]:
        """
        Convert crystal structure to graph representation.
        
        Args:
            structure: Crystal structure object (pymatgen Structure or similar)
            
        Returns:
            Dictionary containing graph tensors
        """
        try:
            # Extract atomic information
            atomic_numbers = self._get_atomic_numbers(structure)
            positions = self._get_positions(structure)
            lattice = self._get_lattice(structure)
            
            # Create node features
            node_features = self._create_node_features(atomic_numbers, structure)
            
            # Find neighbors and create edges
            edge_indices, edge_features = self._create_edges(
                positions, lattice, atomic_numbers
            )
            
            # Create global features
            global_features = self._create_global_features(structure)
            
            # Convert to tensors
            graph_data = {
                'node_features': torch.tensor(node_features, dtype=torch.float32),
                'edge_indices': torch.tensor(edge_indices, dtype=torch.long),
                'edge_features': torch.tensor(edge_features, dtype=torch.float32),
                'global_features': torch.tensor(global_features, dtype=torch.float32),
                'num_nodes': torch.tensor([len(atomic_numbers)], dtype=torch.long),
                'num_edges': torch.tensor([len(edge_indices[0])], dtype=torch.long)
            }
            
            # Validate graph data
            if not self._validate_graph_data(graph_data):
                raise ValueError("Generated graph data is invalid")
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error featurizing crystal structure: {e}")
            raise
    
    def _get_atomic_numbers(self, structure: Any) -> List[int]:
        """Extract atomic numbers from structure."""
        if hasattr(structure, 'atomic_numbers'):
            return structure.atomic_numbers.tolist()
        elif hasattr(structure, 'species'):
            return [specie.Z for specie in structure.species]
        elif hasattr(structure, 'atoms'):
            return [atom.number for atom in structure.atoms]
        else:
            # Fallback: try to parse from string representation
            return self._parse_atomic_numbers_fallback(structure)
    
    def _get_positions(self, structure: Any) -> np.ndarray:
        """Extract atomic positions from structure."""
        if hasattr(structure, 'cart_coords'):
            return structure.cart_coords
        elif hasattr(structure, 'positions'):
            return structure.positions
        elif hasattr(structure, 'coords'):
            return structure.coords
        else:
            # Create dummy positions for demonstration
            num_atoms = len(self._get_atomic_numbers(structure))
            return np.random.rand(num_atoms, 3) * 10
    
    def _get_lattice(self, structure: Any) -> np.ndarray:
        """Extract lattice parameters from structure."""
        if hasattr(structure, 'lattice') and hasattr(structure.lattice, 'matrix'):
            return structure.lattice.matrix
        elif hasattr(structure, 'cell'):
            return structure.cell
        else:
            # Default cubic lattice for demonstration
            return np.eye(3) * 10.0
    
    def _parse_atomic_numbers_fallback(self, structure: Any) -> List[int]:
        """Fallback method to parse atomic numbers."""
        # This is a simplified fallback - in practice, you'd need more robust parsing
        logger.warning("Using fallback atomic number parsing")
        return [6, 6, 8, 1, 1, 1, 1]  # Example: ethanol-like molecule
    
    def _create_node_features(self, atomic_numbers: List[int], structure: Any) -> np.ndarray:
        """
        Create node features for each atom.
        
        Args:
            atomic_numbers: List of atomic numbers
            structure: Original structure object
            
        Returns:
            Node feature matrix
        """
        features = []
        
        for atomic_num in atomic_numbers:
            atom_features = self._get_atomic_features(atomic_num)
            features.append(atom_features)
        
        node_features = np.array(features)
        
        if self.normalize_features:
            node_features = self._normalize_features(
                torch.tensor(node_features, dtype=torch.float32)
            ).numpy()
        
        return node_features
    
    def _get_atomic_features(self, atomic_number: int) -> List[float]:
        """
        Get comprehensive atomic features for a given atomic number.
        
        Args:
            atomic_number: Atomic number
            
        Returns:
            List of atomic features
        """
        # Get element symbol
        element_symbols = {v['atomic_number']: k for k, v in self.ATOMIC_PROPERTIES.items()}
        symbol = element_symbols.get(atomic_number, 'C')  # Default to carbon
        
        if symbol not in self.ATOMIC_PROPERTIES:
            # Use carbon as default
            props = self.ATOMIC_PROPERTIES['C']
        else:
            props = self.ATOMIC_PROPERTIES[symbol]
        
        features = []
        
        # Basic atomic properties
        features.extend([
            props['atomic_number'] / 100.0,  # Normalized atomic number
            props['period'] / 7.0,  # Normalized period
            props['group'] / 18.0,  # Normalized group
            props['electronegativity'] / 4.0,  # Normalized electronegativity
            props['atomic_radius'],  # Atomic radius
            props['covalent_radius'],  # Covalent radius
        ])
        
        # One-hot encoding for common elements (first 36 elements)
        one_hot = [0.0] * 36
        if atomic_number <= 36:
            one_hot[atomic_number - 1] = 1.0
        features.extend(one_hot)
        
        # Electron configuration features
        electron_config = self._get_electron_config_features(atomic_number)
        features.extend(electron_config)
        
        # Oxidation state features (simplified)
        oxidation_features = self._get_oxidation_features(atomic_number)
        features.extend(oxidation_features)
        
        # Ensure we have exactly the expected number of features
        while len(features) < self.node_feature_dim:
            features.append(0.0)
        
        return features[:self.node_feature_dim]
    
    def _get_electron_config_features(self, atomic_number: int) -> List[float]:
        """Get electron configuration features."""
        # Simplified electron configuration features
        # In practice, you'd want more sophisticated orbital features
        features = []
        
        # s, p, d, f electron counts (simplified)
        if atomic_number <= 2:
            s_electrons = atomic_number
            p_electrons = d_electrons = f_electrons = 0
        elif atomic_number <= 10:
            s_electrons = 2
            p_electrons = atomic_number - 2
            d_electrons = f_electrons = 0
        elif atomic_number <= 18:
            s_electrons = 2
            p_electrons = 8
            d_electrons = atomic_number - 10
            f_electrons = 0
        else:
            # Simplified for higher elements
            s_electrons = 2
            p_electrons = 8
            d_electrons = min(10, atomic_number - 18)
            f_electrons = max(0, atomic_number - 28)
        
        features.extend([
            s_electrons / 2.0,
            p_electrons / 8.0,
            d_electrons / 10.0,
            f_electrons / 14.0
        ])
        
        return features
    
    def _get_oxidation_features(self, atomic_number: int) -> List[float]:
        """Get oxidation state features."""
        # Common oxidation states for elements (simplified)
        common_oxidations = {
            1: [-1, 1], 6: [-4, 2, 4], 7: [-3, 3, 5], 8: [-2],
            11: [1], 12: [2], 13: [3], 14: [-4, 4], 15: [-3, 3, 5],
            16: [-2, 4, 6], 17: [-1, 1, 3, 5, 7], 19: [1], 20: [2],
            29: [1, 2], 30: [2]
        }
        
        oxidations = common_oxidations.get(atomic_number, [0])
        
        # Create features for common oxidation states
        features = []
        for ox_state in [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]:
            features.append(1.0 if ox_state in oxidations else 0.0)
        
        return features
    
    def _create_edges(
        self, 
        positions: np.ndarray, 
        lattice: np.ndarray, 
        atomic_numbers: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create edges based on atomic distances.
        
        Args:
            positions: Atomic positions
            lattice: Lattice matrix
            atomic_numbers: Atomic numbers
            
        Returns:
            Tuple of (edge_indices, edge_features)
        """
        num_atoms = len(positions)
        edge_indices = []
        edge_features = []
        
        # Calculate pairwise distances
        for i in range(num_atoms):
            distances = []
            neighbors = []
            
            for j in range(num_atoms):
                if i == j:
                    continue
                
                # Calculate minimum distance considering periodic boundary conditions
                min_dist = float('inf')
                best_vector = None
                
                if self.include_images:
                    # Check periodic images
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                image_vector = np.array([dx, dy, dz])
                                image_pos = positions[j] + np.dot(image_vector, lattice)
                                dist_vector = image_pos - positions[i]
                                dist = np.linalg.norm(dist_vector)
                                
                                if dist < min_dist:
                                    min_dist = dist
                                    best_vector = dist_vector
                else:
                    # Direct distance without periodic images
                    dist_vector = positions[j] - positions[i]
                    min_dist = np.linalg.norm(dist_vector)
                    best_vector = dist_vector
                
                if min_dist <= self.cutoff_radius:
                    distances.append(min_dist)
                    neighbors.append((j, best_vector))
            
            # Sort by distance and keep only max_neighbors
            if neighbors:
                neighbor_data = list(zip(distances, neighbors))
                neighbor_data.sort(key=lambda x: x[0])
                neighbor_data = neighbor_data[:self.max_neighbors]
                
                for dist, (j, vector) in neighbor_data:
                    edge_indices.append([i, j])
                    edge_feat = self._create_edge_features(
                        dist, vector, atomic_numbers[i], atomic_numbers[j]
                    )
                    edge_features.append(edge_feat)
        
        if not edge_indices:
            # Create at least one dummy edge to avoid empty graphs
            edge_indices = [[0, 0]]
            edge_features = [self._create_dummy_edge_features()]
        
        edge_indices = np.array(edge_indices).T
        edge_features = np.array(edge_features)
        
        return edge_indices, edge_features
    
    def _create_edge_features(
        self, 
        distance: float, 
        vector: np.ndarray, 
        atom1_num: int, 
        atom2_num: int
    ) -> List[float]:
        """
        Create edge features for a bond.
        
        Args:
            distance: Bond distance
            vector: Distance vector
            atom1_num: Atomic number of first atom
            atom2_num: Atomic number of second atom
            
        Returns:
            Edge features
        """
        features = []
        
        # Distance features
        features.append(distance)
        features.append(1.0 / (distance + 1e-8))  # Inverse distance
        features.append(np.exp(-distance))  # Exponential decay
        
        # Gaussian distance features (RBF)
        centers = np.linspace(0, self.cutoff_radius, 20)
        gamma = 0.5
        for center in centers:
            features.append(np.exp(-gamma * (distance - center) ** 2))
        
        # Direction features
        unit_vector = vector / (np.linalg.norm(vector) + 1e-8)
        features.extend(unit_vector.tolist())
        
        # Atomic pair features
        features.extend([
            atom1_num / 100.0,
            atom2_num / 100.0,
            abs(atom1_num - atom2_num) / 100.0,
            (atom1_num + atom2_num) / 200.0
        ])
        
        # Bond type features (simplified)
        bond_features = self._get_bond_type_features(distance, atom1_num, atom2_num)
        features.extend(bond_features)
        
        # Ensure we have exactly the expected number of features
        while len(features) < self.edge_feature_dim:
            features.append(0.0)
        
        return features[:self.edge_feature_dim]
    
    def _get_bond_type_features(self, distance: float, atom1: int, atom2: int) -> List[float]:
        """Get bond type features based on distance and atom types."""
        # Simplified bond type classification
        features = []
        
        # Typical bond lengths (Angstroms)
        typical_bonds = {
            (1, 1): 0.74,   # H-H
            (1, 6): 1.09,   # H-C
            (1, 8): 0.96,   # H-O
            (6, 6): 1.54,   # C-C
            (6, 8): 1.43,   # C-O
            (8, 8): 1.48,   # O-O
        }
        
        # Get expected bond length
        bond_key = tuple(sorted([atom1, atom2]))
        expected_length = typical_bonds.get(bond_key, 2.0)
        
        # Bond strength indicator
        strength = np.exp(-(distance - expected_length) ** 2 / 0.1)
        features.append(strength)
        
        # Bond type indicators
        features.extend([
            1.0 if distance < 1.8 else 0.0,  # Strong bond
            1.0 if 1.8 <= distance < 3.0 else 0.0,  # Medium bond
            1.0 if distance >= 3.0 else 0.0,  # Weak bond
        ])
        
        return features
    
    def _create_dummy_edge_features(self) -> List[float]:
        """Create dummy edge features for empty graphs."""
        return [0.0] * self.edge_feature_dim
    
    def _create_global_features(self, structure: Any) -> List[float]:
        """
        Create global/graph-level features for the crystal.
        
        Args:
            structure: Crystal structure
            
        Returns:
            Global features
        """
        features = []
        
        # Crystal system features (simplified)
        if hasattr(structure, 'lattice'):
            lattice = structure.lattice
            if hasattr(lattice, 'abc'):
                a, b, c = lattice.abc
                features.extend([a / 20.0, b / 20.0, c / 20.0])  # Normalized cell parameters
            else:
                features.extend([0.5, 0.5, 0.5])  # Default values
            
            if hasattr(lattice, 'angles'):
                alpha, beta, gamma = lattice.angles
                features.extend([
                    alpha / 180.0, beta / 180.0, gamma / 180.0
                ])  # Normalized angles
            else:
                features.extend([0.5, 0.5, 0.5])  # Default angles (90 degrees)
        else:
            # Default crystal features
            features.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Density and volume features
        if hasattr(structure, 'density'):
            features.append(structure.density / 5.0)  # Normalized density
        else:
            features.append(0.5)  # Default density
        
        # Number of atoms (normalized)
        num_atoms = len(self._get_atomic_numbers(structure))
        features.append(min(num_atoms / 100.0, 1.0))
        
        # Composition features (simplified)
        atomic_numbers = self._get_atomic_numbers(structure)
        unique_elements = len(set(atomic_numbers))
        features.append(unique_elements / 10.0)  # Normalized number of unique elements
        
        # Average atomic number
        avg_atomic_num = np.mean(atomic_numbers) / 50.0
        features.append(avg_atomic_num)
        
        # Ensure we have exactly the expected number of features
        while len(features) < self.global_feature_dim:
            features.append(0.0)
        
        return features[:self.global_feature_dim]
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get the dimensions of different feature types."""
        return {
            'node_features': self.node_feature_dim,
            'edge_features': self.edge_feature_dim,
            'global_features': self.global_feature_dim
        }