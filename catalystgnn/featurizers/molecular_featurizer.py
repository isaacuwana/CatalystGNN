"""
Molecular graph featurizer for organic molecules.

This module implements graph featurization for molecular structures,
particularly useful for catalysts, ligands, and organic molecules.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
import logging

from .base_featurizer import BaseFeaturizer

logger = logging.getLogger(__name__)


class MolecularGraphFeaturizer(BaseFeaturizer):
    """
    Graph featurizer for molecular structures.
    
    This featurizer converts molecular structures into graph representations
    suitable for Message Passing Neural Networks (MPNN) and similar
    architectures. It handles molecular bonds, formal charges, and
    stereochemistry information.
    
    Attributes:
        use_chirality: Whether to include chirality information
        use_formal_charge: Whether to include formal charge information
        use_hybridization: Whether to include hybridization information
        max_atomic_num: Maximum atomic number to consider
    """
    
    # Atomic properties for molecular featurization
    ATOMIC_FEATURES = {
        'atomic_num': list(range(1, 101)),  # 1-100
        'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        'chiral_tag': [0, 1, 2, 3],  # None, R, S, Other
        'num_Hs': [0, 1, 2, 3, 4, 5],
        'hybridization': [0, 1, 2, 3, 4, 5, 6],  # SP, SP2, SP3, SP3D, SP3D2, Other, Unspecified
        'aromaticity': [0, 1],  # False, True
        'mass': 'continuous'
    }
    
    BOND_FEATURES = {
        'bond_type': [0, 1, 2, 3, 4],  # None, Single, Double, Triple, Aromatic
        'conjugated': [0, 1],  # False, True
        'in_ring': [0, 1],  # False, True
        'stereo': [0, 1, 2, 3, 4, 5, 6]  # None, Any, E, Z, Cis, Trans, Other
    }
    
    def __init__(
        self,
        use_chirality: bool = True,
        use_formal_charge: bool = True,
        use_hybridization: bool = True,
        max_atomic_num: int = 100,
        normalize_features: bool = True,
        **kwargs
    ):
        """
        Initialize the molecular graph featurizer.
        
        Args:
            use_chirality: Whether to include chirality information
            use_formal_charge: Whether to include formal charge
            use_hybridization: Whether to include hybridization
            max_atomic_num: Maximum atomic number to consider
            normalize_features: Whether to normalize features
        """
        self.use_chirality = use_chirality
        self.use_formal_charge = use_formal_charge
        self.use_hybridization = use_hybridization
        self.max_atomic_num = max_atomic_num
        self.normalize_features = normalize_features
        
        super().__init__(**kwargs)
    
    def _setup_featurizer(self):
        """Setup molecular-specific featurizer configurations."""
        # Calculate feature dimensions
        self.node_feature_dim = self._calculate_node_feature_dim()
        self.edge_feature_dim = self._calculate_edge_feature_dim()
        self.global_feature_dim = 20  # Molecular descriptors
        
        logger.info(f"Molecular featurizer initialized with {self.node_feature_dim} node features")
    
    def _calculate_node_feature_dim(self) -> int:
        """Calculate the dimension of node features."""
        dim = 0
        
        # Atomic number (one-hot)
        dim += min(self.max_atomic_num, 100)
        
        # Degree (one-hot)
        dim += len(self.ATOMIC_FEATURES['degree'])
        
        # Formal charge (one-hot)
        if self.use_formal_charge:
            dim += len(self.ATOMIC_FEATURES['formal_charge'])
        
        # Chirality (one-hot)
        if self.use_chirality:
            dim += len(self.ATOMIC_FEATURES['chiral_tag'])
        
        # Number of hydrogens (one-hot)
        dim += len(self.ATOMIC_FEATURES['num_Hs'])
        
        # Hybridization (one-hot)
        if self.use_hybridization:
            dim += len(self.ATOMIC_FEATURES['hybridization'])
        
        # Aromaticity (one-hot)
        dim += len(self.ATOMIC_FEATURES['aromaticity'])
        
        # Atomic mass (continuous)
        dim += 1
        
        # Additional chemical properties
        dim += 10  # Electronegativity, radius, etc.
        
        return dim
    
    def _calculate_edge_feature_dim(self) -> int:
        """Calculate the dimension of edge features."""
        dim = 0
        
        # Bond type (one-hot)
        dim += len(self.BOND_FEATURES['bond_type'])
        
        # Conjugated (one-hot)
        dim += len(self.BOND_FEATURES['conjugated'])
        
        # In ring (one-hot)
        dim += len(self.BOND_FEATURES['in_ring'])
        
        # Stereo (one-hot)
        dim += len(self.BOND_FEATURES['stereo'])
        
        # Bond length (continuous)
        dim += 1
        
        # Additional bond features
        dim += 5  # Bond order, rotatable, etc.
        
        return dim
    
    def featurize(self, molecule: Any) -> Dict[str, torch.Tensor]:
        """
        Convert molecular structure to graph representation.
        
        Args:
            molecule: Molecular structure object (RDKit Mol or similar)
            
        Returns:
            Dictionary containing graph tensors
        """
        try:
            # Extract molecular information
            atoms = self._get_atoms(molecule)
            bonds = self._get_bonds(molecule)
            
            # Create node features
            node_features = self._create_node_features(atoms, molecule)
            
            # Create edge features and indices
            edge_indices, edge_features = self._create_edge_features(bonds, len(atoms))
            
            # Create global features
            global_features = self._create_global_features(molecule)
            
            # Convert to tensors
            graph_data = {
                'node_features': torch.tensor(node_features, dtype=torch.float32),
                'edge_indices': torch.tensor(edge_indices, dtype=torch.long),
                'edge_features': torch.tensor(edge_features, dtype=torch.float32),
                'global_features': torch.tensor(global_features, dtype=torch.float32),
                'num_nodes': torch.tensor([len(atoms)], dtype=torch.long),
                'num_edges': torch.tensor([len(edge_indices[0])], dtype=torch.long)
            }
            
            # Validate graph data
            if not self._validate_graph_data(graph_data):
                raise ValueError("Generated graph data is invalid")
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error featurizing molecular structure: {e}")
            raise
    
    def _get_atoms(self, molecule: Any) -> List[Dict]:
        """Extract atom information from molecule."""
        atoms = []
        
        if hasattr(molecule, 'GetAtoms'):
            # RDKit molecule
            for atom in molecule.GetAtoms():
                atom_info = {
                    'atomic_num': atom.GetAtomicNum(),
                    'degree': atom.GetDegree(),
                    'formal_charge': atom.GetFormalCharge(),
                    'chiral_tag': int(atom.GetChiralTag()),
                    'num_Hs': atom.GetTotalNumHs(),
                    'hybridization': int(atom.GetHybridization()),
                    'is_aromatic': atom.GetIsAromatic(),
                    'mass': atom.GetMass(),
                    'symbol': atom.GetSymbol()
                }
                atoms.append(atom_info)
        else:
            # Fallback: create dummy atoms for demonstration
            atoms = self._create_dummy_atoms()
        
        return atoms
    
    def _get_bonds(self, molecule: Any) -> List[Dict]:
        """Extract bond information from molecule."""
        bonds = []
        
        if hasattr(molecule, 'GetBonds'):
            # RDKit molecule
            for bond in molecule.GetBonds():
                bond_info = {
                    'begin_atom': bond.GetBeginAtomIdx(),
                    'end_atom': bond.GetEndAtomIdx(),
                    'bond_type': int(bond.GetBondType()),
                    'is_conjugated': bond.GetIsConjugated(),
                    'is_in_ring': bond.IsInRing(),
                    'stereo': int(bond.GetStereo())
                }
                bonds.append(bond_info)
        else:
            # Fallback: create dummy bonds
            bonds = self._create_dummy_bonds()
        
        return bonds
    
    def _create_dummy_atoms(self) -> List[Dict]:
        """Create dummy atoms for demonstration."""
        return [
            {'atomic_num': 6, 'degree': 4, 'formal_charge': 0, 'chiral_tag': 0, 
             'num_Hs': 3, 'hybridization': 3, 'is_aromatic': False, 'mass': 12.01, 'symbol': 'C'},
            {'atomic_num': 6, 'degree': 4, 'formal_charge': 0, 'chiral_tag': 0,
             'num_Hs': 2, 'hybridization': 3, 'is_aromatic': False, 'mass': 12.01, 'symbol': 'C'},
            {'atomic_num': 8, 'degree': 2, 'formal_charge': 0, 'chiral_tag': 0,
             'num_Hs': 1, 'hybridization': 3, 'is_aromatic': False, 'mass': 15.999, 'symbol': 'O'},
        ]
    
    def _create_dummy_bonds(self) -> List[Dict]:
        """Create dummy bonds for demonstration."""
        return [
            {'begin_atom': 0, 'end_atom': 1, 'bond_type': 1, 'is_conjugated': False, 
             'is_in_ring': False, 'stereo': 0},
            {'begin_atom': 1, 'end_atom': 2, 'bond_type': 1, 'is_conjugated': False,
             'is_in_ring': False, 'stereo': 0},
        ]
    
    def _create_node_features(self, atoms: List[Dict], molecule: Any) -> np.ndarray:
        """
        Create node features for each atom.
        
        Args:
            atoms: List of atom dictionaries
            molecule: Original molecule object
            
        Returns:
            Node feature matrix
        """
        features = []
        
        for atom in atoms:
            atom_features = self._get_atom_features(atom)
            features.append(atom_features)
        
        node_features = np.array(features)
        
        if self.normalize_features:
            node_features = self._normalize_features(
                torch.tensor(node_features, dtype=torch.float32)
            ).numpy()
        
        return node_features
    
    def _get_atom_features(self, atom: Dict) -> List[float]:
        """
        Get comprehensive atomic features.
        
        Args:
            atom: Atom information dictionary
            
        Returns:
            List of atomic features
        """
        features = []
        
        # Atomic number (one-hot)
        atomic_num = min(atom['atomic_num'], self.max_atomic_num)
        one_hot_atomic = [0.0] * self.max_atomic_num
        if atomic_num > 0:
            one_hot_atomic[atomic_num - 1] = 1.0
        features.extend(one_hot_atomic)
        
        # Degree (one-hot)
        degree = min(atom['degree'], len(self.ATOMIC_FEATURES['degree']) - 1)
        one_hot_degree = [0.0] * len(self.ATOMIC_FEATURES['degree'])
        one_hot_degree[degree] = 1.0
        features.extend(one_hot_degree)
        
        # Formal charge (one-hot)
        if self.use_formal_charge:
            formal_charge = atom['formal_charge']
            if formal_charge in self.ATOMIC_FEATURES['formal_charge']:
                idx = self.ATOMIC_FEATURES['formal_charge'].index(formal_charge)
            else:
                idx = self.ATOMIC_FEATURES['formal_charge'].index(0)  # Default to 0
            one_hot_charge = [0.0] * len(self.ATOMIC_FEATURES['formal_charge'])
            one_hot_charge[idx] = 1.0
            features.extend(one_hot_charge)
        
        # Chirality (one-hot)
        if self.use_chirality:
            chiral_tag = min(atom['chiral_tag'], len(self.ATOMIC_FEATURES['chiral_tag']) - 1)
            one_hot_chiral = [0.0] * len(self.ATOMIC_FEATURES['chiral_tag'])
            one_hot_chiral[chiral_tag] = 1.0
            features.extend(one_hot_chiral)
        
        # Number of hydrogens (one-hot)
        num_Hs = min(atom['num_Hs'], len(self.ATOMIC_FEATURES['num_Hs']) - 1)
        one_hot_Hs = [0.0] * len(self.ATOMIC_FEATURES['num_Hs'])
        one_hot_Hs[num_Hs] = 1.0
        features.extend(one_hot_Hs)
        
        # Hybridization (one-hot)
        if self.use_hybridization:
            hybridization = min(atom['hybridization'], len(self.ATOMIC_FEATURES['hybridization']) - 1)
            one_hot_hybrid = [0.0] * len(self.ATOMIC_FEATURES['hybridization'])
            one_hot_hybrid[hybridization] = 1.0
            features.extend(one_hot_hybrid)
        
        # Aromaticity (one-hot)
        one_hot_aromatic = [0.0] * len(self.ATOMIC_FEATURES['aromaticity'])
        one_hot_aromatic[int(atom['is_aromatic'])] = 1.0
        features.extend(one_hot_aromatic)
        
        # Atomic mass (continuous, normalized)
        features.append(atom['mass'] / 200.0)  # Normalize by typical max mass
        
        # Additional chemical properties
        additional_features = self._get_additional_atom_features(atom)
        features.extend(additional_features)
        
        # Ensure we have exactly the expected number of features
        while len(features) < self.node_feature_dim:
            features.append(0.0)
        
        return features[:self.node_feature_dim]
    
    def _get_additional_atom_features(self, atom: Dict) -> List[float]:
        """Get additional chemical properties for atoms."""
        # Electronegativity values (Pauling scale)
        electronegativity = {
            'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
            'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66
        }
        
        # Van der Waals radii (Angstroms)
        vdw_radius = {
            'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
            'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        symbol = atom.get('symbol', 'C')
        
        features = [
            electronegativity.get(symbol, 2.5) / 4.0,  # Normalized electronegativity
            vdw_radius.get(symbol, 1.7) / 2.5,  # Normalized VdW radius
            atom['atomic_num'] / 100.0,  # Normalized atomic number
            1.0 if atom['atomic_num'] <= 10 else 0.0,  # Is light element
            1.0 if atom['atomic_num'] in [6, 7, 8, 9] else 0.0,  # Is CNOF
            1.0 if atom['atomic_num'] in [15, 16, 17] else 0.0,  # Is PSCl
            1.0 if atom['atomic_num'] in range(21, 31) else 0.0,  # Is transition metal
            1.0 if atom['degree'] == 1 else 0.0,  # Is terminal
            1.0 if atom['formal_charge'] != 0 else 0.0,  # Is charged
            float(atom['is_aromatic'])  # Aromaticity (duplicate for emphasis)
        ]
        
        return features
    
    def _create_edge_features(self, bonds: List[Dict], num_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create edge features and indices.
        
        Args:
            bonds: List of bond dictionaries
            num_atoms: Number of atoms in molecule
            
        Returns:
            Tuple of (edge_indices, edge_features)
        """
        edge_indices = []
        edge_features = []
        
        for bond in bonds:
            begin_atom = bond['begin_atom']
            end_atom = bond['end_atom']
            
            # Add both directions for undirected graph
            edge_indices.extend([[begin_atom, end_atom], [end_atom, begin_atom]])
            
            # Create edge features
            edge_feat = self._get_bond_features(bond)
            edge_features.extend([edge_feat, edge_feat])  # Same features for both directions
        
        # Add self-loops
        for i in range(num_atoms):
            edge_indices.append([i, i])
            edge_features.append(self._get_self_loop_features())
        
        if not edge_indices:
            # Create dummy edge if no bonds exist
            edge_indices = [[0, 0]]
            edge_features = [self._get_dummy_edge_features()]
        
        edge_indices = np.array(edge_indices).T
        edge_features = np.array(edge_features)
        
        return edge_indices, edge_features
    
    def _get_bond_features(self, bond: Dict) -> List[float]:
        """
        Get bond features.
        
        Args:
            bond: Bond information dictionary
            
        Returns:
            List of bond features
        """
        features = []
        
        # Bond type (one-hot)
        bond_type = min(bond['bond_type'], len(self.BOND_FEATURES['bond_type']) - 1)
        one_hot_type = [0.0] * len(self.BOND_FEATURES['bond_type'])
        one_hot_type[bond_type] = 1.0
        features.extend(one_hot_type)
        
        # Conjugated (one-hot)
        one_hot_conj = [0.0] * len(self.BOND_FEATURES['conjugated'])
        one_hot_conj[int(bond['is_conjugated'])] = 1.0
        features.extend(one_hot_conj)
        
        # In ring (one-hot)
        one_hot_ring = [0.0] * len(self.BOND_FEATURES['in_ring'])
        one_hot_ring[int(bond['is_in_ring'])] = 1.0
        features.extend(one_hot_ring)
        
        # Stereo (one-hot)
        stereo = min(bond['stereo'], len(self.BOND_FEATURES['stereo']) - 1)
        one_hot_stereo = [0.0] * len(self.BOND_FEATURES['stereo'])
        one_hot_stereo[stereo] = 1.0
        features.extend(one_hot_stereo)
        
        # Bond length (estimated from bond type)
        bond_length = self._estimate_bond_length(bond['bond_type'])
        features.append(bond_length / 3.0)  # Normalized
        
        # Additional bond features
        additional_features = [
            float(bond['bond_type']),  # Bond order
            1.0 if not bond['is_in_ring'] else 0.0,  # Is rotatable
            1.0 if bond['bond_type'] > 1 else 0.0,  # Is multiple bond
            1.0 if bond['is_conjugated'] and bond['is_aromatic'] else 0.0,  # Is aromatic conjugated
            float(bond['stereo'] > 0)  # Has stereochemistry
        ]
        features.extend(additional_features)
        
        # Ensure we have exactly the expected number of features
        while len(features) < self.edge_feature_dim:
            features.append(0.0)
        
        return features[:self.edge_feature_dim]
    
    def _estimate_bond_length(self, bond_type: int) -> float:
        """Estimate bond length from bond type."""
        # Typical bond lengths in Angstroms
        lengths = {0: 2.0, 1: 1.5, 2: 1.3, 3: 1.2, 4: 1.4}  # None, Single, Double, Triple, Aromatic
        return lengths.get(bond_type, 1.5)
    
    def _get_self_loop_features(self) -> List[float]:
        """Get features for self-loop edges."""
        features = [0.0] * self.edge_feature_dim
        features[0] = 1.0  # Mark as self-loop in first position
        return features
    
    def _get_dummy_edge_features(self) -> List[float]:
        """Get dummy edge features for empty graphs."""
        return [0.0] * self.edge_feature_dim
    
    def _create_global_features(self, molecule: Any) -> List[float]:
        """
        Create global/molecular-level features.
        
        Args:
            molecule: Molecular structure
            
        Returns:
            Global features
        """
        features = []
        
        # Molecular weight
        if hasattr(molecule, 'GetMolWt'):
            mol_wt = molecule.GetMolWt()
        else:
            mol_wt = 100.0  # Default
        features.append(mol_wt / 500.0)  # Normalized
        
        # Number of atoms
        if hasattr(molecule, 'GetNumAtoms'):
            num_atoms = molecule.GetNumAtoms()
        else:
            num_atoms = 10  # Default
        features.append(num_atoms / 100.0)  # Normalized
        
        # Number of bonds
        if hasattr(molecule, 'GetNumBonds'):
            num_bonds = molecule.GetNumBonds()
        else:
            num_bonds = 9  # Default
        features.append(num_bonds / 100.0)  # Normalized
        
        # Ring information
        if hasattr(molecule, 'GetRingInfo'):
            ring_info = molecule.GetRingInfo()
            num_rings = ring_info.NumRings()
        else:
            num_rings = 0
        features.append(num_rings / 10.0)  # Normalized
        
        # Aromaticity
        aromatic_atoms = 0
        atoms = self._get_atoms(molecule)
        for atom in atoms:
            if atom['is_aromatic']:
                aromatic_atoms += 1
        features.append(aromatic_atoms / len(atoms) if atoms else 0.0)
        
        # Charge distribution
        total_charge = sum(atom['formal_charge'] for atom in atoms)
        features.append((total_charge + 5) / 10.0)  # Normalized to [0,1]
        
        # Heteroatom ratio
        heteroatoms = sum(1 for atom in atoms if atom['atomic_num'] not in [1, 6])
        hetero_ratio = heteroatoms / len(atoms) if atoms else 0.0
        features.append(hetero_ratio)
        
        # Degree distribution
        if atoms:
            avg_degree = np.mean([atom['degree'] for atom in atoms])
            max_degree = max([atom['degree'] for atom in atoms])
        else:
            avg_degree = max_degree = 0
        features.extend([avg_degree / 6.0, max_degree / 6.0])
        
        # Hybridization distribution
        sp3_count = sum(1 for atom in atoms if atom['hybridization'] == 3)
        sp2_count = sum(1 for atom in atoms if atom['hybridization'] == 2)
        sp_count = sum(1 for atom in atoms if atom['hybridization'] == 1)
        
        total_atoms = len(atoms) if atoms else 1
        features.extend([
            sp3_count / total_atoms,
            sp2_count / total_atoms,
            sp_count / total_atoms
        ])
        
        # Rotatable bonds (estimated)
        rotatable_bonds = 0
        bonds = self._get_bonds(molecule)
        for bond in bonds:
            if bond['bond_type'] == 1 and not bond['is_in_ring']:  # Single bond not in ring
                rotatable_bonds += 1
        features.append(rotatable_bonds / 20.0)  # Normalized
        
        # Hydrogen bond donors/acceptors (simplified)
        hbd = sum(1 for atom in atoms if atom['atomic_num'] in [7, 8] and atom['num_Hs'] > 0)
        hba = sum(1 for atom in atoms if atom['atomic_num'] in [7, 8])
        features.extend([hbd / 10.0, hba / 10.0])
        
        # Molecular complexity (estimated)
        complexity = len(atoms) + len(bonds) + num_rings
        features.append(complexity / 100.0)
        
        # Saturation (ratio of actual to maximum bonds)
        max_possible_bonds = sum(self._get_max_valence(atom['atomic_num']) for atom in atoms) // 2
        actual_bonds = len(bonds)
        saturation = actual_bonds / max_possible_bonds if max_possible_bonds > 0 else 0
        features.append(min(saturation, 1.0))
        
        # Ensure we have exactly the expected number of features
        while len(features) < self.global_feature_dim:
            features.append(0.0)
        
        return features[:self.global_feature_dim]
    
    def _get_max_valence(self, atomic_num: int) -> int:
        """Get maximum valence for an atom."""
        max_valences = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1}
        return max_valences.get(atomic_num, 4)  # Default to 4
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get the dimensions of different feature types."""
        return {
            'node_features': self.node_feature_dim,
            'edge_features': self.edge_feature_dim,
            'global_features': self.global_feature_dim
        }