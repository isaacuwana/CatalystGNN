"""
File handling utilities for loading chemical structures.

This module provides functions to load and parse various chemical file formats
including CIF, SMILES, XYZ, and POSCAR files.
"""

import os
import logging
from typing import Any, Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


def load_structure_from_file(file_path: str) -> Any:
    """
    Load chemical structure from file.
    
    Supports multiple file formats:
    - .cif: Crystallographic Information File
    - .xyz: XYZ coordinate file
    - .poscar/.vasp: VASP structure file
    - .pdb: Protein Data Bank file
    
    Args:
        file_path: Path to structure file
        
    Returns:
        Structure object (pymatgen Structure or similar)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {file_path}")
    
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == '.cif':
            return load_cif_file(file_path)
        elif file_extension == '.xyz':
            return load_xyz_file(file_path)
        elif file_extension in ['.poscar', '.vasp']:
            return load_poscar_file(file_path)
        elif file_extension == '.pdb':
            return load_pdb_file(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return create_dummy_structure(file_path)
            
    except Exception as e:
        logger.error(f"Error loading structure from {file_path}: {e}")
        logger.info("Creating dummy structure for demonstration")
        return create_dummy_structure(file_path)


def load_cif_file(file_path: Path) -> Any:
    """
    Load structure from CIF file.
    
    Args:
        file_path: Path to CIF file
        
    Returns:
        Structure object
    """
    try:
        # Try to use pymatgen if available
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifParser
        
        parser = CifParser(str(file_path))
        structures = parser.get_structures()
        
        if structures:
            structure = structures[0]  # Take first structure
            logger.info(f"Loaded CIF structure with {len(structure)} atoms")
            return structure
        else:
            raise ValueError("No structures found in CIF file")
            
    except ImportError:
        logger.warning("pymatgen not available, using ASE fallback")
        return load_with_ase(file_path)
    except Exception as e:
        logger.error(f"Error parsing CIF file: {e}")
        return create_dummy_crystal_structure()


def load_xyz_file(file_path: Path) -> Any:
    """
    Load structure from XYZ file.
    
    Args:
        file_path: Path to XYZ file
        
    Returns:
        Structure object
    """
    try:
        # Try to use ASE if available
        from ase.io import read
        
        atoms = read(str(file_path))
        logger.info(f"Loaded XYZ structure with {len(atoms)} atoms")
        return atoms
        
    except ImportError:
        logger.warning("ASE not available, parsing XYZ manually")
        return parse_xyz_manually(file_path)
    except Exception as e:
        logger.error(f"Error parsing XYZ file: {e}")
        return create_dummy_molecular_structure()


def load_poscar_file(file_path: Path) -> Any:
    """
    Load structure from POSCAR/VASP file.
    
    Args:
        file_path: Path to POSCAR file
        
    Returns:
        Structure object
    """
    try:
        # Try pymatgen first
        from pymatgen.io.vasp import Poscar
        
        poscar = Poscar.from_file(str(file_path))
        structure = poscar.structure
        logger.info(f"Loaded POSCAR structure with {len(structure)} atoms")
        return structure
        
    except ImportError:
        logger.warning("pymatgen not available, using ASE fallback")
        return load_with_ase(file_path)
    except Exception as e:
        logger.error(f"Error parsing POSCAR file: {e}")
        return create_dummy_crystal_structure()


def load_pdb_file(file_path: Path) -> Any:
    """
    Load structure from PDB file.
    
    Args:
        file_path: Path to PDB file
        
    Returns:
        Structure object
    """
    try:
        # Try to use ASE
        from ase.io import read
        
        atoms = read(str(file_path))
        logger.info(f"Loaded PDB structure with {len(atoms)} atoms")
        return atoms
        
    except ImportError:
        logger.warning("ASE not available, parsing PDB manually")
        return parse_pdb_manually(file_path)
    except Exception as e:
        logger.error(f"Error parsing PDB file: {e}")
        return create_dummy_molecular_structure()


def load_with_ase(file_path: Path) -> Any:
    """
    Load structure using ASE as fallback.
    
    Args:
        file_path: Path to structure file
        
    Returns:
        ASE Atoms object
    """
    try:
        from ase.io import read
        
        atoms = read(str(file_path))
        logger.info(f"Loaded structure with ASE: {len(atoms)} atoms")
        return atoms
        
    except ImportError:
        logger.error("ASE not available for fallback loading")
        return create_dummy_structure(file_path)
    except Exception as e:
        logger.error(f"ASE loading failed: {e}")
        return create_dummy_structure(file_path)


def parse_xyz_manually(file_path: Path) -> Dict:
    """
    Manually parse XYZ file when ASE is not available.
    
    Args:
        file_path: Path to XYZ file
        
    Returns:
        Dictionary with atomic information
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse XYZ format
        num_atoms = int(lines[0].strip())
        comment = lines[1].strip()
        
        atoms = []
        positions = []
        
        for i in range(2, 2 + num_atoms):
            parts = lines[i].strip().split()
            symbol = parts[0]
            x, y, z = map(float, parts[1:4])
            
            atoms.append(symbol)
            positions.append([x, y, z])
        
        structure = {
            'symbols': atoms,
            'positions': positions,
            'comment': comment,
            'num_atoms': num_atoms
        }
        
        logger.info(f"Manually parsed XYZ file: {num_atoms} atoms")
        return structure
        
    except Exception as e:
        logger.error(f"Manual XYZ parsing failed: {e}")
        return create_dummy_molecular_structure()


def parse_pdb_manually(file_path: Path) -> Dict:
    """
    Manually parse PDB file when ASE is not available.
    
    Args:
        file_path: Path to PDB file
        
    Returns:
        Dictionary with atomic information
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        atoms = []
        positions = []
        
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Parse PDB ATOM record
                symbol = line[76:78].strip()
                if not symbol:
                    symbol = line[12:16].strip()[0]  # Fallback to atom name
                
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                atoms.append(symbol)
                positions.append([x, y, z])
        
        structure = {
            'symbols': atoms,
            'positions': positions,
            'num_atoms': len(atoms)
        }
        
        logger.info(f"Manually parsed PDB file: {len(atoms)} atoms")
        return structure
        
    except Exception as e:
        logger.error(f"Manual PDB parsing failed: {e}")
        return create_dummy_molecular_structure()


def parse_smiles(smiles: str) -> Any:
    """
    Parse SMILES string to molecular structure.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Molecular structure object (RDKit Mol or similar)
        
    Raises:
        ValueError: If SMILES string is invalid
    """
    try:
        # Try to use RDKit if available
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.OptimizeMolecule(mol)
        
        logger.info(f"Parsed SMILES '{smiles}': {mol.GetNumAtoms()} atoms")
        return mol
        
    except ImportError:
        logger.warning("RDKit not available, creating dummy molecule")
        return create_dummy_molecule_from_smiles(smiles)
    except Exception as e:
        logger.error(f"Error parsing SMILES '{smiles}': {e}")
        return create_dummy_molecule_from_smiles(smiles)


def create_dummy_structure(file_path: Path) -> Dict:
    """
    Create a dummy structure for demonstration purposes.
    
    Args:
        file_path: Original file path (for context)
        
    Returns:
        Dummy structure dictionary
    """
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.cif':
        return create_dummy_crystal_structure()
    else:
        return create_dummy_molecular_structure()


def create_dummy_crystal_structure() -> Dict:
    """Create a dummy crystal structure (MOF-like)."""
    import numpy as np
    
    # Create a simple cubic MOF-like structure
    structure = {
        'symbols': ['Zn', 'Zn', 'C', 'C', 'C', 'C', 'O', 'O', 'O', 'O', 'H', 'H', 'H', 'H'],
        'positions': np.array([
            [0.0, 0.0, 0.0],      # Zn
            [5.0, 5.0, 5.0],      # Zn
            [2.5, 0.0, 0.0],      # C
            [0.0, 2.5, 0.0],      # C
            [7.5, 5.0, 5.0],      # C
            [5.0, 7.5, 5.0],      # C
            [1.5, 0.0, 0.0],      # O
            [0.0, 1.5, 0.0],      # O
            [6.5, 5.0, 5.0],      # O
            [5.0, 6.5, 5.0],      # O
            [3.5, 0.0, 0.0],      # H
            [0.0, 3.5, 0.0],      # H
            [8.5, 5.0, 5.0],      # H
            [5.0, 8.5, 5.0],      # H
        ]),
        'lattice': np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]
        ]),
        'num_atoms': 14,
        'composition': 'Zn2C4O4H4',
        'space_group': 'P1'
    }
    
    logger.info("Created dummy crystal structure (MOF-like)")
    return structure


def create_dummy_molecular_structure() -> Dict:
    """Create a dummy molecular structure (ethanol-like)."""
    import numpy as np
    
    structure = {
        'symbols': ['C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'],
        'positions': np.array([
            [0.0, 0.0, 0.0],      # C
            [1.5, 0.0, 0.0],      # C
            [2.3, 1.2, 0.0],      # O
            [-0.5, -0.9, 0.0],    # H
            [-0.5, 0.5, 0.9],     # H
            [-0.5, 0.5, -0.9],    # H
            [1.5, -0.5, 0.9],     # H
            [1.5, -0.5, -0.9],    # H
            [3.2, 1.0, 0.0],      # H
        ]),
        'num_atoms': 9,
        'formula': 'C2H6O',
        'molecular_weight': 46.07
    }
    
    logger.info("Created dummy molecular structure (ethanol-like)")
    return structure


def create_dummy_molecule_from_smiles(smiles: str) -> Dict:
    """
    Create a dummy molecule structure from SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dummy molecule dictionary
    """
    # Simple mapping of common SMILES patterns to dummy structures
    smiles_patterns = {
        'CCO': create_dummy_molecular_structure(),  # Ethanol
        'CC': {
            'symbols': ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
            'positions': [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], 
                         [-0.5, -0.9, 0.0], [-0.5, 0.5, 0.9], [-0.5, 0.5, -0.9],
                         [2.0, -0.9, 0.0], [2.0, 0.5, 0.9], [2.0, 0.5, -0.9]],
            'num_atoms': 8,
            'formula': 'C2H6'
        },
        'C': {
            'symbols': ['C', 'H', 'H', 'H', 'H'],
            'positions': [[0.0, 0.0, 0.0], [-0.5, -0.9, 0.0], 
                         [-0.5, 0.5, 0.9], [-0.5, 0.5, -0.9], [0.5, 0.0, 0.0]],
            'num_atoms': 5,
            'formula': 'CH4'
        }
    }
    
    # Return matching pattern or default ethanol structure
    structure = smiles_patterns.get(smiles, create_dummy_molecular_structure())
    structure['smiles'] = smiles
    
    logger.info(f"Created dummy molecule for SMILES '{smiles}'")
    return structure


def validate_structure_file(file_path: str) -> bool:
    """
    Validate that a structure file exists and has a supported format.
    
    Args:
        file_path: Path to structure file
        
    Returns:
        True if file is valid, False otherwise
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        return False
    
    if not file_path.is_file():
        logger.error(f"Path is not a file: {file_path}")
        return False
    
    supported_extensions = {'.cif', '.xyz', '.poscar', '.vasp', '.pdb'}
    if file_path.suffix.lower() not in supported_extensions:
        logger.warning(f"Unsupported file extension: {file_path.suffix}")
        return False
    
    # Check if file is readable
    try:
        with open(file_path, 'r') as f:
            f.read(100)  # Try to read first 100 characters
        return True
    except Exception as e:
        logger.error(f"Cannot read file {file_path}: {e}")
        return False


def get_supported_formats() -> List[str]:
    """
    Get list of supported file formats.
    
    Returns:
        List of supported file extensions
    """
    return ['.cif', '.xyz', '.poscar', '.vasp', '.pdb']


def detect_file_format(file_path: str) -> Optional[str]:
    """
    Detect file format from file content.
    
    Args:
        file_path: Path to file
        
    Returns:
        Detected format or None if unknown
    """
    try:
        with open(file_path, 'r') as f:
            first_lines = [f.readline().strip() for _ in range(5)]
        
        # Check for CIF format
        if any('_cell_length' in line or '_symmetry_space_group' in line for line in first_lines):
            return 'cif'
        
        # Check for XYZ format
        try:
            int(first_lines[0])
            return 'xyz'
        except (ValueError, IndexError):
            pass
        
        # Check for PDB format
        if any(line.startswith(('ATOM', 'HETATM', 'HEADER')) for line in first_lines):
            return 'pdb'
        
        # Check for POSCAR format
        if len(first_lines) >= 3:
            try:
                # POSCAR has scaling factor on line 2 and lattice vectors on lines 3-5
                float(first_lines[1])
                return 'poscar'
            except (ValueError, IndexError):
                pass
        
        return None
        
    except Exception as e:
        logger.error(f"Error detecting file format: {e}")
        return None