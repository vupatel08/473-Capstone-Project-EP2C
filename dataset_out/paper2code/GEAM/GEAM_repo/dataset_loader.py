## dataset_loader.py

import os
import random
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

import numpy as np

# Set a fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class MoleculeDataset(InMemoryDataset):
    """
    A dataset class that loads molecules from a SMILES file, filters invalid molecules,
    converts them into torch_geometric Data objects, and handles dataset splits.
    """

    def __init__(self, root: str, smiles_list: List[str], properties: Optional[List[float]] = None,
                 transform=None, pre_transform=None):
        """
        Args:
            root: Directory for saving processed dataset.
            smiles_list: List of SMILES strings to load.
            properties: Optional list of target property values corresponding to SMILES.
            transform: Optional transform for data.
            pre_transform: Optional pre-processing transform.
        """
        self.raw_smiles = smiles_list
        self.properties = properties
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Placeholder, since raw data is provided as list directly
        return []

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def download(self):
        # No download since data is provided as input
        pass

    def process(self):
        data_list = []

        for idx, smi in enumerate(self.raw_smiles):
            mol = None
            try:
                # Convert SMILES string to RDKit molecule
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue  # skip invalid SMILES
                # Sanitize molecule
                Chem.SanitizeMol(mol)
            except Exception:
                continue  # skip molecules that raise errors during sanitization

            # Check for validity: ensure molecule is not empty or malformed
            if mol is None:
                continue

            # Optional: further filtering for disconnected structures, radicals, etc.
            if not self.is_valid_molecule(mol):
                continue

            # Convert molecule to graph data
            node_feat, edge_index, edge_attr = self.mol_to_graph_data(mol)
            if node_feat is None or edge_index is None:
                continue  # skip if conversion failed

            # Prepare data object
            data = Data(x=torch.tensor(node_feat, dtype=torch.float),
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        mol_smi=smi,
                        mol=mol,
                        index=idx)

            # Attach property if available
            if self.properties is not None:
                prop_value = self.properties[idx]
                data.y = torch.tensor([prop_value], dtype=torch.float)
            else:
                data.y = None

            data_list.append(data)

        # Save the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def is_valid_molecule(self, mol: Chem.Mol) -> bool:
        """
        Checks basic validity of molecule, e.g., sanitization, size, connectivity.
        """
        try:
            # Check molecule's basic properties
            if mol.GetNumAtoms() == 0:
                return False
            # Optional: check for radicals, disconnected parts, etc.
            # For simplicity, skip molecules with radicals
            for atom in mol.GetAtoms():
                if atom.GetExplicitRadicalCount() > 0:
                    return False
            # Check for valence issues
            Chem.SanitizeMol(mol)
            return True
        except Exception:
            return False

    def mol_to_graph_data(self, mol: Chem.Mol) -> Tuple[List[List[float]], torch.Tensor, torch.Tensor]:
        """
        Converts RDKit molecule to graph features suitable for torch_geometric.
        Features: atom types encoded as one-hot vectors or atomic number.
        Edges: bond connectivity in COO format, with bond types as attributes.
        """
        atom_features = []
        for atom in mol.GetAtoms():
            atom_feat = self.atom_feature_vector(atom)
            atom_features.append(atom_feat)

        # Edge list
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            # Add both directions for undirected graph
            edge_index.append([i, j])
            edge_index.append([j, i])
            bond_feat = self.bond_feature_vector(bond)
            edge_attr.append(bond_feat)
            edge_attr.append(bond_feat)

        if len(edge_index) == 0:
            # Molecules with no bonds (e.g., single atom)
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, self.bond_feature_dim()), dtype=torch.float)
        else:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)

        return atom_features, edge_index_tensor, edge_attr_tensor

    def atom_feature_vector(self, atom: Chem.Atom) -> List[float]:
        """
        Encodes atom features, e.g., atomic number, hybridization, aromaticity.
        """
        # Example feature vector: one-hot encoding of atomic number (limited set), aromaticity
        atomic_num = atom.GetAtomicNum()
        aromatic = atom.GetIsAromatic()
        hybridization = atom.GetHybridization()

        # For simplicity, encode atomic number as one-hot for common atoms
        # Define a set of common atoms for one-hot (can extend as needed)
        common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
        atom_one_hot = [1.0 if atomic_num == num else 0.0 for num in common_atoms]

        # Aromaticity as binary feature
        aromatic_feature = [1.0 if aromatic else 0.0]

        # Hybridization one-hot
        hybridization_list = [
            Chem.Hybridization.SP, Chem.Hybridization.SP2,
            Chem.Hybridization.SP3, Chem.Hybridization.SP3D,
            Chem.Hybridization.SP3D2
        ]
        hybrid_hot = [1.0 if hybridization == h else 0.0 for h in hybridization_list]

        feature_vector = atom_one_hot + aromatic_feature + hybrid_hot
        return feature_vector

    def bond_feature_vector(self, bond: Chem.Bond) -> List[float]:
        """
        Encodes bond features, e.g., bond type, conjugation, ring status.
        """
        bond_type_mapping = {
            Chem.BondType.SINGLE: [1, 0, 0, 0],
            Chem.BondType.DOUBLE: [0, 1, 0, 0],
            Chem.BondType.TRIPLE: [0, 0, 1, 0],
            Chem.BondType.AROMATIC: [0, 0, 0, 1],
            Chem.BondType.SINGLE**: [1, 0, 0, 0],
        }
        bond_type = bond.GetBondType()
        bond_feat = bond_type_mapping.get(bond_type, [0, 0, 0, 0])
        # Optionally add conjugation and ring features
        conjugated = 1.0 if bond.GetIsConjugated() else 0.0
        in_ring = 1.0 if bond.IsInRing() else 0.0
        return bond_feat + [conjugated, in_ring]

    def bond_feature_dim(self) -> int:
        """
        Returns the dimension of bond feature vector.
        """
        # 4 (bond types) + 1 (conjugation) + 1 (ring) = 6
        return 6

def load_smiles_from_file(file_path: str) -> List[str]:
    """
    Loads SMILES strings from a text file, each line a SMILES.
    """
    smiles = []
    with open(file_path, 'r') as f:
        for line in f:
            smi = line.strip()
            if smi:
                smiles.append(smi)
    return smiles

def create_datasets_from_smiles(smiles_list: List[str],
                                properties: Optional[List[float]],
                                train_ratio: float = 0.8,
                                val_ratio: float = 0.1,
                                test_ratio: float = 0.1,
                                dataset_root: str = "data") -> Tuple[Dataset, Dataset, Dataset]:
    """
    Creates train, validation, and test datasets from list of SMILES and optional properties.
    """
    total_samples = len(smiles_list)
    indices = list(range(total_samples))
    random.shuffle(indices)

    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def subset_smiles(idxs: List[int]) -> Tuple[List[str], Optional[List[float]]]:
        return [smiles_list[i] for i in idxs], (
            [properties[i] for i in idxs] if properties is not None else None
        )

    train_smiles, train_props = subset_smiles(train_idx)
    val_smiles, val_props = subset_smiles(val_idx)
    test_smiles, test_props = subset_smiles(test_idx)

    train_dataset = MoleculeDataset(os.path.join(dataset_root, "train"),
                                    train_smiles, train_props)
    val_dataset = MoleculeDataset(os.path.join(dataset_root, "val"),
                                  val_smiles, val_props)
    test_dataset = MoleculeDataset(os.path.join(dataset_root, "test"),
                                   test_smiles, test_props)

    return train_dataset, val_dataset, test_dataset

def load_dataset(config: dict) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Main function to load, process, and split dataset based on configuration.
    """
    dataset_path = config.get("dataset_path", "data/ZINC250k.smi")
    train_ratio = config.get("train_split_ratio", 0.8)
    val_ratio = config.get("val_split_ratio", 0.1)
    test_ratio = config.get("test_split_ratio", 0.1)

    # Load raw SMILES data
    smiles_list = load_smiles_from_file(dataset_path)

    # No default properties provided; assuming none or properties are used downstream
    properties = None

    # Create datasets with splits
    train_dataset, val_dataset, test_dataset = create_datasets_from_smiles(
        smiles_list, properties, train_ratio, val_ratio, test_ratio
    )

    return train_dataset, val_dataset, test_dataset
