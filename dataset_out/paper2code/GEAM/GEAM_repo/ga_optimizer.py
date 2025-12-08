## ga_optimizer.py

import random
import copy
from typing import List, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from dataset_loader import MoleculeDataset  # Assuming molecules are RDKit Mol objects
from model import molecule_to_data  # Conversion utility
from evaluation import evaluate_properties  # External oracle for scoring


class GAOptimizer:
    """
    Genetic Algorithm module for molecule evolution in the GEAM framework.
    Implements population initialization, selection, crossover, mutation, and reproduction.
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration parameters from "config.yaml"
                - 'population_size': int
                - 'mutation_rate': float
                - 'max_size': int
                - 'reproduction_count': int
        """
        self.population_size: int = int(config.get('training', {}).get('population_size', 100))
        self.mutation_rate: float = float(config.get('training', {}).get('mutation_rate', 0.1))
        self.max_size: int = int(config.get('training', {}).get('molecule_max_size', 40))
        self.reproduction_count: int = int(config.get('training', {}).get('num_reproduction', 3))
        # Initialize empty population and scores
        self.population: List[Chem.Mol] = []
        self.population_scores: List[float] = []

    def initialize_population(self, init_molecules: List[Chem.Mol]):
        """
        Initialize population with given molecules, filtering invalid.
        """
        valid_molecules = []
        for mol in init_molecules:
            if self.validate_molecule(mol):
                valid_molecules.append(mol)
            if len(valid_molecules) >= self.population_size:
                break
        self.population = valid_molecules
        # Initialize scores as None or evaluate later
        self.population_scores = [None] * len(self.population)

    def validate_molecule(self, mol: Chem.Mol) -> bool:
        """
        Check if molecule is chemically valid after sanitization.
        """
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol)
            if mol.GetNumAtoms() == 0:
                return False
            # Optionally, avoid radicals or invalid valences
            for atom in mol.GetAtoms():
                if atom.GetExplicitRadicalCount() > 0:
                    return False
            return True
        except:
            return False

    def select_parents(self, top_fraction: float = 0.2) -> List[Tuple[Chem.Mol, Chem.Mol]]:
        """
        Select parent pairs based on property scores.
        Args:
            top_fraction (float): fraction of top molecules to consider for selection
        Returns:
            List of tuples (Mol, Mol): parent pairs for crossover
        """
        # If scores are unknown, assign uniform probabilities
        if any(s is None for s in self.population_scores):
            probs = [1.0 / len(self.population) for _ in self.population]
        else:
            # Use scores for weighted selection; higher score = higher chance
            scores_array = np.array([s if s is not None else 0.0 for s in self.population_scores])
            # To avoid negative/zero, shift scores
            min_score = scores_array.min()
            adjusted_scores = scores_array - min_score + 1e-6
            probs = adjusted_scores / adjusted_scores.sum()

        # Select top fraction as best parents
        num_parents = max(2, int(len(self.population) * top_fraction))
        indices = np.random.choice(len(self.population), size=num_parents, replace=False, p=probs)
        parents = [self.population[i] for i in indices]

        # Generate parent pairs (shuffle and pair)
        random.shuffle(parents)
        parent_pairs = []
        for i in range(0, len(parents) - 1, 2):
            parent_pairs.append((parents[i], parents[i + 1]))
        return parent_pairs

    def crossover(self, parent1: Chem.Mol, parent2: Chem.Mol) -> Chem.Mol:
        """
        Perform molecule crossover based on Jensen (2019).
        - Randomly select bonds to cut, swap parts.
        - Validate resulting molecule.
        Returns:
            offspring molecule or None if invalid
        """
        try:
            mol1 = copy.deepcopy(parent1)
            mol2 = copy.deepcopy(parent2)

            # Identify candidate bonds for cutting
            bonds1 = [b for b in mol1.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE]
            bonds2 = [b for b in mol2.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE]
            if not bonds1 or not bonds2:
                return None

            # Randomly select cut bonds
            b1 = random.choice(bonds1)
            b2 = random.choice(bonds2)

            # Cut bonds and split molecules
            fragment1_a, fragment1_b = self.split_molecule_at_bond(mol1, b1)
            fragment2_a, fragment2_b = self.split_molecule_at_bond(mol2, b2)
            if not all([fragment1_a, fragment1_b, fragment2_a, fragment2_b]):
                return None

            # Swap fragments to create offspring
            off1 = self.combine_fragments(fragment1_a, fragment2_b)
            off2 = self.combine_fragments(fragment2_a, fragment1_b)

            # Validate
            for mol in [off1, off2]:
                if mol and self.validate_molecule(mol):
                    Chem.SanitizeMol(mol)
                    return mol
            return None
        except:
            return None

    def split_molecule_at_bond(self, mol: Chem.Mol, bond):
        """
        Cut molecule at specified bond, return two fragments.
        """
        # Mark atoms to keep
        atom_indices = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

        # Break bond
        rw_mol = Chem.RWMol(mol)
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        try:
            rw_mol.RemoveBond(idx1, idx2)
            Chem.SanitizeMol(rw_mol)
        except:
            return None, None

        # Extract fragments
        frags = Chem.GetMolFrags(rw_mol, asMols=True, sanitize=True)
        # Determine which fragment contains atom1 and atom2
        frag1, frag2 = None, None
        for f in frags:
            atoms_in_f = set([atom.GetIdx() for atom in f.GetAtoms()])
            if idx1 in atoms_in_f:
                frag1 = f
            if idx2 in atoms_in_f:
                frag2 = f
        return frag1, frag2

    def combine_fragments(self, frag_a: Chem.Mol, frag_b: Chem.Mol) -> Chem.Mol:
        """
        Merge two fragments into a molecule, trying to connect via a new bond.
        """
        if frag_a is None or frag_b is None:
            return None
        try:
            combo = Chem.CombineMols(frag_a, frag_b)
            rw_combo = Chem.RWMol(combo)

            # Identify candidates for bonding: for simplicity, connect first atoms
            atom_idx_a = 0
            atom_idx_b = frag_a.GetNumAtoms()  # start index of second fragment

            # Add a single bond
            rw_combo.AddBond(atom_idx_a, atom_idx_b, Chem.BondType.SINGLE)
            combined_mol = rw_combo.GetMol()
            Chem.SanitizeMol(combined_mol)
            return combined_mol
        except:
            return None

    def mutate(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Apply random mutation: atom replacement, bond addition/removal.
        """
        try:
            mol_rw = Chem.RWMol(mol)
            atoms = mol_rw.GetAtoms()
            num_atoms = mol_rw.GetNumAtoms()
            if num_atoms == 0:
                return None
            # Random mutation type
            mutation_type = random.choice(['atom_replace', 'add_bond', 'remove_bond'])

            if mutation_type == 'atom_replace':
                atom_idx = random.randint(0, num_atoms - 1)
                atom = mol_rw.GetAtomWithIdx(atom_idx)
                # Replace atom with another valid atom, e.g., carbon
                atom.SetAtomicNum(6)
            elif mutation_type == 'add_bond':
                if num_atoms > 1:
                    a1_idx, a2_idx = random.sample(range(num_atoms), 2)
                    # Ensure no existing bond
                    if not mol_rw.GetBondBetweenAtoms(a1_idx, a2_idx):
                        mol_rw.AddBond(a1_idx, a2_idx, Chem.BondType.SINGLE)
            elif mutation_type == 'remove_bond':
                bonds = mol_rw.GetBonds()
                if bonds:
                    bond = random.choice(bonds)
                    mol_rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

            mutated_mol = mol_rw.GetMol()
            if self.validate_molecule(mutated_mol):
                Chem.SanitizeMol(mutated_mol)
                return mutated_mol
            else:
                return None
        except:
            return None

    def run_cycle(self, current_population: List[Chem.Mol], property_scores: List[float]) -> Tuple[List[Chem.Mol], List[float]]:
        """
        Run a full reproduction cycle:
        - Select parents
        - Generate offspring via crossover
        - Mutate offspring
        - Evaluate
        - Update population
        Returns:
            new_population, new_scores
        """
        new_population = []
        new_scores = []

        # Sort population by scores if available
        if property_scores and all(s is not None for s in property_scores):
            sorted_indices = np.argsort(property_scores)[::-1]
            top_ind = sorted_indices[:self.population_size]
            population_sorted = [current_population[i] for i in top_ind]
        else:
            # Random selection if scores unavailable
            population_sorted = current_population

        # Prepare parent pairs
        parent_pairs = []
        for _ in range(self.reproduction_count):
            parent1, parent2 = random.sample(population_sorted, 2)
            parent_pairs.append((parent1, parent2))

        # Generate offspring through crossover and mutation
        for p1, p2 in parent_pairs:
            child = self.crossover(p1, p2)
            if child is None:
                child = random.choice([p1, p2])  # fallback
            # mutate with certain probability
            if random.random() < self.mutation_rate:
                mutated_child = self.mutate(child)
                if mutated_child is not None:
                    child = mutated_child

            # Validate and append
            if self.validate_molecule(child):
                # Check size constraint
                if child.GetNumAtoms() <= self.max_size:
                    # Evaluate property via oracle
                    score = evaluate_properties(child)
                    new_population.append(child)
                    new_scores.append(score)

        # Optionally, fill remaining slots with random selections
        while len(new_population) < self.population_size:
            # fill with random from existing top population
            mol = random.choice(population_sorted)
            if mol.GetNumAtoms() <= self.max_size:
                score = evaluate_properties(mol)
                new_population.append(mol)
                new_scores.append(score)

        return new_population, new_scores
