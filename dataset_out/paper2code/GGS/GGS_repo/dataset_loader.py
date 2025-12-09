## dataset_loader.py
import os
import csv
import json
import numpy as np
from typing import List, Tuple
from biopython import SeqIO
from Bio.Seq import Seq

# Optional: For Levenshtein distance; if not installed, can implement manually or use difflib
try:
    import Levenshtein
except ImportError:
    # Fallback to a simple implementation or use difflib
    import difflib

    def levenshtein_distance(seq1: str, seq2: str) -> int:
        """
        Compute Levenshtein edit distance via difflib SequenceMatcher as approximation.
        """
        seqmatch = difflib.SequenceMatcher(None, seq1, seq2)
        # Levenshtein distance is approximated by:
        # total length minus twice the number of matching blocks
        matches = sum(n for _, _, n in seqmatch.get matching_blocks())
        return max(len(seq1), len(seq2)) - matches
else:
    def levenshtein_distance(seq1: str, seq2: str) -> int:
        return Levenshtein.distance(seq1, seq2)

# Define standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids

class DatasetLoader:
    def __init__(self, dataset_path: str, dataset_name: str, filters: dict, config: dict):
        """
        Initialize DatasetLoader.
        :param dataset_path: Path to raw dataset file.
        :param dataset_name: Name of dataset: 'GFP' or 'AAV'.
        :param filters: Dict with keys 'percentile_range' and 'mutational_gap'.
        :param config: Full configuration dict for dataset filtering details.
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.filters = filters
        self.config = config
        self.sequences: List[str] = []
        self.fitnesses: List[float] = []

    def load_data(self):
        """
        Load dataset from file. Supports CSV with columns: sequence, fitness.
        Extend this method if datasets are in other formats.
        """
        sequences = []
        fitnesses = []

        # Support CSV format: assume columns 'sequence', 'fitness'
        if self.dataset_path.endswith('.csv'):
            with open(self.dataset_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    seq = row.get('sequence') or row.get('Sequence') or row.get('seq')
                    fit_str = row.get('fitness') or row.get('Fitness') or row.get('score')
                    if seq is None or fit_str is None:
                        continue
                    try:
                        fit = float(fit_str)
                        sequences.append(seq.upper())
                        fitnesses.append(fit)
                    except:
                        continue
        elif self.dataset_path.endswith('.json'):
            with open(self.dataset_path, 'r') as f:
                data = json.load(f)
                # Expect list of dicts with 'sequence' and 'fitness'
                for item in data:
                    seq = item.get('sequence')
                    fit = item.get('fitness')
                    if seq is None or fit is None:
                        continue
                    sequences.append(seq.upper())
                    fitnesses.append(float(fit))
        else:
            # Support for other formats e.g., fasta or fasta-like
            # For simplicity, assume CSV/JSON; user extend as needed
            raise NotImplementedError(f"Unsupported dataset format: {self.dataset_path}")

        self.sequences = sequences
        self.fitnesses = fitnesses

    def get_top_sequences_by_fitness(self, top_fraction=0.01) -> List[Tuple[str, float]]:
        """
        Return top sequences based on fitness.
        :param top_fraction: Fraction of dataset to consider as top (e.g., 0.01 for top 1%)
        """
        num_top = max(1, int(len(self.fitnesses) * top_fraction))
        sorted_indices = np.argsort(self.fitnesses)[::-1]  # descending
        top_indices = sorted_indices[:num_top]
        return [(self.sequences[i], self.fitnesses[i]) for i in top_indices]

    def get_percentile_bounds(self, percentile_range: Tuple[int, int]) -> Tuple[float, float]:
        """
        Compute lower and upper fitness bounds based on percentile range.
        """
        percentiles = percentile_range
        low_pct, high_pct = percentiles
        lower_bound = np.percentile(self.fitnesses, low_pct)
        upper_bound = np.percentile(self.fitnesses, high_pct)
        return lower_bound, upper_bound

    def filter_by_percentile(self, lower_bound: float, upper_bound: float):
        """
        Filter datasets to sequences with fitness within bounds.
        """
        filtered_seqs = []
        filtered_fits = []
        for seq, fit in zip(self.sequences, self.fitnesses):
            if lower_bound <= fit <= upper_bound:
                filtered_seqs.append(seq)
                filtered_fits.append(fit)
        self.sequences = filtered_seqs
        self.fitnesses = filtered_fits

    def compute_mutational_distances(self, reference_sequences: List[str]) -> List[int]:
        """
        Compute minimal mutational distance of each sequence to reference sequences.
        """
        distances = []
        for seq in self.sequences:
            min_dist = np.inf
            for ref_seq in reference_sequences:
                dist = levenshtein_distance(seq, ref_seq)
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)
        return distances

    def filter_by_mutational_gap(self, min_gap: int, reference_sequences: List[str]) -> List[int]:
        """
        Keep only sequences at least min_gap mutations away from references.
        """
        distances = self.compute_mutational_distances(reference_sequences)
        filtered_seqs = []
        filtered_fits = []
        for seq, fit, dist in zip(self.sequences, self.fitnesses, distances):
            if dist >= min_gap:
                filtered_seqs.append(seq)
                filtered_fits.append(fit)
        self.sequences = filtered_seqs
        self.fitnesses = filtered_fits
        return distances

    def get_filtered_dataset(self, difficulty_level: str) -> Tuple[List[str], List[float]]:
        """
        Main method to get filtered sequences and fitnesses based on difficulty.
        """
        # Load raw data
        self.load_data()

        # Get percentile bounds
        lower_bound, upper_bound = self.get_percentile_bounds(self.filters['percentile_range'])

        # Filter by fitness percentile
        self.filter_by_percentile(lower_bound, upper_bound)

        # Identify top sequences for mutational distance comparisons
        top_seqs = [seq for seq, fit in self.get_top_sequences_by_fitness()]

        # Determine mutational gap filter based on difficulty
        mut_gap = self.filters['mutational_gap']
        # For 'easy', often mut_gap=0; for 'medium/hard', use specified values
        # Filter sequences by mutational gap
        if mut_gap > 0:
            self.filter_by_mutational_gap(mut_gap, top_seqs)

        # Additional filtering or data adjustments can be added here if needed

        return self.sequences, self.fitnesses
