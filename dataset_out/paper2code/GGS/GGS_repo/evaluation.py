# evaluation.py
import numpy as np
from typing import List, Optional, Dict
from sklearn.metrics import pairwise_distances

class Evaluation:
    """
    Evaluation class for assessing protein sequence generation quality,
    predictive accuracy, diversity, and extrapolation Metrics.
    """

    def __init__(
        self,
        predictor,
        sequences: List[str],
        true_fitnesses: Optional[np.ndarray] = None,
        train_sequences: Optional[List[str]] = None,
        train_fitnesses: Optional[np.ndarray] = None,
        config: Optional[dict] = None
    ):
        """
        Initialize Evaluation object.
        Args:
            predictor: Predictor model with methods `predict_batch()` (and optionally `predict()`).
            sequences (List[str]): Sequences to evaluate.
            true_fitnesses (Optional[np.ndarray]): Ground-truth fitness scores for sequences.
            train_sequences (Optional[List[str]]): Training sequences for novelty/extrapolation.
            train_fitnesses (Optional[np.ndarray]): Ground truth for training data.
            config (Optional[dict]): Configuration dictionary, if needed for metrics preferences.
        """
        self.predictor = predictor
        self.sequences = sequences
        self.true_fitnesses = true_fitnesses
        self.train_sequences = train_sequences
        self.train_fitnesses = train_fitnesses
        self.config = config if config is not None else {}
        # Storage for results
        self.results: Dict[str, float] = {}

    def evaluate_fitness(self):
        """
        Predict fitness for the sequences and compute max and median metrics.
        """
        predicted_fits = self.predictor.predict_batch(self.sequences)
        # Store max and median fitness predictions
        max_fit = np.max(predicted_fits)
        median_fit = np.median(predicted_fits)
        self.results['max_fitness'] = max_fit
        self.results['median_fitness'] = median_fit

        # If true fitnesses are provided, compute correlation or MAE
        if self.true_fitnesses is not None:
            predicted_true = self.predictor.predict_batch(self.sequences)
            mae = np.mean(np.abs(predicted_true - self.true_fitnesses))
            self.results['mae'] = mae

    def evaluate_extrapolation(self):
        """
        Evaluate predictor's extrapolation ability by MAE on train and hold-out data.
        """
        if self.true_fitnesses is None or self.train_sequences is None or self.train_fitnesses is None:
            # Cannot compute extrapolation metrics without ground truth
            return
        train_pred = self.predictor.predict_batch(self.train_sequences)
        holdout_pred = self.predictor.predict_batch(self.sequences)

        train_mae = np.mean(np.abs(train_pred - self.train_fitnesses))
        holdout_mae = np.mean(np.abs(holdout_pred - self.true_fitnesses))
        self.results['train_mae'] = train_mae
        self.results['holdout_mae'] = holdout_mae

    def compute_sequence_distance(self, seq1: str, seq2: str) -> int:
        """
        Compute Levenshtein distance between two sequences.
        For simplicity, using difflib if Levenshtein library isn't available.
        """
        import difflib
        seqmatch = difflib.SequenceMatcher(None, seq1, seq2)
        match_blocks = seqmatch.get_matching_blocks()
        matches = sum(n for _, _, n in match_blocks)
        return max(len(seq1), len(seq2)) - matches

    def compute_diversity(self):
        """
        Compute the median of all pairwise distances between sampled sequences.
        """
        if len(self.sequences) < 2:
            self.results['diversity'] = 0.0
            return
        n = len(self.sequences)
        dist_list = []
        for i in range(n):
            for j in range(i+1, n):
                dist = self.compute_sequence_distance(self.sequences[i], self.sequences[j])
                dist_list.append(dist)
        median_distance = np.median(dist_list)
        self.results['diversity'] = median_distance

    def compute_novelty(self):
        """
        Compute median of minimal distances from sampled sequences to training set sequences.
        """
        if self.train_sequences is None or len(self.train_sequences) == 0:
            self.results['novelty'] = 0.0
            return
        min_dists = []
        for seq in self.sequences:
            dists = []
            for train_seq in self.train_sequences:
                d = self.compute_sequence_distance(seq, train_seq)
                dists.append(d)
            min_dists.append(np.min(dists))
        median_min_dist = np.median(min_dists)
        self.results['novelty'] = median_min_dist

    def evaluate(self):
        """
        Run complete evaluation: fitness stats, extrapolation, diversity, novelty.
        """
        self.evaluate_fitness()
        self.compute_diversity()
        self.compute_novelty()
        return self.results
