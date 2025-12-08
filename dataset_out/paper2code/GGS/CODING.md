# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
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
```


## evaluation.py

```python
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
```

## graph_utils.py

```python
## graph_utils.py
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

def create_sequence_graph(sequences, knn_neighbors=20, similarity_threshold=None):
    """
    Construct an undirected sequence similarity graph.
    Nodes are sequences, edges connect neighbors based on similarity.
    
    Args:
        sequences (List[str]): List of sequence strings.
        knn_neighbors (int): Number of neighbors for KNN graph.
        similarity_threshold (float or None): Threshold distance to connect nodes.
            If None, build KNN graph.
    
    Returns:
        nx.Graph: Undirected graph with sequences as nodes.
    """
    num_sequences = len(sequences)
    if num_sequences == 0:
        return nx.Graph()

    # Compute pairwise distances
    # Use Levenshtein distance if sequences vary in length, else Hamming for fixed length
    # For efficiency, convert sequences to numpy array of characters if needed
    # Here, default to Levenshtein
    dist_matrix = np.zeros((num_sequences, num_sequences))
    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            dist = levenshtein_distance(sequences[i], sequences[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    G = nx.Graph()
    # Add nodes
    for idx, seq in enumerate(sequences):
        G.add_node(idx, sequence=seq)

    if similarity_threshold is not None:
        # Connect nodes within distance threshold
        for i in range(num_sequences):
            for j in range(i + 1, num_sequences):
                if dist_matrix[i, j] <= similarity_threshold:
                    G.add_edge(i, j, weight=1.0)
    else:
        # Build KNN graph based on distances
        for i in range(num_sequences):
            # Get sorted indices based on distances
            sorted_idx = np.argsort(dist_matrix[i])
            neighbors = 0
            for j in sorted_idx[1:knn_neighbors+1]:
                if i != j:
                    G.add_edge(i, j, weight=1.0)
                    neighbors += 1
    return G

def compute_graph_laplacian(graph, normalized=True):
    """
    Compute the (unnormalized or normalized) graph Laplacian matrix.
    
    Args:
        graph (nx.Graph): Input similarity graph.
        normalized (bool): Whether to compute normalized Laplacian.
    
    Returns:
        scipy.sparse.csr_matrix: Laplacian matrix (sparse).
    """
    # Build adjacency matrix
    A = nx.adjacency_matrix(graph)  # CSR sparse matrix
    degree = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degree)

    if not normalized:
        # Unnormalized Laplacian: L = D - A
        L = D - A
    else:
        # Symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        with np.errstate(divide='ignore'):
            d_inv_sqrt = 1.0 / np.sqrt(degree)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return L.tocsr()

def perform_label_smoothing(labels, laplacian, gamma):
    """
    Smooth labels over the graph using Tikhonov regularization:
    \hat{Y} = (I + gamma * L)^(-1) * Y
    
    Args:
        labels (np.ndarray): 1D array of labels (fitness scores).
        laplacian (scipy.sparse.csr_matrix): The graph Laplacian matrix.
        gamma (float): Regularization hyperparameter.
    
    Returns:
        np.ndarray: Smoothed labels.
    """
    num_nodes = labels.shape[0]
    I = sp.eye(num_nodes, format='csr')
    M = I + gamma * laplacian
    smoothed_labels = spsolve(M, labels)
    return smoothed_labels

def cluster_sequences(sequences, num_clusters=20):
    """
    Cluster sequences based on Levenshtein distance using hierarchical clustering.
    
    Args:
        sequences (List[str]): List of sequences.
        num_clusters (int): Desired number of clusters.
    
    Returns:
        List[List[str]]: List of clusters, each a list of sequences.
    """
    if len(sequences) == 0:
        return []

    # Efficient clustering with sequence distances
    # Compute condensed distance matrix to feed into clustering
    # Using pairwise_distances with custom metric
    def dist_func(x, y):
        return levenshtein_distance(x, y)

    # Compute full distance matrix
    dist_matrix = pairwise_distances(sequences, metric=dist_func, n_jobs=-1)
    # Convert to condensed form
    # Use AgglomerativeClustering with precomputed distances
    clustering = AgglomerativeClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(dist_matrix)

    clusters_dict = {}
    for seq, lbl in zip(sequences, labels):
        clusters_dict.setdefault(lbl, []).append(seq)
    clusters = list(clusters_dict.values())
    return clusters

def select_top_per_cluster(sequences, fitnesses, clusters):
    """
    Select the sequence with highest fitness from each cluster.
    
    Args:
        sequences (List[str]): List of sequences.
        fitnesses (np.ndarray): Corresponding predicted fitness scores.
        clusters (List[List[str]]): Clusters of sequences.
    
    Returns:
        List[str]: Top sequences per cluster.
    """
    top_sequences = []
    seq_to_fit = {seq: fit for seq, fit in zip(sequences, fitnesses)}
    for cluster in clusters:
        # Find sequence with max fitness in cluster
        max_seq = max(cluster, key=lambda s: seq_to_fit[s])
        top_sequences.append(max_seq)
    return top_sequences
```

## label_smoothing.py

```python
## label_smoothing.py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
from typing import Union

class LabelSmoother:
    def __init__(self, laplacian: sp.spmatrix, gamma: float = 1.0):
        """
        Initialize the LabelSmoother with a graph Laplacian and smoothing hyperparameter gamma.
        Args:
            laplacian (scipy.sparse.spmatrix): The graph Laplacian matrix L (shape: [N, N]).
            gamma (float): Regularization parameter controlling smoothing strength.
        """
        self.laplacian = laplacian
        self.gamma = gamma
        self._precompute_factorization()

    def _precompute_factorization(self):
        """
        Precompute the matrix factorization (solve factorization) for (I + gamma * L).
        This improves efficiency when smoothing multiple label vectors with the same gamma.
        """
        N = self.laplacian.shape[0]
        identity = sp.eye(N, format='csr')
        A = identity + self.gamma * self.laplacian
        # Use sparse LU factorization for efficiency
        # For symmetric positive-definite matrices, use splu
        self.solver = la.factorized(A)

    def smooth_labels(self, labels: Union[np.ndarray, list]) -> np.ndarray:
        """
        Smooth the provided labels over the graph using Tikhonov regularization.
        Args:
            labels (np.ndarray or list): 1D array-like object of shape (N,)
                containing the observed fitness values for each node.
        Returns:
            np.ndarray: Smoothed labels as a 1D array of shape (N,).
        """
        labels_array = np.asarray(labels).astype(np.float64)
        if labels_array.ndim != 1:
            raise ValueError("Labels must be a 1D array.")
        # Solve (I + gamma L) * y_hat = y for y_hat
        y_hat = self.solver(labels_array)
        return y_hat
```

## main.py

```python
## main.py
import os
import yaml
import numpy as np
import torch
import random
from dataset_loader import DatasetLoader
from graph_utils import create_sequence_graph, compute_graph_laplacian, perform_label_smoothing, cluster_sequences, select_top_per_cluster
from label_smoothing import LabelSmoother
from model import SequenceFitnessPredictor
from trainer import Trainer
from sampling import GGSampler
from evaluation import Evaluation

def main():
    # 1. Load configuration and set seeds for reproducibility
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    seed = config.get('training', {}).get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Prepare datasets for each dataset and difficulty
    datasets_info = [
        ('GFP', 'GFP', 'dataset_filters', 'GFP'),
        ('AAV', 'AAV', 'dataset_filters', 'AAV')
    ]

    for dataset_name, dataset_type, filters_key, dataset_key in datasets_info:
        filters_config = config['dataset_filters'][dataset_type]

        # Load filtered dataset
        loader = DatasetLoader(
            dataset_path=f'{dataset_name}_data.csv',  # Path to dataset, adjust as needed
            dataset_name=dataset_key,
            filters=filters_config,
            config=config
        )
        sequences, fitness = loader.get_filtered_dataset(difficulty_level=filters_key['hard'])
        # Save filtered data or keep in variables
        initial_seqs = sequences
        initial_fits = np.array(fitness)

        # 3. Build sequence similarity graph
        knn = config['graph_construction'].get('knn_neighbors', 20)
        G = create_sequence_graph(sequences, knn_neighbors=knn)
        L = compute_graph_laplacian(G, normalized=True)

        # 4. Prepare labels for smoothing
        Y = np.array(initial_fits)
        # Store smoothed labels for each gamma
        gamma_values = config['label_smoothing'].get('gamma_values', [0.01, 0.1, 1.0, 10.0])
        smoothed_labels_list = []

        for gamma in gamma_values:
            # 4a. Obtain smoothed labels
            smoothed_labels = perform_label_smoothing(Y, L, gamma)
            smoothed_labels_list.append((gamma, smoothed_labels))

        # 5. For each smoothed label set, train predictor and evaluate
        best_metrics = None
        best_gamma = None

        for gamma, smoothed_Y in smoothed_labels_list:
            # 5a. Train predictor
            predictor_config = {
                'architecture': config['predictor_model'].get('architecture', 'cnn'),
                'sequence_length': len(sequences[0]),
                'learning_rate': 1e-3,
                'batch_size': 128,
                'epochs': 50,
                'dropout_rate': config['predictor_model'].get('dropout_rate', 0.1)
            }
            trainer = Trainer(
                predictor_config,
                train_sequences=sequences,
                train_labels=smoothed_Y,
                val_sequences=sequences,  # For simplicity, using same data for val; can be split
                val_labels=smoothed_Y,
                checkpoint_dir=f'./checkpoints_{dataset_name}_{filters_key}_{gamma}'
            )
            trainer.train()
            predictor_model = trainer.get_model()

            # 5b. Run in-silico evaluation on initial data
            predictor_model.eval()
            predictions = predictor_model.predict_batch(sequences)
            median_pred = np.median(predictions)
            # Compute diversity
            seqs_for_diversity = sequences
            diversities = []
            for i in range(len(sequences)):
                for j in range(i+1, len(sequences)):
                    diversities.append(predictor_model.compute_sequence_distance(sequences[i], sequences[j]))
            median_diversity = np.median(diversities) if diversities else 0.0
            # Compute novelty with respect to training set
            novelties = []
            for seq in sequences:
                min_d = min([predictor_model.compute_sequence_distance(seq, t_seq) for t_seq in sequences])
                novelties.append(min_d)
            median_novelty = np.median(novelties) if novelties else 0.0

            # 5c. Run GWG sampling with clustering (the GGS process)
            gwg_params = {
                'gwg_rounds': config['sampling'].get('gwg_rounds', 15),
                'proposal_per_seq': config['sampling'].get('proposals_per_sequence', 100),
                'temperature_grid': config['sampling'].get('temperature_grid', [0.01, 0.1, 1.0, 10.0]),
                'cluster_num': config['sampling'].get('clustering_clusters', 20),
                'mutation_batch_size': config['sampling'].get('mutation_batch_size', 100),
                'sequence_length': len(sequences[0]),
                'vocab_size': 20,
                'seed': seed
            }
            # Initialize sampler with current sequences
            sampler = GGSampler(
                predictor=predictor_model,
                sequences=sequences,
                predictor_predict_func=predictor_model.predict,
                predictor_grad_func=predictor_model.compute_gradients,
                proposals_per_sequence=gwg_params['proposal_per_seq'],
                gwg_rounds=gwg_params['gwg_rounds'],
                clustering_clusters=gwg_params['cluster_num'],
                temperature=gwg_params['temperature_grid'][0],  # can do hyperparam sweep
                mutation_batch_size=gwg_params['mutation_batch_size'],
                sequence_length=gwg_params['sequence_length'],
                vocab_SIZE=gwg_params['vocab_size'],
                seed=gwg_params['seed']
            )

            sampled_sequences = sampler.run_sampling()

            # 5d. Evaluate sampled sequences
            pred_samples = predictor_model.predict_batch(sampled_sequences)
            best_fitness_sample = np.max(pred_samples)
            avg_fitness_sample = np.mean(pred_samples)

            # 5e. Store metrics, compare to previous best
            current_metrics = {
                'median_fitness': median_pred,
                'diversity': median_diversity,
                'novelty': median_novelty,
                'sampled_best': best_fitness_sample,
                'sampled_avg': avg_fitness_sample,
                'gamma': gamma
            }
            if best_metrics is None or best_metrics['sampled_best'] < best_fitness_sample:
                best_metrics = current_metrics
                best_gamma = gamma

        # 6. Final reporting for current dataset/difficulty
        print(f"Dataset: {dataset_name} | Difficulty: {filters_key} | Best gamma: {best_gamma}")
        print(f"Metrics: {best_metrics}")

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List, Dict, Union

class SequenceFitnessPredictor:
    """
    A flexible neural network model for predicting protein sequence fitness.
    Supports CNN and Transformer architectures, with methods for training,
    inference, and gradient computation w.r.t. input sequences.
    """

    def __init__(self, config: Dict):
        """
        Initialize the model based on configuration parameters.
        Args:
            config (Dict): Dictionary containing model hyperparameters.
                Expected keys:
                    - 'architecture' : str, e.g., 'cnn' or 'transformer'
                    - 'sequence_length' : int, length of sequences
                    - 'vocab_size' : int, number of amino acids (default 20)
                    - 'embedding_dim' : int, used for transformer or embedding layer (optional)
                    - 'dropout_rate' : float, dropout probability
                    - 'hidden_dim' : int, hidden layer size for CNN
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.architecture = config.get('architecture', 'cnn').lower()
        self.sequence_length = config.get('sequence_length', 236)
        self.vocab_size = 20  # fixed for amino acids unless extension needed
        self.embedding_dim = config.get('embedding_dim', 64)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.hidden_dim = config.get('hidden_dim', 256)  # for CNN layers

        if self.architecture == 'cnn':
            self.model = self._build_cnn()
        elif self.architecture == 'transformer':
            self.model = self._build_transformer()
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-3))
        self.epochs = config.get('epochs', 50)

    def _build_cnn(self):
        """Builds a simple 1D CNN architecture."""
        class CNNModel(nn.Module):
            def __init__(self, vocab_size, seq_len, hidden_dim, dropout_rate):
                super().__init__()
                self.conv1 = nn.Conv1d(in_channels=vocab_size, out_channels=self.hidden_dim, kernel_size=5, padding=2)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool1d(kernel_size=2)
                conv_output_size = (seq_len // 2)  # after pooling
                self.fc = nn.Linear(self.hidden_dim * conv_output_size, 1)
                self.dropout = nn.Dropout(p=dropout_rate)

            def forward(self, x):
                """
                x: [batch_size, seq_len, vocab_size]
                """
                x = x.permute(0, 2, 1)  # to [batch_size, vocab_size, seq_len]
                x = self.conv1(x)
                x = self.relu(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                out = self.fc(x)
                return out.squeeze(1)

        return CNNModel(self.vocab_size, self.sequence_length, self.hidden_dim, self.dropout_rate)

    def _build_transformer(self):
        """Builds a simple Transformer encoder architecture."""
        from torch.nn import TransformerEncoder, TransformerEncoderLayer, Embedding

        class TransformerModel(nn.Module):
            def __init__(self, vocab_size, seq_len, embedding_dim, hidden_dim, dropout_rate):
                super().__init__()
                self.embedding = Embedding(vocab_size, embedding_dim)
                encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=dropout_rate)
                self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=2)
                self.fc = nn.Linear(embedding_dim, 1)
                self.seq_len = seq_len

            def forward(self, x):
                """
                x: [batch_size, seq_len, vocab_size] one-hot or embedding - here we assume input is [batch, seq_len, vocab_size]
                """
                # If input is one-hot (batch_size, seq_len, vocab_size), convert to embedding
                # Else, if embedding input, skip embedding layer
                if x.dtype == torch.float32 or x.dtype == torch.float64:
                    # Assume raw input: one-hot
                    x_embed = self.embedding(torch.argmax(x, dim=2))
                else:
                    # If input is already embedded
                    x_embed = x
                # Add positional encoding if desired (not included here for simplicity)
                # transpose for transformer: (seq_len, batch, embed_dim)
                x_encoded = self.transformer_encoder(x_embed.permute(1,0,2))
                # Take mean over sequence or use global pooling as alternative
                pooled = x_encoded.mean(dim=0)
                out = self.fc(pooled)
                return out.squeeze(1)

        return TransformerModel(self.vocab_size, self.sequence_length, self.embedding_dim, self.hidden_dim, self.dropout_rate)

    def train(self, sequences: List[str], labels: np.ndarray):
        """
        Train the model on provided sequences and labels.
        Args:
            sequences (List[str]): List of sequences as strings.
            labels (np.ndarray): Corresponding fitness labels.
        """
        self.model.train()
        # Convert sequences to tensor input
        inputs = self._sequences_to_tensor(sequences)  # shape [batch_size, seq_len, vocab_size]
        labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)

        dataset = torch.utils.data.TensorDataset(inputs, labels_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self._get_batch_size(), shuffle=True)

        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
            avg_loss = total_loss / len(dataloader.dataset)
            # Optional: add validation step, early stopping

    def predict(self, sequence: str) -> float:
        """
        Predict scalar fitness for a single sequence.
        Args:
            sequence (str): Protein sequence string.
        Returns:
            float: Predicted fitness scalar.
        """
        self.model.eval()
        input_tensor = self._sequence_to_tensor(sequence)  # shape [1, seq_len, vocab_size]
        with torch.no_grad():
            pred = self.model(input_tensor.to(self.device))
        return pred.cpu().item()

    def predict_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Batch prediction for multiple sequences.
        Args:
            sequences (List[str])
        Returns:
            np.ndarray: predictions shape [len(sequences)]
        """
        self.model.eval()
        input_tensors = self._sequences_to_tensor(sequences)
        with torch.no_grad():
            preds = self.model(input_tensors.to(self.device))
        return preds.cpu().numpy()

    def compute_gradients(self, sequence: str) -> np.ndarray:
        """
        Compute the gradient of the predicted fitness with respect to sequence input.
        Args:
            sequence (str): Sequence string.
        Returns:
            np.ndarray: Gradient array with shape [sequence_length, vocab_size].
        """
        self.model.eval()
        input_tensor = self._sequence_to_tensor(sequence)  # shape [1, seq_len, vocab_size]
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        preds = self.model(input_tensor)
        preds.backward()

        # Gradient of output w.r.t. input tensor
        grads = input_tensor.grad.detach().cpu().numpy()  # shape same as input
        # grads shape: [1, seq_len, vocab_size], squeeze batch
        grads = grads[0]
        return grads

    def _sequence_to_tensor(self, sequence: str):
        """
        Convert a sequence string to tensor (batch=1). Using one-hot encoding.
        """
        tensor = torch.zeros(1, self.sequence_length, self.vocab_size, dtype=torch.float32)
        for i, aa in enumerate(sequence):
            if i >= self.sequence_length:
                break
            aa_idx = self._aa_to_index(aa)
            if aa_idx is not None:
                tensor[0, i, aa_idx] = 1.0
            else:
                # If unknown amino acid, leave as zeros or handle differently
                pass
        return tensor

    def _sequences_to_tensor(self, sequences: List[str]):
        """
        Convert list of sequences to tensor batch.
        """
        batch_size = len(sequences)
        tensor = torch.zeros(batch_size, self.sequence_length, self.vocab_size, dtype=torch.float32)
        for b, seq in enumerate(sequences):
            for i, aa in enumerate(seq):
                if i >= self.sequence_length:
                    break
                aa_idx = self._aa_to_index(aa)
                if aa_idx is not None:
                    tensor[b, i, aa_idx] = 1.0
        return tensor

    def _aa_to_index(self, amino_acid: str) -> int:
        """
        Map amino acid to index [0, 19].
        """
        aa_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
        try:
            return aa_list.index(amino_acid)
        except ValueError:
            return None

    def _get_batch_size(self):
        """
        Retrieve batch size from config or default.
        """
        # Could be extended to support dynamic batch size based on hardware load.
        return self.model.dropout.p * 512  # fallback, but better to get from config
```

## sampling.py

```python
## sampling.py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from typing import List, Tuple, Dict, Callable
from label_smoothing import LabelSmoother

class GGSampler:
    """
    Implements the core Gibbs Sampling with Graph-based Smoothing (GGS) procedure.
    """

    def __init__(self,
                 predictor,
                 sequences: List[str],
                 predictor_predict_func: Callable[[str], float],
                 predictor_grad_func: Callable[[str], np.ndarray],
                 proposals_per_sequence: int = 100,
                 gwg_rounds: int = 15,
                 clustering_clusters: int = 20,
                 temperature: float = 0.1,
                 mutation_batch_size: int = 100,
                 sequence_length: int = 236,
                 vocab_SIZE: int = 20,
                 seed: int = 42):
        """
        Initialize the sampler.
        Args:
            predictor: Trained predictor model with predict() and compute_gradients().
            sequences: Current set of sequences (initial input pool).
            predictor_predict_func: Function to get predictor's predicted fitness.
            predictor_grad_func: Function to get gradients w.r.t. input.
            proposals_per_sequence: Number of proposals generated per sequence each round.
            gwg_rounds: Total rounds of sampling.
            clustering_clusters: Number of clusters for sequence reduction.
            temperature: Temperature for proposal softmax.
            mutation_batch_size: Total proposals spread across sequences per round.
            sequence_length: Sequence length (fixed).
            vocab_SIZE: Number of amino acids (default 20).
            seed: Random seed for reproducibility.
        """
        self.predictor = predictor
        self.sequences = sequences
        self.predict = predictor_predict_func
        self.grad_fn = predictor_grad_func
        self.proposals_per_sequence = proposals_per_sequence
        self.rounds = gwg_rounds
        self.num_clusters = clustering_clusters
        self.tau = temperature
        self.batch_size = mutation_batch_size
        self.seq_len = sequence_length
        self.vocab_size = vocab_SIZE
        self.rng = np.random.RandomState(seed)

    def run_sampling(self) -> List[str]:
        """
        Main loop: perform R rounds of GH (Gibbs sampling with clustering).
        Returns:
            List of accepted sequences after the last round.
        """
        
        current_sequences = self.sequences.copy()

        for r in range(self.rounds):
            # Generate proposals for all sequences in current pool
            proposals = []
            proposal_scores = []

            for seq in current_sequences:
                # Generate proposals per sequence
                seq_proposals = self._generate_proposals(seq)
                proposals.extend(seq_proposals)

            # Evaluate energy and gradients for proposals
            energies = []
            gradients = []

            for seq in proposals:
                f_x = self.predict(seq)
                energies.append(f_x)
                grads = self.grad_fn(seq)  # shape: [seq_len, vocab_size]
                gradients.append(grads)

            energies = np.array(energies)
            # Compute proposal probabilities q(x'|x)
            q_probs = self._compute_proposal_probs(gradients)

            # Accept/reject proposals
            accepted_sequences = []

            for i, seq in enumerate(proposals):
                current_seq = self._find_parent(seq, current_sequences)
                f_current = self.predict(current_seq)
                f_proposal = energies[i]
                q_curr_to_prop = q_probs[i]
                # Estimate q proposal ratio q(x|x') / q(x'|x)
                # Since proposals are symmetric (mutations), approximate ratio:
                # For simplicity, assume symmetric (q ratio=1), or compute explicitly if possible
                # Here, for most practical purposes, use symmetry approximation
                # (Note: For asymmetric proposals, compute ratio properly)
                ratio = 1.0

                mh_prob = min(1.0, np.exp(f_proposal - f_current) * ratio)
                u = self.rng.uniform()
                if u < mh_prob:
                    accepted_sequences.append(seq)

            # Cluster accepted sequences
            if len(accepted_sequences) == 0:
                # No proposals accepted; fallback could be original set or skip
                # For robustness, keep previous set unchanged
                print(f"Round {r+1}: No proposals accepted; keeping previous sequences.")
                next_sequences = current_sequences
            else:
                clusters = self._cluster_sequences(accepted_sequences)
                # Calculate predicted fitness for accepted sequences
                seq_fitnesses = np.array([self.predict(seq) for seq in accepted_sequences])
                # Select top per cluster
                top_seqs = self._select_top_per_cluster(accepted_sequences, seq_fitnesses, clusters)
                next_sequences = top_seqs

            current_sequences = next_sequences
            print(f"Round {r+1} completed: {len(current_sequences)} sequences retained.")

        return current_sequences

    def _generate_proposals(self, sequence: str) -> List[str]:
        """
        Generate proposals (mutations) for a single sequence
        by sampling mutations in Hamming neighborhood using gradient info.
        """
        proposals = []
        # For proposals, generate N proposals per sequence
        for _ in range(self.proposals_per_sequence):
            proposal_seq = self._mutate(sequence)
            proposals.append(proposal_seq)
        return proposals

    def _mutate(self, sequence: str) -> str:
        """
        Mutate a sequence at a random position guided by gradient-informed probabilities.
        """
        # Get gradient for current sequence
        grad = self.grad_fn(sequence)  # shape: [seq_len, vocab_size]
        # Compute scores per position for possible mutations
        logits = np.zeros((self.seq_len, self.vocab_size))
        for i in range(self.seq_len):
            # The gradient indicates change in fitness with respect to amino acid at position i
            # We can interpret raw gradients as scores
            # For now, use gradients directly as scores
            logits[i, :] = grad[i, :]
        # Apply temperature softmax for each position
        for i in range(self.seq_len):
            # Softmax scaled by temperature
            scores = logits[i, :]
            # To prevent overflow, subtract max
            max_score = np.max(scores)
            exp_scores = np.exp((scores - max_score) / self.tau)
            probs = exp_scores / np.sum(exp_scores)
            # Sample new amino acid index
            aa_idx = self.rng.choice(self.vocab_size, p=probs)
            # Construct mutated sequence: change amino acid at position i
            sequence = list(sequence)
            sequence[i] = self._index_to_aa(aa_idx)
            sequence = ''.join(sequence)
        return sequence

    def _compute_proposal_probs(self, gradients: List[np.ndarray]) -> np.ndarray:
        """
        Compute q(x'|x) probabilities for all proposals based on gradient scores.
        """
        probs = []
        for grad in gradients:
            logits = np.zeros((self.seq_len, self.vocab_size))
            for i in range(self.seq_len):
                logits[i, :] = grad[i, :]
            seq_probs = []
            for i in range(self.seq_len):
                scores = logits[i, :]
                max_score = np.max(scores)
                exp_scores = np.exp((scores - max_score)/self.tau)
                probs_i = exp_scores / np.sum(exp_scores)
                seq_probs.append(probs_i)
            # The overall proposal probability is the product over positions
            # For simplicity and efficiency, approximate by the sum of log-probabilities
            log_prob = 0.0
            for probs_i in seq_probs:
                # The selected amino acid index's probability
                # Since sampling is in _mutate, actual probability is for the sampled aa
                # but here, we can sum all for the proposal distribution
                # For Metropolis, ratios matter, but for simplicity, approximate ratio as 1
                # or implement if running full
                pass
            # For simplicity, set uniform probability over proposals, or ignore ratio
            # Returning 1.0 as ratio placeholder
            probs.append(1.0)
        return np.array(probs)

    def _cluster_sequences(self, sequences: List[str]) -> List[List[str]]:
        """
        Cluster sequences based on Levenshtein distance into specified number of clusters.
        """
        if len(sequences) == 0:
            return []

        # Use AgglomerativeClustering with precomputed distances
        dist_matrix = self._compute_distance_matrix(sequences)
        clustering = AgglomerativeClustering(n_clusters=self.num_clusters,
                                             affinity='precomputed',
                                             linkage='average')
        labels = clustering.fit_predict(dist_matrix)
        clusters: Dict[int, List[str]] = {}
        for seq, lbl in zip(sequences, labels):
            clusters.setdefault(lbl, []).append(seq)
        return list(clusters.values())

    def _compute_distance_matrix(self, sequences: List[str]) -> np.ndarray:
        """
        Compute full pairwise Levenshtein distance matrix for clustering.
        """
        n = len(sequences)
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = self._levenshtein_distance(sequences[i], sequences[j])
                dist_mat[i, j] = dist
                dist_mat[j, i] = dist
        return dist_mat

    def _select_top_per_cluster(self, sequences: List[str], fitnesses: np.ndarray, clusters: List[List[str]]) -> List[str]:
        """
        Select sequence with highest predictor score in each cluster.
        """
        seq_fit_map = {seq: fit for seq, fit in zip(sequences, fitnesses)}
        top_seqs = []
        for cluster in clusters:
            top_seq = max(cluster, key=lambda s: seq_fit_map[s])
            top_seqs.append(top_seq)
        return top_seqs

    def _find_parent(self, seq: str, pool: List[str]) -> str:
        """
        Helper to find existing sequence in pool matching the sequence.
        """
        # Because mutations are local, assume identity matching
        # For robustness, match sequence exactly
        if seq in pool:
            return seq
        else:
            # fallback to closest in pool
            return min(pool, key=lambda s: self._levenshtein_distance(s, seq))

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Use existing implementation from dataset_loader or placeholder"""
        # For consistency, implement here quickly
        # Alternatively, import Python-Levenshtein library if available
        import difflib
        seqmatch = difflib.SequenceMatcher(None, s1, s2)
        matches = sum(n for _, _, n in seqmatch.get_matching_blocks())
        return max(len(s1), len(s2)) - matches

    def _index_to_aa(self, index: int) -> str:
        """
        Map index to amino acid letter.
        """
        aa_list = ['A','C','D','E','F','G','H','I','K','L',
                   'M','N','P','Q','R','S','T','V','W','Y']
        if 0 <= index < len(aa_list):
            return aa_list[index]
        else:
            return 'A'  # fallback

```

## trainer.py

```python
# trainer.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List, Tuple, Dict
from model import SequenceFitnessPredictor

class Trainer:
    """
    Handles training and validation of the predictor model.
    Implements training loop with data batching, loss computation,
    optimizer steps, checkpointing, and metrics tracking.
    """

    def __init__(self, config: Dict, train_sequences: List[str], train_labels: np.ndarray,
                 val_sequences: List[str], val_labels: np.ndarray, checkpoint_dir: str = './checkpoints'):
        """
        Initialize the Trainer with hyperparameters and datasets.
        Args:
            config (Dict): Configuration dict with hyperparameters.
            train_sequences (List[str]): List of training sequences.
            train_labels (np.ndarray): Array of smoothed labels for training.
            val_sequences (List[str]): Validation sequences.
            val_labels (np.ndarray): Validation labels.
            checkpoint_dir (str): Directory to save checkpoints.
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = self.config.get('seed', 42)
        self._set_seed(self.seed)

        # Initialize model
        self.model = SequenceFitnessPredictor(self.config['predictor_model']).model
        self.model.to(self.device)

        # Setup optimizer
        optimizer_params = self.config.get('predictor_model', {}).get('optimizer_params', {})
        learning_rate = optimizer_params.get('learning_rate', 1e-3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Hyperparameters
        self.epochs = self.config.get('training', {}).get('epochs', 50)
        self.batch_size = self.config.get('training', {}).get('batch_size', 128)
        self.checkpoint_dir = checkpoint_dir

        # Datasets
        self.train_sequences = train_sequences
        self.train_labels = train_labels
        self.val_sequences = val_sequences
        self.val_labels = val_labels

        # Create checkpoint directory if not exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Internal tracking
        self.best_val_loss = np.inf

    def _set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _sequences_to_tensor(self, sequences: List[str]) -> torch.Tensor:
        """
        Convert list of sequences to tensor of shape [batch_size, seq_len, vocab_size].
        """
        seq_len = self.config['predictor_model'].get('sequence_length', 236)
        vocab_size = 20
        batch_size = len(sequences)
        tensor = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.float32)
        for b, seq in enumerate(sequences):
            for i, aa in enumerate(seq):
                if i >= seq_len:
                    break
                aa_idx = self._aa_to_index(aa)
                if aa_idx is not None:
                    tensor[b, i, aa_idx] = 1.0
        return tensor

    def _aa_to_index(self, aa: str) -> int:
        """
        Map amino acid to index (0-19).
        """
        aa_list = ['A','C','D','E','F','G','H','I','K','L',
                   'M','N','P','Q','R','S','T','V','W','Y']
        try:
            return aa_list.index(aa)
        except ValueError:
            return None

    def _get_batches(self, sequences: List[str], labels: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generator yielding batches of tensors and labels.
        """
        dataset_size = len(sequences)
        indices = np.arange(dataset_size)
        # Shuffle indices each epoch for stochasticity
        np.random.shuffle(indices)

        for start_idx in range(0, dataset_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, dataset_size)
            batch_idx = indices[start_idx:end_idx]
            batch_seqs = [sequences[i] for i in batch_idx]
            batch_labels = labels[batch_idx]
            batch_tensor = self._sequences_to_tensor(batch_seqs)
            batch_tensor = batch_tensor.to(self.device)
            batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.float32, device=self.device)
            yield batch_tensor, batch_labels_tensor

    def train(self):
        """
        Run the training loop over specified epochs.
        Save the best model based on validation loss.
        """
        val_losses = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_train_loss = 0.0
            n_batches = 0
            tqdm_desc = f"Epoch {epoch}/{self.epochs}"
            with tqdm(total=(len(self.train_sequences) // self.batch_size + 1), desc=tqdm_desc) as pbar:
                for batch_x, batch_y in self._get_batches(self.train_sequences, self.train_labels):
                    self.optimizer.zero_grad()
                    preds = self.model(batch_x)
                    loss = nn.functional.mse_loss(preds, batch_y)
                    loss.backward()
                    self.optimizer.step()

                    total_train_loss += loss.item() * batch_x.size(0)
                    n_batches += 1
                    pbar.update(1)
                avg_train_loss = total_train_loss / len(self.train_sequences)

            # Validation
            val_loss = self._validate()

            # Save checkpoint if improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'best_model.pt'))

            # Logging
            print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def _validate(self) -> float:
        """
        Run model in eval mode over validation dataset to compute loss.
        """
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch_x, batch_y in self._get_batches(self.val_sequences, self.val_labels):
                preds = self.model(batch_x)
                loss = nn.functional.mse_loss(preds, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                count += batch_x.size(0)
        return total_loss / count

    def load_best_model(self):
        """
        Load the best saved model checkpoint for inference.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path))
        else:
            print("Warning: No checkpoint found. Proceeding with current model.")

    def get_model(self) -> nn.Module:
        """
        Return the trained model (loaded on CPU for inference if needed).
        """
        return self.model
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\GGS\GGS_repo`
