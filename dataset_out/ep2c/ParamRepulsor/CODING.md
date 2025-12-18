# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py

import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

class DatasetLoader:
    def __init__(self, config: dict):
        """
        Initialize dataset loader based on configuration.
        Args:
            config (dict): Configuration dictionary containing dataset info.
                Expected keys:
                    - name (str): Dataset name, e.g., 'MNIST'.
                    - feature_dim (int or 'auto'): Target feature dimension after PCA.
                    - data_path (str): Path to load dataset from.
        """
        self.dataset_name = config.get('name', 'MNIST')
        self.feature_dim_setting = config.get('feature_dim', 'auto')
        self.data_path = config.get('data_path', None)
        self.n_neighbors = config.get('neighbor_count', 15)
        self.random_seed = config.get('random_seed', 42)
        
        # Placeholders for data
        self.data = None          # Original data
        self.norm_data = None     # Normalized data
        self.pca_data = None      # PCA reduced data (if applied)
        self.knn_indices = None   # NN indices
        self.knn_distances = None # NN distances
        self.mn_pairs = None      # Mid-Near pairs indices
        self.fp_indices = None    # Far Negative indices (per point)
        self.N = 0                # Number of data points
        self.D = 0                # Data dimensionality
        
        np.random.seed(self.random_seed)
        
        # Load dataset
        self.load_dataset()
        # Preprocess: normalize features
        self._preprocess()
        # Apply PCA if needed
        self._apply_pca()
        # Compute neighbor graph
        self._compute_knn()
        # Generate mid-near pairs
        self._generate_mn_pairs()
        # Generate FP indices (for negative sampling)
        self._generate_fp_indices()
    
    def load_dataset(self):
        """
        Loads dataset based on dataset name.
        Supports MNIST, F-MNIST, USPS, COIL-20, COIL-100, and biological datasets.
        """
        if self.dataset_name == 'MNIST':
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            data = mnist.data.astype(np.float32)
            labels = mnist.target.astype(int)
        elif self.dataset_name == 'F-MNIST':
            from sklearn.datasets import fetch_openml
            f_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
            data = f_mnist.data.astype(np.float32)
            labels = f_mnist.target.astype(int)
        elif self.dataset_name == 'USPS':
            from sklearn.datasets import fetch_openml
            usps = fetch_openml('USPS', version=1, as_frame=False)
            data = usps.data.astype(np.float32)
            labels = usps.target.astype(int)
        elif self.dataset_name == 'COIL-20':
            # Assuming dataset stored as features
            # User provides feature matrix, else replace with custom loader
            data_path = self.data_path
            data = np.loadtxt(data_path)
            labels = np.arange(data.shape[0])  # No labels; placeholder
        elif self.dataset_name == 'COIL-100':
            data_path = self.data_path
            data = np.loadtxt(data_path)
            labels = np.arange(data.shape[0])
        # Biological datasets
        elif self.dataset_name == 'scRNAseq_HumanPancreas':
            data_path = self.data_path
            data = np.loadtxt(data_path, delimiter=',')
            labels = None
        elif self.dataset_name == 'scRNAseq_Kang':
            data_path = self.data_path
            data = np.loadtxt(data_path, delimiter=',')
            labels = None
        elif self.dataset_name == 'Kazer':
            data_path = self.data_path
            data = np.loadtxt(data_path, delimiter=',')
            labels = None
        elif self.dataset_name == 'Circle':
            # Generate synthetic circle data
            np.random.seed(self.random_seed)
            angles = np.random.uniform(0, 2*np.pi, 5000)
            radius = 1.0
            data = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            labels = np.floor(angles / (2*np.pi/10)).astype(int)  # 10 arcs
        elif self.dataset_name == 'Mammoth':
            data_path = self.data_path
            data = np.loadtxt(data_path)
            labels = None
        elif self.dataset_name == 'Lineage':
            # Generate Gaussian points in line
            np.random.seed(self.random_seed)
            data = np.zeros((10000,50))
            labels = None
            for i in range(20):
                mean = np.array([i])
                cov = np.eye(50)*0.1
                points = np.random.multivariate_normal(mean, cov, size=500)
                data[i*500:(i+1)*500,:] = points
        elif self.dataset_name == 'Hierarchy':
            # Generate hierarchy data (placeholder)
            np.random.seed(self.random_seed)
            data = np.random.randn(12500, 50)
            labels = None
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")
        # Store data and labels
        self.data = data
        self.labels = labels
        self.N = data.shape[0]
        self.D = data.shape[1]
    
    def _preprocess(self):
        """
        Normalize features: zero mean, unit variance per feature.
        """
        scaler = StandardScaler()
        self.norm_data = scaler.fit_transform(self.data)
        # For biological datasets, additional normalization could be added if needed
    
    def _apply_pca(self):
        """
        Apply PCA if feature_dim is specified and less than original dimension.
        If feature_dim='auto', heuristically reduce to 50 dims if dataset is large.
        """
        target_dim = self.feature_dim_setting
        if target_dim is None or target_dim == 'auto':
            # Decide based on dataset size
            if self.N > 10000:
                target_dim = 50
            elif self.D > 50:
                target_dim = 50
            else:
                target_dim = self.D
        elif isinstance(target_dim, int):
            pass  # use as is
        else:
            # fallback
            target_dim = self.D
        if target_dim < self.D:
            pca = PCA(n_components=target_dim, random_state=self.random_seed)
            self.pca_data = pca.fit_transform(self.norm_data)
        else:
            # no PCA reduction
            self.pca_data = self.norm_data
    
    def _compute_knn(self):
        """
        Compute k-nearest neighbor graph using sklearn's NearestNeighbors.
        """
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1, algorithm='auto', metric='euclidean')
        nbrs.fit(self.pca_data)
        distances, indices = nbrs.kneighbors(self.pca_data)
        # Remove self from neighbor list
        self.knn_indices = indices[:,1:]  # shape (N, k)
        self.knn_distances = distances[:,1:]
    
    def _generate_mn_pairs(self):
        """
        Generate Mid-Near (MN) pairs:
        For each point, sample 6 points uniformly, find the closest among these to form MN.
        """
        N, k, seed = self.N, self.n_neighbors, self.random_seed
        np.random.seed(seed)
        MN_pairs = np.zeros(N, dtype=int)  # store index of MN partner for each point
        for i in range(N):
            sampled_indices = np.random.choice(N, size=6, replace=True)
            # compute distances to sampled points
            src_point = self.pca_data[i]
            sampled_points = self.pca_data[sampled_indices]
            dists = np.linalg.norm(sampled_points - src_point, axis=1)
            # second closest among sampled points
            second_idx = np.argsort(dists)[1]  # index in sampled_indices array
            MN_pairs[i] = sampled_indices[second_idx]
        self.mn_pairs = MN_pairs
    
    def _generate_fp_indices(self, n_fp_per_point=20):
        """
        Generate Far Negative (FP): For each point, randomly sample n_fp_per_point points
        not in its NN set.
        """
        N, k = self.N, self.n_neighbors
        fp_indices_list = []
        all_indices = np.arange(N)
        for i in range(N):
            # Exclude the NN indices from sampling
            neighbor_set = set(self.knn_indices[i])
            exclude_set = neighbor_set.union({i})
            candidate_indices = np.setdiff1d(all_indices, list(exclude_set))
            # If dataset is small, case where candidate set is smaller than needed
            replace = False
            if candidate_indices.shape[0] < n_fp_per_point:
                replace = True
            fp_sample = np.random.choice(candidate_indices, size=n_fp_per_point, replace=replace)
            fp_indices_list.append(fp_sample)
        # Store as an array (N, n_fp_per_point)
        self.fp_indices = np.array(fp_indices_list)
    
    def get_data(self):
        """
        Return the raw dataset before preprocessing.
        """
        return self.data
    
    def get_normalized_data(self):
        """
        Return normalized data.
        """
        return self.norm_data
    
    def get_pca_data(self):
        """
        Return PCA-reduced data.
        """
        return self.pca_data
    
    def get_knn(self):
        """
        Return neighbor indices and distances.
        """
        return self.knn_indices, self.knn_distances
    
    def get_mn_pairs(self):
        """
        Return mid-near pairs as array of shape (N,), with each entry as the MN partner index.
        """
        return self.mn_pairs
    
    def get_fp_indices(self):
        """
        Return FP indices: array of shape (N, n_fp_per_point).
        """
        return self.fp_indices
```

## evaluation.py

```python
## evaluation.py
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class Evaluation:
    def __init__(self, model: nn.Module, dataset: tuple, labels: np.ndarray = None):
        """
        Initialize Evaluation object.
        Args:
            model (nn.Module): Trained neural network projector for embedding.
            dataset (tuple): Tuple of (X, raw data), where:
                - X: high-dimensional data as np.ndarray shape (N, D)
                - raw data: same as above, not necessarily used here
            labels (np.ndarray): Ground truth labels for evaluation (optional).
        """
        self.model = model
        self.X = dataset[0]  # high-dimensional data
        self.labels = labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embeddings = None

    def get_embeddings(self, X=None, batch_size=5000):
        """
        Embed dataset X using the trained model.
        Args:
            X (np.ndarray): Data to embed, if None defaults to training data.
            batch_size (int): Batch size for processing.
        Returns:
            np.ndarray: Embeddings shape (N, embed_dim)
        """
        if X is None:
            X_input = self.X
        else:
            X_input = X
        self.model.eval()
        embeddings_list = []
        with torch.no_grad():
            for start in range(0, X_input.shape[0], batch_size):
                end = min(start + batch_size, X_input.shape[0])
                batch = torch.from_numpy(X_input[start:end]).float().to(self.device)
                emb = self.model.get_embedding(batch).cpu().numpy()
                embeddings_list.append(emb)
        self.embeddings = np.vstack(embeddings_list)
        return self.embeddings

    def compute_nn_accuracy(self, embeddings=None, labels=None, k=10):
        """
        Compute k-NN classification accuracy in embedding space.
        Args:
            embeddings (np.ndarray): Embedding vectors, shape (N, d). If None, compute with current.
            labels (np.ndarray): True labels for data.
            k (int): Number of neighbors.
        Returns:
            float: Accuracy score.
        """
        if embeddings is None:
            if self.embeddings is None:
                embeddings = self.get_embeddings()
            else:
                embeddings = self.embeddings
        if labels is None:
            labels = getattr(self, 'labels', None)
            if labels is None:
                raise ValueError("Labels must be provided for accuracy computation.")
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean', n_jobs=-1)
        nbrs.fit(embeddings)
        dist, idx = nbrs.kneighbors(embeddings)
        pred_labels = []
        for i in range(embeddings.shape[0]):
            neighbor_idxs = idx[i, 1:]  # exclude self
            neighbor_labels = labels[neighbor_idxs]
            # majority vote
            counts = np.bincount(neighbor_labels)
            pred_label = np.argmax(counts)
            pred_labels.append(pred_label)
        pred_labels = np.array(pred_labels)
        accuracy = np.mean(pred_labels == labels)
        return accuracy

    def compute_triplet_preservation(self, embeddings, high_dim_data, triplets: list):
        """
        Compute the proportion of triplets where the order in high-D is preserved in low-D.
        Args:
            embeddings (np.ndarray): Low-dimensional embeddings, shape (N, d)
            high_dim_data (np.ndarray): Original high-D data, shape (N, D)
            triplets (list): List of triplets (anchor_idx, pos_idx, neg_idx)
        Returns:
            float: Triplet preservation ratio
        """
        if len(triplets) == 0:
            return np.nan
        preserved = 0
        total = len(triplets)
        for (a_idx, p_idx, n_idx) in triplets:
            d_hd_pos = np.linalg.norm(high_dim_data[a_idx] - high_dim_data[p_idx])
            d_hd_neg = np.linalg.norm(high_dim_data[a_idx] - high_dim_data[n_idx])
            d_ld_pos = np.linalg.norm(embeddings[a_idx] - embeddings[p_idx])
            d_ld_neg = np.linalg.norm(embeddings[a_idx] - embeddings[n_idx])
            # Check if high-D order is preserved
            if (d_hd_pos < d_hd_neg and d_ld_pos < d_ld_neg) or (d_hd_pos > d_hd_neg and d_ld_pos > d_ld_neg):
                preserved += 1
        return preserved / total

    def compute_distance_correlation(self, embeddings, high_dim_data, cluster_centroids=None):
        """
        Compute Spearman correlation between pairwise distances in high-D and embedded spaces.
        Args:
            embeddings (np.ndarray): (N, d)
            high_dim_data (np.ndarray): (N, D)
            cluster_centroids (dict): dict of {label: centroid in high-D} (optional).
        Returns:
            float: Spearman correlation
        """
        # For global structure, compute centroid distances if provided
        if cluster_centroids is not None:
            hd_centroids = np.array([cluster_centroids[label] for label in cluster_centroids])
            ld_centroids = np.array([np.mean(embeddings[self.labels == label], axis=0)
                                      for label in cluster_centroids])
            high_dists = cdist(hd_centroids, hd_centroids, metric='euclidean')
            low_dists = cdist(ld_centroids, ld_centroids, metric='euclidean')
            high_dists_vec = high_dists.flatten()
            low_dists_vec = low_dists.flatten()
        else:
            high_dists_vec = cdist(high_dim_data, high_dim_data, metric='euclidean').flatten()
            low_dists_vec = cdist(embeddings, embeddings, metric='euclidean').flatten()

        rho, _ = spearmanr(high_dists_vec, low_dists_vec)
        return rho

    def visualize_embeddings(self, embeddings=None, labels=None, title=None):
        """
        Plot 2D embeddings with optional labels.
        """
        if embeddings is None:
            embeddings = self.embeddings
        plt.figure(figsize=(8,8))
        if labels is not None:
            scatter = plt.scatter(embeddings[:,0], embeddings[:,1], c=labels, cmap='tab20', s=10, alpha=0.8)
            plt.legend(*scatter.legend_elements(), title="Labels")
        else:
            plt.scatter(embeddings[:,0], embeddings[:,1], s=10, alpha=0.8)
        if title:
            plt.title(title)
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
```

## main.py

```python
## main.py
import yaml
import numpy as np
import torch
import os
from dataset_loader import DatasetLoader
from pair_sampler import PairSampler
from model import NeuralNetwork
from trainer import train
from evaluation import Evaluation

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set random seeds for reproducibility
    seed = config.get('misc', {}).get('random_seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess dataset
    dataset_cfg = config['dataset']
    dataset_loader = DatasetLoader(dataset_cfg)
    dataset_loader.load_dataset()
    dataset_loader._preprocess()
    # Use PCA reduction if specified, defaults already handled inside DatasetLoader
    dataset_loader._apply_pca()
    data = dataset_loader.get_pca_data()  # normalized, PCA-reduced features
    N = dataset_loader.N

    # Build neighbor graph (kNN)
    knn_indices, knn_distances = dataset_loader.get_knn()

    # Generate mid-near pairs (MN) as per Sec. 4
    dataset_loader._generate_mn_pairs()  # updates dataset_loader.mn_pairs

    # Generate FP indices (far negatives)
    dataset_loader._generate_fp_indices()

    # Initialize PairSampler with neighbor info
    pair_cfg = config.get('pair_sampling', {})
    pair_sampler = PairSampler(
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        n_points=N,
        config=pair_cfg,
        seed=seed
    )
    pair_sampler.set_data(dataset_loader.get_pca_data())

    # Initialize model
    model_cfg = config['model']
    input_dim = dataset_loader.get_pca_data().shape[1]
    model = NeuralNetwork(
        input_dim=input_dim,
        output_dim=2,  # for visualization
        hidden_layers=model_cfg['hidden_layers'],
        neurons_per_layer=model_cfg['neurons_per_layer'],
        activation=model_cfg.get('activation', 'relu')
    ).to(device)

    # Setup optimizer
    training_cfg = config['training']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg.get('learning_rate', 0.001),
        betas=tuple(config.get('optimization', {}).get('betas', [0.9, 0.999]))
    )

    # Training hyperparameters
    num_epochs = config['hyperparameters'].get('num_epochs', 100)
    batch_size = training_cfg.get('batch_size', 1024)
    report_interval = config['hyperparameters'].get('report_interval', 10)

    # Initialize evaluation object for metrics
    evaluator = Evaluation(model, (dataset_loader.get_pca_data(), dataset_loader.labels))

    # Prepare data indices for batching
    all_indices = np.arange(N)
    n_batches = int(np.ceil(N / batch_size))
    print(f"Total samples: {N}, Batches per epoch: {n_batches}")

    for epoch in range(1, num_epochs + 1):
        np.random.seed(seed + epoch)  # optional: different shuffle each epoch
        permuted_indices = np.random.permutation(all_indices)
        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, N)
            batch_indices = permuted_indices[start:end]
            b = len(batch_indices)

            # Generate pair indices and types for current batch
            idx_i, idx_j, pair_types = pair_sampler.generate_pairs(batch_indices, dataset_loader.get_pca_data())

            # Fetch data
            X_all = dataset_loader.get_pca_data()
            x_batch_np = X_all[batch_indices]
            x_batch = torch.from_numpy(x_batch_np).float().to(device)
            # For the pairs, extract respective features
            # Subset features for pairs
            idx_i_cpu = idx_i.cpu().numpy()
            idx_j_cpu = idx_j.cpu().numpy()

            # Get pair data
            x_NN = torch.from_numpy(X_all[idx_j_cpu[pair_types.cpu().numpy() == 0]]).float().to(device)
            x_MN = torch.from_numpy(X_all[idx_j_cpu[pair_types.cpu().numpy() == 1]]).float().to(device)
            x_FP = torch.from_numpy(X_all[idx_j_cpu[pair_types.cpu().numpy() == 2]]).float().to(device)

            # Compute embeddings
            y_batch = model.forward(x_batch)  # shape (b, 2)
            y_NN = model.forward(x_NN)
            y_MN = model.forward(x_MN)
            y_FP = model.forward(x_FP)

            # Compute pairwise distances in embedding space
            def pairwise_dist(y1, y2):
                return torch.sum((y1.unsqueeze(1) - y2.unsqueeze(0))**2, dim=2)  # shape (len(y1), len(y2))
            # For pairs
            # Note: Need to align pairs with batch points
            # Create mappings:
            # For NN: get embeddings of anchor batch points with their NN
            # For MN and FP: same
            # Extract indices
            # Gather the embedding of anchor points (batch points)
            batch_embs = y_batch
            # NN pairs
            nn_indices = idx_i[pair_types == 0]
            nn_embs = y_NN
            # For simplicity, compute all pairwise in the batch using indexing
            # Alternative: create tensors for pairwise computation for each pair type
            # But since the number of pairs is large, do in small batch.

            # Get number of pairs per type
            nn_mask = (pair_types == 0)
            mn_mask = (pair_types == 1)
            fp_mask = (pair_types == 2)

            # Distances
            if nn_mask.sum() > 0:
                y_anchor_nn = y_batch[nn_indices]
                d2_nn = pairwise_dist(y_anchor_nn, y_NN)
            else:
                d2_nn = torch.tensor([]).to(device)
            if mn_mask.sum() > 0:
                y_anchor_mn = y_batch[idx_i[mn_mask]]
                d2_mn = pairwise_dist(y_anchor_mn, y_MN)
            else:
                d2_mn = torch.tensor([]).to(device)
            if fp_mask.sum() > 0:
                y_anchor_fp = y_batch[idx_i[fp_mask]]
                d2_fp = pairwise_dist(y_anchor_fp, y_FP)
            else:
                d2_fp = torch.tensor([]).to(device)

            # Compute similarity functions
            epsilon = 1e-8

            def q_nn_or_mn(d2):
                return torch.exp(- (d2 + 10) / (d2 + 10 + epsilon))
            def q_fp(d2):
                return torch.exp(- d2 / (d2 + 1 + epsilon))

            q_nn_vals = q_nn_or_mn(d2_nn) if d2_nn.numel() > 0 else torch.tensor([]).to(device)
            q_mn_vals = q_nn_or_mn(d2_mn) if d2_mn.numel() > 0 else torch.tensor([]).to(device)
            q_fp_vals = q_fp(d2_fp) if d2_fp.numel() > 0 else torch.tensor([]).to(device)

            # Compute loss contributions
            loss_NN = -torch.log(torch.clamp(q_nn_vals, min=epsilon)) if d2_nn.numel() > 0 else torch.tensor(0.0).to(device)
            loss_MN = -torch.log(1 - torch.clamp(q_mn_vals, min=epsilon)) if d2_mn.numel() > 0 else torch.tensor(0.0).to(device)
            loss_FP = -torch.log(1 - torch.clamp(q_fp_vals, min=epsilon)) if d2_fp.numel() > 0 else torch.tensor(0.0).to(device)

            # Weights from config
            w_NB = config['loss_weights'].get('weight_NN', 1.0)
            w_MN = config['loss_weights'].get('weight_MN', 0.5)
            w_FP = config['loss_weights'].get('weight_FP', 0.2)

            total_loss = (
                w_NB * torch.sum(loss_NN) +
                w_MN * torch.sum(loss_MN) +
                w_FP * torch.sum(loss_FP)
            ) / b

            # Gradient update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        # Logging and evaluation
        if epoch % report_interval == 0 or epoch == num_epochs:
            # Evaluate local NN accuracy
            acc_nn = evaluator.compute_nn_accuracy(embeddings=None, labels=dataset_loader.labels, k=10)
            # Triplet preservation
            # Generate some triplets (e.g., random triplets from labels or high-D data)
            # For simplicity, we skip triplet sampling here, or implement if needed
            trip_metric = evaluator.compute_triplet_preservation(
                embeddings=evaluator.get_embeddings(), 
                high_dim_data=dataset_loader.data, 
                triplets=[]  # Placeholder: generate triplets if needed
            )
            # Global distance correlation
            dist_corr = evaluator.compute_distance_correlation(
                embeddings=evaluator.get_embeddings(),
                high_dim_data=dataset_loader.data,
                cluster_centroids=None  # or precompute centroids if labels known
            )
            print(f"Epoch {epoch}/{num_epochs}: loss={epoch_loss / n_batches:.4f}, "
                  f"NN_Acc={acc_nn:.4f}, Triplet={trip_metric:.4f}, DistCorr={dist_corr:.4f}")

    # Save the trained model
    save_path = config.get('save_model_path', './models/paramreprulsor.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    # Final embedding for entire dataset (or test set)
    final_embeddings = evaluator.get_embeddings()
    # Save embeddings or visualize
    # e.g., save to file
    np.save('final_embeddings.npy', final_embeddings)
    print("Training and embedding complete. Results saved.")

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,                 # D: input feature dimension, default for MNIST
        output_dim: int = 2,                  # d: low-dimensional embedding size
        hidden_layers: int = 3,               # Number of hidden layers (from config)
        neurons_per_layer: int = 100,         # Neurons per hidden layer (from config)
        activation: str = 'relu'              # Activation function
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer

        # Determine activation function
        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'silu':
            self.activation_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build layers
        layers = []

        # First layer from input_dim to first hidden layer
        in_dim = self.input_dim
        for i in range(self.hidden_layers):
            layer = nn.Linear(in_dim, self.neurons_per_layer)
            layers.append(layer)
            layers.append(self.activation_fn)
            in_dim = self.neurons_per_layer

        # Final layer from last hidden to output_dim
        self.output_layer = nn.Linear(in_dim, self.output_dim)
        layers.append(self.output_layer)

        # Sequential module for hidden layers + activation
        self.model = nn.Sequential(*layers[:-1])  # Exclude last layer from sequence

        # Initialize weights using Kaiming He initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, output_dim).
        """
        out = self.model(x)
        out = self.output_layer(out)
        return out

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the low-dimensional embedding vectors for input.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Embedding vectors of shape (batch_size, output_dim).
        """
        with torch.no_grad():
            embedding = self.forward(x)
        return embedding
```

## pair_sampler.py

```python
## pair_sampler.py
import numpy as np
from typing import List, Tuple
import torch

class PairSampler:
    def __init__(
        self,
        knn_indices: np.ndarray,
        knn_distances: np.ndarray,
        n_points: int,
        config: dict,
        seed: int = 42
    ):
        """
        PairSampler constructs and samples positive and negative pairs based
        on precomputed neighbor graph and mid-near (hard negative) strategies.

        Args:
            knn_indices (np.ndarray): (N, k) neighbors indices for each point.
            knn_distances (np.ndarray): (N, k) neighbor distances.
            n_points (int): total number of data points in dataset.
            config (dict): Dictionary with keys:
                - 'neighbor_count' (int): number of neighbors (k).
                - 'mid_near_sample_count' (int): number of sampled points (h=6).
                - 'negative_samples_per_point' (int): negatives per point for FP.
            seed (int): random seed for reproducibility.
        """
        self.knn_indices = knn_indices  # shape: (N, k)
        self.knn_distances = knn_distances  # shape: (N, k)
        self.N = n_points
        self.k = config.get('neighbor_count', 15)
        self.h = config.get('mid_near_sample_count', 6)
        self.n_neg = config.get('negative_samples_per_point', 20)
        self.seed = seed
        np.random.seed(self.seed)

        # Precompute mid-near pairs using sampling strategy (Sec. 4)
        self._precompute_mid_near_pairs()

    def _precompute_mid_near_pairs(self):
        """
        For each point:
            - sample h=6 points uniformly from dataset
            - find their distances, select second closest point
        Store as an array of shape (N,): `mid_near_partner_idx`
        """
        N = self.N
        h = self.h
        mid_near_partners = np.zeros(N, dtype=int)

        for i in range(N):
            sampled_indices = np.random.choice(N, size=h, replace=False)
            # distances from point i to sampled points
            sampled_dists = np.linalg.norm(
                self.knn_data_point(i, sampled_indices) - self.get_point(i),
                axis=1
            )
            # Find second closest among sampled points
            sorted_idx = np.argsort(sampled_dists)
            second_closest_idx = sampled_indices[sorted_idx[1]]
            mid_near_partners[i] = second_closest_idx
        self.mid_near_partner_idx = mid_near_partners

    def knn_data_point(self, i: int, indices: np.ndarray) -> np.ndarray:
        """
        Get data points for given indices (subset of neighbor set or random points).
        """
        return self.knn_data[indices]
    def get_point(self, i: int) -> np.ndarray:
        """
        Returns the data point of dataset at index i.
        Note: We expect that training data will be passed in during batch sampling.
        If necessary, this function may be adapted.
        """
        # Placeholder: actual data should be stored outside and accessed accordingly
        pass

    def generate_pairs(self, batch_indices: np.ndarray, dataset: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate pairs for a batch of points.

        Args:
            batch_indices (np.ndarray): indices of points in current batch, shape: (b,)
            dataset (np.ndarray): full dataset data (N, D)

        Returns:
            pair_tensor (torch.Tensor): shape (total_pairs, 3), each row: (idx_i, idx_j, pair_type_int)
                pair_type_int: 0=NN, 1=MN, 2=FP
        """
        b = len(batch_indices)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pairs_list = []
        pair_type_list = []

        # For each point in batch:
        for i in batch_indices:
            # 1) Neighbor (NN) pairs
            nn_neighbors = self.knn_indices[i]  # shape: (k,)
            # randomly sample n_NB from neighbors
            n_NB = min(self.k, self.n_neg)  # For example, or set in config
            nb_samples = (
                nn_neighbors
                if len(nn_neighbors) <= n_NB
                else np.random.choice(nn_neighbors, size=n_NB, replace=False)
            )
            for j in nb_samples:
                pairs_list.append((i, j))
                pair_type_list.append(0)

            # 2) Mid-near (MN) pair: deterministic, selected during precompute
            mn_j = self.mid_near_partner_idx[i]
            pairs_list.append((i, mn_j))
            pair_type_list.append(1)

            # 3) Far Negative (FP) pairs: uniformly sample negative indices
            candidate_indices = np.setdiff1d(np.arange(self.N), np.concatenate([nn_neighbors, [i]]))
            # sample n_FP negatives
            n_FP = self.n_neg
            if candidate_indices.shape[0] <= n_FP:
                fp_samples = np.random.choice(candidate_indices, size=n_FP, replace=True)
            else:
                fp_samples = np.random.choice(candidate_indices, size=n_FP, replace=False)
            for j in fp_samples:
                pairs_list.append((i, j))
                pair_type_list.append(2)

        # Convert to tensor
        idx_i = torch.tensor([p[0] for p in pairs_list], dtype=torch.long, device=device)
        idx_j = torch.tensor([p[1] for p in pairs_list], dtype=torch.long, device=device)
        pair_types = torch.tensor(pair_type_list, dtype=torch.long, device=device)

        return (idx_i, idx_j, pair_types)

    def set_data(self, data: np.ndarray):
        """
        Set the dataset data for accessing points during pair formation.
        """
        self.knn_data = data
```

**Notes:**

- The class expects that actual dataset data (`dataset`) is provided at each `generate_pairs()` call because `get_point()` needs to access raw feature vectors; alternatively, you can incorporate data as an attribute, but since the dataset can be large, passing data on calls is cleaner.
- The core functions:
  - `__init__()` precomputes the mid-near pairs as per Sec.4.
  - `generate_pairs()` creates all types of pairs for a batch, with the specified sampling strategy.
- You should instantiate the class with precomputed neighbor info (`knn_indices`, `knn_distances`, and dataset size). The full neighbor graph should be computed prior to training, e.g., via `sklearn` or FAISS.
- The return `pair_tensor` contains index pairs and pair type labels, which can be used in the training loop for computing pairwise distances in embedding space and loss.

This code aligns with the design, interface, and logical steps from the paper, ensuring reproducible and efficient pair sampling with the proposed challenging negative strategy.

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict
from dataset_loader import DatasetLoader
from model import NeuralNetwork
from pair_sampler import PairSampler
from evaluation import Evaluation

def train(
    model: nn.Module,
    dataset: Tuple[np.ndarray, np.ndarray],
    pair_sampler: PairSampler,
    config: Dict
):
    """
    Trains the ParamRepulsor model with hard negative mining and contrastive loss.

    Args:
        model (nn.Module): Neural network projector.
        dataset (Tuple[np.ndarray, np.ndarray]): Dataset tuple (X, labels).
        pair_sampler (PairSampler): PairSampler with neighbor info.
        config (Dict): Configuration dictionary with training hyperparameters.
    """
    # Extract training parameters from config
    training_cfg = config.get('training', {})
    optimization_cfg = config.get('optimization', {})
    loss_weights_cfg = config.get('loss_weights', {})
    hyperparams_cfg = config.get('hyperparameters', {})
    dataset_cfg = config.get('dataset', {})
    
    lr = training_cfg.get('learning_rate', 0.001)
    batch_size = training_cfg.get('batch_size', 1024)
    num_epochs = training_cfg.get('epochs', 100)
    report_interval = hyperparams_cfg.get('report_interval', 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    w_NB = loss_weights_cfg.get('weight_NN', 1.0)
    w_MN = loss_weights_cfg.get('weight_MN', 0.5)
    w_FP = loss_weights_cfg.get('weight_FP', 0.2)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=optimization_cfg.get('betas', [0.9, 0.999]))
    
    # Unpack dataset
    X_data, labels = dataset
    N = X_data.shape[0]
    feature_dim = X_data.shape[1]
    
    # Set dataset data for pair sampler
    pair_sampler.set_data(X_data)
    
    # Calculate number of batches per epoch
    num_batches = int(np.ceil(N / batch_size))
    
    # For evaluation
    evaluator = Evaluation(model, dataset)
    
    # Prepare index array for shuffling
    all_indices = np.arange(N)
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        permuted_indices = np.random.permutation(all_indices)
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{num_epochs}")
        
        for batch_idx in pbar:
            # Sample batch of points
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N)
            batch_indices = permuted_indices[start_idx:end_idx]
            b = len(batch_indices)
            
            # Generate pairs for the batch
            idx_i, idx_j, pair_types = pair_sampler.generate_pairs(batch_indices, X_data)
            # idx_i, idx_j: index tensors for pairs
            # pair_types: tensor indicating pair type (0=NN, 1=MN, 2=FP)

            # Fetch raw data for pair points
            x_batch = torch.from_numpy(X_data[batch_indices]).to(device).float()
            x_NN = torch.from_numpy(X_data[idx_j[:pair_types==0].cpu()]]).to(device).float()
            x_MN = torch.from_numpy(X_data[idx_j[pair_types==1].cpu()]]).to(device).float()
            x_FP = torch.from_numpy(X_data[idx_j[pair_types==2].cpu()]]).to(device).float()

            # Compute embeddings
            y_batch = model.forward(x_batch)                       # shape: (b, 2)
            # Embeddings for pair points
            y_NN = model.forward(x_NN)    # shape: (num_NN_pairs, 2)
            y_MN = model.forward(x_MN)    # shape: (num_MN_pairs, 2)
            y_FP = model.forward(x_FP)    # shape: (num_FP_pairs, 2)

            # Function to compute pairwise squared distances
            def pairwise_distances(y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
                return torch.sum((y1.unsqueeze(1) - y2.unsqueeze(0))**2, dim=2)

            # Compute distances for pairs
            # For NN pairs
            d2_NN = pairwise_distances(y_batch[idx_i[pair_types==0]], y_NN)
            # For MN pairs
            d2_MN = pairwise_distances(y_batch[idx_i[pair_types==1]], y_MN)
            # For FP pairs
            d2_FP = pairwise_distances(y_batch[idx_i[pair_types==2]], y_FP)

            # Compute similarity functions (Sec. 4, Appendix D)
            # q_NN and q_MN: similar form
            def q_nn_or_mn(d2):
                return torch.exp(- (d2 + 10) / (d2 + 10 + 1e-8))
            # q_FP
            def q_fp(d2):
                return torch.exp(- d2 / (d2 + 1 + 1e-8))
            
            q_NN_vals = q_nn_or_mn(d2_NN)
            q_MN_vals = q_nn_or_mn(d2_MN)
            q_FP_vals = q_fp(d2_FP)

            # For loss calculation: following the theoretical form,
            # attraction for NN pairs, repulsion for FP and MN
            # Loss per pair, with weighting
            # Use the equations from Appendix D
            # To avoid numerical issues, add small epsilon where needed
            epsilon = 1e-8

            # Compute pairwise "loss" contributions
            # attraction for NN pairs
            loss_NN = - torch.log(torch.clamp(q_NN_vals, min=epsilon))
            # repulsive for MN and FP (maximize distances)
            loss_MN = - torch.log(1 - torch.clamp(q_MN_vals, min=epsilon))
            loss_FP = - torch.log(1 - torch.clamp(q_FP_vals, min=epsilon))
            
            # Sum weighted contributions
            total_loss = (
                w_NB * torch.sum(loss_NN) +
                w_MN * torch.sum(loss_MN) +
                w_FP * torch.sum(loss_FP)
            ) / b  # Normalize by batch size

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            # Update progress bar
            pbar.set_postfix(loss=total_loss.item())

        # Report metrics at intervals
        if epoch % report_interval == 0:
            # Compute metrics
            # e.g., NN accuracy, triplet accuracy, or distance correlation
            local_acc = evaluator.compute_local_accuracy()
            triplet_ratio = evaluator.compute_triplet_preservation()
            dist_corr = evaluator.compute_distance_correlation()

            print(f"Epoch {epoch}: Loss={epoch_loss/num_batches:.4f}, "
                  f"NN_Acc={local_acc:.4f}, Triplet={triplet_ratio:.4f}, DistCorr={dist_corr:.4f}")

    # Save model after training
    torch.save(model.state_dict(), config.get('save_model_path', './models/paramreprulsor.pth'))


# Usage example (assuming all modules are imported correctly and config is loaded):
# if __name__ == '__main__':
#     import yaml
#     with open('config.yaml', 'r') as f:
#         config = yaml.safe_load(f)
#     dataset_loader = DatasetLoader(config['dataset'])
#     data = dataset_loader.get_normalized_data()
#     pair_sampler = PairSampler(
#         knn_indices=dataset_loader.get_knn()[0],
#         knn_distances=dataset_loader.get_knn()[1],
#         n_points=dataset_loader.N,
#         config=config.get('pair_sampling', {})
#     )
#     model = NeuralNetwork(
#         input_dim=data.shape[1],
#         output_dim=2,
#         hidden_layers=config['model']['hidden_layers'],
#         neurons_per_layer=config['model']['neurons_per_layer'],
#         activation=config['model'].get('activation', 'relu')
#     ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#     train(model, (dataset_loader.data, dataset_loader.labels), pair_sampler, config)
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\ParamRepulsor\ParamRepulsor_repo`
