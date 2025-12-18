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
