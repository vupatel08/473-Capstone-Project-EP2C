## losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from math import sqrt
from config import TRAINING

class AlignmentLoss:
    """
    Computes the alignment loss between augmented and original features,
    weighted by reciprocal neighbor Jaccard similarity.
    """
    def __init__(self, neighbor_k: int = 10):
        self.k = neighbor_k

    def compute_reciprocal_neighbors(self, features: torch.Tensor):
        """
        Compute reciprocal neighbor sets R_k for each sample in features.
        Args:
            features (Tensor): shape [N, D], normalized features
        Returns:
            rec_neighbors_list (list): list of sets containing reciprocal neighbors indices
        """
        neighbor_indices = compute_neighbors(features, self.k)
        N = features.shape[0]
        neighbor_sets = [set(neighbor_indices[i].cpu().numpy()) for i in range(N)]
        rec_neighbors_list = []
        for i in range(N):
            R_i = neighbor_sets[i]
            reciprocal_set = set()
            for j in R_i:
                if i in neighbor_sets[j]:
                    reciprocal_set.add(j)
            rec_neighbors_list.append(reciprocal_set)
        return rec_neighbors_list

    def compute_weight_matrix(self, rec_neighbors_list, positive_pairs):
        """
        Compute normalized weights w_{ij} for positive pairs.
        Args:
            rec_neighbors_list (list): reciprocal neighbors list
            positive_pairs (list of tuples): each (i,j) with same label
        Returns:
            torch.Tensor: normalized weights [num_POS]
        """
        W = []
        for (i, j) in positive_pairs:
            R_i = rec_neighbors_list[i]
            R_j = rec_neighbors_list[j]
            intersection = R_i.intersection(R_j)
            union = R_i.union(R_j)
            weight = len(intersection) / len(union) if len(union) > 0 else 0.0
            W.append(weight)
        W = np.array(W)
        sum_W = np.sum(W)
        if sum_W > 0:
            W /= sum_W  # normalize to sum=1
        return torch.from_numpy(W).float()

    def __call__(self, aug_features: torch.Tensor, orig_features: torch.Tensor, labels: torch.LongTensor):
        """
        Computes the weighted alignment loss over positive pairs in the batch.
        Args:
            aug_features (Tensor): shape [N, D], features from augmented images
            orig_features (Tensor): shape [N, D], features from original images
            labels (LongTensor): shape [N], IDs
        Returns:
            Scalar tensor: alignment loss
        """
        device = aug_features.device
        # Compute reciprocal neighbor sets
        rec_neighbors_list = self.compute_reciprocal_neighbors(orig_features)
        # Find positive pairs (matching labels)
        positive_pairs = []
        label_np = labels.cpu().numpy()
        for i in range(len(label_np)):
            for j in range(i+1, len(label_np)):
                if label_np[i] == label_np[j]:
                    positive_pairs.append((i,j))
        if len(positive_pairs) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        # Compute weights
        W = self.compute_weight_matrix(rec_neighbors_list, positive_pairs).to(device)
        # Calculate loss
        loss = 0.0
        total_weight = W.sum()
        for idx, (i, j) in enumerate(positive_pairs):
            w_ij = W[idx] / total_weight if total_weight > 0 else 1.0 / len(positive_pairs)
            diff = aug_features[i] - orig_features[j]
            loss += w_ij * torch.sum(diff ** 2)
        return loss

def compute_neighbors(features: torch.Tensor, k: int):
    """
    Compute k nearest neighbors for each feature vector.
    Args:
        features (Tensor): shape [N, D]
    Returns:
        neighbor_indices (LongTensor): shape [N, k]
    """
    features_np = features.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean').fit(features_np)
    distances, indices = nbrs.kneighbors(features_np)
    return torch.from_numpy(indices[:,1:])  # exclude self

def compute_pairwise_distances(x: torch.Tensor):
    """
    Compute pairwise Euclidean distance matrix.
    Args:
        x (Tensor): shape [N, D]
    Returns:
        dist (Tensor): shape [N, N]
    """
    sq = torch.sum(x ** 2, dim=1, keepdim=True)
    dist = sq - 2 * torch.mm(x, x.t()) + sq.t()
    dist = torch.clamp(dist, min=0.0)
    return torch.sqrt(dist + 1e-8)

def compute_uniformity(features: torch.Tensor):
    """
    Compute the uniformity loss for features.
    Args:
        features (Tensor): shape [N, D], assumed normalized
    Returns:
        Scalar tensor
    """
    N = features.shape[0]
    pairwise_dists = compute_pairwise_distances(features)
    # Exclude diagonal
    mask = torch.ones_like(pairwise_dists) - torch.eye(N, device=features.device)
    pairwise_dists = pairwise_dists * mask
    exp_term = torch.exp(-2 * pairwise_dists ** 2)
    sum_exp = torch.sum(exp_term) / (N * (N -1))
    return torch.log(sum_exp + 1e-8)

def compute_domain_uniformity(features: torch.Tensor, prototypes: torch.Tensor, domain_labels: torch.LongTensor, num_domains: int, N_proto: int=5):
    """
    Computes domain-specific uniformity loss by distributing features around domain prototypes.
    Args:
        features (Tensor): [batch_size, D], normalized features
        prototypes (Tensor): [num_classes, D], class prototypes
        domain_labels (LongTensor): [batch_size], domain index for each sample
        num_domains (int): total number of domains
        N_proto (int): number of nearest prototypes to consider
    Returns:
        loss (Tensor): scalar
    """
    total_loss = 0.0
    for d in range(num_domains):
        idxs = (domain_labels == d).nonzero(as_tuple=False).squeeze(1)
        if len(idxs) == 0:
            continue
        domain_feats = features[idxs]  # [num_samples_d, D]
        # For each feature, find nearest N_proto prototypes
        protos_d = prototypes  # assuming all prototypes; extension possible
        dists = torch.cdist(domain_feats, protos_d)
        topk_vals, topk_idxs = torch.topk(dists, N_proto, largest=False)  # [num_samples_d, N_proto]
        # Sum e^{-2 * dist^2}
        exp_terms = torch.exp(-2 * topk_vals ** 2)
        loss_d = torch.log(exp_terms.mean() + 1e-8)
        total_loss += loss_d
    return total_loss / (num_domains if num_domains > 0 else 1)

class CrossEntropyLossWrapper:
    """
    Wrapper for standard cross entropy loss.
    """
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, logits: torch.Tensor, labels: torch.LongTensor):
        return self.criterion(logits, labels)

class TripletLossWrapper:
    """
    Batch-hard triplet loss.
    """
    def __init__(self, margin=0.3):
        self.margin = margin

    def __call__(self, embeddings: torch.Tensor, labels: torch.LongTensor):
        """
        Implements batch hard triplet loss.
        """
        # Compute pairwise distances
        pdist = torch.nn.PairwiseDistance(p=2)
        dist_mat = compute_pairwise_distances(embeddings)

        batch_size = embeddings.size(0)
        mask_positive = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask_negative = ~mask_positive

        loss = torch.tensor(0.0, device=embeddings.device)
        eps = 1e-16
        for i in range(batch_size):
            # Hard positive
            pos_mask = mask_positive[i]
            neg_mask = mask_negative[i]
            if pos_mask.sum() <=1:
                continue
            dist_pos = dist_mat[i][pos_mask]
            dist_pos = dist_pos[dist_pos != 0]
            if len(dist_pos)==0:
                continue
            hardest_positive = dist_pos.max()
            # Hard negative
            dist_neg = dist_mat[i][neg_mask]
            if len(dist_neg)==0:
                continue
            hardest_negative = dist_neg.min()
            triplet_loss = F.relu(hardest_positive - hardest_negative + self.margin)
            loss += triplet_loss
        loss = loss / batch_size
        return loss

# Additional functions to be used for metrics can be added here, but they are auxiliary and can be imported elsewhere.

