# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UniformityLoss(nn.Module):
    """
    Implements the uniformity metric based on Wasserstein distance (-W2) as described in the paper.
    Computes the empirical mean and covariance of features, then calculates the Wasserstein distance
    between the feature distribution and the approximate uniform spherical distribution modeled as
    Gaussian with zero mean and scaled identity covariance.
    """
    
    def __init__(self, loss_type='InfoNCE', uniformity_lambda=0.1, tau=0.2, feature_dim=128):
        """
        Args:
            loss_type (str): 'InfoNCE', 'MSE', or 'Covariance' specifying base SSL loss type.
            uniformity_lambda (float): Weight for the uniformity Wasserstein loss.
            tau (float): Temperature parameter for contrastive loss (if applicable).
            feature_dim (int): Dimensionality of the feature embeddings.
        """
        super().__init__()
        self.loss_type = loss_type
        self.uniformity_lambda = uniformity_lambda
        self.tau = tau
        self.feature_dim = feature_dim

        # Small epsilon for numerical stability in eigen decomposition
        self.epsilon = 1e-6

    def compute_wasserstein_distance(self, features):
        """
        Compute the negative uniformity metrics (-W2) based on features.
        Args:
            features (Tensor): Shape [batch_size, feature_dim]
        Returns:
            torch.Tensor: scalar tensor containing -W2 value
        """
        # Normalize features to the unit sphere
        z = F.normalize(features, p=2, dim=1)  # shape: [batch_size, feature_dim]
        n = z.shape[0]
        m = z.shape[1]
        device = z.device

        # Step 1: Compute empirical mean
        mu_hat = torch.mean(z, dim=0)  # [feature_dim]
        
        # Center features
        z_centered = z - mu_hat  # [n, m]
        # Step 2: Covariance estimation
        # Covariance: (Z_centered^T Z_centered) / (n -1)
        sigma_hat = (z_centered.T @ z_centered) / (n - 1)  # [m, m]
        
        # Eigen-decomposition of covariance matrix
        # Use torch.linalg.eigh since sigma_hat is symmetric positive semi-definite
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma_hat)
        # Clamp eigenvalues to non-negative for numerical stability
        eigenvalues = torch.clamp(eigenvalues, min=0)
        # Compute trace of sigma_hat
        trace_sigma = torch.sum(eigenvalues)
        # Compute sqrt of sigma_hat: V * diag(sqrt(eigenvalues)) * V^T
        sqrt_eigenvalues = torch.sqrt(eigenvalues)
        sigma_half = (eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T)
        trace_sigma_half = torch.sum(sqrt_eigenvalues)

        # Compute the Wasserstein distance
        W2 = torch.sqrt(
            torch.norm(mu_hat, p=2) ** 2  # ||mu_hat||^2
            + 1
            + trace_sigma
            - (2.0 / math.sqrt(m)) * trace_sigma_half
        )
        # Negative uniformity loss
        neg_W2 = -W2
        return neg_W2

    def compute_base_loss(self, features_view1, features_view2=None):
        """
        Compute the base SSL loss based on loss_type.
        If 'InfoNCE', features_view1 and view2 are positive pairs.
        If 'MSE', features are from two augmented views.
        For 'Covariance', assume features are used for decorrelation losses.
        """
        if self.loss_type == 'InfoNCE':
            # Expect features from two views
            # features_view1 and features_view2: [batch_size, feature_dim]
            # Normalize
            z1 = F.normalize(features_view1, p=2, dim=1)
            z2 = F.normalize(features_view2, p=2, dim=1)
            # Similarity matrix
            sim_matrix = torch.matmul(z1, z2.T) / self.tau  # scaled by temperature
            # Labels: diagonal entries are positives
            labels = torch.arange(z1.size(0), device=z1.device)
            # Contrastive loss (InfoNCE)
            loss = nn.CrossEntropyLoss()
            loss_val = loss(sim_matrix, labels)
            return loss_val
        elif self.loss_type == 'MSE':
            # For BYOL: mean squared error between normalized features
            z1 = F.normalize(features_view1, p=2, dim=1)
            z2 = F.normalize(features_view2, p=2, dim=1)
            loss_val = torch.mean((z1 - z2).pow(2))
            return loss_val
        elif self.loss_type == 'Covariance':
            # Covariance-based decorrelation loss (e.g., Barlow Twins)
            # features: [batch_size, feature_dim]
            z = F.normalize(features_view1, p=2, dim=1)
            # Cross-correlation matrix
            c = (z.T @ z) / z.shape[0]
            # Loss: sum of squared off-diagonal elements
            off_diag = c - torch.eye(c.size(0), device=c.device)
            loss_val = torch.sum(off_diag ** 2)
            return loss_val
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def forward(self, features_view1, features_view2=None):
        """
        Compute the total loss as base SSL loss + weighted uniformity loss.
        Args:
            features_view1 (Tensor): Batch features from view 1
            features_view2 (Tensor): Batch features from view 2 (if applicable)
        Returns:
            total_loss (Tensor), dict of individual losses for logging
        """
        # Compute base SSL loss
        base_loss = self.compute_base_loss(features_view1, features_view2)

        # Compute uniformity loss (-W2)
        neg_W2 = self.compute_wasserstein_distance(features_view1)

        # Total loss sum
        total_loss = base_loss + self.uniformity_lambda * neg_W2

        # Optional logging dict
        log_dict = {
            'base_loss': base_loss,
            'uniformity_loss': neg_W2,
            'total_loss': total_loss
        }

        return total_loss, log_dict
