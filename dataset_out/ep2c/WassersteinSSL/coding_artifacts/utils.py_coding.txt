## utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

# Constants
EPSILON = 1e-6

def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """
    Normalize feature vectors to unit \ell_2 norm.
    Args:
        features (torch.Tensor): shape (batch_size, feature_dim)
    Returns:
        torch.Tensor: normalized features, same shape
    """
    norms = features.norm(p=2, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=EPSILON)  # prevent division by zero
    normalized = features / norms
    return normalized

def compute_empirical_statistics(features: torch.Tensor) -> tuple:
    """
    Compute empirical mean and covariance of features.
    Args:
        features (torch.Tensor): shape (batch_size, feature_dim)
    Returns:
        mean (torch.Tensor): shape (feature_dim,)
        cov (torch.Tensor): shape (feature_dim, feature_dim)
    """
    mean = torch.mean(features, dim=0)
    centered = features - mean.unsqueeze(0)
    cov = (centered.T @ centered) / (features.shape[0] - 1)
    return mean, cov

def covariance_sqrt_eigen(covariance: torch.Tensor) -> torch.Tensor:
    """
    Compute the square root of a symmetric positive semi-definite matrix via eigen-decomposition.
    Args:
        covariance (torch.Tensor): shape (m, m)
    Returns:
        cov_sqrt (torch.Tensor): shape (m, m)
    """
    # Eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    # Clamp eigenvalues for numerical stability
    eigvals_clamped = torch.clamp(eigvals, min=0)
    sqrt_eigvals = torch.sqrt(eigvals_clamped)
    cov_sqrt = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
    return cov_sqrt

def compute_uniformity_metric_W2(mean: torch.Tensor, cov: torch.Tensor) -> float:
    """
    Compute the -W2 uniformity score based on the empirical mean and covariance.
    Args:
        mean (torch.Tensor): shape (feature_dim,)
        cov (torch.Tensor): shape (feature_dim, feature_dim)
    Returns:
        float: negative Wasserstein distance (-W2)
    """
    feature_dim = mean.shape[0]
    trace_cov = torch.trace(cov).item()
    cov_sqrt = covariance_sqrt_eigen(cov)
    trace_sqrt = torch.trace(cov_sqrt).item()
    mu_norm_sq = torch.sum(mean ** 2).item()
    W2 = np.sqrt(
        mu_norm_sq + 1 + trace_cov - (2.0 / np.sqrt(feature_dim)) * trace_sqrt
    )
    return -W2

def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    Args:
        preds (torch.Tensor): predicted labels, shape (n,)
        labels (torch.Tensor): true labels, shape (n,)
    Returns:
        float: accuracy percentage
    """
    correct = (preds == labels).sum().item()
    total = labels.shape[0]
    return correct / total

def plot_spectrum(singular_values: np.ndarray):
    """
    Plot the log-scaled singular values to visualize spectral decay.
    Args:
        singular_values (np.ndarray): shape (feature_dim,)
    """
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(singular_values)+1), np.log10(singular_values + EPSILON))
    plt.xlabel('Component index')
    plt.ylabel('Log of singular value')
    plt.title('Spectral decay of representation covariance matrix')
    plt.grid(True)
    plt.show()

def visualize_distribution(features: torch.Tensor, title: str = 'Feature distribution'):
    """
    Visualize the 2D distribution of features projected onto top 2 principal components.
    Args:
        features (torch.Tensor): shape (batch_size, feature_dim)
        title (str): plot title
    """
    # Convert to numpy for plotting
    features_np = features.detach().cpu().numpy()
    # Compute principal components via PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features_np)
    plt.figure(figsize=(6,6))
    plt.scatter(proj[:,0], proj[:,1], alpha=0.5, s=10)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Additional utility functions could include: loading eigen-decomposition with fallback,
# robust eigenvalue handling, normalization with device management, etc.

# Note: For eigen-decomposition, scipy.linalg.eigh is used, but here we rely on torch.linalg.eigh.
# If eigen-decomposition of covariance is slow or unstable, consider batching or regularization.
# Also, ensure that features are normalized to the sphere before covariance calculation if needed.
# This code assumes features are already prepared accordingly during training/evaluation
# and that the user calls these functions with proper tensors on the correct device.
