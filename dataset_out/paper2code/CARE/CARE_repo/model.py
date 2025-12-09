### model.py
"""
This module defines neural network encoder classes for different data modalities:
- ResNetEncoder: a standard ResNet-50 based encoder for images.
- DeepSetEncoder: a permutation-invariant encoder for protein point clouds.

Both encoders output normalized features on the unit sphere, suitable for contrastive and equivariance objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def normalize_features(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Normalize the input tensor along specified dimension to have unit norm.
    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to normalize.
    Returns:
        torch.Tensor: Normalized tensor.
    """
    return F.normalize(x, p=2, dim=dim)


class ResNetEncoder(nn.Module):
    """
    ResNet-50 based encoder for images.
    Outputs normalized features before the projection head.
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary with keys:
                - 'embedding_dim' (int): dimension of the embedding output.
                - 'projection_head' (bool): whether to include a projection MLP.
                - 'pretrained' (bool): whether to load ImageNet pretrained weights.
        """
        super(ResNetEncoder, self).__init__()
        self.embedding_dim = config.get('embedding_dim', 128)
        self.projection_head_enabled = config.get('projection_head', True)
        pretrained = config.get('pretrained', False)

        # Load ResNet-50 backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        # Remove the fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # all layers except fc

        # Define projection head if enabled
        if self.projection_head_enabled:
            self.projection_head = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, self.embedding_dim)
            )
        else:
            self.projection_head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x (torch.Tensor): Input images tensor of shape (batch_size, 3, H, W)
        Returns:
            torch.Tensor: Normalized embedding of shape (batch_size, embedding_dim)
        """
        features = self.backbone(x)  # shape: (batch_size, 2048, 1, 1)
        features = torch.flatten(features, start_dim=1)  # shape: (batch_size, 2048)
        projected = self.projection_head(features)  # shape: (batch_size, embedding_dim)
        normalized = normalize_features(projected, dim=1)
        return normalized


class DeepSetEncoder(nn.Module):
    """
    Permutation-invariant encoder for protein point clouds.
    Uses shared point-wise MLPs followed by a pooling operation.
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary with keys:
                - 'n_points' (int): number of points in the point cloud.
                - 'embedding_dim' (int): output feature dimension.
                - 'use_projection' (bool): whether to include a projection head.
        """
        super(DeepSetEncoder, self).__init__()
        self.n_points = config.get('n_points', 1024)
        self.embedding_dim = config.get('embedding_dim', 128)
        self.use_projection = config.get('use_projection', True)

        # Point-wise embedding MLP
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Final set-level embedding
        self.pooling = nn.AdaptiveAvgPool1d(1)  # pooling over points

        # Optional projection head
        if self.use_projection:
            self.projection_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.embedding_dim)
            )
        else:
            self.projection_head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input point clouds of shape (batch_size, n_points, 3)
        Returns:
            torch.Tensor: Normalized embedding of shape (batch_size, embedding_dim)
        """
        # Process each point individually
        batch_size, n_points, _ = x.shape
        x_flat = x.view(-1, 3)  # (batch_size * n_points, 3)
        point_features = self.point_mlp(x_flat)  # (batch_size * n_points, 256)
        point_features = point_features.view(batch_size, n_points, -1)  # (batch, n_points, 256)
        # Pool over points
        pooled = torch.mean(point_features, dim=1)  # (batch_size, 256)
        projected = self.projection_head(pooled)  # (batch_size, embedding_dim)
        normalized = normalize_features(projected, dim=1)
        return normalized
