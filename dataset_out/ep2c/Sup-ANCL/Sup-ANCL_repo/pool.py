## pool.py
import numpy as np
import torch
from collections import defaultdict, deque

class FeaturePool:
    """
    Implements class-specific feature queues for supervised ANCL.
    Supports enqueueing features, sampling positives, optional EMA updates,
    and flexible pool management strategies.
    """
    def __init__(self, size: int = 8192, num_classes: int = 100,
                 feature_dim: int = 128, update_with_ema: bool = True,
                 ema_m: float = 0.99, device: torch.device = torch.device('cpu')):
        """
        Args:
            size (int): total size of the feature pool across all classes.
            num_classes (int): total number of classes.
            feature_dim (int): dimension of feature vectors.
            update_with_ema (bool): whether to update features via EMA (SUPBYOL).
            ema_m (float): EMA momentum coefficient.
            device (torch.device): device for storing features.
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.update_with_ema = update_with_ema
        self.ema_m = ema_m

        # Compute per-class buffer size (floored division)
        self.buffer_size_per_class = size // num_classes
        self.total_size = self.buffer_size_per_class * num_classes
        # For classes where size isn't divisible, last class can get remaining slots
        remainder = size - self.total_size
        self.class_buffer_sizes = [self.buffer_size_per_class] * num_classes
        for y in range(remainder):
            self.class_buffer_sizes[y] += 1

        # Initialize buffers: dict: class_label -> tensor buffer (maxlen=buffer size)
        # Using torch tensors stored on device
        self.buffers = {}
        # Maintain current insertion index for each class for FIFO overwriting
        self.next_idx = {}
        for y, buf_size in enumerate(self.class_buffer_sizes):
            self.buffers[y] = torch.zeros((buf_size, feature_dim), device=self.device)
            self.next_idx[y] = 0

    def enqueue(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Insert features into class buffers, updating with optional EMA in SUPBYOL.
        Args:
            features (torch.Tensor): shape [batch_size, feature_dim]
            labels (torch.Tensor): shape [batch_size]
        """
        batch_size = features.shape[0]
        for i in range(batch_size):
            y = int(labels[i].item())
            feat = features[i]
            # Normalize feature
            feat = torch.nn.functional.normalize(feat, p=2, dim=0)
            # Determine insertion index
            idx = self.next_idx[y]
            buf_size = self.class_buffer_sizes[y]
            # Enqueue by overwriting oldest features (cyclic)
            self.buffers[y][idx] = feat
            # Update pointer
            self.next_idx[y] = (idx + 1) % buf_size
            # If EMA is enabled (SUPBYOL), update the buffer feature via EMA
            if self.update_with_ema:
                # current stored feature
                old_feat = self.buffers[y][idx]
                # EMA update
                new_feat = self.ema_m * old_feat + (1 - self.ema_m) * feat
                new_feat = torch.nn.functional.normalize(new_feat, p=2, dim=0)
                self.buffers[y][idx] = new_feat

    def get_positives(self, y: int, M: int = None):
        """
        Retrieve all or M randomly sampled features for class y.
        Args:
            y (int): class label
            M (int or None): number of positives to sample; if None, return all
        Returns:
            torch.Tensor: shape [num_samples, feature_dim]
        """
        feats = self.buffers[y]
        valid_mask = (feats.norm(p=2, dim=1) > 0)  # To identify non-init zero entries
        valid_feats = feats[valid_mask]

        num_feats = valid_feats.shape[0]
        if num_feats == 0:
            # No features stored yet; return zeros
            return torch.zeros((1, self.feature_dim), device=self.device)
        if M is None or M == 'all':
            # Return all features
            return valid_feats
        else:
            # Sample M features, with replacement if needed
            M = int(M)
            if num_feats >= M:
                indices = torch.randint(0, num_feats, (M,), device=self.device)
            else:
                # Less features than M: sample with replacement
                indices = torch.randint(0, num_feats, (M,), device=self.device)
            sampled_feats = valid_feats[indices]
            return sampled_feats

    def get_buffer_for_class(self, y: int):
        """
        Return current features stored for class y.
        """
        return self.buffers[y]

    def buffer_size(self, y: int):
        """
        Return current active size of class y buffer
        """
        feats = self.buffers[y]
        mask = feats.norm(p=2, dim=1) > 0
        return int(mask.sum().item())

