# dataset_loader.py
import os
import math
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, List, Optional

class DatasetLoader:
    def __init__(self, dataset_names: List[str], config: Dict):
        """
        Initialize DatasetLoader with dataset names and configuration.
        Args:
            dataset_names (List[str]): List of dataset names to load.
            config (Dict): Configuration dictionary loaded from config.yaml.
        """
        self.dataset_names = dataset_names
        self.config = config

        # Extract dataset parameters
        self.dataset_name = config.get('dataset', {}).get('name', '')
        self.data_path = config.get('dataset', {}).get('data_path', '')
        self.normalizer = config.get('dataset', {}).get('normalizer', 'standard')
        # Data split ratios
        self.train_ratio = config.get('dataset', {}).get('train_split', 0.7)
        self.val_ratio = config.get('dataset', {}).get('val_split', 0.15)
        self.test_ratio = config.get('dataset', {}).get('test_split', 0.15)

        # Input and patch parameters
        self.seq_len = config.get('model', {}).get('input_sequence_length', 96)
        self.patch_size = config.get('model', {}).get('patch_size', 24)

        # Internal variables
        self.raw_data = {}  # to store original data
        self.normalized_data = {}  # to store normalized data
        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}

        # Normalization parameters (mean, std) per variable
        self.norm_params = {}

        # Load datasets
        self.load_data()

    def load_data(self):
        """
        Load datasets specified in dataset_names, apply normalization and segmentation.
        """
        for name in self.dataset_names:
            data = self._load_dataset_file(name)
            self.raw_data[name] = data

            # Normalize data based on training set
            norm_data, norm_params = self._normalize_data(data)
            self.normalized_data[name] = norm_data
            self.norm_params[name] = norm_params

            # Generate patches
            patches, patch_count = self._create_patches(norm_data)

            # Store patches
            setattr(self, f"{name}_patches", patches)

        # After loading all datasets, split into train/val/test
        self._create_splits()

    def _load_dataset_file(self, name: str) -> np.ndarray:
        """
        Load dataset file based on name.
        Args:
            name (str): Dataset name.
        Returns:
            np.ndarray: Data array of shape (timesteps, features).
        """
        file_path = os.path.join(self.data_path, name + '.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load CSV
        df = pd.read_csv(file_path)
        data = df.values  # shape (timesteps, features)

        # For univariate datasets (e.g., synthetic), ensure shape correctness
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data  # shape (T, F)

    def _normalize_data(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Normalize data using specified method.
        Args:
            data (np.ndarray): Data to normalize, shape (T, F).
        Returns:
            Tuple[np.ndarray, Dict]: Normalized data and normalization params.
        """
        if self.normalizer == 'standard':
            mean = data[:int(len(data)*self.train_ratio)].mean(axis=0)
            std = data[:int(len(data)*self.train_ratio)].std(axis=0)
            # Prevent division by zero
            std[std == 0] = 1.0
            norm_data = (data - mean) / std
            norm_params = {'mean': mean, 'std': std}
        elif self.normalizer == 'minmax':
            min_val = data[:int(len(data)*self.train_ratio)].min(axis=0)
            max_val = data[:int(len(data)*self.train_ratio)].max(axis=0)
            denom = max_val - min_val
            denom[denom == 0] = 1.0
            norm_data = (data - min_val) / denom
            norm_params = {'min': min_val, 'max': max_val}
        else:
            # Default fallback: no normalization
            norm_data = data
            norm_params = {}
        return norm_data, norm_params

    def _create_patches(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Segment the data into patches.
        Args:
            data (np.ndarray): Normalized data, shape (T, F).
        Returns:
            Tuple[np.ndarray, int]: Patches (N_patches, patch_size, features),
                                     number of patches.
        """
        T, F = data.shape
        patch_size = self.patch_size
        stride = patch_size  # non-overlapping patches

        patches = []
        for start_idx in range(0, T - patch_size + 1, stride):
            patch = data[start_idx:start_idx + patch_size]
            patches.append(patch)
        patches = np.stack(patches, axis=0)
        return patches, patches.shape[0]

    def _create_splits(self):
        """
        Split datasets into train, validation, and test sets based on ratios.
        For synthetic data or if dataset is large enough, split by index.
        """
        for name in self.dataset_names:
            data_length = len(getattr(self, f"{name}_patches"))
            train_end = int(data_length * self.train_ratio)
            val_end = train_end + int(data_length * self.val_ratio)

            # Slicing patches accordingly
            patches = getattr(self, f"{name}_patches")

            self.train_data[name] = patches[:train_end]
            self.valid_data[name] = patches[train_end:val_end]
            self.test_data[name] = patches[val_end:]

    def get_train_test_split(self, dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns train and test data tensors for specified dataset.
        Args:
            dataset_name (str): Name of dataset to retrieve.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (train_tensor, test_tensor)
        """
        train_np = self.train_data[dataset_name]
        test_np = self.test_data[dataset_name]

        # Convert to torch tensors (float32)
        train_tensor = torch.tensor(train_np, dtype=torch.float32)
        test_tensor = torch.tensor(test_np, dtype=torch.float32)
        return train_tensor, test_tensor
    
    def create_synthetic_data(self, length: int = 720, num_signals: int = 2,
                              tau: int = 24, S: int = 8, k: int = 5) -> np.ndarray:
        """
        Generate synthetic signals with periodic shocks for interpretability analysis.
        Args:
            length (int): Length of the generated series.
            num_signals (int): Number of independent signals.
            tau (int): Period of base signal.
            S (int): Shock pattern periodicity.
            k (int): Shock amplitude.
        Returns:
            np.ndarray: Synthetic data of shape (length, num_signals).
        """
        signals = []
        for _ in range(num_signals):
            base = np.random.randn(length)
            for t in range(length):
                if t % tau == 0:
                    base[t] += k
                if t % S == S // 2:
                    base[t] -= k
            signals.append(base)
        signals = np.stack(signals, axis=1)
        return signals  # shape (length, num_signals)
