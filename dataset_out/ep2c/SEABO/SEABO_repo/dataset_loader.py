## dataset_loader.py
import os
import numpy as np

class DatasetLoader:
    """
    Class responsible for loading expert demonstration data and unlabeled offline dataset.
    It handles data format interpretation based on whether demonstrations contain only observations
    or full state-action tuples, performs normalization, and prepares datasets for downstream processing.
    """
    def __init__(self, dataset_path: str = None, expert_path: str = None,
                 only_observations: bool = False, normalize: bool = True):
        """
        Initialize the DatasetLoader with dataset paths and configuration.
        
        Args:
            dataset_path (str): Path to the unlabeled dataset (.npy file).
            expert_path (str): Path to the expert demonstration dataset (.npy file).
            only_observations (bool): True if expert demos contain only observations.
            normalize (bool): Whether to normalize features for distance metric.
        """
        self.dataset_path = dataset_path
        self.expert_path = expert_path
        self.only_observations = only_observations
        self.normalize = normalize

        # Placeholders for loaded data
        self.expert_raw = None
        self.unlabeled_raw = None

        # Processed datasets
        self.expert_features = None
        self.unlabeled_features = None
        self.expert_full = None  # entire expert data (full transitions if applicable)
        self.unlabeled_full = None  # entire unlabeled data (full transitions)
        # Normalization statistics
        self.obs_mean = None
        self.obs_std = None
        self.action_dim = None

        self.load_data()

    def load_data(self):
        """
        Loads datasets from given file paths and performs preprocessing.
        """
        # Verify file existence
        if not os.path.isfile(self.expert_path):
            raise FileNotFoundError(f"Expert demonstration file not found at {self.expert_path}")
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"Unlabeled dataset file not found at {self.dataset_path}")

        # Load data
        self.expert_raw = np.load(self.expert_path)
        self.unlabeled_raw = np.load(self.dataset_path)

        # Check data shapes
        if self.expert_raw.size == 0:
            raise ValueError("Expert demonstration data is empty.")
        if self.unlabeled_raw.size == 0:
            raise ValueError("Unlabeled dataset is empty.")

        # Determine feature parsing based on only_observations
        if self.only_observations:
            # Expert demos: shape (T_expert, obs_dim)
            if self.expert_raw.ndim != 2:
                raise ValueError(f"Expert demo data shape {self.expert_raw.shape} inconsistent with only_observations=True.")
            obs_dim = self.expert_raw.shape[1]
            # Unlabeled data: shape (N, obs_dim + act_dim + obs_dim_next)
            if self.unlabeled_raw.ndim != 2:
                raise ValueError(f"Unlabeled data shape {self.unlabeled_raw.shape} should be 2D.")
            total_dim = self.unlabeled_raw.shape[1]
            # We plan to use only observations for neighbor search
            # For reward labeling, will use observation parts
            # Save features accordingly
            # Expert features: array of shape (T_expert, obs_dim)
            self.expert_features = self.expert_raw.copy().astype(np.float32)
            # Unlabeled features: shape (N, obs_dim)
            # assuming last obs_dim in each transition corresponds to s'
            # For neighbor search, use current state observations
            self.unlabeled_features = self.unlabeled_raw[:, :obs_dim].astype(np.float32)
        else:
            # Expert demos: shape (T_expert, obs_dim + act_dim)
            if self.expert_raw.ndim != 2:
                raise ValueError(f"Expert demo data shape {self.expert_raw.shape} inconsistent with only_observations=False.")
            feature_dim = self.expert_raw.shape[1]
            # Determine act_dim
            # For unlabeled data: shape (N, obs_dim + act_dim + obs_dim_next)
            if self.unlabeled_raw.ndim != 2:
                raise ValueError(f"Unlabeled data shape {self.unlabeled_raw.shape} should be 2D.")
            # Infer act_dim by comparing expert feature dimension
            # For simplicity, assume last part of expert is action dimension
            # but if expert contains only obs, then expert_raw shape differs
            # Let's assume act_dim is the difference between total dims if expert has actions
            # Otherwise, we need to check dataset consistency
            # This implementation assumes expert contains full transitions if only_observations=False
            # For unlabeled data transitions: shape (N, obs_dim + act_dim + obs_dim_next)
            total_dim = self.unlabeled_raw.shape[1]
            # Here, assumption: expert_raw shape matches (T_expert, obs_dim + act_dim)
            # For now, infer act_dim as the difference in dimensions if possible; else default to 0
            # But problem states that expert contains states and actions (not necessarily same as dataset)
            # User must verify consistency; here, for simplicity:
            self.expert_full = self.expert_raw.copy().astype(np.float32)
            # Extract features for neighbor search:
            # Use the full (s, a) concatenation
            self.expert_features = self.expert_full  # shape (T_expert, obs_dim + act_dim)

            # For unlabeled data: extract features (s, a)
            # Assuming columns: s, a, s' (plus possibly others)
            # For simplicity, extract first (obs_dim + act_dim) columns
            self.unlabeled_full = self.unlabeled_raw
            self.unlabeled_features = self.unlabeled_full[:, :self.expert_full.shape[1]].astype(np.float32)

        # Optional normalization
        if self.normalize:
            self._normalize_features()

        # Save full datasets
        # For potential later use, store as dicts
        if self.only_observations:
            self.expert_data = {
                'features': self.expert_features,  # shape (T_expert, obs_dim)
                'raw': self.expert_raw
            }
            self.unlabeled_data = {
                'features': self.unlabeled_features,  # shape (N, obs_dim)
                'raw': self.unlabeled_raw
            }
        else:
            self.expert_data = {
                'features': self.expert_features,  # shape (T_expert, obs_dim + act_dim)
                'raw': self.expert_full
            }
            self.unlabeled_data = {
                'features': self.unlabeled_features,  # shape (N, obs_dim + act_dim)
                'raw': self.unlabeled_full
            }

    def _normalize_features(self):
        """
        Normalize features to zero mean and unit std for each dimension.
        """
        # Concatenate all datasets for consistent stats if needed
        all_data = np.vstack([self.expert_features, self.unlabeled_features])
        self.obs_mean = np.mean(all_data, axis=0)
        self.obs_std = np.std(all_data, axis=0) + 1e-8  # prevent division by zero

        # Normalize expert features
        self.expert_features = (self.expert_features - self.obs_mean) / self.obs_std
        # Normalize unlabeled features
        self.unlabeled_features = (self.unlabeled_features - self.obs_mean) / self.obs_std

        # Update datasets
        if self.only_observations:
            self.expert_data['features'] = self.expert_features
            self.unlabeled_data['features'] = self.unlabeled_features
        else:
            self.expert_data['features'] = self.expert_features
            self.unlabeled_data['features'] = self.unlabeled_features

    def get_expert_data(self):
        """
        Returns:
            dict: containing expert features and raw data.
        """
        return self.expert_data

    def get_unlabeled_data(self):
        """
        Returns:
            dict: containing unlabeled features and raw data.
        """
        return self.unlabeled_data
