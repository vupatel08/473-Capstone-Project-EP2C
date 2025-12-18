## reward_labeler.py
import numpy as np
from sklearn.neighbors import KDTree

class RewardLabeler:
    """
    Assigns pseudo-rewards to unlabeled data based on proximity to expert demonstration data.
    Implements search with KD-tree for nearest neighbor discovery, and computes rewards via 
    an exponential decay function related to distance.
    """
    def __init__(self, expert_samples: np.ndarray, alpha: float = 1.0, beta: float = 0.5, neighbor_count: int = 1):
        """
        Initialize the RewardLabeler with expert samples and hyperparameters.
        
        Args:
            expert_samples (np.ndarray): Expert transition samples, shape (T_expert, feature_dim).
            alpha (float): Reward scaling factor.
            beta (float): Decay parameter for exponential reward.
            neighbor_count (int): Number of nearest neighbors to consider, default is 1.
        """
        self.expert_samples = expert_samples
        self.alpha = alpha
        self.beta = beta
        self.neighbor_count = neighbor_count

        # Build KD-tree index for expert samples for fast nearest neighbor queries
        # Using default leaf_size=40 for balanced performance
        self.kdtree = KDTree(self.expert_samples, leaf_size=40)

    def assign_rewards(self, unlabeled_data: np.ndarray, action_dim: int) -> np.ndarray:
        """
        Assigns reward labels to the unlabeled dataset based on proximity to expert samples.
        
        Args:
            unlabeled_data (np.ndarray): Unlabeled transitions, shape (N, feature_dim_in_dataset).
                Expected format: for each transition, features should include:
                - observation(s) and action(s), depending on data format.
            action_dim (int): Dimension of the action space, used for distance normalization.
        
        Returns:
            np.ndarray: Dataset array with an added reward column, shape (N, feature_dim+1).
        """
        # Validate inputs
        if not isinstance(unlabeled_data, np.ndarray):
            raise TypeError("unlabeled_data must be a numpy.ndarray")
        if not isinstance(action_dim, int):
            raise TypeError("action_dim must be an integer")
        if unlabeled_data.ndim != 2:
            raise ValueError("unlabeled_data should be 2D array with shape (N, feature_dim)")

        # Perform batched query for nearest neighbors
        distances, _ = self.kdtree.query(unlabeled_data, k=self.neighbor_count)
        # 'distances' shape: (N, neighbor_count)

        # For simplicity, consider only the nearest neighbor (closest)
        min_distances = distances[:, 0]

        # Compute rewards using the specified exponential decay formula
        # Divide distance by action_dim for normalization
        scaled_distances = min_distances / max(action_dim, 1e-8)  # avoid division by zero
        rewards = self.alpha * np.exp(-self.beta * scaled_distances)

        # Append rewards to the unlabeled data
        # Create a new array with an extra column for reward
        labeled_dataset = np.hstack((unlabeled_data, rewards.reshape(-1, 1)))

        return labeled_dataset
