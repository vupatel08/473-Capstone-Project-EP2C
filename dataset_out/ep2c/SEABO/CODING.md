# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
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
```

## evaluation.py

```python
## evaluation.py
import gym
import numpy as np
import torch

class Evaluation:
    """
    Loads a trained policy and evaluates it in a specified environment over a given number of episodes.
    Computes metrics such as average return, standard deviation, and normalized score.
    """
    def __init__(self, policy, env_name: str, expert_return: float, env_random_return: float = 0.0, eval_episodes: int = 10):
        """
        Initialize the Evaluation class with the trained policy and environment details.
        
        Args:
            policy: Loaded policy object with a predict(obs) method returning actions.
            env_name (str): Name of the environment for evaluation (must be registered in gym).
            expert_return (float): Return achieved by the expert (used for normalization).
            env_random_return (float): Average return of a random policy (set to 0.0 if unknown).
            eval_episodes (int): Number of episodes to run for evaluation.
        """
        self.policy = policy
        self.env_name = env_name
        self.eval_episodes = eval_episodes
        self.expert_return = expert_return
        self.random_return = env_random_return
        # Initialize environment
        self.env = gym.make(self.env_name)
        # Optional: set seed for reproducibility
        # self.env.seed(42)
    
    def evaluate(self):
        """
        Run the policy in the environment for the specified number of episodes.
        
        Returns:
            dict: Contains 'average_return', 'std_return', and 'normalized_score'.
        """
        returns = []
        for ep in range(self.eval_episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                # Policy prediction
                # Ensure observation shape is (1, obs_dim)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = self.policy.predict(obs_tensor.cpu().numpy())[0]
                # Consume environment step
                next_obs, reward, done, info = self.env.step(action)
                total_reward += reward
                obs = next_obs
            returns.append(total_reward)
        # Compute metrics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        # Compute normalized score (assuming J_r=0 for random, J_e provided)
        normalized_score = ((mean_return - self.random_return) / (self.expert_return - self.random_return)) * 100
        return {
            'average_return': mean_return,
            'std_return': std_return,
            'normalized_score': normalized_score
        }
```

## main.py

```python
## main.py
import os
import yaml
import numpy as np
import gym
from dataset_loader import DatasetLoader
from reward_labeler import RewardLabeler
from model import OfflineModel
from trainer import Trainer
from evaluation import Evaluation

def main():
    # Step 1: Load configuration from 'config.yaml'
    config_path = 'config.yaml'
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract dataset paths and settings
    dataset_cfg = config.get('dataset', {})
    expert_demo_path = dataset_cfg.get('expert_demo_path', 'path/to/expert_demonstration.npy')
    unlabeled_data_path = dataset_cfg.get('unlabeled_data_path', 'path/to/unlabeled_dataset.npy')
    only_observations = dataset_cfg.get('only_observations', False)
    
    # Step 2: Initialize environment for evaluation
    env_name = config.get('evaluation', {}).get('environment', None)
    eval_episodes = config.get('evaluation', {}).get('eval_episodes', 10)
    if env_name is None:
        raise ValueError("Environment name for evaluation must be specified in config.")
    env = gym.make(env_name)
    
    # Step 3: Load datasets
    dataset_loader = DatasetLoader(
        dataset_path=unlabeled_data_path,
        expert_path=expert_demo_path,
        only_observations=only_observations,
        normalize=True
    )
    expert_data = dataset_loader.get_expert_data()
    unlabeled_data = dataset_loader.get_unlabeled_data()

    # Determine feature dimension and action dimension
    # Use expert data features for neighbor search dimensionality
    expert_features = expert_data['features']
    feature_dim = expert_features.shape[1]
    is_observation_only = only_observations
    
    # Step 4: Construct reward labels
    reward_cfg = config.get('reward_labeling', {})
    alpha = reward_cfg.get('alpha', 1.0)
    beta = reward_cfg.get('beta', 0.5)
    neighbor_count = reward_cfg.get('neighbors', 1)

    reward_labeler = RewardLabeler(
        expert_samples=expert_features,
        alpha=alpha,
        beta=beta,
        neighbor_count=neighbor_count
    )
    # Assign rewards to unlabeled data transitions
    labeled_dataset_array = reward_labeler.assign_rewards(
        unlabeled_data['features'],
        action_dim=expert_data['features'].shape[1] - (0 if only_observations else 0)
    )
    # Combine with raw data for offline RL training
    # Retrieve raw data for "actions" and "next observations"
    raw_unlabeled = unlabeled_data['raw']
    if only_observations:
        # features: current obs
        dataset_for_rl = {
            'observations': labeled_dataset_array[:, :feature_dim],  # s
            'actions': None,  # no actions if only observations/if desired
            'rewards': labeled_dataset_array[:, -1],  # pseudo rewards
            'next_observations': None
        }
        # Note: For algorithm compatibility, may need to handle missing actions accordingly.
        # For now, assume action dimension is zero or handle as per dataset.
        # Alternatively, if dataset contains actions, process accordingly.
    else:
        # Dataset includes full transitions: (s, a, s') and rewards
        # Extract s, a, s' from raw dataset
        dataset_for_rl = {
            'observations': raw_unlabeled[:, :feature_dim],  # s
            'actions': raw_unlabeled[:, feature_dim:feature_dim + expert_data['features'].shape[1] - feature_dim],  # a
            'rewards': labeled_dataset_array[:, -1],  # pseudo rewards
            'next_observations': raw_unlabeled[:, feature_dim + expert_data['features'].shape[1] - feature_dim:]  # s'
        }

    # Step 5: Initialize offline RL model
    model_type = config.get('model', {}).get('type', 'iql')
    model_params = config.get('model', {}).get('params', {})
    offline_model = OfflineModel(model_type=model_type, params=model_params)
    
    # Step 6: Train offline RL policy
    # Prepare dataset in required format
    dataset_tensor = {
        'observations': np.array(dataset_for_rl['observations']),
        'actions': np.array(dataset_for_rl['actions']),
        'rewards': np.array(dataset_for_rl['rewards']),
        'next_observations': np.array(dataset_for_rl['next_observations'])
    }
    total_timesteps = config.get('training', {}).get('epochs', 50) * (
        len(dataset_tensor['observations']) // dataset_params.get('batch_size', 256))
    trainer = Trainer(
        model=offline_model,
        dataset=dataset_tensor,
        env_name=env_name,
        total_steps=int(total_timesteps),
        batch_size=dataset_params.get('batch_size', 256),
        epochs=int(config.get('training', {}).get('epochs', 50)),
        eval_interval=int(config.get('evaluation', {}).get('eval_interval', 10000))
    )
    trainer.train()

    # Step 7: Save the trained policy
    save_path = 'seabo_trained_policy.pth'
    offline_model.save(save_path)
    print(f"Saved trained policy to {save_path}")

    # Step 8: Load and evaluate the policy
    # For evaluation, instantiate Evaluation class
    # For the ground-truth expert return, try to load or set a placeholder
    # Here, for demonstration, use a placeholder value; replace with actual if known
    # Alternatively, precompute or pass from dataset
    # For now, set to None and handle accordingly
    expert_return = None  # Placeholder; should be set based on dataset info
    # For simplicity, set to zero
    expert_return = 0.0

    evaluation = Evaluation(
        policy=offline_model,
        env_name=env_name,
        expert_return=expert_return,
        env_random_return=0.0,
        eval_episodes=eval_episodes
    )
    eval_metrics = evaluation.evaluate()
    print(f"Evaluation Results:\n"
          f"Average Return: {eval_metrics['average_return']}\n"
          f"Normalized Score: {eval_metrics['normalized_score']:.2f}\n"
          f"Std of Return: {eval_metrics['std_return']:.2f}")

if __name__ == "__main__":
    main()
```

## model.py

```python
# model.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Optional
import os

class MLP(nn.Module):
    """
    A simple multi-layer perceptron network with configurable hidden layers.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list = [256, 256], activation=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class OfflineModel:
    """
    Encapsulates neural networks for offline RL, primarily supporting IQL.
    Supports initialization, training, inference, saving/loading.
    """
    def __init__(self, model_type: str = "iql", params: Optional[Dict] = None):
        """
        Initialize the OfflineModel with specified algorithm type and parameters.
        """
        if params is None:
            params = {}
        self.model_type = model_type.lower()
        self.params = params

        # Extract hyperparameters with defaults
        hidden_layers = params.get("hidden_layers", [256, 256])
        self.lr = params.get("learning_rate", 3e-4)
        self.batch_size = params.get("batch_size", 256)
        self.total_timesteps = int(params.get("total_timesteps", 1e6))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Placeholders for networks and optimizers
        self.actor = None
        self.critic1 = None
        self.critic2 = None
        self.value = None

        # For IQL, we implement:
        # - Value network (V)
        # - Two Q-networks (Q1, Q2)
        # Setup the networks depending on model_type
        self._build_networks(input_dim=None, action_dim=None)

        # Setup optimizers
        self._init_optimizers()

        # Optionally, load existing weights
        self.save_path = "iql_model.pth"
        if os.path.exists(self.save_path):
            self.load(self.save_path)

    def _build_networks(self, input_dim: int = None, action_dim: int = None):
        """
        Build neural networks based on model type.
        """
        # For demonstration, assume input_dim and action_dim are given
        # For actual use, user should set input/output dims accordingly
        # Alternatively, define a method to initialize with dataset dims later
        # Here, assuming complete info is provided during train() call for flexibility
        # For now, we initialize with dummy dims, to be replaced
        if input_dim is None or action_dim is None:
            # During __init__, not enough info; networks are initialized during train
            self.initialized = False
        else:
            self.initialized = True
            # For IQL:
            obs_input_dim = input_dim
            act_input_dim = action_dim
            value_output_dim = 1
            q_output_dim = 1

            self.value_net = MLP(obs_input_dim, value_output_dim, self.params.get("hidden_layers", [256, 256]))
            self.q1_net = MLP(obs_input_dim + act_input_dim, q_output_dim, self.params.get("hidden_layers", [256, 256]))
            self.q2_net = MLP(obs_input_dim + act_input_dim, q_output_dim, self.params.get("hidden_layers", [256, 256]))
            
            self.value_net.to(self.device)
            self.q1_net.to(self.device)
            self.q2_net.to(self.device)

    def _init_optimizers(self):
        """
        Initialize optimizers for networks.
        """
        if hasattr(self, "value_net"):
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
            self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=self.lr)
            self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=self.lr)

    def initialize_networks(self, input_dim: int, action_dim: int):
        """
        Initialize networks based on dataset dimensions.
        This should be called before training if dims are unknown at init.
        """
        self._build_networks(input_dim, action_dim)
        self._init_optimizers()
        
    def save(self, filename: str):
        """
        Save model weights to file.
        """
        torch.save({
            'value_net_state_dict': self.value_net.state_dict(),
            'q1_net_state_dict': self.q1_net.state_dict(),
            'q2_net_state_dict': self.q2_net.state_dict(),
        }, filename)

    def load(self, filename: str):
        """
        Load model weights from file.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.q1_net.load_state_dict(checkpoint['q1_net_state_dict'])
        self.q2_net.load_state_dict(checkpoint['q2_net_state_dict'])

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Given observations, output actions.
        For IQL, typically use the actor network, if available.
        """
        # For simplicity, implementing a deterministic policy: select action max Q
        # As placeholder, use Q-values to select actions (not in original IQL)
        # Here, just return an action approximation, e.g., argmax over Q-values
        # Note: The original IQL does not provide an explicit actor network.
        # To align with the role, we can implement a simple deterministic policy:
        # For demonstration, output a zero vector (or previous best estimate).
        # In practice, user can implement a learned actor.
        # Here, we implement a naive policy: action maximizing Q-value
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        # Generate an estimated action via a placeholder (e.g., zeros)
        # or implement a learned actor network if available.
        # For now, pick q1 and q2 to get an action—this is just illustrative.
        # Since no actor network, just output zeros.
        action_dim = getattr(self, "action_dim", obs.shape[1])  # fallback
        actions = np.zeros((obs.shape[0], action_dim))
        return actions

    def train(self, dataset: Dict, total_timesteps: int = 1_000_000):
        """
        Train the policy using dataset of (s, a, r, s') tuples.
        This is a simplified placeholder for training IQL.
        """
        # Dataset expected: dict with keys: 'observations', 'actions', 'rewards', 'next_observations'
        # For modularity, check for keys and prepare data
        # Also, during first training, initialize network dims if not done
        if not hasattr(self, "initialed") or not getattr(self, "initialized", False):
            # Infer dims from dataset
            sample = dataset['observations'][0]
            self.action_dim = dataset['actions'].shape[1]
            input_dim = sample.shape[0]
            self.initialize_networks(input_dim, self.action_dim)

        # Extract data
        obs = torch.tensor(dataset['observations'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(dataset['actions'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(dataset['rewards'], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_obs = torch.tensor(dataset['next_observations'], dtype=torch.float32).to(self.device)

        dataset_size = obs.shape[0]
        indices = np.arange(dataset_size)

        # Training loop
        for timestep in range(total_timesteps):
            # Randomly sample a batch
            batch_idx = np.random.choice(indices, self.batch_size, replace=False)
            batch_obs = obs[batch_idx]
            batch_actions = actions[batch_idx]
            batch_rewards = rewards[batch_idx]
            batch_next_obs = next_obs[batch_idx]

            # Update Critic Q1 and Q2
            with torch.no_grad():
                # Compute target V for next state
                target_V = self.value_net(batch_next_obs)
                target_Q = batch_rewards + 0.99 * target_V

            # Critic loss (MSE)
            current_Q1 = self.q1_net(torch.cat([batch_obs, batch_actions], dim=1))
            current_Q2 = self.q2_net(torch.cat([batch_obs, batch_actions], dim=1))
            loss_q1 = F.mse_loss(current_Q1, target_Q)
            loss_q2 = F.mse_loss(current_Q2, target_Q)
            # Optimize critics
            self.q1_optimizer.zero_grad()
            loss_q1.backward()
            self.q1_optimizer.step()
            self.q2_optimizer.zero_grad()
            loss_q2.backward()
            self.q2_optimizer.step()

            # Update Value network (using expectile or similar)
            with torch.no_grad():
                q1_val = self.q1_net(torch.cat([batch_obs, batch_actions], dim=1))
                q2_val = self.q2_net(torch.cat([batch_obs, batch_actions], dim=1))
                min_q = torch.min(q1_val, q2_val)
            value_pred = self.value_net(batch_obs)
            # Loss for value (here, following IQL's value update with expectile or L2)
            # For simplicity, use L2 loss with min_q
            loss_v = F.mse_loss(value_pred, min_q.detach())
            self.value_optimizer.zero_grad()
            loss_v.backward()
            self.value_optimizer.step()

            # Optional: log losses every certain steps
            if (timestep + 1) % 10000 == 0:
                print(f"Step {timestep+1}/{total_timesteps} - Q Loss: {(loss_q1.item() + loss_q2.item())/2:.4f}, V Loss: {loss_v.item():.4f}")

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute the value estimates for observation batch.
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            v = self.value_net(obs_tensor).cpu().numpy()
        return v

    def get_q_value(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Compute Q-values for given obs and actions.
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q1 = self.q1_net(torch.cat([obs_tensor, actions_tensor], dim=1)).cpu().numpy()
            q2 = self.q2_net(torch.cat([obs_tensor, actions_tensor], dim=1)).cpu().numpy()
        return np.minimum(q1, q2)
```

## reward_labeler.py

```python
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
```

## trainer.py

```python
## trainer.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
import os

class Trainer:
    """
    Manages training of an offline RL algorithm (e.g., IQL) using datasets with pseudo-rewards.
    Implements the full training loop: batching, model updates, periodic evaluation, and checkpointing.
    """
    def __init__(
        self,
        model,
        dataset: Dict,
        env_name: str,
        total_steps: int = 1_000_000,
        batch_size: int = 256,
        epochs: int = 50,
        eval_interval: int = 10_000,
        save_path: str = "checkpoint.pt",
        device: Optional[torch.device] = None,
        validation_data: Optional[Dict] = None
    ):
        """
        Args:
            model (OfflineModel): the offline RL model instance (e.g., IQL).
            dataset (Dict): dataset with keys 'observations', 'actions', 'rewards', 'next_observations'.
            env_name (str): environment name for evaluation.
            total_steps (int): total number of training steps.
            batch_size (int): batch size for SGD.
            epochs (int): number of full passes over data.
            eval_interval (int): steps between evaluations.
            save_path (str): path to save the trained model.
            device (torch.device): device to run training on.
            validation_data (Optional[Dict]): optional validation dataset.
        """
        self.model = model
        self.env_name = env_name
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.save_path = save_path
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.validation_data = validation_data

        # Prepare dataset
        self._prepare_dataset(dataset)

        # Initialize optimizer, etc., if not already set in model
        if hasattr(self.model, "initialize_networks") and not hasattr(self.model, "initialized") or not self.model.initialized:
            # Infer dimension info
            obs_sample = self.dataset['observations'][0]
            act_sample = self.dataset['actions'][0]
            input_dim = obs_sample.shape[0]
            action_dim = self.dataset['actions'].shape[1]
            self.model.initialize_networks(input_dim, action_dim)
            self.model._init_optimizers()

        # Set model to device
        self.model.value_net.to(self.device)
        self.model.q1_net.to(self.device)
        self.model.q2_net.to(self.device)

    def _prepare_dataset(self, dataset: Dict):
        """
        Converts dataset arrays to tensors on the correct device.
        Expects keys: 'observations', 'actions', 'rewards', 'next_observations'.
        """
        self.dataset = {}
        for key in ['observations', 'actions', 'rewards', 'next_observations']:
            arr = dataset[key]
            self.dataset[key] = torch.tensor(arr, dtype=torch.float32).to(self.device)
        self.dataset_size = self.dataset['observations'].shape[0]
        self.indices = np.arange(self.dataset_size)

    def train(self):
        """
        Main training loop: iterate over epochs and mini-batches, perform model updates,
        periodically evaluate, and checkpoint.
        """
        total_training_steps = self.epochs * (self.dataset_size // self.batch_size)
        current_step = 0

        for epoch in range(self.epochs):
            # Shuffle dataset indices at each epoch
            np.random.shuffle(self.indices)

            for batch_start in range(0, self.dataset_size, self.batch_size):
                batch_indices = self.indices[batch_start:batch_start + self.batch_size]
                # Select batch data
                batch_obs = self.dataset['observations'][batch_indices]
                batch_actions = self.dataset['actions'][batch_indices]
                batch_rewards = self.dataset['rewards'][batch_indices]
                batch_next_obs = self.dataset['next_observations'][batch_indices]

                # Perform model update
                self.model.train({
                    'observations': batch_obs,
                    'actions': batch_actions,
                    'rewards': batch_rewards,
                    'next_observations': batch_next_obs
                }, total_timesteps=1)

                current_step += 1

                # Periodic evaluation
                if current_step % self.eval_interval == 0:
                    print(f"Step {current_step}/{total_training_steps} - Evaluating policy...")
                    eval_metrics = self.evaluate()
                    print(f"Evaluation results at step {current_step}: {eval_metrics}")
                    # Save checkpoint
                    self.model.save(self.save_path)

        # Save final model after training
        print("Training completed. Saving final model...")
        self.model.save(self.save_path)

    def evaluate(self, n_eval_episodes: int = 10) -> Dict:
        """
        Run policy in environment for specified episodes, compute mean return.
        Args:
            n_eval_episodes (int): number of episodes for evaluation.
        Returns:
            dict: metrics, e.g., mean normalized return.
        """
        import gym
        env = gym.make(self.env_name)
        total_return = 0.0

        for ep in range(n_eval_episodes):
            obs = env.reset()
            done = False
            ep_return = 0.0
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                # Get action from model (deterministic)
                action = self.model.predict(obs_tensor.cpu().numpy())[0]
                obs, reward, done, info = env.step(action)
                ep_return += reward
            total_return += ep_return

        avg_return = total_return / n_eval_episodes

        # Normalize the score if needed; for simplicity, return raw
        return {'avg_return': avg_return}

```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\SEABO\SEABO_repo`
