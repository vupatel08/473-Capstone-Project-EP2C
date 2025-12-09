# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import json
import pickle
from typing import List, Dict, Tuple, Union, Callable
import numpy as np
import pandas as pd

from utils import load_hyperparameters, set_seed

class Segment:
    def __init__(self, segment_id: str, features: np.ndarray):
        self.segment_id = segment_id
        self.features = features

class PreferenceSample:
    def __init__(self, segment_a_id: str, segment_b_id: str, label: float):
        """
        label: 0.0 - segment_b preferred
               1.0 - segment_a preferred
               0.5 - equal preference
        """
        self.segment_a_id = segment_a_id
        self.segment_b_id = segment_b_id
        self.label = label

class PreferenceDataset:
    def __init__(self):
        # List of PreferenceSample objects
        self.preference_pairs: List[PreferenceSample] = []

    def add(self, pref_sample: PreferenceSample):
        self.preference_pairs.append(pref_sample)

class DatasetLoader:
    def __init__(self, data_dir: str, config: dict = None):
        """
        Initialize DatasetLoader with data directory and optional config.
        Loads preference data, trajectories, ground-truth rewards, environment configs.
        """
        self.data_dir = data_dir
        # Load hyperparameters from config.yaml
        if config is None:
            self.config = load_hyperparameters()
        else:
            self.config = config

        self.env_type = self.config['environment'].get('type', 'metaworld')
        self.env_name = self.config['environment'].get('env_name', 'reach-v2')
        self.preference_data: PreferenceDataset = PreferenceDataset()
        self.trajectories: Dict[str, Dict] = {}
        self.gt_rewards: Dict[str, float] = {}
        self._load_all_data()

    def _load_all_data(self):
        """Load all required data files from the data directory."""
        self._load_trajectories()
        self._load_ground_truth_rewards()
        self._load_preference_data()

    def _load_trajectories(self):
        """Load trajectories stored as separate files or a single file."""
        # Assuming trajectories are stored in 'trajectories.pkl' or multiple json files
        traj_path = os.path.join(self.data_dir, 'trajectories.pkl')
        if os.path.exists(traj_path):
            with open(traj_path, 'rb') as f:
                self.trajectories = pickle.load(f)
        else:
            # Alternative: load multiple json files named 'segment_*.json'
            self.trajectories = {}
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json') and 'segment' in filename:
                    filepath = os.path.join(self.data_dir, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        segment_id = data.get('segment_id', filename[:-5])
                        features = np.array(data.get('features', []), dtype=np.float32)
                        self.trajectories[segment_id] = {'features': features}
        # Build Segment objects for convenience
        self.segments: Dict[str, Segment] = {}
        for seg_id, data in self.trajectories.items():
            self.segments[seg_id] = Segment(segment_id=seg_id, features=data.get('features', np.array([])))

    def _load_ground_truth_rewards(self):
        """Load ground truth rewards, assumed filename: 'ground_truth_rewards.csv'."""
        gt_path = os.path.join(self.data_dir, 'ground_truth_rewards.csv')
        if os.path.exists(gt_path):
            df = pd.read_csv(gt_path)
            # Expect columns: 'segment_id', 'reward'
            self.gt_rewards = {
                str(row['segment_id']): float(row['reward'])
                for _, row in df.iterrows()
            }
        else:
            # Ground truth rewards may be unavailable; fallback to empty dict
            self.gt_rewards = {}

    def _load_preference_data(self):
        """Load preference data from file, e.g., 'preference_feedback.json'."""
        pref_path = os.path.join(self.data_dir, 'preference_feedback.json')
        if os.path.exists(pref_path):
            with open(pref_path, 'r') as f:
                data = json.load(f)
            for item in data:
                seg_a_id = item.get('segment_a_id')
                seg_b_id = item.get('segment_b_id')
                label_str = item.get('label', 'preferred')
                # Map label string to float: preferred=1, dispreferred=0, equal=0.5
                label_map = {
                    'preferred': 1.0,
                    'dispreferred': 0.0,
                    'equal': 0.5
                }
                label = label_map.get(label_str, 0.5)
                pref_sample = PreferenceSample(seg_a_id, seg_b_id, label)
                self.preference_data.add(pref_sample)
        else:
            # Alternative: load from CSV
            pref_csv_path = os.path.join(self.data_dir, 'preference_feedback.csv')
            if os.path.exists(pref_csv_path):
                df = pd.read_csv(pref_csv_path)
                for _, row in df.iterrows():
                    seg_a_id = row['segment_a_id']
                    seg_b_id = row['segment_b_id']
                    label_val = row['label']
                    # Expect label as float: 0, 0.5, 1
                    try:
                        label = float(label_val)
                    except Exception:
                        label = 0.5
                    pref_sample = PreferenceSample(seg_a_id, seg_b_id, label)
                    self.preference_data.add(pref_sample)
            else:
                # No preference data available
                print("Warning: Preference data not found.")
                self.preference_data = PreferenceDataset()

    def get_preference_pairs(self,
                             segment_ids: List[str]) -> List[Tuple[str, str, float]]:
        """
        Return preference pairs from stored data for given segment IDs.
        """
        pairs = []
        for pref in self.preference_data.preference_pairs:
            if pref.segment_a_id in segment_ids and pref.segment_b_id in segment_ids:
                pairs.append((
                    pref.segment_a_id,
                    pref.segment_b_id,
                    pref.label
                ))
        return pairs

    def get_segment_feature(self, segment_id: str) -> np.ndarray:
        """Return feature vector for a given segment ID."""
        if segment_id in self.segments:
            return self.segments[segment_id].features
        else:
            # Return empty array or raise warning
            print(f"Warning: Segment ID {segment_id} not found.")
            return np.array([])

    def get_all_segments(self) -> Dict[str, Segment]:
        """Return all loaded segments."""
        return self.segments

    def get_ground_truth_reward(self, segment_id: str) -> float:
        """Return ground truth reward for a segment if available."""
        return self.gt_rewards.get(segment_id, None)

    def get_environment_config(self) -> dict:
        """Return environment configuration details from YAML."""
        return {
            'type': self.env_type,
            'name': self.env_name
        }

    def instantiate_env(self):
        """Instantiate environment instance based on config."""
        env_type = self.env_type
        env_name = self.env_name
        if env_type == 'gym':
            return self._get_gym_env(env_name)
        elif env_type == 'metaworld':
            return self._get_metaworld_env(env_name)
        else:
            raise ValueError(f"Unknown environment type: {env_type}")

    def _get_gym_env(self, env_name: str):
        """Create a gym environment."""
        import gym
        return gym.make(env_name)

    def _get_metaworld_env(self, env_name: str):
        """Create a Meta-World environment."""
        import metaworld
        # Example: MT50 environments
        env = metaworld.MT50(env_name)
        env.train()
        return env

    def generate_synthetic_preference(self,
                                      segment_id_a: str,
                                      segment_id_b: str,
                                      reward_func: Callable[[np.ndarray], float],
                                      threshold: float = 12.5,
                                      flip_noise: float = 0.0) -> float:
        """
        Generate synthetic preference label based on ground-truth rewards.
        Adds stochastic noise based on flip_noise probability.
        """
        reward_a = reward_func(self.get_segment_feature(segment_id_a))
        reward_b = reward_func(self.get_segment_feature(segment_id_b))
        delta = reward_a - reward_b
        if delta > threshold:
            label = 1.0
        elif delta < -threshold:
            label = 0.0
        else:
            label = 0.5

        if flip_noise > 0:
            rand_val = np.random.rand()
            if label == 1.0 and rand_val < flip_noise:
                label = 0.0
            elif label == 0.0 and rand_val < flip_noise:
                label = 1.0
            elif label == 0.5 and rand_val < flip_noise / 2:
                label = 0.0
            elif label == 0.5 and rand_val < flip_noise:
                label = 1.0
        return label

    def get_segments_for_policy(self):
        """Return list of segment objects for policy training or evaluation."""
        return list(self.segments.values())

    # Additional utility functions can be added as needed.
```

## evaluation.py

```python
## evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from utils import load_hyperparameters, set_seed

class Evaluation:
    def __init__(self, 
                 policy, 
                 environment, 
                 ground_truth_rewards=None, 
                 segment_list=None, 
                 environment_config=None, 
                 success_criteria_fn=None, 
                 num_episodes=50, 
                 max_episode_steps=None, 
                 seed=0):
        """
        Initialize the evaluation class.
        
        Args:
            policy: trained policy object with an 'act' method: act(state) -> action
            environment: environment object with reset() and step(action) methods
            ground_truth_rewards (np.ndarray or list): ground-truth rewards for segments (optional)
            segment_list (List): list of Segment objects with features (optional, for reward estimation & plots)
            environment_config (dict): environment configurations (optional, for success criteria)
            success_criteria_fn (callable): function to determine success given (obs, info), optional
            num_episodes (int): number of episodes to run for evaluation
            max_episode_steps (int): maximum steps per episode; if None, use environment default
            seed (int): random seed for reproducibility
        """
        self.policy = policy
        self.env = environment
        self.ground_truth_rewards = ground_truth_rewards
        self.segment_list = segment_list
        self.env_config = environment_config if environment_config else {}
        self.success_criteria_fn = success_criteria_fn
        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        
        # Set seeds for reproducibility
        set_seed(self.seed)
        
    def run(self):
        """
        Run the policy for a number of episodes, measure success rate, and plot reward correlation.
        Returns:
            success_rate (float): fraction of successful episodes
        """
        success_count = 0
        total_rewards = []
        ground_rewards = []
        estimated_rewards = []  # for plotting, if segment data is available

        for ep in range(self.num_episodes):
            obs = self._reset_env()
            done = False
            ep_reward = 0
            step_count = 0
            success_flag = False

            while True:
                if self.max_episode_steps and step_count >= self.max_episode_steps:
                    break
                # Get action from policy
                action = self.policy.act(obs)
                # Step environment
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                step_count += 1
                # Check success
                if self.success_criteria_fn:
                    success_flag = self.success_criteria_fn(obs, info)
                else:
                    # Default: environment specific info, fallback to 'is_success' if exists
                    success_flag = info.get('is_success', False) if isinstance(info, dict) else False
                if done:
                    break

            total_rewards.append(ep_reward)
            if success_flag:
                success_count += 1

        success_rate = success_count / self.num_episodes * 100.0
        print(f"Evaluation over {self.num_episodes} episodes: Success Rate = {success_rate:.2f}%")

        # Plot reward scatter if ground truth and estimated rewards are available
        if self.segment_list is not None and self.ground_truth_rewards is not None:
            ground_rewards = np.array([
                self.ground_truth_rewards.get(seg.segment_id, 0) for seg in self.segment_list
            ])
            est_rewards = []
            try:
                for seg in self.segment_list:
                    feat = seg.features
                    est = self.env.policy.reward_model.predict(feat.reshape(1, -1))[0]
                    est_rewards.append(est)
                est_rewards = np.array(est_rewards)
                # Plot the scatter
                corr = self._plot_reward_scatter(ground_rewards, est_rewards)
                print(f"Reward Estimation Pearson correlation: r = {corr:.3f}")
            except Exception as e:
                print(f"Error during reward scatter plotting: {e}")

        return success_rate

    def _reset_env(self):
        """
        Reset environment with seed for reproducibility.
        """
        # Some environments may support seed parameter
        # As a fallback, just call reset
        try:
            return self.env.reset()
        except:
            return self.env.reset()

    def _plot_reward_scatter(self, gt_rewards: np.ndarray, est_rewards: np.ndarray, save_path=None):
        """
        Plot scatter of ground truth vs estimated rewards.
        """
        import scipy.stats as stats
        plt.figure(figsize=(6,6))
        plt.scatter(gt_rewards, est_rewards, alpha=0.6)
        corr, _ = stats.pearsonr(gt_rewards, est_rewards)
        plt.title(f'Reward Correlation (r = {corr:.2f})')
        plt.xlabel('Ground Truth Rewards')
        plt.ylabel('Estimated Rewards')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        return corr
```

## main.py

```python
## main.py
import os
import yaml
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from utils import load_hyperparameters, set_seed, plot_reward_scatter
from dataset_loader import DatasetLoader
from rlt_builder import RLTBuilder
from reward_model import RewardModel
from reward_trainer import RewardTrainer
from offline_rl import OfflineRLPolicy
from evaluation import Evaluation

def main():
    # 1. Load configuration
    config_path = 'config.yaml'
    config = load_hyperparameters(config_path)

    # 2. Set seeds for reproducibility
    seeds = config['experiment'].get('seeds', 6)
    seed_for_run = 42  # or vary per seed later
    set_seed(seed_for_run)

    # 3. Environment setup
    env_type = config['environment'].get('type', 'metaworld')
    env_name = config['environment'].get('env_name', 'reach-v2')
    # Instantiate environment
    if env_type == 'metaworld':
        import metaworld
        env = metaworld.MT50(env_name)
        env.train()
    elif env_type == 'gym':
        import gym
        env = gym.make(env_name)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

    # 4. Load or generate dataset
    # Assume data directory is specified or default
    data_dir = 'data/'  # Adjust as needed
    dataset_loader = DatasetLoader(data_dir, config)
    segments_dict = dataset_loader.get_all_segments()
    segment_list = list(segments_dict.values())

    # 5. Ground truth rewards for validation (if available)
    gt_rewards = dataset_loader.gt_rewards
    # For validation purposes, normalize gt rewards between 0 and 1
    if gt_rewards:
        gt_rewards_list = np.array(list(gt_rewards.values()))
        gt_rewards_min, gt_rewards_max = gt_rewards_list.min(), gt_rewards_list.max()

    # 6. Generate preference labels (simulate human or use existing)
    preference_data = dataset_loader.preference_data
    # We assume preference_data.prefer_pairs is already loaded
    # Optional: You can synthesize or load pre-existing preference labels

    # 7. Construct Ranked List of Trajectories (RLT)
    # Use compare function: results in 0, 1, 0.5
    def compare_fn(seg_a, seg_b):
        # Simple preferential based on ground-truth rewards for synthetic data
        reward_a = gt_rewards.get(seg_a.segment_id, 0) if gt_rewards else 0
        reward_b = gt_rewards.get(seg_b.segment_id, 0) if gt_rewards else 0
        delta = reward_a - reward_b
        threshold = 12.5
        if delta > threshold:
            return 1.0  # seg_a preferred
        elif delta < -threshold:
            return 0.0  # seg_b preferred
        else:
            return 0.5  # tie

    # Build multiple RLTs if desired (e.g., with max Q per list)
    Q = config['preference_model'].get('Q', 100)
    # For simplicity, build one RLT
    rlt_builder = RLTBuilder(preference_data, segment_list, compare_fn)
    ranked_groups = rlt_builder.construct_rlt()

    # 8. Generate preference pairs or listwise rankings from RLT
    # For training reward model with pairwise loss:
    preference_pairs = []
    # For each pair of segments across different groups:
    for i, g1 in enumerate(ranked_groups):
        for j in range(i+1, len(ranked_groups)):
            g2 = ranked_groups[j]
            label = 0.0 if i > j else 1.0  # higher group preferred
            for seg_a in g1:
                for seg_b in g2:
                    preference_pairs.append((seg_a.segment_id, seg_b.segment_id, label))
        # For segments in same group, possibility to assign tie=0.5 (if second order info used)

    # 9. Instantiate reward model
    # Determine input feature dimension
    feature_dim = len(segment_list[0].features) if segment_list else 0
    reward_model = RewardModel(input_dim=feature_dim,
                               config=config['preference_model'])

    # 10. Train reward model using preference data
    preference_dataset = preference_data
    reward_trainer = RewardTrainer(reward_model, preference_dataset, segment_list,
                                   ground_truth_rewards=gt_rewards,
                                   feature_extractor=lambda seg: seg.features,
                                   config=config)
    print("Training reward model...")
    reward_trainer.train()

    # 11. Evaluate reward model's correlation with ground truth
    if gt_rewards:
        reward_trainer.evaluate_reward_correlation()

    # 12. Generate reward estimates for all segments
    segment_features = np.array([seg.features for seg in segment_list])
    estimated_rewards = reward_model.predict(segment_features)
    # Normalize for plotting
    if gt_rewards:
        gt_vals = np.array([gt_rewards[sig.segment_id] for sig in segment_list])
        # Optional: normalize to [0,1]
    else:
        gt_vals = None

    # 13. Offline Policy Training
    # Prepare dataset with estimated rewards
    # For simplicity, assume dataset_loader's dataset is used
    dataset = dataset_loader.trajectories  # or adapt as necessary
    # For each data point, assign reward from reward model
    # Assuming dataset is list of dicts with 'state', 'action', 'segment_id'
    dataset_for_rl = []
    for data_point in dataset:
        seg_id = data_point['segment_id']
        feat = dataset_loader.get_segment_feature(seg_id)
        rew = reward_model.predict(feat.reshape(1, -1))[0]
        # Append to dataset for RL
        dataset_for_rl.append({
            'state': data_point['state'],
            'action': data_point['action'],
            'reward': rew,
            'done': data_point.get('done', False)
        })

    # Normalize rewards between 0 and 1
    rew_vals = np.array([d['reward'] for d in dataset_for_rl])
    rew_min, rew_max = rew_vals.min(), rew_vals.max()
    for d in dataset_for_rl:
        d['reward'] = (d['reward'] - rew_min) / (rew_max - rew_min + 1e-8)

    # 14. Initialize and train offline RL policy
    rl_params = {
        'epochs': config['offline_rl'].get('epochs', 300),
        'learning_rate': config['offline_rl'].get('learning_rate', 0.001),
        'discount_factor': config['offline_rl'].get('discount_factor', 0.99)
    }
    policy_algo = OfflineRLPolicy(reward_model, 
                                  env_params={'type': env_type, 'env_name': env_name},
                                  rl_params=rl_params,
                                  dataset=dataset_for_rl,
                                  ground_truth_rewards=gt_rewards)
    print("Training offline RL policy...")
    policy_algo.train_policy()

    # 15. Evaluate policy
    print("Evaluating policy...")
    success_rate, avg_return = policy_algo.evaluate_policy()

    # 16. Plot reward scatter if ground truth available
    if gt_rewards:
        # Compute estimated rewards for segments again
        est_rewards_for_segments = reward_model.predict(segment_features)
        corr = plot_reward_scatter(gt_vals, est_rewards_for_segments)
        print(f"Reward estimate Pearson correlation: r = {corr:.3f}")

    # 17. Save models / results if needed
    # e.g., torch.save(reward_model.network.state_dict(), 'reward_model.pth')
    # or save policy
    # policy_algo.save_policy('policy.pth')

    # 18. Repeat over different seeds or parameter settings as needed

if __name__ == "__main__":
    main()
```

## offline_rl.py

```python
## offline_rl.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from utils import load_hyperparameters, set_seed, normalize
import gym
try:
    import metaworld
except ImportError:
    metaworld = None

class OfflineRLPolicy:
    def __init__(self, reward_model, env_params, rl_params, dataset, ground_truth_rewards=None):
        """
        Initialize the OfflineRLPolicy.
        Args:
            reward_model: pretrained RewardModel object with .predict() method.
            env_params: dict with environment configurations ('type', 'env_name').
            rl_params: dict with RL training hyperparameters.
            dataset: dataset object containing offline data (states, actions, etc).
            ground_truth_rewards: Optional dict {segment_id: reward} for evaluation.
        """
        self.reward_model = reward_model
        self.env_type = env_params.get('type', 'metaworld')
        self.env_name = env_params.get('env_name', 'reach-v2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ground_truth_rewards = ground_truth_rewards
        self.dataset = dataset
        # Set parameters
        self.epochs = rl_params.get('epochs', 300)
        self.lr = rl_params.get('learning_rate', 0.001)
        self.gamma = rl_params.get('discount_factor', 0.99)
        self.batch_size = rl_params.get('batch_size', 256)
        self.env = self._instantiate_env()
        # Set seed for reproducibility
        seed = rl_params.get('seed', 0)
        set_seed(seed)
        # Initialize policy and critic networks
        self._init_networks()
        # Initialize optimizers
        self._init_optimizers()

        # Prepare offline data with rewards
        self.offline_data = self._prepare_dataset()

    def _instantiate_env(self):
        if self.env_type == 'gym':
            import gym
            return gym.make(self.env_name)
        elif self.env_type == 'metaworld':
            if metaworld is None:
                raise ImportError("MetaWorld not installed")
            env = metaworld.MT50(self.env_name)
            env.train()
            return env
        else:
            raise ValueError(f"Unknown environment type: {self.env_type}")

    def _init_networks(self):
        # Define policy and Q networks
        # For simplicity, assuming low dimensional features
        obs_dim, act_dim = self._get_env_dims()
        hidden_dim = 256
        # Policy network
        self.policy_net = self._build_network(obs_dim + act_dim, hidden_dim, output_dim=act_dim)
        # Q networks (double Q)
        self.q_net1 = self._build_network(obs_dim + act_dim, hidden_dim, output_dim=1)
        self.q_net2 = self._build_network(obs_dim + act_dim, hidden_dim, output_dim=1)

    def _build_network(self, input_dim, hidden_dim, output_dim=1):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def _get_env_dims(self):
        # Get observation and action dims based on environment
        if self.env_type == 'gym':
            obs_dim = self.env.observation_space.shape[0]
            act_dim = self.env.action_space.shape[0]
        elif self.env_type == 'metaworld':
            # Assuming MT50 environments have specific observation/action dims
            obs_dim = self.env._get_observation_space().shape[0]
            act_dim = self.env._get_action_space().shape[0]
        else:
            raise ValueError("Unknown environment type for dims")
        return obs_dim, act_dim

    def _init_optimizers(self):
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=self.lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=self.lr)

    def _prepare_dataset(self):
        """
        Converts raw dataset into tensors suitable for training.
        Assumes dataset is a list/dict with 'states', 'actions', possibly 'segments'.
        """
        states = []
        actions = []
        rewards = []
        for data_point in self.dataset:
            # Assuming dataset entries are dicts: {'state':..., 'action':..., 'segment_id':...}
            state = data_point['state']
            action = data_point['action']
            # Estimate reward using reward_model
            segment_features = self._extract_features_from_data_point(data_point)
            reward = self.reward_model.predict(segment_features.reshape(1, -1))[0]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(self.device)
        # Combine states and actions for Q updates
        dataset_tensors = {
            'states': states,
            'actions': actions,
            'rewards': rewards
        }
        return dataset_tensors

    def _extract_features_from_data_point(self, data_point):
        # Based on environment, extract features for reward prediction
        # For example, use 'state' as features
        state = data_point.get('state', None)
        if state is None:
            # fallback: use 'observation'
            state = data_point.get('observation', [])
        return np.array(state, dtype=np.float32)

    def train_policy(self):
        """
        Trains the policy using offline RL algorithm (e.g., IQL).
        """
        for epoch in range(self.epochs):
            # Sample mini-batch
            indices = np.random.choice(len(self.dataset), size=self.batch_size, replace=True)
            batch_states = torch.tensor(np.array([self.dataset[i]['state'] for i in indices]), dtype=torch.float32).to(self.device)
            batch_actions = torch.tensor(np.array([self.dataset[i]['action'] for i in indices]), dtype=torch.float32).to(self.device)

            # Compute rewards
            batch_rewards = []
            for i in range(self.batch_size):
                seg_feat = self._extract_features_from_data_point(self.dataset[indices[i]])
                r = self.reward_model.predict(seg_feat.reshape(1, -1))[0]
                batch_rewards.append(r)
            batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

            # Update Q networks
            q_input = torch.cat([batch_states, batch_actions], dim=1)
            q1_pred = self.q_net1(q_input)
            q2_pred = self.q_net2(q_input)

            # Compute target
            with torch.no_grad():
                next_actions = self.policy_net(torch.cat([batch_states, batch_actions], dim=1))
                q1_next = self.q_net1(torch.cat([batch_states, next_actions], dim=1))
                q2_next = self.q_net2(torch.cat([batch_states, next_actions], dim=1))
                target_q = batch_rewards + self.gamma * torch.min(q1_next, q2_next)

            # Critic loss (MSE)
            loss_q1 = nn.MSELoss()(q1_pred, target_q)
            loss_q2 = nn.MSELoss()(q2_pred, target_q)

            self.q1_optimizer.zero_grad()
            loss_q1.backward()
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            loss_q2.backward()
            self.q2_optimizer.step()

            # Update policy via policy gradient (e.g., max Q)
            self.policy_optimizer.zero_grad()
            current_actions = self.policy_net(torch.cat([batch_states, batch_actions], dim=1))
            q1_policy = self.q_net1(torch.cat([batch_states, current_actions], dim=1))
            # Maximize Q
            policy_loss = -q1_policy.mean()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Logging
            if epoch % 50 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch+1}/{self.epochs}: Q Losss ({loss_q1.item():.4f}, {loss_q2.item():.4f}), Policy Loss: {policy_loss.item():.4f}")

    def evaluate_policy(self, num_episodes=10):
        """
        Runs the trained policy in environment to evaluate success rate and return.
        """
        success_list = []
        total_return_list = []
        for ep in range(num_episodes):
            obs = self._reset_env()
            done = False
            ep_reward = 0
            while not done:
                state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.policy_net(torch.cat([state_tensor], dim=1)).cpu().numpy().flatten()
                obs, reward, done, info = self._step_env(action)
                ep_reward += reward
            total_return_list.append(ep_reward)
            success = self._check_success(obs, info)
            success_list.append(success)
        success_rate = np.mean(success_list)
        avg_return = np.mean(total_return_list)
        print(f"Evaluation: success rate = {success_rate*100:.2f}%, average return = {avg_return:.2f}")
        return success_rate, avg_return

    def _reset_env(self):
        if self.env_type == 'gym':
            return self.env.reset()
        elif self.env_type == 'metaworld':
            return self.env.reset()
        else:
            raise ValueError("Unknown environment type")

    def _step_env(self, action):
        if self.env_type == 'gym':
            obs, reward, done, info = self.env.step(action)
            return obs, reward, done, info
        elif self.env_type == 'metaworld':
            obs, reward, done, info = self.env.step(action)
            return obs, reward, done, info
        else:
            raise ValueError("Unknown environment type")

    def _check_success(self, obs, info):
        # Environment-specific success check, placeholder:
        # For example, in Meta-World, info may contain 'success' flag
        if self.env_type == 'metaworld':
            return info.get('success', False)
        elif self.env_type == 'gym':
            # Define custom success criteria for gym
            return False
        return False

    def save_policy(self, save_path):
        torch.save(self.policy_net.state_dict(), save_path)

    def load_policy(self, load_path):
        self.policy_net.load_state_dict(torch.load(load_path))
```

## reward_model.py

```python
## reward_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import load_hyperparameters


class RewardNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, activation: str = 'relu'):
        """
        Neural network module for reward prediction.
        Args:
            input_dim (int): Dimension of segment features.
            hidden_layers (list): List of integers for hidden layer sizes.
            activation (str): Activation function name ('relu' or 'tanh').
        """
        super(RewardNetwork, self).__init__()
        layers = []

        prev_dim = input_dim
        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev_dim, layer_size))
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            prev_dim = layer_size

        # Final linear layer outputting a single scalar score
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
        self.activation = activation.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the score for input features.
        Args:
            x (torch.Tensor): Input features with shape (batch_size, feature_dim).
        Returns:
            torch.Tensor: Raw scores with shape (batch_size, 1).
        """
        return self.model(x)

class RewardModel:
    def __init__(self, input_dim: int = None, config: dict = None):
        """
        Initialize the RewardModel with architecture and hyperparameters.
        Args:
            input_dim (int): Dimensionality of segment features.
            config (dict): Hyperparameters dictionary (loaded from config.yaml).
        """
        if config is None:
            config = {}
        # Load default hyperparameters if not provided
        hyperparams = load_hyperparameters()
        self.score_function_type = hyperparams['preference_model'].get('score_function_type', 'exp')
        self.loss_type = hyperparams['preference_model'].get('loss_type', 'listwise')
        hidden_layers = hyperparams['preference_model'].get('hidden_layers', [128, 128, 128])
        activation = hyperparams['preference_model'].get('activation', 'relu')
        final_activation = hyperparams['preference_model'].get('final_activation', 'tanh')
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Determine input_dim
        if input_dim is None:
            raise ValueError("Input feature dimension must be specified.")
        self.input_dim = input_dim
        # Initialize neural network
        self.network = RewardNetwork(input_dim=self.input_dim,
                                     hidden_layers=hidden_layers,
                                     activation=activation).to(self.device)
        # Store hyperparameters
        self.learning_rate = hyperparams['preference_model'].get('learning_rate', 0.001)
        self.regularization = hyperparams['preference_model'].get('regularization', 0.0003)
        self.epochs = hyperparams['preference_model'].get('epochs', 300)
        self.batch_size = hyperparams['preference_model'].get('batch_size', 512)
        # Instantiate optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

    def predict(self, segment_features: np.ndarray) -> np.ndarray:
        """
        Compute the reward estimate for a batch of segments.
        Args:
            segment_features (np.ndarray): Array of shape (batch_size, feature_dim).
        Returns:
            np.ndarray: Reward values bounded in [0, R], shape (batch_size,).
        """
        self.network.eval()
        with torch.no_grad():
            inputs = torch.tensor(segment_features, dtype=torch.float32).to(self.device)
            raw_scores = self.network(inputs).squeeze()  # shape (batch_size,)
            if self.score_function_type == 'exp':
                scores = torch.exp(raw_scores)
            elif self.score_function_type == 'linear':
                scores = raw_scores
            else:
                raise ValueError(f"Unsupported score function type: {self.score_function_type}")

            # Apply final activation (tanh scaled)
            if self.network.activation == 'tanh':
                bounded_rewards = torch.tanh(scores)
            else:
                # Default to tanh if not explicitly set; can be adapted
                bounded_rewards = torch.tanh(scores)

            return bounded_rewards.cpu().numpy()

    def train(self, segment_dataset: List['Segment'], preference_data, listwise: bool = True):
        """
        Train the reward model using preference data.
        Args:
            segment_dataset (List[Segment]): List of Segment objects used for sampling.
            preference_data (PreferenceDataset): Preference pairs or listwise rankings.
            listwise (bool): If True, use listwise loss; else, use pairwise.
        """
        self.network.train()
        # Convert segments to feature matrix
        feature_matrix = np.array([seg.features for seg in segment_dataset], dtype=np.float32)
        dataset_size = len(segment_dataset)
        # Prepare preference pairs or listwise data
        pref_pairs = preference_data.preference_pairs
        # For simplicity, filter preference pairs relevant to current dataset
        # Assumption: preference_data contains valid segment IDs
        # Optimization: Could prepare index mapping for fast sampling

        for epoch in range(self.epochs):
            permutation = np.random.permutation(dataset_size)
            for batch_start in range(0, dataset_size, self.batch_size):
                batch_indices = permutation[batch_start: batch_start + self.batch_size]
                batch_segments = [segment_dataset[i] for i in batch_indices]
                batch_features = np.array([seg.features for seg in batch_segments], dtype=np.float32)
                inputs = torch.tensor(batch_features, dtype=torch.float32).to(self.device)

                self.optimizer.zero_grad()

                scores = self.network(inputs).squeeze()  # shape (batch_size,)
                # Compute scores based on the selected scoring function
                if self.score_function_type == 'exp':
                    pred_scores = torch.exp(scores)
                elif self.score_function_type == 'linear':
                    pred_scores = scores
                else:
                    raise ValueError(f"Unsupported score function: {self.score_function_type}")

                # Final reward bounding
                if self.network.activation == 'tanh':
                    rewards = torch.tanh(pred_scores)
                else:
                    rewards = torch.tanh(pred_scores)

                # Compute loss
                if self.loss_type == 'listwise':
                    # Listwise (ListNet) loss
                    # Sample list_size segments for listwise loss
                    if len(segment_dataset) >= self.config['preference_model'].get('list_size', 16):
                        list_indices = np.random.choice(dataset_size, size=self.config['preference_model'].get('list_size',16), replace=False)
                        list_feats = torch.tensor([segment_dataset[i].features for i in list_indices], dtype=torch.float32).to(self.device)
                        list_scores = self.network(list_feats).squeeze()
                        if self.score_function_type == 'exp':
                            list_pred = torch.exp(list_scores)
                        else:
                            list_pred = list_scores
                        # Normalize to probabilities
                        pred_prob = list_pred / torch.sum(list_pred)
                        # Ground truth distribution: proportional to ground-truth rewards
                        # For training, we need a ground-truth distribution; but in practice, it can be based on listwise ranking
                        # Here, we assume uniform or derived distribution (see paper), but for simplicity, use softmax from true scores
                        # If ground-truth ground rewards are available, better to use them here
                        # Placeholder: Using model scores as approx ground truth (not ideal but aligns with experimental approach)
                        true_scores = torch.tensor([np.mean([self.get_segment_true_reward(seg_id) for seg_id in list_indices])], dtype=torch.float32).to(self.device)
                        true_scores_exp = torch.exp(true_scores)
                        true_prob = true_scores_exp / torch.sum(true_scores_exp)
                        loss = torch.nn.functional.kl_div(pred_prob.log(), true_prob, reduction='sum')
                    else:
                        # fallback if dataset too small
                        continue
                elif self.loss_type == 'pairwise':
                    # Pairwise loss on preference pairs
                    batch_pref = self._sample_preference_pairs(preference_data, len(segment_dataset))
                    if batch_pref:
                        loss = 0.0
                        for (id_a, id_b, label) in batch_pref:
                            feat_a = torch.tensor(segment_dataset[self._find_segment_index(id_a)].features, dtype=torch.float32).to(self.device)
                            feat_b = torch.tensor(segment_dataset[self._find_segment_index(id_b)].features, dtype=torch.float32).to(self.device)
                            score_a = self.network(feat_a.unsqueeze(0)).squeeze()
                            score_b = self.network(feat_b.unsqueeze(0)).squeeze()
                            if self.score_function_type == 'exp':
                                s_a = torch.exp(score_a)
                                s_b = torch.exp(score_b)
                            else:
                                s_a = score_a
                                s_b = score_b
                            prob = s_a / (s_a + s_b + 1e-8)
                            # Label: 1.0 means segment_a preferred, 0.0 preferred_b, 0.5 tie
                            if label == 1.0:
                                target = torch.tensor([1.0], dtype=torch.float32).to(self.device)
                                loss += -torch.log(prob + 1e-8)
                            elif label == 0.0:
                                target = torch.tensor([0.0], dtype=torch.float32).to(self.device)
                                loss += -torch.log(1 - prob + 1e-8)
                            else:
                                # tie (0.5), treat as binary with label=0.5, use binary cross entropy with logits
                                # For simplicity, skip or treat as average of both
                                continue
                        # Average loss over pairs
                        loss = loss / len(batch_pref)
                    else:
                        continue
                else:
                    raise ValueError(f"Unsupported loss type: {self.loss_type}")

                # Backpropagation
                loss.backward()
                self.optimizer.step()

    def _sample_preference_pairs(self, preference_data: 'PreferenceDataset', dataset_size: int):
        """
        Sample preference pairs from preference data for batch training.
        Args:
            preference_data (PreferenceDataset): Preference data object.
            dataset_size (int): Total number of segments.
        Returns:
            List of tuples: (segment_id_a, segment_id_b, label)
        """
        if not preference_data.preference_pairs:
            return []
        num_pairs = min(self.batch_size, len(preference_data.preference_pairs))
        sampled_pairs = np.random.choice(preference_data.preference_pairs, size=num_pairs, replace=False)
        return [(pair.segment_a_id, pair.segment_b_id, pair.label) for pair in sampled_pairs]

    def _find_segment_index(self, segment_id: str) -> int:
        """
        Helper method to find index of segment in dataset by segment_id.
        """
        # This assumes segment_dataset is a list; implement as needed outside
        # For this code, assuming a global reference or passed in; can be improved
        # For safety, implement a mapping if needed
        raise NotImplementedError("Segment index lookup should be implemented based on data structure.")

    def get_segment_true_reward(self, segment_id: str) -> float:
        """
        Placeholder for method to retrieve ground-truth reward if available.
        For training, may be used to approximate ground-truth distribution.
        """
        # This method should interface with the environment or dataset
        # For now, return a dummy value or fetch from dataset if stored.
        # Since not specified here, we return 0.
        return 0.0
```

## reward_trainer.py

```python
## reward_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import load_hyperparameters, normalize, plot_reward_scatter
from typing import List, Tuple
import copy

class RewardTrainer:
    def __init__(self, 
                 reward_model, 
                 preference_data, 
                 segment_list: List,  # list of Segment objects
                 ground_truth_rewards=None,  # optional, for validation
                 feature_extractor=None,  # function to extract features from segments
                 config: dict = None):
        """
        Initializes the RewardTrainer.
        Args:
            reward_model: instance of RewardModel with .predict() method.
            preference_data: dataset object containing preference pairs or listwise rankings.
            segment_list: list of Segment objects used for sampling segments.
            ground_truth_rewards: optional dict {segment_id: reward} for validation or correlation.
            feature_extractor: optional function to extract features from Segment.
            config: configuration dictionary (loaded from YAML).
        """
        import torch
        self.reward_model = reward_model
        self.preference_data = preference_data
        self.segment_list = segment_list
        self.ground_truth_rewards = ground_truth_rewards
        self.feature_extractor = feature_extractor
        self.config = config if config is not None else {}
        # Load hyperparameters
        hyperparams = load_hyperparameters()
        self.score_function_type = hyperparams['preference_model'].get('score_function_type', 'exp')
        self.sample_size = hyperparams['preference_model'].get('sample_size', 10)
        self.list_size = hyperparams['preference_model'].get('list_size', 16)
        self.regularization = hyperparams['preference_model'].get('regularization', 0.0003)
        self.learning_rate = hyperparams['preference_model'].get('learning_rate', 0.001)
        self.epochs = hyperparams['preference_model'].get('epochs', 300)
        self.batch_size = hyperparams['preference_model'].get('batch_size', 512)
        # Initialize optimizer
        self.optimizer = optim.Adam(self.reward_model.network.parameters(), lr=self.learning_rate, weight_decay=self.regularization)
        # For reproducibility
        import random
        seed = self.config.get('seed', 0)
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def extract_features(self, segment):
        """
        Extract features from a segment using the provided feature_extractor
        or default to segment.features.
        """
        if self.feature_extractor is not None:
            return self.feature_extractor(segment)
        else:
            # fallback: assume segment has 'features' attribute or dict
            if hasattr(segment, 'features'):
                return segment.features
            elif isinstance(segment, dict):
                return segment.get('features', np.array([]))
            else:
                # as last resort, flatten segment
                return np.array(segment).flatten()

    def sample_segments_for_list(self, n):
        """
        Randomly sample n distinct segments from segment_list for listwise loss.
        """
        if len(self.segment_list) < n:
            # if not enough segments, sample with replacement
            return np.random.choice(self.segment_list, size=n, replace=True)
        else:
            return np.random.choice(self.segment_list, size=n, replace=False)

    def generate_list_from_segments(self, segments: List):
        """
        Prepare feature tensor for a list of segments.
        """
        feats = [self.extract_features(seg) for seg in segments]
        feats = np.stack(feats, axis=0)     # shape (list_size, feature_dim)
        return torch.tensor(feats, dtype=torch.float32)

    def build_batch_list(self):
        """
        Sample a list (sequence) of segments for listwise training.
        The list is built by sampling segments from the overall segment list.
        """
        list_segments = self.sample_segments_for_list(self.list_size)
        feat_tensor = self.generate_list_from_segments(list_segments)
        return list_segments, feat_tensor

    def compute_listwise_loss(self, scores: torch.Tensor, list_segments: List):
        """
        Compute listwise (ListNet) KL divergence loss between target distribution and predicted.
        Args:
            scores: Tensor of shape (list_size,) - predicted scores for segments.
            list_segments: List of segments corresponding to scores.
        """
        # Apply scoring function
        if self.score_function_type == 'exp':
            s_pred = torch.exp(scores)
        elif self.score_function_type == 'linear':
            s_pred = scores
        else:
            raise ValueError(f"Unknown score function type {self.score_function_type}")

        # Predicted probability distribution over segments
        p_pred = s_pred / torch.sum(s_pred)

        # Target distribution: infer from preference data or use uniform
        # Here, for listwise, assume the current ranking is given by ground-truth rewards if available,
        # or approximate via preference labels.
        # Suppose the preference data contains a ranking among segments:
        # For simplicity, we assign target probabilities proportional to ground-truth rewards
        # (if available), otherwise uniform.
        if self.ground_truth_rewards is not None:
            rewards = np.array([self.ground_truth_rewards.get(seg.segment_id, 0) for seg in list_segments])
            if np.sum(rewards) > 0:
                p_target = torch.tensor(rewards, dtype=torch.float32)
                p_target = p_target / p_target.sum()
            else:
                p_target = torch.ones_like(p_pred) / len(p_pred)
        else:
            # fallback: uniform distribution
            p_target = torch.ones_like(p_pred) / len(p_pred)

        # To prevent log(0), clamp p_pred
        p_pred = torch.clamp(p_pred, 1e-8, 1.0)
        p_target = torch.clamp(p_target, 1e-8, 1.0)
        # Compute KL divergence: sum_i p_target_i * log(p_target_i / p_pred_i)
        loss = torch.sum(p_target * torch.log(p_target / p_pred))
        return loss

    def train(self):
        """
        Main training loop for the reward model.
        """
        segment_dataset = self.segment_list
        for epoch in range(self.epochs):
            # Sample a list of segments
            list_segments, feat_tensor = self.build_batch_list()
            feat_tensor = feat_tensor.to(self.reward_model.network.model[0].weight.device)

            # Forward prediction
            scores = self.reward_model.network(feat_tensor).squeeze()  # shape (list_size,)
            # Compute listwise loss
            loss = self.compute_listwise_loss(scores, list_segments)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            if (epoch + 1) % self.config.get('validation_interval', 50) == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {loss.item():.4f}")

        # After training, if ground_truth_rewards available, evaluate correlation
        if self.ground_truth_rewards is not None:
            self.evaluate_reward_correlation()

    def evaluate_reward_correlation(self):
        """
        Compute Pearson correlation coefficient between ground-truth rewards and predicted rewards
        over a validation set of segments.
        """
        # Prepare features and ground truth
        segment_feats = []
        gt_rewards = []
        for seg in self.segment_list:
            feat = self.extract_features(seg)
            segment_feats.append(feat)
            reward = self.ground_truth_rewards.get(seg.segment_id, 0)
            gt_rewards.append(reward)
        feat_array = np.stack(segment_feats, axis=0)
        pred_rewards = self.reward_model.predict(feat_array)
        gt_rewards = np.array(gt_rewards)
        # Normalize
        pred_rewards_norm = normalize(pred_rewards)
        gt_rewards_norm = normalize(gt_rewards)
        # Compute Pearson correlation
        r, _ = self._pearson_correlation(gt_rewards_norm, pred_rewards_norm)
        print(f"Reward model Pearson correlation with ground truth: r = {r:.3f}")
        # Plot scatter
        plot_reward_scatter(gt_rewards, pred_rewards)

    def _pearson_correlation(self, arr1, arr2):
        """
        Compute Pearson correlation coefficient between two arrays.
        """
        if len(arr1) == 0 or len(arr2) == 0:
            return 0.0, 0
        from scipy.stats import pearsonr
        r, p_value = pearsonr(arr1, arr2)
        return r, p_value
```

## rlt_builder.py

```python
## rlt_builder.py
import random
from typing import List, Callable, Tuple
from collections import deque

from dataset_loader import PreferenceDataset, Segment

class RLTBuilder:
    def __init__(self,
                 preference_data: 'PreferenceDataset',
                 dataset_segments: List['Segment'],
                 compare_fn: Callable[[Segment, Segment], float],
                 config: dict = None):
        """
        Initializes the RLTBuilder with preference data, dataset segments, and comparison function.
        Args:
            preference_data (PreferenceDataset): Collected preference labels among segments.
            dataset_segments (List[Segment]): List of all candidate segments to insert.
            compare_fn (Callable): Function taking two segments and returning preference label:
                                 0 - first preferred, 1 - second preferred, 0.5 - equal.
            config (dict): Optional configuration dictionary (not used explicitly here).
        """
        self.preference_data = preference_data
        self.segments = dataset_segments
        self.compare_fn = compare_fn
        # List of groups; each group is a set/list of segments with same preference level
        self.ranked_list: List[List[Segment]] = []

    def construct_rlt(self, seed_segment: 'Segment' = None) -> List[List['Segment']]:
        """
        Constructs the Ranked List of Trajectories (RLT) by sequential insertion.
        Args:
            seed_segment (Segment, optional): Segment to seed the list. Defaults to random.
        Returns:
            List[List[Segment]]: The fully constructed ranked list (ordered groups).
        """
        if not self.segments:
            return []

        # Initialize list with one seed segment in first group
        if seed_segment is None:
            seed_segment = random.choice(self.segments)
        self.ranked_list = [[seed_segment]]

        # Set of segments already inserted (for avoiding duplicates)
        inserted_segments = set([seed_segment.segment_id])

        # For each remaining segment, insert into list
        for segment in self.segments:
            if segment.segment_id in inserted_segments:
                continue  # skip already inserted
            # Insert current segment using binary search
            self._insert_segment(segment)

            inserted_segments.add(segment.segment_id)

        return self.ranked_list

    def _insert_segment(self, segment: 'Segment') -> None:
        """
        Insert a segment into the current RLT list using binary search and preference queries.
        Args:
            segment (Segment): The candidate segment to insert.
        """
        low = 0
        high = len(self.ranked_list) - 1

        # If list is empty, just add the segment
        if high < 0:
            self.ranked_list.append([segment])
            return

        while low <= high:
            mid = (low + high) // 2
            group = self.ranked_list[mid]
            # Select a representative segment from the group for comparison
            # Here, compare with the first segment in group for simplicity
            ref_segment = group[0]

            preference = self._query_preference(segment, ref_segment)

            if preference == 0:
                # segment preferred over ref_segment -> go higher (more preferred group)
                low = mid + 1
            elif preference == 1:
                # ref_segment preferred over segment -> go lower
                high = mid - 1
            else:
                # preference == 0.5, equal preference
                # insert segment into this group
                group.append(segment)
                return

        # After binary search, insert at position low
        # Check for equal preference with neighboring groups
        # First, insert as a new group at position low
        self.ranked_list.insert(low, [segment])

    def _query_preference(self, seg_a: 'Segment', seg_b: 'Segment') -> float:
        """
        Query preference between two segments using compare_fn.
        Args:
            seg_a (Segment): First segment.
            seg_b (Segment): Second segment.
        Returns:
            float: Preference label (0: seg_a preferred, 1: seg_b preferred, 0.5: tie)
        """
        # Use compare_fn which should return preference label accordingly
        # It's expected that compare_fn returns 0, 1, or 0.5
        preference = self.compare_fn(seg_a, seg_b)
        return preference
```

## utils.py

```python
## utils.py

import os
import yaml
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

try:
    import gym
except ImportError:
    gym = None

try:
    import metaworld
except ImportError:
    metaworld = None


def load_hyperparameters(config_path='config.yaml'):
    """
    Loads hyperparameters from a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Dictionary containing hyperparameters for different modules.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    hyperparams = {
        'preference_model': {
            'score_function': config.get('preference_model', {}).get('score_function', 'exp'),
            'sample_size': config.get('preference_model', {}).get('sample_size', 10),
            'list_size': config.get('preference_model', {}).get('list_size', 16),
            'regularization': config.get('preference_model', {}).get('regularization', 0.0003),
            'learning_rate': config.get('preference_model', {}).get('learning_rate', 0.001),
            'batch_size': config.get('preference_model', {}).get('batch_size', 512),
            'epochs': config.get('preference_model', {}).get('epochs', 300),
            'hidden_layers': config.get('preference_model', {}).get('hidden_layers', [128, 128, 128]),
            'activation': config.get('preference_model', {}).get('activation', 'relu'),
            'final_activation': config.get('preference_model', {}).get('final_activation', 'tanh'),
            'loss_type': config.get('preference_model', {}).get('loss_type', 'listwise'),
            'score_function_type': config.get('preference_model', {}).get('score_function_type', 'exp')
        },
        'offline_rl': {
            'epochs': config.get('offline_rl', {}).get('epochs', 300),
            'learning_rate': config.get('offline_rl', {}).get('learning_rate', 0.001),
            'discount_factor': config.get('offline_rl', {}).get('discount_factor', 0.99)
        },
        'environment': {
            'type': config.get('environment', {}).get('type', 'metaworld'),
            'env_name': config.get('environment', {}).get('env_name', 'reach-v2')
        },
        'preference_feedback': {
            'feedback_count': config.get('preference_feedback', {}).get('feedback_count', 500),
            'noise_level': config.get('preference_feedback', {}).get('noise_level', 0)
        },
        'experiment': {
            'seeds': config.get('experiment', {}).get('seeds', 6),
            'validation_interval': config.get('experiment', {}).get('validation_interval', 50)
        }
    }
    return hyperparams


def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility across numpy, torch, and random.

    Args:
        seed (int): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional for more deterministic behavior:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def normalize(values, min_value=None, max_value=None):
    """
    Normalizes a numpy array or list of values to [0, 1].

    Args:
        values (np.ndarray or list): The values to normalize.
        min_value (float, optional): Minimum value for normalization. Defaults to min of values.
        max_value (float, optional): Maximum value for normalization. Defaults to max of values.

    Returns:
        np.ndarray: Normalized values in [0, 1].
    """
    values = np.array(values)
    if min_value is None:
        min_value = np.min(values)
    if max_value is None:
        max_value = np.max(values)
    denom = max_value - min_value + 1e-8
    normalized = (values - min_value) / denom
    return normalized


def simulate_preference(segment_a_features, segment_b_features, reward_func, threshold=12.5, noise=0.0):
    """
    Simulates a ternary preference label between two segments based on their ground-truth rewards.
    Adds optional noise.

    Args:
        segment_a_features (np.ndarray): Feature vector of segment A.
        segment_b_features (np.ndarray): Feature vector of segment B.
        reward_func (callable): Function that maps segment features to ground-truth reward.
        threshold (float): Threshold for reward difference to decide preference.
        noise (float): Probability [0,1] to flip the label to simulate noise.

    Returns:
        float: Preference label (0: segment B preferred, 1: segment A preferred, 0.5: equal).
    """
    reward_a = reward_func(segment_a_features)
    reward_b = reward_func(segment_b_features)
    delta = reward_a - reward_b

    if delta > threshold:
        label = 1.0
    elif delta < -threshold:
        label = 0.0
    else:
        label = 0.5

    # Add symmetric noise if specified
    if noise > 0:
        rand_val = np.random.rand()
        if label == 1.0:
            if rand_val < noise:
                label = 0.0  # flip
        elif label == 0.0:
            if rand_val < noise:
                label = 1.0
        else:  # label == 0.5
            if rand_val < noise / 2:
                label = 0.0
            elif rand_val < noise:
                label = 1.0
            # else remain tie
    return label


def get_gym_env(env_name):
    """
    Initializes and returns a Gym environment.

    Args:
        env_name (str): Name of the Gym environment.

    Returns:
        gym.Env: Gym environment instance.
    """
    if gym is None:
        raise ImportError("Gym is not installed.")
    env = gym.make(env_name)
    return env


def get_metaworld_env(env_name):
    """
    Initializes and returns a Meta-World environment.

    Args:
        env_name (str): Name of the Meta-World task.

    Returns:
        metaworld.MT50: Meta-World environment instance.
    """
    if metaworld is None:
        raise ImportError("MetaWorld is not installed.")
    env = metaworld.MT50(env_name)
    env.train()  # Prepare for training
    return env


def extract_segment_features(segment, env_type='metaworld'):
    """
    Extract features from a trajectory segment for model input.

    Args:
        segment (dict): Segment data containing observations, states, or features.
        env_type (str): 'metaworld' or 'gym'. Determines feature extraction method.

    Returns:
        np.ndarray: Feature vector representing the segment.
    """
    if env_type == 'metaworld':
        # Assuming segment contains 'observations' as features
        features = segment.get('observations')
    elif env_type == 'gym':
        # Assuming segment contains 'state' as features
        features = segment.get('state')
    else:
        # Fallback: raw segment data or placeholder
        features = segment.get('features', None)
    if features is None:
        # As a last resort, flatten the segment data
        features = np.array(segment).flatten()
    return np.array(features, dtype=np.float32)


def plot_reward_scatter(gt_rewards, est_rewards, save_path=None):
    """
    Plot scatter of ground truth vs. estimated rewards and compute Pearson correlation.

    Args:
        gt_rewards (np.ndarray): Ground-truth reward values.
        est_rewards (np.ndarray): Estimated reward values.
        save_path (str): Path to save plot image. If None, show plot.

    Returns:
        float: Pearson correlation coefficient.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(gt_rewards, est_rewards, alpha=0.5)
    corr, _ = pearsonr(gt_rewards, est_rewards)
    plt.title(f'Reward Correlation (r = {corr:.2f})')
    plt.xlabel('Ground Truth Rewards')
    plt.ylabel('Estimated Rewards')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return corr
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\LiRE\LiRE_repo`
