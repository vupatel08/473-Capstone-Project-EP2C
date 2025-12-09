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
