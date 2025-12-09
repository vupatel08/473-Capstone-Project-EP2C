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
