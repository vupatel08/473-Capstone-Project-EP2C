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
