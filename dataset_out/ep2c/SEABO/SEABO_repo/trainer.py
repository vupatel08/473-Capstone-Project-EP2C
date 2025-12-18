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

