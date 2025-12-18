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
