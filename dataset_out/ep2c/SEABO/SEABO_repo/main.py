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
