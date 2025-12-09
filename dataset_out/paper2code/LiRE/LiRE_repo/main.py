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
