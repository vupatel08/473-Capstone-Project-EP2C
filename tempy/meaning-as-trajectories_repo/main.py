## main.py
import yaml
import torch
import numpy as np
import random
from collections import defaultdict
from dataset_loader import DatasetLoader
from model import ModelWrapper
from sampling import Sampler
from likelihood import LikelihoodCalculator
from distance import DistanceEvaluator
from evaluation import evaluate_similarity, evaluate_relation

def main():
    # 1. Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. Set seeds for reproducibility
    seed = config['evaluation'].get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # 3. Load datasets
    dataset_type = config['dataset'].get('type', 'STS')
    dataset_split = config['dataset'].get('split', 'validation')
    dataset_loader = DatasetLoader(dataset_type, dataset_split)
    datasets = dataset_loader.load_datasets()
    # Get pairs for evaluation based on dataset type
    if dataset_type == "STS":
        pairs = datasets.get('pairs', [])
    elif dataset_type == "SNLI":
        pairs = datasets.get('pairs', [])
    elif dataset_type == "WordNet":
        pairs = datasets.get('pairs', [])
    elif dataset_type == "CxC":
        pairs = datasets.get('pairs', [])
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # 4. Initialize model
    model_name = config['model'].get('name', 'gpt2')
    device = config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model_wrapper = ModelWrapper(model_name, device)

    # 5. Setup sampler
    sampler_params = config['sampling']
    n_traj = int(sampler_params.get('num_trajectories', 20))
    max_len = int(sampler_params.get('max_length', 20))
    temperature = float(sampler_params.get('temperature', 1.0))
    sampler = Sampler(model_wrapper, n_traj, max_len, temperature, seed=seed)

    # 6. Setup likelihood calculator
    likelihood = LikelihoodCalculator(model_wrapper)

    # 7. Initialize distance evaluator
    eval_params = config['evaluation']
    distance_metric = eval_params.get('distance_metric', 'log-l1')
    distance_eval = DistanceEvaluator(n_samples=n_traj)

    # 8. Prepare results containers
    similarities = []
    true_scores = []  # For correlation metrics
    relations_results = {'entailment': [], 'hypernym': []}

    # 9. Loop over pairs for evaluation
    for pair in pairs:
        u, v = pair
        # Generate trajectories for u and v
        T_u = sampler.sample_trajectories(u)
        T_v = sampler.sample_trajectories(v)

        # Compute log probabilities for each trajectory
        log_mu_u = []
        log_mu_v = []

        for t in T_u:
            log_prob = likelihood.compute_log_prob(t, u)
            log_mu_u.append(log_prob)
        for t in T_v:
            log_prob = likelihood.compute_log_prob(t, v)
            log_mu_v.append(log_prob)

        # Approximate distance
        dist = distance_eval.compute_distance_from_logs(
            log_mu_u, log_mu_v
        )
        similarities.append(dist)

        # For semantic similarity evaluation
        if dataset_type == "STS":
            # Assumed true similarity scores (normalized)
            true_score = pair[-1]  # if dataset provides
            true_scores.append(true_score)  # adapt based on dataset format

        # Relation inference examples
        # -- Entailment
        entailment = distance_eval.infer_entailment(
            log_mu_u, log_mu_v
        )
        relations_results['entailment'].append(entailment)

        # -- Hyponym
        hypernym = distance_eval.infer_hyponym(
            log_mu_u, log_mu_v
        )
        relations_results['hypernym'].append(hypernym)

    # 10. Final evaluation metrics
    # a. Similarity: compute Spearman correlation
    if dataset_type == "STS" and true_scores:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(true_scores, similarities)
        print(f"Spearman correlation (STS): {rho:.4f}")

    # b. Relation tasks (accuracy)
    # For simplicity, convert boolean list to numpy array
    for rel_type, preds in relations_results.items():
        preds_np = np.array(preds)
        # Ground truth Labels: assume availability, here placeholder
        # e.g., Chinese set for entailment, hypernym relations from dataset
        # For this code, we cannot get ground-truth here, so skipping actual accuracy calculation
        # To implement: compare preds with ground-truth labels from dataset
        pass

    # 11. Summarize results
    print("Evaluation completed.")
    # Implement logging or saving results as needed

if __name__ == "__main__":
    main()
