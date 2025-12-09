## main.py
import os
import yaml
import numpy as np
import torch
import random
from dataset_loader import DatasetLoader
from graph_utils import create_sequence_graph, compute_graph_laplacian, perform_label_smoothing, cluster_sequences, select_top_per_cluster
from label_smoothing import LabelSmoother
from model import SequenceFitnessPredictor
from trainer import Trainer
from sampling import GGSampler
from evaluation import Evaluation

def main():
    # 1. Load configuration and set seeds for reproducibility
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    seed = config.get('training', {}).get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Prepare datasets for each dataset and difficulty
    datasets_info = [
        ('GFP', 'GFP', 'dataset_filters', 'GFP'),
        ('AAV', 'AAV', 'dataset_filters', 'AAV')
    ]

    for dataset_name, dataset_type, filters_key, dataset_key in datasets_info:
        filters_config = config['dataset_filters'][dataset_type]

        # Load filtered dataset
        loader = DatasetLoader(
            dataset_path=f'{dataset_name}_data.csv',  # Path to dataset, adjust as needed
            dataset_name=dataset_key,
            filters=filters_config,
            config=config
        )
        sequences, fitness = loader.get_filtered_dataset(difficulty_level=filters_key['hard'])
        # Save filtered data or keep in variables
        initial_seqs = sequences
        initial_fits = np.array(fitness)

        # 3. Build sequence similarity graph
        knn = config['graph_construction'].get('knn_neighbors', 20)
        G = create_sequence_graph(sequences, knn_neighbors=knn)
        L = compute_graph_laplacian(G, normalized=True)

        # 4. Prepare labels for smoothing
        Y = np.array(initial_fits)
        # Store smoothed labels for each gamma
        gamma_values = config['label_smoothing'].get('gamma_values', [0.01, 0.1, 1.0, 10.0])
        smoothed_labels_list = []

        for gamma in gamma_values:
            # 4a. Obtain smoothed labels
            smoothed_labels = perform_label_smoothing(Y, L, gamma)
            smoothed_labels_list.append((gamma, smoothed_labels))

        # 5. For each smoothed label set, train predictor and evaluate
        best_metrics = None
        best_gamma = None

        for gamma, smoothed_Y in smoothed_labels_list:
            # 5a. Train predictor
            predictor_config = {
                'architecture': config['predictor_model'].get('architecture', 'cnn'),
                'sequence_length': len(sequences[0]),
                'learning_rate': 1e-3,
                'batch_size': 128,
                'epochs': 50,
                'dropout_rate': config['predictor_model'].get('dropout_rate', 0.1)
            }
            trainer = Trainer(
                predictor_config,
                train_sequences=sequences,
                train_labels=smoothed_Y,
                val_sequences=sequences,  # For simplicity, using same data for val; can be split
                val_labels=smoothed_Y,
                checkpoint_dir=f'./checkpoints_{dataset_name}_{filters_key}_{gamma}'
            )
            trainer.train()
            predictor_model = trainer.get_model()

            # 5b. Run in-silico evaluation on initial data
            predictor_model.eval()
            predictions = predictor_model.predict_batch(sequences)
            median_pred = np.median(predictions)
            # Compute diversity
            seqs_for_diversity = sequences
            diversities = []
            for i in range(len(sequences)):
                for j in range(i+1, len(sequences)):
                    diversities.append(predictor_model.compute_sequence_distance(sequences[i], sequences[j]))
            median_diversity = np.median(diversities) if diversities else 0.0
            # Compute novelty with respect to training set
            novelties = []
            for seq in sequences:
                min_d = min([predictor_model.compute_sequence_distance(seq, t_seq) for t_seq in sequences])
                novelties.append(min_d)
            median_novelty = np.median(novelties) if novelties else 0.0

            # 5c. Run GWG sampling with clustering (the GGS process)
            gwg_params = {
                'gwg_rounds': config['sampling'].get('gwg_rounds', 15),
                'proposal_per_seq': config['sampling'].get('proposals_per_sequence', 100),
                'temperature_grid': config['sampling'].get('temperature_grid', [0.01, 0.1, 1.0, 10.0]),
                'cluster_num': config['sampling'].get('clustering_clusters', 20),
                'mutation_batch_size': config['sampling'].get('mutation_batch_size', 100),
                'sequence_length': len(sequences[0]),
                'vocab_size': 20,
                'seed': seed
            }
            # Initialize sampler with current sequences
            sampler = GGSampler(
                predictor=predictor_model,
                sequences=sequences,
                predictor_predict_func=predictor_model.predict,
                predictor_grad_func=predictor_model.compute_gradients,
                proposals_per_sequence=gwg_params['proposal_per_seq'],
                gwg_rounds=gwg_params['gwg_rounds'],
                clustering_clusters=gwg_params['cluster_num'],
                temperature=gwg_params['temperature_grid'][0],  # can do hyperparam sweep
                mutation_batch_size=gwg_params['mutation_batch_size'],
                sequence_length=gwg_params['sequence_length'],
                vocab_SIZE=gwg_params['vocab_size'],
                seed=gwg_params['seed']
            )

            sampled_sequences = sampler.run_sampling()

            # 5d. Evaluate sampled sequences
            pred_samples = predictor_model.predict_batch(sampled_sequences)
            best_fitness_sample = np.max(pred_samples)
            avg_fitness_sample = np.mean(pred_samples)

            # 5e. Store metrics, compare to previous best
            current_metrics = {
                'median_fitness': median_pred,
                'diversity': median_diversity,
                'novelty': median_novelty,
                'sampled_best': best_fitness_sample,
                'sampled_avg': avg_fitness_sample,
                'gamma': gamma
            }
            if best_metrics is None or best_metrics['sampled_best'] < best_fitness_sample:
                best_metrics = current_metrics
                best_gamma = gamma

        # 6. Final reporting for current dataset/difficulty
        print(f"Dataset: {dataset_name} | Difficulty: {filters_key} | Best gamma: {best_gamma}")
        print(f"Metrics: {best_metrics}")

if __name__ == '__main__':
    main()
