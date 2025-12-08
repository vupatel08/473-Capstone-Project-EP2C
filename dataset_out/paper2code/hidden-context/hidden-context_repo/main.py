## main.py
import yaml
import os
import torch
import numpy as np
from tqdm import tqdm

from dataset_loader import DatasetLoader
from model import PreferenceModel
from trainer import Trainer
from evaluation import Evaluation

def main():
    # 1. Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Parse training parameters
    train_cfg = config.get('training', {})
    learning_rate = train_cfg.get('learning_rate', 3e-6)
    min_lr = train_cfg.get('min_learning_rate', 3e-7)
    batch_size = train_cfg.get('batch_size', 2)
    epochs = train_cfg.get('epochs', 2)
    lambda_reg = train_cfg.get('lambda_reg', 0.0001)
    reg_type = train_cfg.get('regularization_type', 'l2')
    use_regularization = train_cfg.get('use_regularization', True)
    optimizer_name = train_cfg.get('optimizer', 'AdamW')
    lr_schedule = train_cfg.get('lr_schedule', 'cosine')
    # total_steps can be computed later based on dataset size and epochs

    # Parse model parameters
    model_cfg = config.get('model', {})
    base_model_name = model_cfg.get('base_model', 'llama-2-7b-hf')
    head_type = model_cfg.get('head_type', 'scalar')
    num_outputs = model_cfg.get('num_outputs', 1)
    lora_rank = model_cfg.get('lora_rank', 8)

    # Parse dataset setup
    dataset_cfg = config.get('dataset', {})
    dataset_type = dataset_cfg.get('dataset_type', 'synthetic')
    synthetic_size = dataset_cfg.get('synthetic_size', 10000)
    relabel = dataset_cfg.get('relabel', False)

    # Save directory for checkpoints
    save_dir = 'saved_model'
    os.makedirs(save_dir, exist_ok=True)

    # 2. Load or generate dataset
    print("Loading dataset...")
    data_loader_obj = DatasetLoader(dataset_type, relabel)

    if dataset_type == 'synthetic':
        # Generate synthetic dataset with known hidden context effects
        alternatives, pairs = data_loader_obj.generate_synthetic_data({'synthetic_size': synthetic_size})
        # For evaluation, store true utilities if needed
        true_utilities = {}
        for a in alternatives:
            # example: true U(a,z) with z ~ Bernoulli(0.5)
            # But in code, actual true utility is defined in synthetic_data, so optional
            pass
        dataset = data_loader_obj.load_data()
    elif dataset_type == 'relabeled_hh_rlhf':
        # Load existing dataset, then relabel if needed
        dataset = data_loader_obj.load_data()
    else:
        # For real datasets, implement actual loading here
        dataset = data_loader_obj.load_data()

    # 3. Initialize model
    print("Initializing model...")
    model_config = {
        'base_model': base_model_name,
        'head_type': head_type,
        'num_outputs': num_outputs,
        'lora_rank': lora_rank
    }
    model = PreferenceModel(model_config)

    # 4. Setup Trainer
    print("Setting up training...")
    total_dataset_size = len(dataset.pairs)
    steps_per_epoch = (total_dataset_size + batch_size - 1) // batch_size
    total_training_steps = steps_per_epoch * epochs

    trainer_args = {
        'batch_size': batch_size,
        'epochs': epochs,
        'lambda_reg': lambda_reg,
        'regularization_type': reg_type,
        'use_regularization': use_regularization,
        'learning_rate': learning_rate,
        'min_learning_rate': min_lr,
        'optimizer': optimizer_name,
        'scheduler': lr_schedule,
        'total_steps': total_training_steps
    }

    trainer = Trainer(model, dataset, trainer_args)

    # 5. Train the model
    print("Starting training...")
    trainer.train()

    # Save final model
    final_path = os.path.join(save_dir, 'final_model')
    print(f"Saving final model to {final_path}")
    model.transformer.save_pretrained(final_path)

    # 6. Evaluation
    print("Evaluating model...")
    # For synthetic data, true utilities are known (if stored); for real, may be absent
    # Here, just demonstrate evaluation
    eval_obj = Evaluation(model, dataset)
    metrics = eval_obj.evaluate()

    # Print evaluation metrics
    print("\n=== Evaluation Results ===")
    print(f"Preference accuracy: {metrics.get('preference_accuracy', 'N/A'):.4f}")
    spearman = metrics.get('spearman_correlation', None)
    if spearman is not None:
        print(f"Spearman correlation to true utilities: {spearman:.4f}")
    kendall = metrics.get('kendall_tau', None)
    if kendall is not None:
        print(f"Kendall tau: {kendall:.4f}")
    print("Borda counts:")
    for alt, score in metrics.get('borda_counts', {}).items():
        print(f"  Alt {alt:.3f}: {score:.3f}")
    print("Hidden context detection (variance):")
    high_var_alts = metrics.get('hidden_context_detection', {}).get('high_variance_alternatives', [])
    if high_var_alts:
        print("Alternatives with high variance indicating hidden context influence:")
        for alt in high_var_alts:
            print(f"  {alt}")
    else:
        print("No high variance signals detected.")

    # Optional: Save metrics to file
    metrics_path = os.path.join(save_dir, 'evaluation_metrics.yaml')
    with open(metrics_path, 'w') as f:
        yaml.safe_dump(metrics, f)

    print("Experiment complete.")

if __name__ == "__main__":
    main()
