# main.py
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
import logging

from dataset import DatasetLoader
from model import TransformerModel
from utils import set_random_seed, log_metrics
from trainer import Trainer
from evaluation import Evaluation

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    seed = int(config.get('misc', {}).get('seed', 42))
    set_random_seed(seed)

    # Setup device
    device_str = config.get('misc', {}).get('device', 'cuda:0')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # Verbose logging setup
    verbose = config.get('misc', {}).get('verbose_logging', True)
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # 1. Dataset Loading
    dataset_cfg = config.get('dataset', {})
    dataset_loader = DatasetLoader(dataset_cfg)

    print(f"Loading dataset: {dataset_cfg.get('dataset_name', 'Unknown')}")
    # Load train, validation, test datasets
    train_dataset = dataset_loader.get_dataset(split=dataset_cfg.get('train_split', 'train'), is_training=True)
    val_dataset = dataset_loader.get_dataset(split=dataset_cfg.get('validation_split', 'validation'), is_training=False)
    test_dataset = dataset_loader.get_dataset(split=dataset_cfg.get('test_split', 'test'), is_training=False)

    # DataLoader creation
    batch_size = int(config.get('training', {}).get('batch_size', 16))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model Initialization
    model_cfg = config.get('model', {})
    pretrained_name = model_cfg.get('pretrained_model_name', '')
    peft_method = model_cfg.get('peft_method', 'LoRA')
    peft_rank = int(model_cfg.get('peft_rank', 16))
    adapter_scale = float(model_cfg.get('adapter_params', 1.0))
    perturb_sigma = float(model_cfg.get('perturbation_sigma', 0.2))
    adapter_perturbation = bool(model_cfg.get('adapter_perturbation', True))
    output_regulation = bool(model_cfg.get('output_regularization', True))
    
    # Instantiate model
    model = TransformerModel(
        pretrained_model_name=pretrained_name,
        config={
            'peft_method': peft_method,
            'peft_rank': peft_rank,
            'adapter_params': adapter_scale,
            'perturbation_sigma': perturb_sigma,
            'adapter_perturbation': adapter_perturbation,
            'output_regularization': output_regulation
        }
    ).to(device)

    # 3. Optimizer setup
    learning_rate = float(config.get('training', {}).get('learning_rate', 2e-5))
    weight_decay = float(config.get('training', {}).get('weight_decay', 1e-4))
    optimizer_type = config.get('training', {}).get('optimizer', 'AdamW')
    # For simplicity, use AdamW
    params = list(model.get_parameters())  # Only PEFT + head params typically
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # 4. Trainer initialization
    trainer_cfg = {
        'epochs': int(config.get('training', {}).get('epochs', 300)),
        'batch_size': batch_size,
        'lambda_consistency': float(config.get('training', {}).get('lambda_consistency', 0.01)),
        'sigma_noise': float(config.get('training', {}).get('sigma_noise', 0.2)),
        'regularization_type': config.get('training', {}).get('regularization_type', 'standard'),
        'lazy_update_interval': int(config.get('training', {}).get('lazy_update_interval', 10)),
        'use_previous_epoch_outputs': bool(config.get('training', {}).get('use_previous_epoch_outputs', True))
    }
    trainer = Trainer(model, train_loader, optimizer, trainer_cfg, device=device_str)

    # 5. Training Loop
    for epoch in range(1, trainer_cfg['epochs'] + 1):
        print(f"Epoch {epoch}/{trainer_cfg['epochs']}")
        trainer.train_one_epoch()

        # Validation & Model Saving
        if epoch % int(config.get('evaluation', {}).get('evaluation_interval', 10)) == 0:
            val_metrics = trainer.evaluate(split='validation')
            # Check for best model
            val_acc = val_metrics.get('accuracy', 0.0)
            if trainer.best_val_metric is None or val_acc > trainer.best_val_metric:
                trainer.best_val_metric = val_acc
                # Save best model state dict
                trainer._save_best_model()

    # Load the best checkpoint after training
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load('best_model.pth') if os.path.exists('best_model.pth') else model.state_dict())

    # 6. Final Evaluation on Test Set
    print("Evaluating on test dataset...")
    test_metrics = trainer.evaluate(split='test')

    # Log final test metrics
    print(f"Final Test Metrics: {test_metrics}")

def save_best_model(trainer: 'Trainer', path='best_model.pth'):
    torch.save(trainer.model.state_dict(), path)

# This is the main entry point.
if __name__ == '__main__':
    main()
