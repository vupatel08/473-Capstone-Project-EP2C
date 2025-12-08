# main.py
import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

# Import the modules as per design
import dataset_loader
import prompt_builder
import model
import trainer
import evaluation

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 1. Parse command-line args for optional config path
    parser = argparse.ArgumentParser(description='AutoTimes Reproduction Main')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    # 2. Load config.yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 3. Set seed for reproducibility
    seed = 42
    set_seed(seed)

    # 4. Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 5. Load datasets using dataset_loader
    dataset_paths = config['dataset']
    segment_size = config['hyperparameters'].get('segment_size', 96)
    
    dl_module = dataset_loader.DataLoader(dataset_paths, segment_size=segment_size)
    all_datasets = dl_module.load_all()

    # For demonstration:
    # For training, pick one dataset or multiple datasets for training
    # and prepare splits with respect to realistic scenario:
    # Here, we pick one dataset e.g., "etth1" (adjust as needed)
    train_dataset_name = 'etth1'   # in practice, can be based on training setup
    val_dataset_name = 'etth1'     # For simplicity, use same dataset for val
    test_dataset_name = 'etth1'    # Similarly for test

    train_samples = all_datasets[train_dataset_name]
    val_samples = all_datasets[val_dataset_name]
    test_samples = all_datasets[test_dataset_name]

    # 6. Initialize PromptBuilder with tokenizer
    pretrained_model_name = config['model']['pretrained_model_name']
    # Using transformers tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    prompt_builder_obj = prompt_builder.PromptBuilder(
        tokenizer=tokenizer,
        segment_size=config['hyperparameters'].get('segment_size', 96),
        prompt_strategy=config['hyperparameters'].get('prompt_strategy', 'firstF'),
        prompt_length=config['hyperparameters'].get('text_prompt_length', 48)
    )

    # 7. Initialize Model (load pretrained LM, freeze backbone)
    model_obj = model.Model(
        pretrained_model_path=pretrained_model_name,
        embedding_dim=config['model'].get('embedding_dim', 768),
        segment_size=config['hyperparameters'].get('segment_size', 96)
    )
    model_obj.freeze_backbone()

    # 8. Initialize Trainer
    hyperparams = trainer.Hyperparameters(config)
    trainer_obj = trainer.Trainer(
        model=model_obj,
        dataset_samples=train_samples,
        prompt_builder=prompt_builder_obj,
        hyperparams=hyperparams,
        device=device
    )

    # 9. Setup DataLoader for training
    # Note: Implement batching with collate_fn based on your dataset loader
    # For simplicity, assuming dataset_samples are already in appropriate form
    # Here, just replicate DataLoader with batch_size from config
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_samples, batch_size=hyperparams.batch_size, shuffle=True, collate_fn=trainer_obj._collate_fn)
    val_loader = DataLoader(val_samples, batch_size=hyperparams.batch_size, shuffle=False, collate_fn=trainer_obj._collate_fn)
    # Save the loaders in trainer
    trainer_obj.set_dataloaders(train_samples, val_samples, test_samples)
        
    # 10. Train the model
    trainer_obj.train()

    # 11. Load the best checkpoint
    checkpoint_path = os.path.join(trainer_obj.save_dir, 'best_model.pt')
    # Assuming trainer has method to load
    model_obj.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model_obj.eval()

    # 12. Inference / Evaluation on test dataset
    eval_obj = evaluation.Evaluation(
        dataset_samples=test_samples,
        model_path=checkpoint_path,
        dataset_name='Test Dataset',
        forecast_horizon=hyperparams.forecast_horizon,
        prompt_strategy=hyperparams.prompt_strategy,
        prompt_length=hyperparams.text_prompt_length,
        device=device
    )

    metrics_results = eval_obj.evaluate()
    print("Test Metrics:", metrics_results)

    # 13. Save predictions, possibly as plots or files
    # For demonstration, evaluation class already produces plots/logs

if __name__ == '__main__':
    main()
