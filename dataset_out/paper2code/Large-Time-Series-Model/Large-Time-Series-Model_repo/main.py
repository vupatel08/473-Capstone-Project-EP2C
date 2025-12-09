## main.py
import os
import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path

# Import modules following the defined interfaces
from dataset_loader import DatasetLoader
from tokenization import Tokenizer
from model import TimerTransformer
from trainer import Trainer
from evaluation import Evaluation

def main():
    # 1. Parse configuration
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Set Random seeds for reproducibility
    seed = cfg.get('training', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 3. Prepare logging/checkpoint directories
    save_dir = cfg.get('logging', {}).get('save_dir', 'checkpoints/')
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. Instantiate DatasetLoader
    dataset_paths = []
    for ds_cfg in cfg['dataset']['pretraining_datasets']:
        dataset_paths.append(ds_cfg['path'])
    dataset_loader = DatasetLoader(dataset_paths, cfg)
    
    # 5. Instantiate Tokenizer for pretraining
    hierarchy_levels = cfg['dataset'].get('dataset_species_levels', [])
    # Extract segment lengths for hierarchy levels
    segment_lengths = []
    for level in hierarchy_levels:
        level_name = level.get('name', '')
        # Default pattern; adjust if necessary
        if '96' in level_name:
            segment_lengths.append(96)
        elif '672' in level_name:
            segment_lengths.append(672)
        elif '1440' in level_name:
            segment_lengths.append(1440)
        else:
            segment_lengths.append(96)  # fallback

    tokenizer = Tokenizer(
        hierarchy_levels=['small','medium','large'],
        segment_lengths=segment_lengths,
        use_timestamps=True,
        max_sequence_length=cfg['training'].get('max_sequence_length', 1440),
        embedding_dim=cfg['model'].get('hidden_size', 512)
    )
    
    # 6. Convert datasets to tokenized sequences for pretraining
    # Gather all training series (from all datasets and split)
    train_series_list = dataset_loader.get_series_split('train')
    # Convert to tokenized sequences (list of tensors)
    tokenized_train_sequences = []
    for series in train_series_list:
        seq_obj = tokenizer.convert_series_to_sequence(series)
        seq_ids, _ = seq_obj.to_id_tensor()
        tokenized_train_sequences.append(seq_ids)
    
    # 7. Build model
    model_size = cfg['model'].get('size_m', 50)  # in millions
    size_multiplier = {
        1: 1, 50: 1, 91: 2, 311: 3, 385: 4
    }.get(model_size, 50)
    model_params = {
        'size': cfg['model'].get('hidden_size', 512),
        'num_layers': cfg['model'].get('num_layers', 6),
        'num_heads': cfg['model'].get('num_heads', 8),
        'max_position_embeddings': cfg['model'].get('max_position_embeddings', 1024),
        'dropout': cfg['model'].get('dropout_rate', 0.1),
        'input_token_length': cfg['model'].get('input_token_length', 96),
        'use_positional_embedding': True,
        'use_timestamp_embedding': True
    }
    model = TimerTransformer(**model_params).to(device)
    
    # 8. Load pretrained checkpoint if enabled (e.g., for fine-tuning)
    pretrained_path = None
    # Decide based on typical configs, or add config option
    if cfg.get('training', {}).get('pretrained_checkpoint_path'):
        pretrained_path = cfg['training']['pretrained_checkpoint_path']
        if os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"Pretrained checkpoint not found at {pretrained_path}")
    
    # 9. Initialize Trainer with dataset, model, hyperparameters
    trainer_cfg = cfg['training']
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_sequences,
        val_dataset=None,  # Could set validation sequences similarly if validation data is available
        config=trainer_cfg
    )
    
    # 10. Pretraining phase
    pretrain_flag = cfg.get('training', {}).get('do_pretrain', True)
    if pretrain_flag:
        print("Starting pretraining...")
        trainer.train()
        # Save final pretrained checkpoint
        final_ckpt_path = os.path.join(save_dir, 'pretrain_final.pt')
        trainer.save_checkpoint(final_ckpt_path)
        print(f"Pretraining completed. Checkpoint saved at {final_ckpt_path}")
        # Load the final checkpoint into model
        ckpt = torch.load(final_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("Skipping pretraining as per configuration.")
    
    # 11. Fine-tuning / downstream tasks
    downstream_task = cfg.get('task', 'forecasting')  # default to forecasting
    task_params = cfg.get('task_params', {})
    # Load or prepare datasets for downstream task, similar to training but task-specific
    # For simplicity, assume datasets for downstream are prepared similarly
    # For real usage, datasets are loaded and tokenized per task
    # For this code, as per instruction, invoke evaluation only
    
    # Prepare dataset for downstream task
    # For example, for forecasting:
    downstream_dataset_sequences = []
    # Assume task datasets are stored or loaded accordingly
    # Here, we just assume test series (mocked)
    test_series_list = dataset_loader.get_series_split('test')
    # Convert each to token sequences with task-specific parameters
    for series in test_series_list:
        seq_obj = tokenizer.convert_series_to_sequence(series)
        seq_ids, _ = seq_obj.to_id_tensor()
        downstream_dataset_sequences.append(seq_ids)
    
    # Initialize Evaluation object
    eval_obj = Evaluation(
        model=model,
        dataset_loader=dataset_loader,
        task=downstream_task,
        task_params=task_params,
        config=cfg
    )
    # Run evaluation (can be forecast, impute, detect)
    eval_obj.evaluate()
    print(f"Evaluation for task '{downstream_task}' completed. Results:")
    print(eval_obj.results)
    
    # 12. Save evaluation results and optionally generate figures
    # Save results json
    results_path = os.path.join(cfg.get('logging', {}).get('save_dir', 'checkpoints/'), 'evaluation_results.json')
    import json
    with open(results_path, 'w') as f:
        json.dump(eval_obj.results, f, indent=4)
    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    main()
