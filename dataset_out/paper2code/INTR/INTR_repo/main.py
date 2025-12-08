# main.py
import os
import sys
import torch
import numpy as np
from utils import load_config, set_seed
from dataset_loader import DatasetLoader
from model import INTRModel
from trainer import Trainer
from evaluation import Evaluation

def main():
    # 1. Determine config file path (e.g., passed as argument or default)
    # For simplicity, assume 'config.yaml' in the current directory
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"Configuration file '{config_path}' not found.")
        sys.exit(1)

    # 2. Load configuration
    config = load_config(config_path)

    # 3. Set seed for reproducibility
    seed = config.get('misc', {}).get('seed', 42)
    set_seed(seed)

    # 4. Set device: GPU if available, else CPU
    device_str = config.get('training', {}).get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() and device_str=='cuda' else 'cpu')
    print(f"Using device: {device}")

    # 5. Initialize dataset loader
    dataset_cfg = config.get('dataset', {})
    dataset_path = dataset_cfg.get('path', '')  # assume path provided in config if needed
    train_split_path = dataset_cfg.get('train_split', '')
    test_split_path = dataset_cfg.get('test_split', '')
    image_size = dataset_cfg.get('image_size', 224)
    batch_size = dataset_cfg.get('batch_size', 16)

    dataset_loader = DatasetLoader(
        dataset_path=dataset_path,
        train_split=train_split_path,
        test_split=test_split_path,
        image_size=image_size,
        batch_size=batch_size,
        dataset_name=dataset_cfg.get('name', ''),
        use_fully_finetune_backbone=dataset_cfg.get('use_fully_finetune_backbone', False)
    )

    # Extract DataLoaders
    train_loader = dataset_loader.train_loader
    test_loader = dataset_loader.test_loader

    # 6. Prepare model
    model_cfg = config.get('model', {})
    # Add dataset-specific class count
    num_classes = dataset_loader.num_classes
    model_cfg['class_queries'] = num_classes
    model_cfg['pretrained_weights'] = model_cfg.get('pretrained_weights', '')

    # Instantiate model
    model = INTRModel(model_cfg)
    model.to(device)

    # 7. Set up optimizer, scheduler
    training_cfg = config.get('training', {})
    lr = training_cfg.get('learning_rate', 1e-4)
    weight_decay = training_cfg.get('weight_decay', 0.05)
    epochs = training_cfg.get('epochs', 50)

    # Only fine-tune backbone if specified
    use_finetune = dataset_cfg.get('use_fully_finetune_backbone', False)

    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Scheduler: cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 8. Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # 9. Initialize Trainer
    trainer = Trainer(
        model=model,
        data_loader={'train': train_loader, 'val': test_loader},
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        config=training_cfg
    )

    # 10. Run training
    print("Starting training...")
    trainer.train()

    # 11. Load best model for evaluation and interpretability
    best_model_path = os.path.join(
        training_cfg.get('save_dir', 'outputs/checkpoints'), 'best_model.pth'
    )
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
    else:
        print("Best model checkpoint not found. Using last epoch model.")

    # 12. Final evaluation and visualization
    print("Evaluating and visualizing attention maps...")
    evaluator = Evaluation(
        model=model,
        data_loader=test_loader,
        config=training_cfg,
        device=str(device)
    )
    metrics = evaluator.evaluate()
    print("Evaluation metrics:", metrics)

    # Optional: Visualize some attention maps for interpretability
    # For example, visualize attention for first few images
    # (This can be integrated into Evaluation or called separately)
    # e.g.,
    # evaluator.visualize_attention_maps(...)

if __name__ == "__main__":
    main()
