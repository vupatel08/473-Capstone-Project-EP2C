## main.py
import os
import yaml
import torch
import numpy as np
import random
from dataset_loader import DatasetLoader
from model import CATSModel
from trainer import Trainer
from evaluation import Evaluation
from utils import plot_attention_map, plot_forecast_and_attention
from datetime import datetime

def main():
    # 1. Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. Set random seeds for reproducibility
    seed = config.get('misc', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 3. Setup device based on GPU availability
    num_gpus = config.get('hardware', {}).get('gpus', 1)
    if torch.cuda.is_available() and num_gpus > 0:
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        print(f"Using {torch.cuda.device_count()} GPU(s).")
    else:
        device = torch.device('cpu')
        print("Using CPU.")

    # 4. Load Dataset
    dataset_name = config['dataset'].get('name', 'ETTm1')
    data_path = config['dataset'].get('data_path', './datasets')
    dataset_loader = DatasetLoader([dataset_name], config)
    # Load train, val, test sets for the target dataset
    train_data, _ = dataset_loader.get_train_test_split(dataset_name + '_train')
    val_data, _ = dataset_loader.get_train_test_split(dataset_name + '_val')
    test_data, _ = dataset_loader.get_train_test_split(dataset_name + '_test')

    # 5. Initialize Model
    model_params = {
        'model': {
            'input_sequence_length': config['model'].get('input_sequence_length', 96),
            'forecast_horizon': config['model'].get('forecast_horizon', 72),
            'patch_size': config['model'].get('patch_size',24),
            'num_layers': config['model'].get('num_layers',3),
            'num_heads': config['model'].get('num_heads',2),
            'embed_dim': config['model'].get('embed_dim',256),
            'horizon_embeddings': True,
            'parameter_sharing': True
        },
        'training': {
            'mask_probability': config['training'].get('mask_probability', 0.2),
            'dropout_rate': config['training'].get('dropout_rate',0.1)
        }
    }
    model = CATSModel(model_params['model'])
    model.to(device)

    # 6. Set up optimizer, scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['training'].get('learning_rate',1e-3),
                                 weight_decay=config['training'].get('weight_decay',1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                           patience=config['training'].get('patience',10),
                                                           verbose=True)

    # 7. Prepare datasets for training
    train_tensor, _ = dataset_loader.get_train_test_split(dataset_name+'_train')
    val_tensor, _ = dataset_loader.get_train_test_split(dataset_name+'_val')
    test_tensor, _ = dataset_loader.get_train_test_split(dataset_name+'_test')

    # Wrap tensors in DataLoader
    def get_loader(tensor):
        dataset = torch.utils.data.TensorDataset(tensor)
        return torch.utils.data.DataLoader(dataset, batch_size=config['training'].get('batch_size',32), shuffle=True, drop_last=False)

    train_loader = get_loader(train_tensor)
    val_loader = get_loader(val_tensor)
    test_loader = get_loader(test_tensor)

    # 8. Initialize Trainer
    trainer = Trainer(model, {'train': train_tensor, 'val': val_tensor}, config)

    # 9. Run training with early stopping
    print(f"Starting training for {trainer.epochs} epochs...")
    trainer.train()

    # 10. Load best model weights
    trainer._load_checkpoint('best.pth')

    # 11. Run evaluation
    evaluator = Evaluation(checkpoint_path='best.pth',
                             dataset_name=dataset_name,
                             dataset_path=data_path,
                             config=config,
                             device=device)
    evaluator.evaluate()

    # Optional: Visualize some attention maps / forecasts
    # For example, plot attention maps on sample predictions
    # (Assuming evaluator.store attentions during evaluation as needed)

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Experiment started at {start_time}")
    main()
    end_time = datetime.now()
    print(f"Experiment finished at {end_time}")
    print(f"Total duration: {end_time - start_time}")

