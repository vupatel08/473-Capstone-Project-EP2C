## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
import copy

from model import GraphGRED
from dataset_loader import GraphDataset
from utils import (
    compute_shortest_paths,
    create_neighborhood_masks,
)
from torch_geometric.data import Batch

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def main():
    # Load configuration
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset parameters
    dataset_cfg = cfg['dataset']
    dataset_name = dataset_cfg.get('name', 'CIFAR10')
    dataset_path = dataset_cfg.get('path', None)
    max_K = cfg['model'].get('neighborhood_K', None)
    
    # Load dataset
    dataset = GraphDataset(dataset_name, dataset_path, max_K=max_K)
    
    # Data splitting: for simplicity, assume all data is for training
    # For realistic scenario, split into train/val/test
    total_samples = len(dataset)
    indices = list(range(total_samples))
    # Here, for demonstration, use all for training; adapt as necessary for validation/testing
    train_indices = indices
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=cfg['training'].get('batch_size', 32), shuffle=True, collate_fn=collate_fn)
    
    # Initialize model
    model_cfg = cfg['model']
    input_dim = dataset[0]['features'].shape[1]  # feature dimension
    model = GraphGRED(model_cfg, input_dim=input_dim).to(device)
    
    # Initialize optimizer
    lr = cfg['training'].get('learning_rate', 1e-3)
    weight_decay = cfg['training'].get('weight_decay', 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler (optional)
    scheduler = None
    if 'scheduler' in cfg['training']:
        if cfg['training']['scheduler'] == 'ExponentialDecay':
            decay_rate = cfg['training'].get('lr_decay_rate', 0.99)
            decay_steps = cfg['training'].get('lr_decay_steps', 1000)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        # Add other schedulers if needed
    
    # Loss criterion
    task_type = cfg['evaluation'].get('metrics', ['accuracy'])[0]
    if task_type == 'accuracy':
        criterion = nn.CrossEntropyLoss()
    elif task_type == 'MAE':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    
    num_epochs = cfg['training'].get('epochs', 600)
    eval_interval = cfg['evaluation'].get('eval_interval', 10)
    save_dir = cfg.get('save_dir', 'checkpoints/')
    os.makedirs(save_dir, exist_ok=True)
    save_model = cfg.get('save_model', True)
    experiment_name = cfg.get('experiment_name', 'GRED_training')
    
    # Track best performance
    best_metric = None
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            features = batch['features'].to(device)            # (V, d)
            masks_list = batch['neighborhood_masks']           # list of dicts per layer
            labels = batch['label'].to(device)                 # (batch_size,) for graph tasks or (V,) for node tasks
            # For node classification, batch features may need splitting; alternatively, process graph-wise
            
            optimizer.zero_grad()
            outputs = model(features, masks_list)
            if outputs.dim() > 1 and labels.dim() == 1:
                # For node classification: outputs shape (V, num_classes), labels (V,)
                loss = criterion(outputs, labels)
            else:
                # For graph classification: aggregate node embeddings if needed
                # Here assume graph-level labels, do mean pooling
                # But for this code, assume node-level tasks
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Constrain eigenvalues after each update
            for layer in model.layers:
                layer.rnn_encoder.constrain_eigenvalues()
            epoch_loss += loss.item() * features.shape[0]
            pbar.set_postfix(loss=loss.item())
        epoch_loss /= len(dataset)  # average loss over all samples

        # Step learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Logging training info
        print(f"Epoch {epoch}: loss={epoch_loss:.4f}")

        # Evaluation
        if epoch % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                # For evaluation, process the entire dataset
                all_features = torch.stack([d['features'] for d in dataset], dim=0).to(device)
                all_masks = [d['neighborhood_masks'] for d in dataset]
                all_labels = torch.stack([d['label'] for d in dataset], dim=0).to(device)
                outputs = model(all_features, all_masks)
                if task_type == 'accuracy':
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == all_labels).float().mean().item()
                    print(f"Validation Accuracy at epoch {epoch}: {acc:.4f}")
                    # Save best model
                    if best_metric is None or acc > best_metric:
                        best_metric = acc
                        best_state = copy.deepcopy(model.state_dict())
                elif task_type == 'MAE':
                    mae = nn.L1Loss()(outputs, all_labels).item()
                    print(f"Validation MAE at epoch {epoch}: {mae:.6f}")
                    if best_metric is None or mae < best_metric:
                        best_metric = mae
                        best_state = copy.deepcopy(model.state_dict())
                else:
                    # Add other metrics as needed
                    pass
                # Save checkpoint
                if save_model:
                    torch.save(model.state_dict(), os.path.join(save_dir, f"{experiment_name}_epoch{epoch}.pt"))
    
    # After training, load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best model based on validation performance.")
    # Final evaluation
    with torch.no_grad():
        all_features = torch.stack([d['features'] for d in dataset], dim=0).to(device)
        all_masks = [d['neighborhood_masks'] for d in dataset]
        all_labels = torch.stack([d['label'] for d in dataset], dim=0).to(device)
        outputs = model(all_features, all_masks)
        if task_type == 'accuracy':
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == all_labels).float().mean().item()
            print(f"Final Test Accuracy: {acc:.4f}")
        elif task_type == 'MAE':
            mae = nn.L1Loss()(outputs, all_labels).item()
            print(f"Final Test MAE: {mae:.6f}")
        # Add other metrics as needed

def collate_fn(batch):
    """
    Collate function to batch graph data with neighborhood masks.
    """
    features_list = []
    label_list = []
    masks_list = []
    for item in batch:
        features_list.append(item['features'])
        label_list.append(item['label'])
        masks_list.append(item['neighborhood_masks'])
    return {
        'features': torch.cat(features_list, dim=0),
        'label': torch.stack(label_list),
        'neighborhood_masks': masks_list
    }

if __name__ == '__main__':
    main()
