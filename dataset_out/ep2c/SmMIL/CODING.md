# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
import pickle
import pandas as pd

class CustomDataset(Dataset):
    """
    Dataset class for MIL bags containing precomputed features, labels, optional instance labels,
    and adjacency matrices for each bag.
    """
    def __init__(self, features_list, bag_labels, instance_labels_list=None, adjacency_matrices=None):
        """
        Args:
            features_list (list of torch.Tensor): list of tensors, each tensor shape (num_instances, feature_dim)
            bag_labels (list or array): list of binary labels for each bag
            instance_labels_list (list of list or tensor): optional, ground truth instance labels for validation
            adjacency_matrices (list of torch.Tensor): optional, normalized adjacency matrices for each bag
        """
        self.features = features_list
        self.labels = torch.tensor(bag_labels, dtype=torch.float)
        self.instance_labels = instance_labels_list
        self.adjacency_matrices = adjacency_matrices

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        data = {
            'features': self.features[idx],  # Tensor (num_instances, feature_dim)
            'label': self.labels[idx],       # float tensor
        }
        if self.instance_labels is not None:
            data['instance_labels'] = self.instance_labels[idx]
        if self.adjacency_matrices is not None:
            data['adjacency'] = self.adjacency_matrices[idx]
        return data

class DatasetLoader:
    """
    Loads dataset features, labels, constructs adjacency matrices, performs dataset splits,
    and provides DataLoader for training/evaluation.
    """
    def __init__(self, data_paths, split_seed=42, split_type='k-NN', k=8, dataset_name='RSNA', batch_size=8, device='cuda'):
        """
        Args:
            data_paths (dict): contains 'features' and 'labels' paths
            split_seed (int): seed for reproducibility of splits
            split_type (str): 'k-NN' for adjacency, can be extended
            k (int): number of neighbors for k-NN graph construction
            dataset_name (str): name identifier, e.g., 'RSNA', 'PANDA', 'CAMELYON16'
            batch_size (int): batch size for DataLoader
            device (str): 'cuda' or 'cpu'
        """
        self.features_path = data_paths['features']
        self.labels_path = data_paths['labels']
        self.split_seed = split_seed
        self.split_type = split_type
        self.k = k
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Internal variables
        self.dataset = None
        self.splits = None  # Dict: {'train': indices, 'val': indices, 'test': indices}
        self.adjacency_matrices = None

        # Load data
        self.features_list, self.bag_labels, self.instance_labels_list = self.load_raw_data()

        # Construct dataset object
        self.construct_dataset()

        # Generate splits
        self.create_splits()

        # Compute adjacency matrices for each bag
        self.compute_adjacency_matrices()

    def load_raw_data(self):
        """
        Loads features and labels from disk.
        Assumes features are stored as individual files per bag or as a large array.
        Adjust according to data format.
        """
        # Load features
        # For this implementation, assume features are stored as a pickle/dict with bag_id as key
        features_list = []
        with open(self.features_path, 'rb') as f:
            features_data = pickle.load(f)
        # features_data: dict {bag_id: np.array or torch.Tensor (num_instances, feature_dim)}
        for key in sorted(features_data.keys()):
            feat = features_data[key]
            if not isinstance(feat, torch.Tensor):
                feat = torch.tensor(feat, dtype=torch.float)
            features_list.append(feat)

        # Load labels
        # Assume labels are stored as a CSV or pickle with same key order
        labels_df = pd.read_csv(self.labels_path)
        # Expect columns: 'bag_id', 'label', possibly 'instance_labels'
        labels_df = labels_df.sort_values('bag_id')
        bag_labels = labels_df['label'].astype(int).tolist()

        # Optionally load instance labels if available
        instance_labels_list = None
        if 'instance_labels' in labels_df.columns:
            # Assuming stored as stringified list or pickled format per row
            # Here, implement accordingly if needed
            # For simplicity, assume not available
            instance_labels_list = None

        return features_list, bag_labels, instance_labels_list

    def construct_dataset(self):
        """Constructs the dataset object."""
        self.dataset = CustomDataset(
            features_list=self.features_list,
            bag_labels=self.bag_labels,
            instance_labels_list=self.instance_labels_list
        )

    def create_splits(self):
        """
        Creates train/validation/test splits using stratified or random sampling.
        Uses the fixed seed for reproducibility.
        """
        np.random.seed(self.split_seed)
        total_bags = len(self.dataset)
        indices = np.arange(total_bags)
        # For simplicity, perform a random split stratified if labels are binary
        # Otherwise, random split
        # Here, as in paper, assume fixed splits are provided; or generate a stratified split
        from sklearn.model_selection import StratifiedShuffleSplit

        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.split_seed)
        train_val_idx, test_idx = next(strat_split.split(indices, self.dataset.labels.numpy()))
        strat_split_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.split_seed+1)
        train_idx, val_idx = next(strat_split_val.split(train_val_idx, np.array(self.dataset.labels)[train_val_idx]))

        self.splits = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }

    def compute_adjacency_matrices(self):
        """
        For each bag, construct adjacency matrix based on feature similarity using k-NN.
        Use Euclidean distance, then build symmetric adjacency.
        For small bags, fallback to fully connected if instances < k.
        """
        adjacency_list = []
        for feat in self.features_list:
            num_instances = feat.shape[0]
            # Handle small bags
            if num_instances <= self.k:
                adj = torch.ones((num_instances, num_instances))
            else:
                # Convert to numpy for sklearn
                feat_np = feat.numpy()
                nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='auto', metric='euclidean').fit(feat_np)
                distances, indices = nbrs.kneighbors(feat_np)
                # Build adjacency matrix
                adj = torch.zeros((num_instances, num_instances))
                for i in range(num_instances):
                    # skip the first neighbor if it's the point itself
                    for j in indices[i][1:]:
                        adj[i, j] = 1
                        adj[j, i] = 1  # symmetrize
            # Normalize adjacency later in the pipeline if needed
            # Save adjacency as dense tensor
            adjacency_list.append(adj)
        self.adjacency_matrices = adjacency_list

    def get_dataloader(self, split_name='train'):
        """
        Returns a DataLoader for the specified split.
        """
        indices = self.splits[split_name]
        subset = torch.utils.data.Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=(split_name=='train'), pin_memory=True, collate_fn=self.collate_fn)

    def get_full_dataset(self):
        """Returns the full dataset object."""
        return self.dataset

    def collate_fn(self, batch):
        """
        Custom collate function to handle variable-sized bags.
        Batches will be a list of dicts from __getitem__.
        Returns batch dictionaries with padded tensors or list of tensors.
        """
        features = [item['features'] for item in batch]
        labels = torch.stack([item['label'] for item in batch], dim=0)
        adjacency = [item['adjacency'] for item in batch] if 'adjacency' in batch[0] else None
        instance_labels = [item.get('instance_labels', None) for item in batch]

        # Pad features to max length in batch if necessary
        max_instances = max(feat.shape[0] for feat in features)
        feat_dim = features[0].shape[1]
        batch_features = torch.zeros((len(features), max_instances, feat_dim))
        batch_adjacency = []
        for i, feat in enumerate(features):
            length = feat.shape[0]
            batch_features[i, :length, :] = feat
            if adjacency is not None:
                # Pad adjacency
                adj = adjacency[i]
                pad_size = max_instances - adj.shape[0]
                adj_pad = torch.zeros((max_instances, max_instances))
                adj_pad[:adj.shape[0], :adj.shape[1]] = adj
                batch_adjacency.append(adj_pad)
        if adjacency is not None:
            batch_adjacency = torch.stack(batch_adjacency)

        return {
            'features': batch_features.to(self.device),
            'labels': labels.to(self.device),
            'adjacency': batch_adjacency.to(self.device) if adjacency is not None else None,
            'instance_labels': instance_labels
        }
```

## evaluation.py

```python
# evaluation.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score
import torch

def calculate_auc_score(preds, labels):
    """
    Computes the AUROC score, handling robust edge cases.
    Args:
        preds (array-like): Predicted scores/probabilities.
        labels (array-like): Ground truth binary labels.
    Returns:
        float: AUROC score.
    """
    labels = np.array(labels)
    preds = np.array(preds)
    # Handle edge cases where labels are homogeneous
    if np.all(labels == 0) or np.all(labels == 1):
        return float('nan')  # Undefined AUROC
    try:
        return roc_auc_score(labels, preds)
    except ValueError:
        return float('nan')

def calculate_f1_score(preds, labels):
    """
    Calculates F1 score at the optimal threshold between 0 and 1.
    Args:
        preds (array-like): Predicted scores.
        labels (array-like): Ground truth labels.
    Returns:
        best_f1 (float): Max F1 over thresholds.
        best_thresh (float): Threshold yielding max F1.
    """
    labels = np.array(labels)
    preds = np.array(preds)
    thresholds = np.linspace(0,1,101)
    best_f1 = -1
    best_thresh = 0.5
    for thresh in thresholds:
        pred_bin = preds >= thresh
        f1 = f1_score(labels, pred_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_f1, best_thresh

def normalize_attention_scores(scores):
    """
    Normalizes attention or scores between 0 and 1 for visualization.
    Args:
        scores (Tensor or np.array): Scores to normalize.
    Returns:
        np.array: normalized scores in [0,1]
    """
    scores_np = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
    min_val = scores_np.min()
    max_val = scores_np.max()
    if max_val - min_val < 1e-8:
        return np.ones_like(scores_np)
    else:
        return (scores_np - min_val) / (max_val - min_val)

def plot_attention_map(attention_weights, images, save_path=None, colormap='jet'):
    """
    Overlays attention scores as heatmaps on images for visualization.
    Args:
        attention_weights (np.array): 1D array of shape (N), normalized between 0 and 1.
        images (list of np.array or Tensor): list of images (or patches), shape (H,W,C).
        save_path (str): Path to save overlay figure. If None, just display.
        colormap (str): Colormap for heatmap.
    """
    for i, (attn, img) in enumerate(zip(attention_weights, images)):
        # Convert tensor to np.array if needed
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            if img.shape[0] == 3 or img.shape[0] == 1:
                img = np.transpose(img, (1,2,0))
        # Resize attn to match image size if needed
        attn_img = plt.cm.get_cmap(colormap)(attn)[:,:,:3]
        # Overlay: weighted sum
        overlay = (0.5 * img/np.max(img) + 0.5 * attn_img)
        overlay = np.clip(overlay, 0, 1)
        plt.figure(figsize=(4,4))
        plt.imshow(overlay)
        plt.axis('off')
        if save_path:
            filename = os.path.join(save_path, f'attention_map_{i}.png')
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

def visualize_attention_samples(attention_maps, images, save_dir, top_k=5):
    """
    Generate overlays for top-k positive and negative samples based on attention scores.
    Args:
        attention_maps (list): list of tuples (attention_scores, images_tensor)
        images (list): list of raw images corresponding to attention_maps
        save_dir (str): Directory to save visualizations
        top_k (int): Number of top positive/negative samples to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    # Flatten list if nested
    all_scores = []
    for attn_scores, imgs in attention_maps:
        all_scores.extend(attn_scores.detach().cpu().numpy())
    if not all_scores:
        return
    threshold_pos = np.percentile(all_scores, 95)
    threshold_neg = np.percentile(all_scores, 5)

    # Visualize top positive
    for attn_scores, imgs in attention_maps:
        scores_np = attn_scores.detach().cpu().numpy()
        # Select samples above 95 percentile
        top_pos_idxs = np.where(scores_np >= threshold_pos)[0]
        for i in top_pos_idxs[:top_k]:
            attn = scores_np[i]
            img = imgs[i]
            normalized_score = normalize_attention_scores(torch.tensor(attn))
            plot_attention_map([normalized_score], [img], save_path=os.path.join(save_dir, f'pos_{i}.png'))
        # Similarly for negative
        top_neg_idxs = np.where(scores_np <= threshold_neg)[0]
        for i in top_neg_idxs[:top_k]:
            attn = scores_np[i]
            img = imgs[i]
            normalized_score = normalize_attention_scores(torch.tensor(attn))
            plot_attention_map([normalized_score], [img], save_path=os.path.join(save_dir, f'neg_{i}.png'))

def evaluate(model, dataloader, device, visualize=False, save_dir='attention_maps', colormap='jet'):
    """
    Runs inference on dataloader, computes metrics, and optionally visualizes attention maps.
    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for evaluation.
        device (str): 'cpu' or 'cuda'.
        visualize (bool): Whether to produce attention maps visualizations.
        save_dir (str): Directory to save attention maps if visualization is True.
        colormap (str): Colormap name for attention heatmaps.
    Returns:
        dict: Aggregated metrics including AUROC, F1, with optional sample attention overlays.
    """
    model.eval()
    all_bag_preds = []
    all_bag_labels = []
    all_instance_scores = []
    all_attention_maps = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)  # shape (B, N, D)
            adjacency = batch['adjacency'].to(device) if batch['adjacency'] is not None else None
            labels = batch['labels'].cpu().numpy()

            outputs = model.forward(pre_extracted_features=features, adjacency=adjacency)
            # outputs keys: 'bag_pred', 'instance_scores', 'attention_weights'
            bag_preds = outputs['bag_pred'].cpu().numpy()
            inst_scores = outputs['instance_scores'].cpu().numpy()
            attn_weights = outputs['attention_weights'].cpu().numpy()

            all_bag_preds.extend(bag_preds)
            all_bag_labels.extend(labels)
            all_instance_scores.extend(inst_scores)

            # Store attention maps and corresponding images for visualization if needed
            if visualize:
                imgs_list = batch.get('raw_images', None)
                if imgs_list is not None:
                    # If raw images not provided, skip visualization
                    if isinstance(imgs_list, list):
                        imgs = imgs_list
                    else:
                        imgs = [img for img in imgs_list]
                else:
                    imgs = [np.zeros((64,64,3)) for _ in range(len(attn_weights))]  # dummy placeholder
                all_attention_maps.append((attn_weights, imgs))

    # Compute AUROC and F1
    auroc_bag = calculate_auc_score(all_bag_preds, all_bag_labels)
    f1_bag, best_thresh = calculate_f1_score(all_bag_preds, all_bag_labels)

    metrics = {
        'AUROC_bag': auroc_bag,
        'F1_bag': f1_bag,
        'best_thresh': best_thresh,
        'AUROC_inst': calculate_auc_score(all_instance_scores, all_bag_labels),  # approximate
    }

    # Visualization
    if visualize:
        save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        # For simplicity, visualize a sample of the dataset
        sample_attn_maps = all_attention_maps[:min(20, len(all_attention_maps))]
        visualize_attention_samples(sample_attn_maps, imgs, save_dir, top_k=3)

    return metrics
```

## main.py

```python
## main.py
import os
import sys
import random
import yaml
import torch
import numpy as np
from tqdm import tqdm

# Import project modules
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import evaluate
from utils import set_seed, print_separator

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    # Set seeds for reproducibility
    seed = config['training'].get('seed', 42)
    set_seed(seed)

    # Prepare dataset paths and parameters
    dataset_name = config['dataset'].get('name', 'RSNA')
    data_paths = {
        'features': config['dataset']['data_paths'].get('features', ''),
        'labels': config['dataset']['data_paths'].get('labels', '')
    }
    batch_size = config['training'].get('batch_size', 8)
    split_seed = config['dataset'].get('split_seed', 42)
    n_folds = config['misc'].get('cross_validation_folds', 5)
    n_runs = config['misc'].get('num_runs', 1)  # For averaging if multiple runs

    # Load dataset with cross-validation splits
    print("Loading dataset and constructing adjacency matrices...")
    dataset_loader = DatasetLoader(
        data_paths=data_paths,
        split_seed=split_seed,
        dataset_name=dataset_name,
        batch_size=batch_size,
        device=str(device)
    )

    # Generate cross-validation splits
    # dataset_loader.Splits already created inside
    splits = dataset_loader.splits

    # Prepare output directory
    output_dir = config['misc'].get('output_dir', 'outputs/')
    os.makedirs(output_dir, exist_ok=True)

    # Collect metrics over folds
    fold_metrics = []

    for fold_idx in range(1, n_folds+1):
        print_separator(f"Starting Fold {fold_idx}/{n_folds}")
        # Define train and val indices
        train_idx = splits['train']
        val_idx = splits['val']
        test_idx = splits['test']

        # Create datasets for this fold
        train_subset = torch.utils.data.Subset(dataset_loader.dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset_loader.dataset, val_idx)
        test_subset = torch.utils.data.Subset(dataset_loader.dataset, test_idx)

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=dataset_loader.collate_fn, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=dataset_loader.collate_fn, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=dataset_loader.collate_fn, pin_memory=True)

        # Initialize Model
        print("Initializing model...")

        # Map config parameters
        feat_type = config['model'].get('feature_extractor', 'resnet18')
        use_transformer = config['model'].get('use_transformer', False)
        sm_points = config['model'].get('sm_points', 'early')
        sm_num_steps = config['model'].get('sm_num_steps', 10)
        sm_alpha_init = config['model'].get('sm_alpha_init', 0.5)
        sm_trainable_alpha = config['model'].get('sm_trainable_alpha', True)
        use_spectral_norm = config['model'].get('use_spectral_norm', True)

        # Instantiate model
        model = Model({
            'feature_extractor': feat_type,
            'freeze_feature_extractor': True,
            'use_transformer': use_transformer,
            'transformer_layers': config['model'].get('transformer_layers', 2),
            'transformer_heads': config['model'].get('transformer_heads', 4),
            'attention_points': sm_points,
            'sm_enabled': True,
            'sm_points': sm_points,
            'sm_num_steps': sm_num_steps,
            'sm_alpha_init': sm_alpha_init,
            'sm_trainable_alpha': sm_trainable_alpha,
            'use_spectral_norm': use_spectral_norm
        })
        model.to(device)

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training'].get('learning_rate', 1e-4), weight_decay=config['training'].get('weight_decay', 1e-4))
        # LR scheduler (optional)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

        # Loss criterion
        criterion = nn.BCELoss()

        # Instantiate trainer
        trainer = Trainer(model, {'train': train_loader, 'val': val_loader, 'test': test_loader},
                          optimizer, criterion, {'training': config['training'], 'model': config['model'], 'misc': config.get('misc', {})})

        # Train the model with early stopping
        print("Starting training...")
        best_val_score = -np.inf
        patience_counter = 0
        max_epochs = config['training'].get('epochs', 50)

        for epoch in range(1, max_epochs + 1):
            print_separator(f"Epoch {epoch}/{max_epochs}")
            train_loss = trainer.train_epoch()
            val_metrics = trainer.validate()

            # Step scheduler on validation AUROC
            scheduler.step(val_metrics['auroc'])

            # Save best model if improved
            if val_metrics['auroc'] > best_val_score:
                best_val_score = val_metrics['auroc']
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"New best model at epoch {epoch} with AUROC: {best_val_score:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config['training'].get('early_stopping_patience', 10):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        # Load best model weights
        model.load_state_dict(best_model_weights)
        print("Evaluating on test set with best model...")
        test_metrics = evaluate(model, test_loader, device=device, visualize=True, save_dir=os.path.join(output_dir, f'fold_{fold_idx}_attention'))
        print(f"Fold {fold_idx} Test AUROC: {test_metrics['AUROC_bag']:.4f} | F1: {test_metrics['F1_bag']:.4f}")
        fold_metrics.append({
            'fold': fold_idx,
            'AUROC_bag': test_metrics['AUROC_bag'],
            'F1_bag': test_metrics['F1_bag']
        })

    # After all folds, compute mean and std
    print_separator("Summary over cross-validation folds")
    auroc_vals = [m['AUROC_bag'] for m in fold_metrics]
    f1_vals = [m['F1_bag'] for m in fold_metrics]
    print(f"Average AUROC: {np.mean(auroc_vals):.4f} ± {np.std(auroc_vals):.4f}")
    print(f"Average F1: {np.mean(f1_vals):.4f} ± {np.std(f1_vals):.4f}")

if __name__ == '__main__':
    main()


# Additional helpers not shown here:
# - import copy
# - print_separator() from utils
# - set_seed() from utils
# These are assumed to be implemented in utils.py as per the modular design.
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import resnet18, resnet50
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sm_operator import SmOperator
from math import sqrt

class FeatureExtractor(nn.Module):
    """
    Pre-trained feature extractor, configurable to be frozen or trainable.
    Supports ResNet18, ResNet50, and ViT-B-32.
    Assumes input images of size 512x512.
    """
    def __init__(self, name='resnet18', freeze=True):
        super().__init__()
        if name == 'resnet18':
            self.model = resnet18(pretrained=True)
            feature_dim = 512
        elif name == 'resnet50':
            self.model = resnet50(pretrained=True)
            feature_dim = 2048
        elif name == 'vit-b-32':
            # Use torchvision's ViT or a placeholder; here, assume we have a pre-trained ViT
            # For simplicity, use a dummy linear layer as placeholder
            # Replace with proper ViT model if available
            self.model = nn.Identity()
            feature_dim = 768
        else:
            raise ValueError(f"Unknown feature extractor {name}")
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.feature_dim = feature_dim

    def forward(self, x):
        # Input x: batch x C x H x W
        if hasattr(self.model, 'children'):
            # For ResNet
            features = self.model.conv1(x)
            features = self.model.bn1(features)
            features = self.model.relu(features)
            features = self.model.maxpool(features)
            features = self.model.layer1(features)
            features = self.model.layer2(features)
            features = self.model.layer3(features)
            features = self.model.layer4(features)
            features = self.model.avgpool(features)
            features = torch.flatten(features, 1)
        else:
            # Placeholder for ViT: assume features are provided
            # Should be replaced with actual ViT forward
            features = torch.flatten(x, 1)  # Dummy
        return features  # shape: batch x feature_dim

class PositionalEncoding(nn.Module):
    """
    Optional positional encoding if needed. Not required for current design.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: batch x seq_len x d_model
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class Model(nn.Module):
    def __init__(self, config: dict):
        """
        Constructs the MIL model with optional transformer and Sm integration.
        Args:
            config (dict): from config.yaml
        """
        super().__init__()
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            name=config['model'].get('feature_extractor', 'resnet18'),
            freeze=config['model'].get('freeze_feature_extractor', True)
        )
        self.feature_dim = self.feature_extractor.feature_dim

        # Optional transformer encoder
        self.use_transformer = config['model'].get('use_transformer', False)
        if self.use_transformer:
            encoder_layers = TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=config['model'].get('transformer_heads', 4),
                dim_feedforward=2048,
                dropout=0.1,
                activation='relu'
            )
            self.transformer_encoder = TransformerEncoder(
                encoder_layer=encoder_layers,
                num_layers=config['model'].get('transformer_layers', 2)
            )
        else:
            self.transformer_encoder = None

        # Attention pooling weights
        # As in ABMIL: learnable attention parameters
        self.attn_W = nn.Linear(self.feature_dim, 1, bias=False)  # for attention score
        # Final classifier
        self.classifier = nn.Linear(self.feature_dim, 1)

        # Configuration for Sm
        self.sm_enabled = config['model'].get('sm_enabled', False)
        self.sm_points = config['model'].get('sm_points', 'early')  # 'early', 'mid', 'late', 'both'
        self.sm_num_steps = config['model'].get('sm_num_steps', 10)
        self.trainable_alpha = config['model'].get('sm_trainable_alpha', True)
        self.use_spectral_norm = config['model'].get('use_spectral_norm', True)

        if self.sm_enabled:
            init_alpha = config['model'].get('sm_alpha_init', 0.5)
            self.sm_operator = SmOperator(
                num_steps=self.sm_num_steps,
                alpha_init=init_alpha,
                use_spectral_norm=self.use_spectral_norm
            )
            if self.trainable_alpha:
                # Alpha as a parameter
                self.alpha_param = nn.Parameter(torch.tensor(init_alpha))
            else:
                self.alpha_param = torch.tensor(init_alpha)

    def get_alpha(self):
        if self.trainable_alpha:
            return torch.sigmoid(self.alpha_param).item()
        else:
            return self.alpha_param

    def apply_sm(self, embeddings, adjacency, point=''):
        """
        Applies 'Sm' operator conditioned on 'point'.
        """
        alpha = torch.sigmoid(self.alpha_param) if self.trainable_alpha else self.alpha_param
        if self.sm_enabled:
            if point == 'early':
                return self.sm_operator(embeddings, adjacency, alpha=alpha)
            elif point == 'mid':
                return self.sm_operator(embeddings, adjacency, alpha=alpha)
            elif point == 'late':
                return self.sm_operator(embeddings, adjacency, alpha=alpha)
            elif point == 'both':
                # Apply twice or sequentially at both points, in calling order
                return self.sm_operator(embeddings, adjacency, alpha=alpha)
            else:
                return embeddings
        else:
            return embeddings

    def forward(self, input_images=None, pre_extracted_features=None, adjacency=None):
        """
        Args:
            input_images: tensor batch of images if feature extractor is used raw.
            pre_extracted_features: tensor batch of features (from dataset loader) shape (N, D)
            adjacency: Tensor (N, N), adjacency matrix for the instances of the batch.
        Returns:
            dict with 'bag_pred', 'instance_scores', 'attention_weights'
        """
        # Get instance features
        if pre_extracted_features is not None:
            H = pre_extracted_features  # shape (N, D)
        else:
            # Assume input_images of shape (batch, C, H, W), batch size = number of bags
            # For multiple bags, the dataset collate will give batch of bags; but as per code, process per bag
            # Here, implement for batch of size 1 or extend as needed
            H = self.feature_extractor(input_images)  # shape (N, D)

        # Apply 'early' Sm if configured
        if self.sm_enabled and self.sm_points in ['early', 'both']:
            if adjacency is not None:
                H = self.apply_sm(H, adjacency, point='early')
            else:
                # fallback if adjacency not provided, do nothing or compute adjacency
                pass

        # Pass through transformer if used
        if self.use_transformer:
            H_input = H.unsqueeze(0)  # batch dimension, shape: 1 x N x D
            if self.sm_enabled and self.sm_points in ['mid', 'both']:
                H_input = self.apply_sm(H_input.squeeze(0), adjacency, point='mid')
                H_input = H_input.unsqueeze(0)
            # Transformer encoder
            H_transform = self.transformer_encoder(H_input)  # shape: 1 x N x D
            H = H_transform.squeeze(0)  # shape: N x D
            # Apply 'late' Sm if configured
            if self.sm_enabled and self.sm_points == 'late':
                if adjacency is not None:
                    H = self.apply_sm(H, adjacency, point='late')
        else:
            # No transformer
            pass

        # Compute attention scores
        f = self.attn_W(H)  # shape: N x 1
        attention_scores = f.squeeze(-1)  # shape: N

        # Attention weights
        attn_weights = F.softmax(attention_scores, dim=0)  # normalize over instances

        # Compute bag feature as weighted sum
        bag_feature = torch.sum(attn_weights.unsqueeze(-1) * H, dim=0)  # shape: D

        # Bag-level prediction
        bag_logit = self.classifier(bag_feature)  # scalar
        bag_pred = torch.sigmoid(bag_logit)

        # Instance scores are attention scores (or can be class scores if needed)
        instance_scores = attention_scores  # for localization

        # For visualization or localization, attention weights are relevant
        return {
            'bag_pred': bag_pred,
            'instance_scores': instance_scores,
            'attention_weights': attn_weights
        }
```

## requirements.txt

### requirements.txt

# Deep learning framework
torch==1.11.0
torchvision==0.12.0

# Numerical computations and scientific tools
numpy==1.21.0
scipy==1.7.3

# Evaluation metrics and data handling
scikit-learn==0.24.2
pandas==1.3.5

# Progress visualization
tqdm==4.62.3

# Optional: visualization for attention maps and histograms
matplotlib==3.4.3

## sm_operator.py

```python
## sm_operator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmOperator(nn.Module):
    """
    Implements the smooth operator as an iterative graph Laplacian smoothing based on the paper.
    It approximates the inverse (I + gamma * L)^(-1) via T iteration steps, promoting
    local consistency of instance embeddings.
    """
    def __init__(self, num_steps: int = 10, alpha_init: float = 0.5, use_spectral_norm: bool = True):
        """
        Args:
            num_steps (int): Number of T steps in the iterative approximation.
            alpha_init (float): Initial value for the alpha parameter, in (0,1). It will be learned.
            use_spectral_norm (bool): Whether to apply spectral normalization to alpha (if learnable).
        """
        super().__init__()
        self.num_steps = num_steps
        # Initialize alpha as a learnable parameter, constrained between 0 and 1
        # Use sigmoid parametrization for stability
        self._logit_alpha = nn.Parameter(torch.logit(torch.tensor(alpha_init)))
        self.use_spectral_norm = use_spectral_norm

        if self.use_spectral_norm:
            # Optional: apply spectral normalization to the alpha parameter
            self.alpha = nn.utils.spectral_norm(self._logit_alpha.unsqueeze(0)).squeeze()
        else:
            self.alpha = self._logit_alpha

    def forward(self, embeddings: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings (Tensor): shape (N, D), instance embeddings before smoothing.
            adjacency (Tensor): shape (N, N), adjacency matrix (symmetric, non-negative).
        Returns:
            smoothed_embeddings (Tensor): shape (N, D), smoothed instance embeddings.
        """
        # Validate inputs
        assert embeddings.dim() == 2, "embeddings should be (N, D)"
        assert adjacency.dim() == 2 and adjacency.size(0) == adjacency.size(1) == embeddings.size(0), \
            "adjacency should be (N, N)"

        N = embeddings.shape[0]
        device = embeddings.device

        # Step 1: Compute degree matrix D
        degrees = torch.clamp(adjacency.sum(dim=1), min=1e-12)  # avoid division by zero
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees))
        # Step 2: Compute normalized adjacency: A_norm = D^{-1/2} A D^{-1/2}
        A_norm = D_inv_sqrt @ adjacency @ D_inv_sqrt  # shape (N, N)

        # Step 3: Compute the normalized Laplacian: L = I - A_norm
        L = torch.eye(N, device=device) - A_norm  # shape (N, N)
        I = torch.eye(N, device=device)

        # Convert alpha to a sigmoid for stability, ensure it's in [0,1]
        alpha = torch.sigmoid(self._logit_alpha)

        # Initialize G(0) = embeddings
        G = embeddings

        # Precompute (I - L) matrix
        # Note: (I - L) = A_norm
        A_tilde = A_norm  # (N, N)

        # Iterative process
        for _ in range(self.num_steps):
            # Update based on eq: G(t) = alpha * (I - L) * G(t-1) + (1 - alpha) * embeddings
            G = alpha * (A_tilde @ G) + (1 - alpha) * embeddings

        return G

    def get_alpha(self) -> float:
        """
        Returns current alpha value (in [0,1])
        """
        return torch.sigmoid(self._logit_alpha).item()

    def set_alpha(self, new_alpha: float):
        """
        Set alpha as a float in (0, 1), updates the parameter accordingly.
        """
        # Convert to logit
        new_logit = torch.logit(torch.tensor(new_alpha))
        self._logit_alpha.data = new_logit
```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
from tqdm import tqdm
import os
import copy

class Trainer:
    def __init__(self, model, dataset, optimizer, criterion, config):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The MIL model with optional Sm integration.
            dataset (dict): Contains 'train', 'val', 'test' DataLoaders.
            optimizer (torch.optim.Optimizer): Optimizer, e.g., Adam.
            criterion (callable): Loss function, e.g., BCELoss.
            config (dict): Configuration parameters from YAML.
        """
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        
        self.device = torch.device(config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        # Learning rate scheduler (optional), here using ReduceLROnPlateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # maximize validation metric (AUROC)
            factor=0.1,
            patience=3,
            verbose=True,
            min_lr=1e-6
        )
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_score = -np.inf
        self.early_stop_patience = self.config['training'].get('early_stopping_patience', 10)
        self.patience_counter = 0

        # For reproducibility
        seed = self.config['training'].get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Log storage
        self.train_logs = []
        self.val_logs = []

        # Other parameters
        self.epochs = self.config['training'].get('epochs', 50)
        self.use_spectral_norm = self.config['model'].get('use_spectral_norm', True)

    def train(self):
        """
        Main training loop with validation and early stopping.
        """
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")
            train_loss = self.train_one_epoch()
            val_metrics = self.validate()

            # Log training
            self.train_logs.append({'epoch': epoch, 'loss': train_loss})
            # Log validation
            val_score = val_metrics['auroc']
            self.val_logs.append({'epoch': epoch, **val_metrics})

            # Step scheduler
            self.scheduler.step(val_score)
            print(f"Validation AUROC: {val_metrics['auroc']:.4f} - Validation F1: {val_metrics['f1']:.4f}")

            # Check if it's the best model
            if val_metrics['auroc'] > self.best_score:
                self.best_score = val_metrics['auroc']
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
                print("New best model saved.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"Early stopping: no improvement for {self.early_stop_patience} epochs.")
                    break

        # Load best model Weights
        self.model.load_state_dict(self.best_model_wts)
        print("Training complete. Best validation AUROC: {:.4f}".format(self.best_score))
        # Final evaluation on test set
        test_metrics = self.validate(split='test', save_attention=True)
        return test_metrics

    def train_one_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.dataset['train'], desc='Training', leave=False)
        
        for batch in progress_bar:
            # Move data to device
            features = batch['features'].to(self.device)  # shape: (B, N, D)
            adjacency = batch['adjacency'].to(self.device) if batch['adjacency'] is not None else None
            labels = batch['labels'].to(self.device)      # shape: (B,)
            instance_labels = batch.get('instance_labels', None)  # optional

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model.forward(pre_extracted_features=features, adjacency=adjacency)
            # outputs: dict with keys 'bag_pred', 'instance_scores', 'attention_weights'
            bag_pred = outputs['bag_pred'].squeeze()
            # Compute loss
            loss = self.criterion(bag_pred, labels)
            # Backpropagate
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.dataset['train'])
        return avg_loss

    def validate(self, split='val', save_attention=False):
        """
        Evaluates model on validation/test set.
        Args:
            split (str): 'val' or 'test'
            save_attention (bool): if True, save attention maps for visualization
        Returns:
            dict: metrics (AUROC, F1)
        """
        self.model.eval()
        all_labels = []
        all_preds = []
        all_instance_scores = []
        all_attention_maps = []

        dataloader = self.dataset[split]
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'Validation ({split})', leave=False):
                features = batch['features'].to(self.device)
                adjacency = batch['adjacency'].to(self.device) if batch['adjacency'] is not None else None
                labels = batch['labels'].cpu().numpy()

                outputs = self.model.forward(pre_extracted_features=features, adjacency=adjacency)
                bag_pred = outputs['bag_pred'].cpu().numpy()
                instance_scores = outputs['instance_scores'].cpu().numpy()
                attention_weights = outputs['attention_weights'].cpu().numpy()

                all_labels.extend(labels)
                all_preds.extend(bag_pred)
                all_instance_scores.extend(instance_scores)
                if save_attention:
                    # Store attention maps, optionally for visualization
                    all_attention_maps.append((attention_weights, features.cpu()))
        
        # Compute metrics
        auroc = roc_auc_score(all_labels, all_preds)
        # Optimize threshold for F1 on validation set
        thresholds = np.linspace(0, 1, 101)
        f1_max = -1
        f1_best_thresh = 0.5
        for thresh in thresholds:
            preds_binary = np.array(all_preds) >= thresh
            f1 = f1_score(all_labels, preds_binary)
            if f1 > f1_max:
                f1_max = f1
                f1_best_thresh = thresh
        print(f"Split: {split} - AUROC: {auroc:.4f} - F1 (thresh={f1_best_thresh:.2f}): {f1_max:.4f}")

        metrics = {
            'auroc': auroc,
            'f1': f1_max,
            'best_thresh': f1_best_thresh
        }

        # Save attention maps / visualization if needed
        if save_attention:
            self.save_attention_maps(all_attention_maps, split)

        return metrics

    def save_attention_maps(self, attention_data, split):
        """
        Saves attention maps overlayed on input images for visualization.
        """
        # Placeholder implementation: actual overlay depends on data
        # Here, save attention weights and features for later visualization
        save_dir = self.config.get('misc', {}).get('log_dir', 'outputs')
        os.makedirs(save_dir, exist_ok=True)
        for i, (attn_weights, feats) in enumerate(attention_data):
            # Save attention weights
            filename = os.path.join(save_dir, f"{split}_attention_{i}.npy")
            np.save(filename, attn_weights)
            # Visualization could be added here as needed
        print(f"Saved {len(attention_data)} attention maps for {split} split.")
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\SmMIL\SmMIL_repo`
