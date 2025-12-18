# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import numpy as np
import scipy.sparse as sp
import torch

class DatasetLoader:
    """
    DatasetLoader loads graph datasets (heterophilic and homophilic),
    processes adjacency matrices, features, labels, and masks.
    """

    def __init__(self, config: dict):
        """
        Initialize the dataset loader with the provided configuration.

        Args:
            config (dict): Dictionary containing dataset parameters.
                Should include keys:
                    - 'dataset_name' (str)
                    - 'seed' (int)
                    - 'train_ratio' (float)
                    - 'val_ratio' (float)
                    - 'test_ratio' (float)
                    - 'splits' (int)
                    - 'data_dir' (str): path to data directory or base URL
        """
        self.dataset_name: str = config.get('dataset_name', '')
        self.seed: int = config.get('seed', 42)
        self.train_ratio: float = config.get('train_ratio', 0.6)
        self.val_ratio: float = config.get('val_ratio', 0.2)
        self.test_ratio: float = config.get('test_ratio', 0.2)
        self.splits: int = config.get('splits', 10)
        self.data_dir: str = config.get('data_dir', './datasets')

        # Dataset-specific info: file names or URLs, dataset formats
        self.dataset_info = self._get_dataset_info()

    def _get_dataset_info(self):
        """
        Map dataset names to file paths or URLs, and to dataset specific processing info.
        """
        info = {}
        # Placeholder: provide actual file paths or URLs as needed.
        # For the purpose of this implementation, assume local files are prepared.
        # Extend this dict with actual dataset files or URLs.
        if self.dataset_name.lower().startswith('minesweeper'):
            info = {
                'features_file': os.path.join(self.data_dir, 'mines_features.npy'),
                'edges_file': os.path.join(self.data_dir, 'mines_edges.npy'),
                'labels_file': os.path.join(self.data_dir, 'mines_labels.npy'),
                'node_ids_file': os.path.join(self.data_dir, 'mines_node_ids.npy')
            }
        elif self.dataset_name.lower().startswith('roman-empire'):
            info = {
                'edges_file': os.path.join(self.data_dir, 'roman_edges.npy'),
                'features_file': os.path.join(self.data_dir, 'roman_features.npy'),
                'labels_file': os.path.join(self.data_dir, 'roman_labels.npy')
            }
        elif self.dataset_name.lower().startswith('questions'):
            info = {
                'edges_file': os.path.join(self.data_dir, 'questions_edges.npy'),
                'features_file': os.path.join(self.data_dir, 'questions_features.npy'),
                'labels_file': os.path.join(self.data_dir, 'questions_labels.npy')
            }
        elif self.dataset_name.lower().startswith('squirrel'):
            info = {
                'edges_file': os.path.join(self.data_dir, 'squirrel_edges.npy'),
                'features_file': os.path.join(self.data_dir, 'squirrel_features.npy'),
                'labels_file': os.path.join(self.data_dir, 'squirrel_labels.npy')
            }
        elif self.dataset_name.lower().startswith('cora'):
            info = {
                'inductive': False,
                'edges_file': os.path.join(self.data_dir, 'cora_adj.npz'),
                'features_file': os.path.join(self.data_dir, 'cora_features.npy'),
                'labels_file': os.path.join(self.data_dir, 'cora_labels.npy'),
                'split_indices': os.path.join(self.data_dir, 'cora_splits.npy')  # if available
            }
        elif self.dataset_name.lower().startswith('citeseer'):
            info = {
                'inductive': False,
                'edges_file': os.path.join(self.data_dir, 'citeseer_adj.npz'),
                'features_file': os.path.join(self.data_dir, 'citeseer_features.npy'),
                'labels_file': os.path.join(self.data_dir, 'citeseer_labels.npy')
            }
        elif self.dataset_name.lower().startswith('pubmed'):
            info = {
                'inductive': False,
                'edges_file': os.path.join(self.data_dir, 'pubmed_adj.npz'),
                'features_file': os.path.join(self.data_dir, 'pubmed_features.npy'),
                'labels_file': os.path.join(self.data_dir, 'pubmed_labels.npy')
            }
        # Add other datasets similarly...
        # For actual implementation, fill in with real data locations.
        return info

    def load_data(self):
        """
        Loads dataset according to dataset name and processes into tensors.

        Returns:
            adj (torch.sparse.FloatTensor): Sparse adjacency matrix with self-loops.
            features (torch.FloatTensor): Node feature matrix.
            labels (torch.LongTensor): Node labels.
            train_mask (torch.BoolTensor): Mask for training nodes.
            val_mask (torch.BoolTensor): Mask for validation nodes.
            test_mask (torch.BoolTensor): Mask for test nodes.
        """
        # Based on dataset name, call the specific load function
        ds_name = self.dataset_name.lower()
        if ds_name.startswith('minesweeper'):
            return self._load_minesweeper()
        elif ds_name.startswith('roman-empire'):
            return self._load_roman_empire()
        elif ds_name.startswith('questions'):
            return self._load_questions()
        elif ds_name.startswith('squirrel'):
            return self._load_squirrel()
        elif ds_name.startswith('cora'):
            return self._load_cora()
        elif ds_name.startswith('citeseer'):
            return self._load_citeseer()
        elif ds_name.startswith('pubmed'):
            return self._load_pubmed()
        # Add other dataset load calls as needed
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported or not implemented.")

    def _load_minesweeper(self):
        """
        Load Minesweeper dataset: features, adjacency, labels.
        Assume files exists in npy format: features, edges, labels.
        """
        features_path = self.dataset_info['features_file']
        edges_path = self.dataset_info['edges_file']
        labels_path = self.dataset_info['labels_file']
        node_ids_path = self.dataset_info.get('node_ids_file', None)

        # Load node features
        features_np = np.load(features_path)  # shape: [num_nodes, feature_dim]
        features = torch.FloatTensor(features_np)

        # Load labels
        labels_np = np.load(labels_path)  # shape: [num_nodes]
        labels = torch.LongTensor(labels_np)

        # Load edges
        edges_np = np.load(edges_path)  # shape: [2, num_edges]
        edge_index = torch.LongTensor(edges_np)
        # Convert to sparse adjacency
        num_nodes = features.shape[0]
        adj = sp.coo_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes)
        )
        # Make undirected
        adj = adj + adj.T.multiply(adj.T > adj)
        adj = adj.tocoo()

        # Add self-loops
        adj = adj + sp.eye(adj.shape[0])
        # Normalize adjacency for GCN
        degs = np.array(adj.sum(1)).flatten()
        degs_inv_sqrt = np.power(degs, -0.5)
        degs_inv_sqrt[np.isinf(degs_inv_sqrt)] = 0.
        D_inv_sqrt = sp.diags(degs_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        # Convert to torch sparse tensor
        indices = torch.LongTensor([adj_norm.row, adj_norm.col])
        values = torch.FloatTensor(adj_norm.data)
        shape = adj_norm.shape
        adj_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Generate masks for splits
        train_mask, val_mask, test_mask = self._generate_splits(features.shape[0])

        return adj_tensor, features, labels, train_mask, val_mask, test_mask

    def _load_roman_empire(self):
        """
        Similar to above, load Roman-empire dataset from files.
        """
        edges_path = self.dataset_info['edges_file']
        features_path = self.dataset_info['features_file']
        labels_path = self.dataset_info['labels_file']

        # Load features
        features_np = np.load(features_path)
        features = torch.FloatTensor(features_np)
        # Load labels
        labels_np = np.load(labels_path)
        labels = torch.LongTensor(labels_np)
        # Load edges
        edge_idx = np.load(edges_path)  # shape: [2, num_edges]
        edge_index = torch.LongTensor(edge_idx)

        # Build adjacency
        num_nodes = features.shape[0]
        adj = sp.coo_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes)
        )
        # Make symmetric
        adj = adj + adj.T.multiply(adj.T > adj)
        adj = adj.tocoo()

        # Add self-loops
        adj = adj + sp.eye(adj.shape[0])
        # Normalize adjacency
        degs = np.array(adj.sum(1)).flatten()
        degs_inv_sqrt = np.power(degs, -0.5)
        degs_inv_sqrt[np.isinf(degs_inv_sqrt)] = 0.
        D_inv_sqrt = sp.diags(degs_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        indices = torch.LongTensor([adj_norm.row, adj_norm.col])
        values = torch.FloatTensor(adj_norm.data)
        shape = adj_norm.shape
        adj_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        # Generate splits
        train_mask, val_mask, test_mask = self._generate_splits(features.shape[0])
        return adj_tensor, features, labels, train_mask, val_mask, test_mask

    def _load_questions(self):
        """
        Load Questions dataset similarly.
        """
        edges_path = self.dataset_info['edges_file']
        features_path = self.dataset_info['features_file']
        labels_path = self.dataset_info['labels_file']

        features_np = np.load(features_path)
        features = torch.FloatTensor(features_np)
        labels_np = np.load(labels_path)
        labels = torch.LongTensor(labels_np)
        edge_idx = np.load(edges_path)
        edge_index = torch.LongTensor(edge_idx)

        num_nodes = features.shape[0]
        adj = sp.coo_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes)
        )
        # Symmetrize
        adj = adj + adj.T.multiply(adj.T > adj)
        adj = adj.tocoo()

        adj = adj + sp.eye(adj.shape[0])
        degs = np.array(adj.sum(1)).flatten()
        degs_inv_sqrt = np.power(degs, -0.5)
        degs_inv_sqrt[np.isinf(degs_inv_sqrt)] = 0
        D_inv_sqrt = sp.diags(degs_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        indices = torch.LongTensor([adj_norm.row, adj_norm.col])
        values = torch.FloatTensor(adj_norm.data)
        shape = adj_norm.shape
        adj_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        train_mask, val_mask, test_mask = self._generate_splits(features.shape[0])
        return adj_tensor, features, labels, train_mask, val_mask, test_mask

    def _load_squirrel(self):
        """
        Load Squirrel dataset with potential filtering for heterophily.
        Similar steps; additional filtering can be implemented here.
        """
        edges_path = self.dataset_info['edges_file']
        features_path = self.dataset_info['features_file']
        labels_path = self.dataset_info['labels_file']

        features_np = np.load(features_path)
        features = torch.FloatTensor(features_np)

        labels_np = np.load(labels_path)
        labels = torch.LongTensor(labels_np)

        edge_idx = np.load(edges_path)
        edge_index = torch.LongTensor(edge_idx)

        num_nodes = features.shape[0]
        adj = sp.coo_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes)
        )
        # Make symmetric
        adj = adj + adj.T.multiply(adj.T > adj)
        adj = adj.tocoo()

        # Optionally filter edges here if needed (e.g., remove edges connecting train/test nodes)
        # For simplicity, assume no filtering unless specified.
        adj = adj + sp.eye(adj.shape[0])  # self-loops
        degs = np.array(adj.sum(1)).flatten()
        degs_inv_sqrt = np.power(degs, -0.5)
        degs_inv_sqrt[np.isinf(degs_inv_sqrt)] = 0
        D_inv_sqrt = sp.diags(degs_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        indices = torch.LongTensor([adj_norm.row, adj_norm.col])
        values = torch.FloatTensor(adj_norm.data)
        shape = adj_norm.shape
        adj_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        train_mask, val_mask, test_mask = self._generate_splits(features.shape[0])
        return adj_tensor, features, labels, train_mask, val_mask, test_mask

    def _load_cora(self):
        """
        Load Cora dataset from .npz sparse adjacency, features, labels, optionally splits.
        """
        import scipy.sparse.linalg as la
        adj_path = self.dataset_info['edges_file']
        feat_path = self.dataset_info['features_file']
        label_path = self.dataset_info['labels_file']
        split_path = self.dataset_info.get('split_indices', None)

        # Load adjacency
        loader = np.load(adj_path)
        adj = sp.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])
        adj = adj + adj.T.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])  # add self-loops

        # Normalize adjacency
        degs = np.array(adj.sum(1)).flatten()
        degs_inv_sqrt = np.power(degs, -0.5)
        degs_inv_sqrt[np.isinf(degs_inv_sqrt)] = 0
        D_inv_sqrt = sp.diags(degs_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        indices = torch.LongTensor([adj_norm.row, adj_norm.col])
        values = torch.FloatTensor(adj_norm.data)
        shape = adj_norm.shape
        adj_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        features_np = np.load(feat_path)
        features = torch.FloatTensor(features_np)
        labels_np = np.load(label_path)
        labels = torch.LongTensor(labels_np)

        if split_path and os.path.exists(split_path):
            splits = np.load(split_path)
            train_mask, val_mask, test_mask = self._create_masks_from_splits(splits, labels.shape[0])
        else:
            train_mask, val_mask, test_mask = self._generate_splits(labels.shape[0])

        return adj_tensor, features, labels, train_mask, val_mask, test_mask

    def _load_citeseer(self):
        """
        Load Citeseer dataset similarly.
        """
        # Implement similarly as Cora
        pass

    def _load_pubmed(self):
        """
        Load PubMed dataset similarly.
        """
        # Implement similarly as Cora
        pass

    def _generate_splits(self, num_nodes):
        """
        Generate random train/val/test masks for the nodes.
        """
        np.random.seed(self.seed)
        indices = np.random.permutation(num_nodes)
        train_end = int(num_nodes * self.train_ratio)
        val_end = train_end + int(num_nodes * self.val_ratio)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        return train_mask, val_mask, test_mask

    def _create_masks_from_splits(self, splits, num_nodes):
        """
        Create masks from pre-saved split indices.
        """
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[splits['train']] = True
        val_mask[splits['val']] = True
        test_mask[splits['test']] = True

        return train_mask, val_mask, test_mask
```

## evaluation.py

```python
## evaluation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

class Evaluation:
    """
    Evaluation class to perform inference on test data, compute metrics,
    and optionally visualize node representations.
    """

    def __init__(self, model, predictor, data, device):
        """
        Initialize with trained diffusion model, predictor, dataset split data, and device.
        
        Args:
            model (DiffusionModel): diffusion-based GNN with inverse capabilities.
            predictor (torch.nn.Module): trained classification head.
            data (tuple): (features, labels, train_mask, val_mask, test_mask)
            device (torch.device): computation device.
        """
        self.model = model
        self.predictor = predictor
        self.features, self.labels, self.train_mask, self.val_mask, self.test_mask = data
        self.device = device
        # Set model to eval mode
        self.model.eval()
        self.predictor.eval()
        # Extract diffusion hyperparameters for inference
        self.diff_TF = getattr(self.model, 'T_F', 1.0)
        self.diff_TR = getattr(self.model, 'T_R', -1.0)
        self.fixed_point_iter = getattr(self.model, 'M', 16)
        # Visualization flag from config could be passed here if needed
        self.visualization_enabled = False

    def evaluate(self, n_splits=10, seed=42, visualize_layers=None):
        """
        Perform inference over multiple splits, compute mean and std of accuracy,
        and produce visualizations if enabled.
        
        Args:
            n_splits (int): number of dataset splits.
            seed (int): for reproducibility.
            visualize_layers (list): layers to visualize; e.g., ['forward', 'reverse', 'concatenate']
        
        Returns:
            metrics_dict (dict): contains mean and std of accuracy over splits.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        accuracies = []

        for split_idx in tqdm(range(n_splits), desc='Evaluating splits'):
            # For reproducibility, ensure dataset splits are consistent only if dataset is
            # configured accordingly; assume test_mask is fixed for each split
            test_mask = self.test_mask
            
            # Run diffusion forward at TF
            with torch.no_grad():
                x_fwd = self.model.forward(self.features)
                # Run inverse diffusion at TR
                x_rev = self.model.inverse(x_fwd)
            
            # For test nodes only
            test_indices = torch.nonzero(test_mask, as_tuple=True)[0]

            # Extract representations for test nodes
            # Depending on "layers to visualize", pick representations
            # For this code, simply use final diffused and inverse features
            # If detailed layer features needed, modify accordingly

            # Get representations
            fwd_repr = x_fwd[test_indices]
            rev_repr = x_rev[test_indices]

            # Concatenate representations
            combined = torch.cat([fwd_repr, rev_repr], dim=1)
            # Pass through predictor
            logits = self.predictor(combined)

            preds = torch.argmax(logits, dim=1)

            # Compute accuracy
            lbls = self.labels[test_indices]
            correct = (preds == lbls).sum().item()
            total = lbls.shape[0]
            accuracy = correct / total
            accuracies.append(accuracy)

            # Visualization if enabled
            if self.visualization_enabled:
                self._visualize_embeddings(
                    fwd_repr.cpu().numpy(),
                    rev_repr.cpu().numpy(),
                    lbls.cpu().numpy(),
                    preds.cpu().numpy(),
                    split_idx,
                    features=self.features.cpu().numpy(),
                    indices=test_indices.cpu().numpy()
                )

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        metrics_dict = {
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "accuracy_list": accuracies
        }
        print(f"\nEvaluation over {n_splits} splits:")
        print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        return metrics_dict

    def _visualize_embeddings(self, fwd_repr, rev_repr, labels, preds, split_idx, features=None, indices=None):
        """
        Generate scatter plot of representations for qualitative analysis.
        
        Args:
            fwd_repr (np.ndarray): forward diffusion features
            rev_repr (np.ndarray): inverse diffusion features
            labels (np.ndarray): true labels
            preds (np.ndarray): predicted labels
            split_idx (int): index of split for saving figure
            features (np.ndarray): original features (optional)
            indices (np.ndarray): node indices (optional)
        """
        plt.figure(figsize=(12, 4))
        # Plot forward representations
        plt.subplot(1, 3, 1)
        self._plot_repr(fwd_repr, labels, title='Forward Diffusion', indices=indices)
        # Plot reverse representations
        plt.subplot(1, 3, 2)
        self._plot_repr(rev_repr, labels, title='Reverse Diffusion', indices=indices)
        # Plot concatenated
        plt.subplot(1, 3, 3)
        concat_repr = np.concatenate([fwd_repr, rev_repr], axis=1)
        self._plot_repr(concat_repr, labels, title='Concatenated', indices=indices)

        plt.tight_layout()
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig(f"visualizations/representation_split_{split_idx}.png")
        plt.close()

    def _plot_repr(self, reprs, labels, title='Representation', indices=None):
        """
        Plot 2D visualization of node representations using t-SNE.
        """
        if reprs.shape[1] > 2:
            embedding = TSNE(n_components=2, random_state=42).fit_transform(reprs)
        else:
            embedding = reprs
        plt.scatter(embedding[:,0], embedding[:,1], c=labels, cmap='tab10', s=15)
        plt.title(title)
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.colorbar()
```


## main.py

```python
## main.py
import os
import sys
import yaml
import random
import torch
import numpy as np
from tqdm import tqdm

from dataset_loader import DatasetLoader
from model import DiffusionModel
from trainer import Trainer
from evaluation import Evaluation

def main():
    # ------------------------------
    # Step 1: Load configuration and dataset parameters
    # ------------------------------
    CONFIG_PATH = 'config.yaml'
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Set default configs if missing
    training_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    evaluation_cfg = config.get('evaluation', {})

    # ------------------------------
    # Step 2: Set random seeds for reproducibility
    # ------------------------------
    seed = dataset_cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ------------------------------
    # Step 3: Select device
    # ------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ------------------------------
    # Step 4: Load the dataset
    # ------------------------------
    dataset_loader = DatasetLoader({
        'dataset_name': dataset_cfg.get('dataset_name', ''),
        'seed': seed,
        'train_ratio': dataset_cfg.get('train_ratio', 0.6),
        'val_ratio': dataset_cfg.get('val_ratio', 0.2),
        'test_ratio': dataset_cfg.get('test_ratio', 0.2),
        'splits': dataset_cfg.get('splits', 10),
        'data_dir': dataset_cfg.get('data_dir', './datasets')  # adjust as needed
    })

    print("Loading dataset...")
    adj, features, labels, train_mask, val_mask, test_mask = dataset_loader.load_data()

    # Move tensors to device
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    adj = adj.to(device)

    # ------------------------------
    # Step 5: Instantiate the diffusion model
    # ------------------------------
    diffusion_time_TF = model_cfg.get('diffusion_time_TF', 1.0)
    diffusion_time_TR = model_cfg.get('diffusion_time_TR', -1.0)
    num_forward_layers = model_cfg.get('num_forward_layers', 4)
    num_reverse_layers = model_cfg.get('num_reverse_layers', 4)
    fixed_point_iter = model_cfg.get('fixed_point_iter', 16)
    max_diffusion_steps = model_cfg.get('max_diffusion_steps', 100)
    step_size = model_cfg.get('diffusion_step_size', 0.1)
    attention_heads = model_cfg.get('attention_heads', 1)
    hidden_dim = model_cfg.get('hidden_dim', features.shape[1])
    normalize_weights = model_cfg.get('normalize_weights', True)

    print("Initializing diffusion model...")
    diffusion_model = DiffusionModel(
        adj=adj,
        in_features=features.shape[1],
        hidden_dim=hidden_dim,
        num_forward_layers=num_forward_layers,
        diffusion_time_TF=diffusion_time_TF,
        num_reverse_layers=num_reverse_layers,
        diffusion_time_TR=diffusion_time_TR,
        fixed_point_iter=fixed_point_iter,
        attention_heads=attention_heads,
        max_diffusion_steps=max_diffusion_steps,
        step_size=step_size,
        normalize_weights=normalize_weights,
        device=device
    )

    # ------------------------------
    # Step 6: Create predictor head
    # ------------------------------
    num_classes = int(labels.max().item() + 1)
    class PredictorHead(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.linear = nn.Linear(input_dim * 2, num_classes)

        def forward(self, fwd_repr, rev_repr):
            concat = torch.cat([fwd_repr, rev_repr], dim=1)
            logits = self.linear(concat)
            return logits

    predictor = PredictorHead(
        input_dim=features.shape[1],
        num_classes=num_classes
    ).to(device)

    # ------------------------------
    # Step 7: Initialize the trainer
    # ------------------------------
    learning_rate = training_cfg.get('learning_rate', 1e-3)
    batch_size = training_cfg.get('batch_size', features.shape[0])  # full batch if small dataset
    epochs = training_cfg.get('epochs', 100)
    weight_decay = training_cfg.get('weight_decay', 5e-4)
    dropout_rate = training_cfg.get('dropout', 0.5)
    patience = training_cfg.get('patience', 10)

    trainer = Trainer(
        model=diffusion_model,
        predictor=predictor,
        data=(features, labels, train_mask, val_mask, test_mask),
        hyperparams={
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'weight_decay': weight_decay,
            'dropout': dropout_rate,
            'patience': patience
        },
        device=device
    )

    # ------------------------------
    # Step 8: Train the model
    # ------------------------------
    print("Starting training...")
    trainer.train()

    # ------------------------------
    # Step 9: Load best model after training
    # ------------------------------
    print("Loading best model for evaluation...")
    trainer.load_model()

    # ------------------------------
    # Step 10: Evaluate on test set
    # ------------------------------
    print("Evaluating on test set...")
    evaluator = Evaluation(
        model=diffusion_model,
        predictor=predictor,
        data=(features, labels, train_mask, val_mask, test_mask),
        device=device
    )

    metrics = evaluator.evaluate()

    # Print final test metrics
    print("Test Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")

    # ------------------------------
    # Step 11: Visualization (if enabled)
    # ------------------------------
    if evaluation_cfg.get('visualization', False):
        # Generate representations at certain layers
        # Forward, reverse, concatenated, etc.
        # This part is model-specific and may include hidden state extraction
        try:
            import matplotlib.pyplot as plt
            # Example: visualize representations
            # Note: actual visualization code depends on further implementation
            # For simplicity, placeholder code
            print("Generating visualization figures...")
            # e.g., feature representations before and after inverse, labels, prediction overlays
            # Save figures
            # plt.savefig('representation_vis.png')
        except ImportError:
            print("matplotlib not installed. Skipping visualization.")

if __name__ == "__main__":
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionModel(nn.Module):
    """
    DiffusionModel implements the core forward diffusion by Euler's method, the inverse process via fixed-point iteration,
    and the attention mechanism for calculating the diffusion matrix \(\mathbf{A}(\mathbf{X})\). It also manages the stacking
    of multiple forward and inverse layers with residual/invertibility considerations.
    """

    def __init__(
        self,
        adj: torch.sparse.FloatTensor,
        in_features: int = 64,
        hidden_dim: int = 64,
        num_forward_layers: int = 4,
        diffusion_time_TF: float = 1.0,
        num_reverse_layers: int = 4,
        diffusion_time_TR: float = -1.0,
        fixed_point_iter: int = 16,
        attention_heads: int = 1,
        max_diffusion_steps: int = 100,
        step_size: float = 0.1,
        normalize_weights: bool = True,
        weight_scale: float = 0.9,
        device: torch.device = torch.device('cpu')
    ):
        super(DiffusionModel, self).__init__()
        self.adj = adj  # sparse tensor, shape: [N, N]
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_forward_layers = num_forward_layers
        self.num_reverse_layers = num_reverse_layers
        self.T_F = diffusion_time_TF
        self.T_R = diffusion_time_TR
        self.M = fixed_point_iter
        self.max_diffusion_steps = max_diffusion_steps
        self.dt = step_size
        self.heads = attention_heads
        self.normalize_weights = normalize_weights
        self.weight_scale = weight_scale
        self.device = device

        # Initialize attention parameters
        # For simplicity, using single-head attention here; extend for multi-head
        self.W_K = nn.Parameter(torch.randn(self.in_features, self.in_features))
        self.W_Q = nn.Parameter(torch.randn(self.in_features, self.in_features))
        self.a = nn.Parameter(torch.randn(self.heads, self.in_features * 2))
        # Initialize residual layer weights
        self.W = nn.Parameter(torch.randn(self.in_features, self.in_features))
        # After parameter init, normalize weights for invertibility if needed
        if self.normalize_weights:
            self.normalize_weights_fn()

        # Activation
        self.activation = nn.ReLU()

    def compute_attention(self, X: torch.FloatTensor):
        """
        Compute the attention matrix A(X) based on current features X.
        """
        # Compute queries and keys
        Q = X @ self.W_Q  # shape: [N, d]
        K = X @ self.W_K  # shape: [N, d]

        # For multi-head attention, extend here if needed
        # Compute scaled dot-product
        d_prime = self.in_features
        scores = (K @ Q.t()) / math.sqrt(d_prime)  # shape: [N, N]

        # For edges, mask scores: set non-edges to -inf to prevent attention
        # but since A is sparse, masking is implicit
        # Convert scores to dense for softmax or use sparse softmax
        # For efficiency with large graphs, handle only edges
        # We'll proceed with dense here for simplicity
        mask = torch.sparse.full_like(self.adj, fill_value=0)
        # NOTE: For large graphs, implement sparse-softmax. For now, approximate dense.
        # The adjacency provides edge mask: non-edges set to -inf
        dense_mask = torch.zeros_like(scores)
        dense_mask[self.adj.indices()[0], self.adj.indices()[1]] = scores[self.adj.indices()[0], self.adj.indices()[1]]
        # Assign large negative value to non-edges to zero-out after softmax
        scores_masked = torch.full_like(scores, fill_value=-1e9)
        scores_masked[self.adj.indices()[0], self.adj.indices()[1]] = dense_mask[self.adj.indices()[0], self.adj.indices()[1]]

        # Compute attention weights
        alpha = F.softmax(scores_masked, dim=1)  # shape: [N, N]
        # Convert to sparse tensor
        indices = self.adj.indices()
        values = alpha[indices[0], indices[1]]
        A = torch.sparse.FloatTensor(indices, values, self.adj.shape).to(self.device)
        return A

    def normalize_weights_fn(self):
        """
        Normalize the weights matrix W to satisfy the Lipschitz constraint.
        """
        W_norm = torch.norm(self.W, p='fro')
        with torch.no_grad():
            if W_norm > 0:
                scale_factor = self.weight_scale / W_norm
                self.W.data.mul_(scale_factor)

    def forward_diffusion(self, x0: torch.FloatTensor, M: int = 100, dt: float = 0.1):
        """
        Simulate forward diffusion from initial features x0 over time T_F using Euler method.
        """
        x = x0.clone()
        steps = M
        delta_t = self.T_F / steps
        for _ in range(steps):
            A_x = self.compute_attention(x)  # sparse tensor
            # Compute diffusive update: (A - I)X
            Ax = torch.sparse.mm(A_x, x)
            x = x + delta_t * (Ax - x)
        return x

    def inverse_process(self, xT: torch.FloatTensor, T_R: float, M: int = 16):
        """
        Approximate the back-in-time features at T_R < 0 using fixed-point iteration.
        """
        # Initialize with features at T_F (xT)
        x = xT.clone()
        for _ in range(M):
            x_prev = x.clone()
            h_x = self._compute_inverse_h(x)
            x = x_prev - h_x
        return x

    def _compute_inverse_h(self, x: torch.FloatTensor):
        """
        Residual operator h in inverse process.
        """
        # Approximate negative diffusion step: (A - I)X
        A_x = self.compute_attention(x)
        Ax = torch.sparse.mm(A_x, x)
        h = Ax - x  # residual
        return h

    def gnn_layer(self, X: torch.FloatTensor, W: torch.nn.Parameter, A: torch.sparse.FloatTensor):
        """
        Forward residual GNN layer with residual connection and activation.
        """
        # f(X) = X + activation(AX W)
        AXW = torch.sparse.mm(A, X) @ W
        out = X + self.activation(AXW)
        return out

    def inverse_gnn_layer(self, X: torch.FloatTensor, W: torch.nn.Parameter, A: torch.sparse.FloatTensor):
        """
        Inverse GNN layer estimated via fixed-point iteration.
        """
        X_est = X.clone()
        for _ in range(self.M):
            X_est_prev = X_est.clone()
            # Inverse step: solve X ≈ X - activation(A X W) for X
            # Here, we assume residual h is contraction, so do fixed-point
            # For simplicity, linear approximation
            AXW = torch.sparse.mm(A, X_est) @ W
            X_est = X_est - self.activation(AXW)  # Might need more sophisticated inverse
        return X_est

    def forward(self, x0: torch.FloatTensor):
        """
        Run the entire forward diffusion process with multiple layers.
        """
        x = x0.clone()
        for _ in range(self.num_forward_layers):
            A_x = self.compute_attention(x)
            Ax = torch.sparse.mm(A_x, x)
            x = x + self.dt * (Ax - x)
        return x

    def inverse(self, xT: torch.FloatTensor):
        """
        Run the inverse process with multiple inverse layers.
        """
        x = xT.clone()
        for _ in range(self.num_reverse_layers):
            x = self.inverse_process(x, self.T_R, self.M)
        return x

    def generate_reverse_features(self, xT: torch.FloatTensor):
        """
        Generate features at T_R using multiple reverse layers.
        """
        x = xT.clone()
        for _ in range(self.num_reverse_layers):
            x = self.inverse_process(x, self.T_R, self.M)
        return x

    def combine_representations(self, fwd_repr: torch.Tensor, rev_repr: torch.Tensor):
        """
        Concatenate forward and reverse representations for downstream prediction.
        """
        return torch.cat([fwd_repr, rev_repr], dim=1)

    def prepare_diffusion_attention(self, features: torch.FloatTensor):
        """
        Precompute or initialize attention matrices if needed.
        """
        # For efficiency, could cache attention matrices per epoch
        return self.compute_attention(features)

    def normalize_all_weights(self):
        """
        Normalize all relevant weight matrices after training step for invertibility.
        """
        self.normalize_weights_fn()

```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, predictor, data, hyperparams, device):
        """
        Initialize the Trainer with model, predictor, dataset, hyperparameters, and device.
        Args:
            model (DiffusionModel): diffusion-based GNN with inverse process capabilities.
            predictor (nn.Module): classification head (e.g., linear or MLP).
            data (tuple): (adj, features, labels, train_mask, val_mask, test_mask)
            hyperparams (dict): hyperparameters like learning rate, epochs, diffusion times, etc.
            device (torch.device): CPU or CUDA device.
        """
        self.model = model
        self.predictor = predictor
        self.adj, self.features, self.labels, self.train_mask, self.val_mask, self.test_mask = data
        self.device = device

        # Extract hyperparameters with defaults
        self.lr = hyperparams.get('learning_rate', 1e-3)
        self.batch_size = hyperparams.get('batch_size', self.features.shape[0])
        self.epochs = hyperparams.get('epochs', 100)
        self.weight_decay = hyperparams.get('weight_decay', 5e-4)
        self.dropout = hyperparams.get('dropout', 0.5)
        self.patience = hyperparams.get('patience', 10)
        self.M = hyperparams.get('fixed_point_iter', 16)
        self.L_F = hyperparams.get('num_forward_layers', 4)
        self.L_R = hyperparams.get('num_reverse_layers', 4)
        self.diff_TF = hyperparams.get('diffusion_time_TF', 1.0)
        self.diff_TR = hyperparams.get('diffusion_time_TR', -1.0)
        self.diff_step_size = hyperparams.get('diffusion_step_size', 0.1)
        self.device = device

        # Set optimizer: params of model and predictor
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.predictor.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.best_val_acc = 0.0
        self.wait = 0  # For early stopping
        self.best_state_dict = None

        # Ensure model weights have correct initial normalization if needed
        self._normalize_weights()

    def _normalize_weights(self):
        """
        Normalize weight matrices in the model to satisfy Lipschitz < 1 for invertibility.
        Specifically, for W matrices, scale so that ||A||_2 * ||W||_F < 1.
        """
        # Assuming model has method to normalize residual layer weights
        # e.g., model.normalize_all_weights()
        if hasattr(self.model, 'normalize_weights_fn'):
            self.model.normalize_weights_fn()

    def train(self):
        """
        Full training loop with early stopping and model checkpointing.
        """
        for epoch in range(1, self.epochs + 1):
            loss_value = self._train_one_epoch(epoch)
            val_metrics = self._validate()

            val_acc = val_metrics.get('accuracy', 0.0)
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.wait = 0
                self.best_state_dict = {
                    'model': self.model.state_dict(),
                    'predictor': self.predictor.state_dict()
                }
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        # Load best model
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict['model'])
            self.predictor.load_state_dict(self.best_state_dict['predictor'])

    def _train_one_epoch(self, epoch):
        """
        Perform one epoch of training: diffusion forward, inverse, representation, loss, backprop.
        """
        self.model.train()
        self.predictor.train()

        # Reset gradients
        self.optimizer.zero_grad()

        # Forward diffusion: obtain diffused features at T_F
        x0 = self.features
        xT = self.model.forward_diffusion(x0=x0, M=self.M, dt=self.diff_step_size)
        # Inverse process: approximate features at T_R
        xT_R = self.model.inverse_process(xT=xT, T_R=self.diff_TR, M=self.M)

        # Compute forward representation from features (could be features directly or diffused)
        # Here, for consistent usage, we take features as initial x0
        # and run forward through L_F layers (or use diffusion output)
        # For simplicity, run diffusion for L_F + 1 steps (or just take xT)
        # but following the paper, we can perform multiple residual layers
        # For clarity, assume features go through L_F-layer stacking
        # For this implementation, we perform diffusions with L_F steps separately
        # Alternatively, we can batch process diffusion over full steps or reuse per-layer features.
        # Here, we do a simple approach: run diffusion once, and use the final diffused features.
        # For consistency with the paper, you may want to run multiple diffusion steps per layer.

        # For simplicity, here:
        fwd_repr = self.model.forward(x0)  # forward diffusion layers: call model.forward
        # For multiple layers, in practice, run step-by-step; but here, suffice

        # Similarly, get reverse representation at L_R layers (inverse process)
        rev_repr = self.model.inverse(xT)  # apply inverse layers

        # Compute logits: concatenate representations
        logits = self.predictor.predict_logits(fwd_repr, rev_repr)

        # Loss computation (classification)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits[self.train_mask], self.labels[self.train_mask])

        # Backpropagation
        loss.backward()
        self._clip_gradients()
        self.optimizer.step()

        # Normalize weights for invertibility
        self._normalize_weights()

        # Return loss for logging
        return loss.item()

    def _clip_gradients(self, max_norm=1.0):
        """
        Optional gradient clipping for training stability.
        """
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm)

    def _validate(self):
        """
        Run validation: diffusion forward, inverse, compute accuracy.
        """
        self.model.eval()
        self.predictor.eval()

        with torch.no_grad():
            # Similar to train, but no backprop
            x0 = self.features
            xT = self.model.forward_diffusion(x0=x0, M=self.M, dt=self.diff_step_size)
            xT_R = self.model.inverse_process(xT=xT, T_R=self.diff_TR, M=self.M)

            # Obtain forward features
            fwd_repr = self.model.forward(x0)
            # Obtain reverse features
            rev_repr = self.model.inverse(xT)

            logits = self.predictor.predict_logits(fwd_repr, rev_repr)
            preds = logits.argmax(dim=1)

            correct = (preds[self.val_mask] == self.labels[self.val_mask]).sum().item()
            total = self.val_mask.sum().item()
            accuracy = correct / total

        return {'accuracy': accuracy}

    def save_model(self, path):
        """
        Save the model and predictor states.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'predictor_state_dict': self.predictor.state_dict()
        }, path)

    def load_model(self, path=None):
        """
        Load saved model: if path is given, load from file.
        """
        if path is not None:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        # else, load best saved weights if saved during training
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\reverse-gnn\reverse-gnn_repo`
