# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py

import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional


class DatasetLoader:
    """
    Class: DatasetLoader
    Purpose: Load datasets from specified paths, normalize variates (channel-wise), generate sequences for training, validation, and testing,
             and provide batch iteration for training and evaluation.
    """

    def __init__(self,
                 data_paths: Dict[str, str],
                 seq_len: int,
                 pred_len: int,
                 variate_normalization: bool = True):
        """
        Initialize DatasetLoader.

        Args:
            data_paths (dict): Dictionary with keys 'train', 'val', 'test' mapping to dataset file paths.
            seq_len (int): Length of input sequence (T).
            pred_len (int): Length of prediction horizon (S).
            variate_normalization (bool): Whether to normalize variates independently.
        """
        self.data_paths = data_paths
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.variate_normalization = variate_normalization

        # Placeholders for raw data arrays
        self.raw_data_train: Optional[np.ndarray] = None
        self.raw_data_val: Optional[np.ndarray] = None
        self.raw_data_test: Optional[np.ndarray] = None

        # Normalization parameters
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None

        # Sequences for train, val, test: list of (input_seq, target_seq)
        self.train_sequences: List[Tuple[np.ndarray, np.ndarray]] = []
        self.val_sequences: List[Tuple[np.ndarray, np.ndarray]] = []
        self.test_sequences: List[Tuple[np.ndarray, np.ndarray]] = []

        # Load and preprocess data upon initialization
        self.load_data()

    def load_csv(self, file_path: str) -> np.ndarray:
        """
        Load dataset from CSV file.
        Assumption: The dataset contains only variate columns, no timestamp column.
        If timestamp exists, it should be excluded beforehand or handled accordingly.

        Args:
            file_path (str): Path to CSV file.

        Returns:
            np.ndarray: Data array of shape (time_points, variates)
        """
        df = pd.read_csv(file_path)
        # If there's a timestamp column named 'timestamp' or similar, exclude it
        # For robustness, drop columns not numeric or with known timestamp name
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data_array = df[numeric_cols].values
        return data_array

    def compute_normalization(self, data: np.ndarray):
        """
        Compute mean and std for each variate (channel).
        """
        self.means = np.mean(data, axis=0)
        self.stds = np.std(data, axis=0)
        # Prevent division by zero
        self.stds[self.stds == 0] = 1.0

    def normalize(self, data: np.ndarray):
        """
        Normalize data with stored means and stds.
        """
        return (data - self.means) / self.stds

    def generate_sequences(self, data: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate overlapping sequences of input-target pairs from data.

        Args:
            data (np.ndarray): Data array of shape (time_points, variates)

        Returns:
            List of tuples: [(input_seq, target_seq), ...]
            shape of input_seq: (seq_len, variates)
            shape of target_seq: (pred_len, variates)
        """
        sequences = []
        total_points = data.shape[0]
        max_idx = total_points - self.seq_len - self.pred_len + 1
        for i in range(max_idx):
            input_seq = data[i : i + self.seq_len]  # shape: (T, N)
            target_seq = data[i + self.seq_len : i + self.seq_len + self.pred_len]  # shape: (S, N)
            sequences.append((input_seq, target_seq))
        return sequences

    def load_data(self):
        """
        Load raw datasets, compute normalization parameters if enabled, generate sequences.
        """
        # Load raw datasets
        self.raw_data_train = self._load_single_dataset(self.data_paths['train'])
        if 'val' in self.data_paths:
            self.raw_data_val = self._load_single_dataset(self.data_paths['val'])
        if 'test' in self.data_paths:
            self.raw_data_test = self._load_single_dataset(self.data_paths['test'])

        # Normalization parameters based on training data
        if self.variate_normalization:
            self.compute_normalization(self.raw_data_train)

        # Generate sequences for train, val, test
        self.train_sequences = self.generate_sequences(self._normalize(self.raw_data_train))
        if self.raw_data_val is not None:
            self.val_sequences = self.generate_sequences(self._normalize(self.raw_data_val))
        if self.raw_data_test is not None:
            self.test_sequences = self.generate_sequences(self._normalize(self.raw_data_test))

    def _load_single_dataset(self, file_path: str) -> np.ndarray:
        """
        Load a dataset from CSV, returning np.ndarray.

        Args:
            file_path (str): Path to dataset CSV file.

        Returns:
            np.ndarray: Data array (time_points, variates)
        """
        data = self.load_csv(file_path)
        return data

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data if normalization is enabled.

        Args:
            data (np.ndarray): Raw data.

        Returns:
            np.ndarray: Normalized data.
        """
        if self.variate_normalization and self.means is not None and self.stds is not None:
            return self.normalize(data)
        else:
            return data

    def get_train_batches(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of training data: input sequences and target sequences.

        Args:
            batch_size (int): Number of sequences per batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X_batch, Y_batch)
                - X_batch: shape (batch_size, T, N)
                - Y_batch: shape (batch_size, pred_len, N)
        """
        return self._get_batches(self.train_sequences, batch_size)

    def get_val_batches(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of validation data.

        Args:
            batch_size (int): Batch size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        return self._get_batches(self.val_sequences, batch_size)

    def get_test_batches(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of testing data.

        Args:
            batch_size (int): Batch size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        return self._get_batches(self.test_sequences, batch_size)

    def _get_batches(self, sequences: List[Tuple[np.ndarray, np.ndarray]], batch_size: int):
        """
        Internal method to generate batched data with shuffling for training.

        Args:
            sequences (list): List of (input_seq, target_seq) tuples.
            batch_size (int): Batch size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: batched input and target tensors
        """
        indices = np.arange(len(sequences))
        np.random.shuffle(indices)
        for start_idx in range(0, len(sequences), batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            batch_input_seqs = [sequences[i][0] for i in batch_indices]
            batch_target_seqs = [sequences[i][1] for i in batch_indices]

            X_batch = torch.tensor(np.stack(batch_input_seqs), dtype=torch.float32)  # shape: (batch, T, N)
            Y_batch = torch.tensor(np.stack(batch_target_seqs), dtype=torch.float32)  # shape: (batch, pred_len, N)

            yield X_batch, Y_batch
```

## evaluation.py

```python
## evaluation.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_config, compute_metrics, plot_series, plot_attention_matrix
from model import InvertedTransformer
from dataset_loader import DatasetLoader

class Evaluation:
    """
    Class: Evaluation
    Purpose:
        Load a trained iTransformer model, run inference on validation or test datasets,
        compute metrics (MSE, MAE), and visualize attention maps and forecasted series.
    """

    def __init__(
        self,
        model: InvertedTransformer,
        data_paths: dict,
        dataset_split: str,
        config_path: str = 'config.yaml',
        device: str = 'cpu',
        attention_maps: bool = True,
        forecast_plots: bool = True,
        num_samples_to_plot: int = 3
    ):
        """
        Initialize Evaluation object.

        Args:
            model (InvertedTransformer): The trained model instance.
            data_paths (dict): Paths to datasets (train/val/test).
            dataset_split (str): 'val' or 'test'.
            config_path (str): Path to config.yaml file.
            device (str): 'cpu' or 'cuda'.
            attention_maps (bool): Whether to visualize attention weights.
            forecast_plots (bool): Whether to plot predicted series vs true.
            num_samples_to_plot (int): Number of samples to visualize.
        """
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.attention_maps = attention_maps
        self.forecast_plots = forecast_plots
        self.num_samples_to_plot = num_samples_to_plot

        # Load config for dataset parameters and normalization
        self.config = load_config(config_path)
        self.dataset_params = self.config['dataset']
        self.seq_len = self.dataset_params['sequence_length']
        self.forecast_len = self.dataset_params['forecast_length']

        # Initialize dataset loader for the specified split
        self.data_paths = data_paths
        self.dataset_split = dataset_split.lower()
        self.data_loader = DatasetLoader(
            data_paths=self.data_paths,
            seq_len = self.seq_len,
            pred_len = self.forecast_len,
            variate_normalization=self.dataset_params.get('variate_normalization', True)
        )

        # To store predictions and ground truth
        self.predictions: list = []
        self.targets: list = []
        self.attention_weights_per_layer: list = []  # optional, if model exposes
        self.sample_indices_vis: list = []

        # For reproducibility, set seed if needed
        # import random, torch
        # random.seed(42)
        # torch.manual_seed(42)

    def run_evaluation(self):
        """
        Run inference over the entire dataset split, compute metrics and produce visualizations.
        """
        total_mse = 0.0
        total_mae = 0.0
        total_batches = 0

        # Gather batches
        if self.dataset_split == 'val':
            batch_generator = self.data_loader.get_val_batches(self.dataset_params.get('batch_size', 64))
        elif self.dataset_split == 'test':
            batch_generator = self.data_loader.get_test_batches(self.dataset_params.get('batch_size', 64))
        else:
            raise ValueError(f"Unknown dataset split: {self.dataset_split}")

        # For attention visualization
        # We assume the model can output attention weights when specified
        # Thus, we will modify the inference code to retrieve attention weights if model supports

        with torch.no_grad():
            for X_batch, Y_batch in batch_generator:
                X_batch = X_batch.to(self.device)   # shape: [B, T, N]
                Y_batch = Y_batch.to(self.device)   # shape: [B, S, N]
                # Permute predictions to (B, N, S) for convenience
                preds = self.model.forward(X_batch)  # shape: [B, N, S]
                preds = preds.permute(0, 2, 1)       # shape: [B, S, N]

                # Save predictions and targets
                self.predictions.append(preds.cpu().numpy())
                self.targets.append(Y_batch.cpu().numpy())

                total_batches += 1

        # Concatenate all batches
        preds_concat = np.concatenate(self.predictions, axis=0)  # [samples, S, N]
        targets_concat = np.concatenate(self.targets, axis=0)  # same shape

        # Transpose for metrics: shape (samples, N, S) or (samples, S, N)
        # Metrics are symmetric whether per-float variate or combined
        # Here, we flatten all variates for overall metrics
        preds_flat = preds_concat.reshape(-1)
        targets_flat = targets_concat.reshape(-1)

        metrics = compute_metrics(preds_flat, targets_flat)

        print(f"Evaluation on {self.dataset_split} set:")
        print(f"  MSE: {metrics['MSE']:.6f}")
        print(f"  MAE: {metrics['MAE']:.6f}")

        # Optional: visualize attention maps for select samples
        if self.attention_maps:
            self._visualize_attention_maps()

        # Optional: visualize forecast series for some samples
        if self.forecast_plots:
            self._visualize_forecasts(preds_concat, targets_concat)

        return metrics

    def _visualize_attention_maps(self):
        """
        Visualize attention matrices over variates for some selected samples.
        This requires that the model exposes attention weights.
        """
        # Assuming the model saves attention weights during forward pass.
        # Here, we do a forward pass with hooks or via model returns.
        # For simplicity, if the model doesn't expose, we skip.
        # Otherwise, one may have added methods or hooks inside the model.

        # We try to get attention matrices for first few samples
        num_samples = min(self.num_samples_to_plot, len(self.predictions))
        for i in range(num_samples):
            # Re-instantiate a batch for visualization
            X, Y = list(self.data_loader.get_val_batches(1))[0]
            X = X.to(self.device)
            # Run inference with attention store enabled
            # For this, assume model has a method to return attentions, or hooks are registered
            # For robust code, check model attribute / method
            attn_matrices = self._get_attention_weights(X)

            if attn_matrices is not None:
                for layer_idx, attn_mat in enumerate(attn_matrices):
                    # attn_mat: [batch, n_heads, variates, variates]
                    # For batch=1, take first
                    attn = attn_mat[0]  # shape: [n_heads, N, N]
                    for head_idx in range(attn.shape[0]):
                        fig, ax = plt.subplots(figsize=(8,6))
                        sns.heatmap(attn[head_idx].cpu().numpy(), annot=False, cmap='viridis', ax=ax)
                        ax.set_title(f"Sample {i+1} Layer {layer_idx+1} Head {head_idx+1}")
                        plt.show()

    def _get_attention_weights(self, X: torch.Tensor):
        """
        Retrieve attention weights during forward pass.
        This method depends on model implementation exposing attention matrices.
        """
        # Placeholder: This function needs model modifications to expose attentions.
        # If not implemented, return None.
        # For example, you can modify model.py to store attention weights on each layer.
        # Here, for completeness, we assume such a feature:
        if hasattr(self.model, 'attention_weights'):
            return self.model.attention_weights
        else:
            return None

    def _visualize_forecasts(self, preds: np.ndarray, targets: np.ndarray):
        """
        Plot sample forecasted series vs. ground truth for visualization.
        """
        num_samples = min(self.num_samples_to_plot, preds.shape[0])
        for i in range(num_samples):
            # Choose a random variate
            N = preds.shape[2]
            n_var_idx = np.random.randint(0, N)
            # Plot forecasted vs true for selected variate
            input_series = self.data_loader.raw_data_test[:, n_var_idx] if hasattr(self.data_loader, 'raw_data_test') else None
            # Input series: underlying original series if available; else use normalized
            # Here, we prefer to plot normalized data
            input_series = self.data_loader.raw_data_test[:, n_var_idx] if hasattr(self.data_loader, 'raw_data_test') else None
            if input_series is None:
                # fallback: no input series for plotting
                pass
            predicted_series = preds[i, :, n_var_idx]
            true_series = targets[i, :, n_var_idx]

            title = f"{self.dataset_split.upper()} Sample {i+1} - Variate {n_var_idx}"
            # For better visualization, plot the prediction horizon only
            # or the entire sequence
            # Here, plot prediction horizon:
            plot_series(
                series=input_series[-self.seq_len:],  # last input sequence
                forecast=true_series,
                title=title,
                save_path=None
            )

    def load_trained_model(self, checkpoint_path: str):
        """
        Load the trained model parameters from a checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

```

## main.py

```python
# main.py

import os
import sys
import yaml
import torch
import numpy as np
from dataset_loader import DatasetLoader
from model import InvertedTransformer
from trainer import Trainer
from evaluation import Evaluation
from utils import load_config, save_model

def main():
    # Load configuration from 'config.yaml'
    config = load_config('config.yaml')
    
    # Determine device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset parameters
    dataset_cfg = config['dataset']
    data_paths = dataset_cfg['data_paths']
    variate_norm = dataset_cfg.get('variate_normalization', True)
    T = dataset_cfg['sequence_length']
    S = dataset_cfg['forecast_length']
    dataset_name = dataset_cfg.get('name', 'Unknown')
    
    # Initialize DatasetLoader for train, val, test
    data_loader = DatasetLoader(
        data_paths=data_paths,
        seq_len=T,
        pred_len=S,
        variate_normalization=variate_norm
    )
    
    # Extract dataset-specific variations
    N = data_loader.raw_data_train.shape[1]  # number of variates
    
    # Model hyperparameters
    model_cfg = config['model']
    embed_dim = model_cfg.get('embedding_dim', 128)
    num_layers = model_cfg.get('num_layers', 4)
    num_heads = model_cfg.get('num_heads', 4)
    dropout_rate = model_cfg.get('dropout_rate', 0.1)
    ff_dim = model_cfg.get('feedforward_dim', 512)
    
    # Instantiate model
    model = InvertedTransformer(
        num_variates=N,
        seq_len=T,
        forecast_len=S,
        embedding_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        feedforward_dim=ff_dim,
        dropout=dropout_rate
    ).to(device)
    print(f"Model instantiated with {N} variates, embedding dim {embed_dim}, {num_layers} layers.")
    
    # Training parameters
    train_cfg = config.get('training', {})
    lr = train_cfg.get('learning_rate', 1e-3)
    batch_size = train_cfg.get('batch_size', 64)
    epochs = train_cfg.get('epochs', 50)
    optimizer_type = train_cfg.get('optimizer', 'adamw')
    weight_decay = train_cfg.get('weight_decay', 0.01)
    save_dir = train_cfg.get('save_dir', 'checkpoints/')
    save_freq = train_cfg.get('save_frequency', 10)
    
    # Initialize optimizer
    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # Loss function
    criterion = torch.nn.MSELoss()
    
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=criterion,
        data_loader=data_loader,
        device=str(device),
        config={
            'epochs': epochs,
            'batch_size': batch_size,
            'save_dir': save_dir,
            'save_frequency': save_freq
        }
    )
    
    # Training loop
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch()
        print(f"Epoch [{epoch}/{epochs}] - Training Loss: {train_loss:.6f}")
        
        # Validation step
        if epoch % save_freq == 0 or epoch == epochs:
            val_metrics = trainer.evaluate(data_loader, split='val')
            val_mse = val_metrics.get('MSE', None)
            print(f"Validation MSE at epoch {epoch}: {val_mse:.6f}")
            # Save best model based on validation MSE
            save_path = os.path.join(save_dir, 'best_model.pth')
            save_model(model, save_path)
            print(f"Model saved at {save_path}")
        else:
            # periodically save checkpoints
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            save_model(model, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    print("Training completed. Loading best model for evaluation...")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    model.eval()

    # Run evaluation on test set
    test_eval = Evaluation(
        model=model,
        data_paths=data_paths,
        dataset_split='test',
        config_path='config.yaml',
        device=str(device),
        attention_maps=False,
        forecast_plots=True,
        num_samples_to_plot=3
    )
    test_metrics = test_eval.run_evaluation()
    print(f"Test set metrics: MSE={test_metrics['MSE']:.6f}, MAE={test_metrics['MAE']:.6f}")

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class LayerNorm(nn.Module):
    """
    Implements Layer Normalization over the feature dimension.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize over last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable layers and activations.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, activation=F.relu, dropout: float = 0.0):
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        # Hidden layers with activation
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module over variate tokens (sequence length = number of variates).
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query, key, value: shape [batch_size, N, embed_dim]
        """
        B, N, _ = query.shape

        # Linear projections
        Q = self.q_linear(query).view(B, N, self.num_heads, self.head_dim).transpose(1,2)  # B, heads, N, head_dim
        K = self.k_linear(key).view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v_linear(value).view(B, N, self.num_heads, self.head_dim).transpose(1,2)

        # Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # B, heads, N, N
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)  # B, heads, N, N
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)  # B, heads, N, head_dim

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        output = self.out_linear(output)
        return output

class FeedForwardNetwork(nn.Module):
    """
    FFN module with two linear layers and activation.
    """
    def __init__(self, embed_dim: int, ff_dim: int, activation=F.relu, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformLayer(nn.Module):
    """
    Single transformer layer (attention + FFN) with residual and variate-wise normalization.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, activation=F.relu, dropout=dropout)
        self.norm3 = LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape [batch_size, N, D]
        """
        # Variate-wise normalization on input
        x_norm = self.norm1(x)
        # Self-attention over variate tokens
        attn_output = self.attn(x_norm, x_norm, x_norm)
        # Residual connection
        x2 = x + attn_output
        # Variate-wise normalization
        x2_norm = self.norm2(x2)
        # FFN applied per variate token
        ffn_output = self.ffn(x2_norm)
        # Second residual
        out = x2 + ffn_output
        # Final normalization
        out = self.norm3(out)
        return out

class InvertedTransformer(nn.Module):
    """
    Inverted Transformer for multivariate time series forecasting.
    Embeds each variate series independently, models multivariate correlations via attention
    over variates, and learns series representations via FFN.
    """
    def __init__(self,
                 num_variates: int,
                 seq_len: int,
                 forecast_len: int,
                 embedding_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 feedforward_dim: int = 512,
                 dropout: float = 0.1):
        """
        Args:
            num_variates (int): Number of variates (channels)
            seq_len (int): Input sequence length T
            forecast_len (int): Prediction horizon S
            embedding_dim (int): Dimension D for variate tokens
            num_layers (int): Number of stacked transformer layers
            num_heads (int): Number of attention heads
            feedforward_dim (int): Dimension of FFN inner layer
            dropout (float): Dropout rate
        """
        super().__init__()
        self.N = num_variates
        self.T = seq_len
        self.S = forecast_len
        self.D = embedding_dim
        self.num_layers = num_layers

        # Variate-wise embedding MLP: maps series of shape (batch, T) -> (batch, D)
        self.series_embedding = MLP(
            input_dim=seq_len,
            hidden_dim=2*embedding_dim,
            output_dim=embedding_dim,
            num_layers=3,
            activation=F.relu,
            dropout=dropout
        )

        # Stack of inverted transformer layers
        self.layers: nn.ModuleList = nn.ModuleList([
            TransformLayer(embed_dim=embedding_dim,
                           num_heads=num_heads,
                           ff_dim=feedforward_dim,
                           dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final Projection from variate tokens to forecasted series
        # Maps (batch, N, D) -> (batch, N, S)
        self.proj = MLP(
            input_dim=embedding_dim,
            hidden_dim=2*embedding_dim,
            output_dim=forecast_len,
            num_layers=3,
            activation=F.relu,
            dropout=dropout
        )

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        """
        Args:
            series: shape (batch_size, N, T)
        Returns:
            pred: shape (batch_size, N, S)
        """
        B = series.shape[0]
        N = self.N
        T = self.T

        # Variate-wise embedding: process each variate series independently
        # series: (B, N, T)
        # Reshape for embedding: flatten batch and variates: (B*N, T)
        series_reshaped = series.reshape(B * N, T)
        # Pass through embedding MLP
        series_emb = self.series_embedding(series_reshaped)  # (B*N, D)
        # Reshape back to (B, N, D)
        H = series_emb.reshape(B, N, self.D)

        # Pass through stacked transformer layers
        for layer in self.layers:
            H = layer(H)  # shape remains (B, N, D)

        # Final projection to forecast horizon
        # Apply projection MLP per variate
        out = self.proj(H)  # shape: (B, N, S)
        return out

    def predict(self, series: torch.Tensor) -> torch.Tensor:
        """
        Generate forecast for a given input series.
        series: shape (B, N, T)
        Returns:
            forecasted series: shape (B, N, S)
        """
        return self.forward(series)
```

## requirements.txt

# requirements.txt

numpy==1.21.0          # Numerical computations and data manipulation
pandas==1.3.0          # Data loading and processing
scikit-learn==0.24.2   # Auxiliary tools such as normalization
torch==1.9.0           # Model implementation, training, inference
matplotlib==3.4.3      # Visualization of results and attention maps

# Notes:
# - Assumes datasets are provided in CSV format as specified in 'data_paths'.
# - For normalization, variate-wise scaling to zero-mean, unit-variance will be applied if enabled.
# - Use default PyTorch modules; no specialized packages for attention or sparse transformers are included.
# - Set random seed for reproducibility in the scripts to ensure consistent training/evaluation outputs.

## trainer.py

```python
## trainer.py

import torch
from torch.optim import Optimizer
from typing import Callable, Dict, Any

class Trainer:
    """
    Class: Trainer
    Purpose:
        Manage the training loop for the InvertedTransformer model.
        Handles per-epoch training, logging, validation, and checkpointing.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: Optimizer,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 data_loader: Any,
                 device: str = 'cpu',
                 config: Dict[str, Any] = None):
        """
        Initialize the trainer.

        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer instance.
            loss_fn (callable): Loss function, e.g., nn.MSELoss().
            data_loader (Any): Dataset loader object providing get_*__batches methods.
            device (str): Device to run computation on ('cpu' or 'cuda').
            config (dict): Configuration dictionary, optional, including training hyperparams.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.device = device
        self.config = config if config is not None else {}

        # Extract hyperparameters with defaults
        self.epochs = self.config.get('epochs', 50)
        self.batch_size = self.config.get('batch_size', 64)
        self.learning_rate = self.config.get('learning_rate', 1e-3)
        self.clip_norm = self.config.get('clip_norm', None)  # e.g., 1.0
        self.save_dir = self.config.get('save_dir', 'checkpoints/')
        self.save_freq = self.config.get('save_frequency', 10)
        self.validation_interval = self.config.get('validation_interval', 1)  # in epochs

        # Create checkpoints directory if not exists
        import os
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self):
        """
        Run one epoch of training over the training dataset.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        batch_count = 0

        # Get iterable batch generator
        train_batches = self.data_loader.get_train_batches(self.batch_size)

        for X_batch, Y_batch in train_batches:
            # Move to device
            X_batch = X_batch.to(self.device)  # shape: [batch, T, N]
            Y_batch = Y_batch.to(self.device)  # shape: [batch, S, N]

            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model.forward(X_batch)  # shape: [batch, N, S]
            predictions = predictions.permute(0, 2, 1)  # to shape: [batch, S, N]

            # Compute loss
            loss = self.loss_fn(predictions, Y_batch)
            loss.backward()

            # Gradient clipping if specified
            if self.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

            # Optimizer update
            self.optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        return avg_loss

    def train(self):
        """
        Run training over all epochs, with validation and checkpointing.
        """
        best_val_metric = float('inf')
        for epoch in range(1, self.epochs + 1):
            # Train epoch
            train_loss = self.train_epoch()

            # Log training loss
            print(f"Epoch [{epoch}/{self.epochs}] - Training Loss: {train_loss:.6f}")

            # Validation step
            if epoch % self.validation_interval == 0:
                val_metrics = self.evaluate(self.data_loader, split='val')
                val_mse = val_metrics.get('MSE', None)
                print(f"Epoch [{epoch}] - Validation MSE: {val_mse:.6f}")

                # Save best model based on validation MSE
                if val_mse is not None and val_mse < best_val_metric:
                    best_val_metric = val_mse
                    save_path = os.path.join(self.save_dir, 'best_model.pth')
                    self.save_checkpoint(save_path)
                    print(f"Saved new best model at epoch {epoch}")

            # Save checkpoint periodically
            if epoch % self.save_freq == 0:
                save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
                self.save_checkpoint(save_path)
                print(f"Saved checkpoint at epoch {epoch}")

    def save_checkpoint(self, save_path: str):
        """
        Save model and optimizer state_dicts.

        Args:
            save_path (str): Path where to save checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, save_path)

    def evaluate(self, data_loader: Any, split: str = 'val') -> Dict[str, float]:
        """
        Evaluate model on validation or test set.

        Args:
            data_loader (Any): Dataset loader with get_*_batches methods.
            split (str): 'val' or 'test'.

        Returns:
            dict: Metrics, e.g., {'MSE': value, 'MAE': value}
        """
        self.model.eval()
        total_preds = []
        total_targets = []

        # Get validation/test batches
        if split == 'val':
            batches = data_loader.get_val_batches(self.batch_size)
        elif split == 'test':
            batches = data_loader.get_test_batches(self.batch_size)
        else:
            raise ValueError(f"Unknown split: {split}")

        with torch.no_grad():
            for X_batch, Y_batch in batches:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                preds = self.model.forward(X_batch)  # shape: [batch, N, S]
                preds = preds.permute(0, 2, 1)  # shape: [batch, S, N]

                total_preds.append(preds.cpu().numpy())
                total_targets.append(Y_batch.cpu().numpy())

        # Concatenate all batches
        preds_concat = np.concatenate(total_preds, axis=0)  # shape: [samples, S, N]
        targets_concat = np.concatenate(total_targets, axis=0)  # same shape

        # Compute metrics
        metrics = {}
        for metric_name in self.config.get('metrics', ['MSE', 'MAE']):
            if metric_name == 'MSE':
                mse = np.mean((preds_concat - targets_concat) ** 2)
                metrics['MSE'] = mse
            elif metric_name == 'MAE':
                mae = np.mean(np.abs(preds_concat - targets_concat))
                metrics['MAE'] = mae
            # Add other metrics if needed

        return metrics
```

## utils.py

```python
## utils.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os

# ==============================
# Configuration Handling
# ==============================
def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# ==============================
# Normalization Utilities
# ==============================
def normalize_variates(series: np.ndarray, epsilon: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize each variate in the series independently to zero mean and unit variance.

    Args:
        series (np.ndarray): Series of shape (T, N).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            normalized_series of shape (T, N),
            means of shape (N,),
            stds of shape (N,)
    """
    means = np.mean(series, axis=0)
    stds = np.std(series, axis=0)
    stds = np.where(stds == 0, 1.0, stds)  # Prevent division by zero
    normalized_series = (series - means) / stds
    return normalized_series, means, stds

def denormalize_variates(normalized_series: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """
    Revert normalized variate series to original scale.

    Args:
        normalized_series (np.ndarray): Normalized data, shape (T, N).
        means (np.ndarray): Means used for normalization, shape (N,).
        stds (np.ndarray): Standard deviations used for normalization, shape (N,).

    Returns:
        np.ndarray: Original scale series, shape (T, N).
    """
    return normalized_series * stds + means

# ==============================
# Plot Series and Forecast
# ==============================
def plot_series(series: np.ndarray,
                forecast: np.ndarray,
                input_seq: np.ndarray = None,
                title: str = '',
                save_path: str = None) -> None:
    """
    Plot input series, forecasted series, and ground truth (if available).

    Args:
        series (np.ndarray): Input series, shape (T, N) or (T,).
        forecast (np.ndarray): Forecasted series, shape (S, N) or (S,).
        input_seq (np.ndarray): Original input sequence, optional, for overlay.
        title (str): Plot title.
        save_path (str): If specified, save the plot to this path.
    """
    plt.figure(figsize=(10, 6))
    if input_seq is not None:
        plt.plot(range(len(input_seq)), input_seq, label='Input Series', color='blue')
    plt.plot(range(len(series)), series, label='Ground Truth', color='green')
    plt.plot(range(len(series), len(series)+len(forecast)), forecast, label='Forecast', color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Series Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ==============================
# Plot Attention Matrix
# ==============================
def plot_attention_matrix(matrix: np.ndarray,
                          title: str = 'Attention Map',
                          save_path: str = None) -> None:
    """
    Visualize attention score matrix as heatmap.

    Args:
        matrix (np.ndarray): Attention score matrix of shape (N, N).
        title (str): Plot title.
        save_path (str): If specified, save plot to file.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=False, cmap='viridis', cbar=True,
                xticklabels=np.arange(matrix.shape[1]),
                yticklabels=np.arange(matrix.shape[0]))
    plt.xlabel('Variate index')
    plt.ylabel('Variate index')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ==============================
# Save and Load Model Checkpoints
# ==============================
def save_model(model: torch.nn.Module, save_path: str) -> None:
    """
    Save model state_dict to disk.

    Args:
        model (torch.nn.Module): The model to save.
        save_path (str): Path where to save the model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

def load_model(model: torch.nn.Module, load_path: str) -> torch.nn.Module:
    """
    Load model state_dict from disk.

    Args:
        model (torch.nn.Module): Model architecture to load into.
        load_path (str): Path to checkpoint file.

    Returns:
        torch.nn.Module: Model with loaded parameters.
    """
    state_dict = torch.load(load_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model

# ==============================
# Metrics computation
# ==============================
def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute evaluation metrics: MSE and MAE.

    Args:
        preds (np.ndarray): Predictions, shape (N, S) or (batch, N, S)
        targets (np.ndarray): Ground truth, same shape as preds

    Returns:
        dict: Dictionary with 'MSE' and 'MAE'
    """
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    return {'MSE': mse, 'MAE': mae}

# ==============================
# Additional Helper Functions (Optional)
# ==============================
# Could add functions for statistical summaries, tensor shape checks, etc.
# But for core functionality, above are most essential utilities.
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\iTransformer\iTransformer_repo`
