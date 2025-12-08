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

