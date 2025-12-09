## evaluation.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataset_loader import DatasetLoader
from model import CATSModel
from utils import plot_attention_map, plot_forecast_and_attention, normalize_tensor

class Evaluation:
    def __init__(
        self,
        model_checkpoint_path: str,
        dataset_name: str,
        dataset_path: str,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the evaluation with:
        - Load dataset (test split)
        - Load model weights
        - Setup device
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.dataset_name = dataset_name
        self.model_checkpoint_path = model_checkpoint_path

        # Load dataset if not already loaded
        self.dataset_loader = DatasetLoader([dataset_name], {
            **self._get_dataset_config(),
            'dataset': {
                'name': dataset_name,
                'data_path': dataset_path,
                'normalizer': 'standard',  # assume normalization exactly same as training
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15
            }
        })
        # Get test data in patch-embedded form
        self.test_input, self.test_target = self.dataset_loader.get_train_test_split(dataset_name)
        # Move to device
        self.test_input = self.test_input.to(self.device)  # shape (num_samples, N_patches, D)
        self.test_target = self.test_target.to(self.device)  # shape (num_samples, T, D or 1)

        # Load model with architecture
        self.model = self._initialize_model()
        # Load trained weights
        self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _get_dataset_config(self) -> Dict:
        """
        Extract dataset-related configs (path, normalization, etc.)
        """
        ds_cfg = {
            'input_sequence_length': self.config.get('model', {}).get('input_sequence_length', 96),
            'patch_size': self.config.get('model', {}).get('patch_size', 24),
            'forecast_horizon': self.config.get('model', {}).get('forecast_horizon', 72),
            'num_layers': self.config.get('model', {}).get('num_layers', 3),
            'num_heads': self.config.get('model', {}).get('num_heads', 2),
            'embed_dim': self.config.get('model', {}).get('embed_dim', 256),
            'horizon_embeddings': True
        }
        return ds_cfg

    def _initialize_model(self) -> nn.Module:
        """
        Instantiate the model with hyperparameters from config
        """
        model_params = self._get_dataset_config()
        model = CATSModel(model_params)
        return model

    def _load_model(self):
        """
        Load saved weights, assuming checkpoint is a torch state_dict
        """
        state_dict = torch.load(self.model_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def evaluate(self):
        """
        Run inference over the test set, compute metrics and visualize attention maps.
        """
        total_samples = self.test_input.shape[0]
        batch_size = self.config.get('training', {}).get('batch_size', 32)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.test_input, self.test_target),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        mse_sum = np.zeros(self.model.forecast_horizon)
        mae_sum = np.zeros(self.model.forecast_horizon)
        total_batches = len(dataloader)

        # Store predictions and attention weights for visualization
        all_predictions = []
        all_targets = []
        all_attention_maps = []  # Will be a list of attention maps per batch

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_seq, true_seq = batch
                input_seq = input_seq.to(self.device)     # (B, N_patches, D)
                true_seq = true_seq.to(self.device)       # (B, T, D or 1)

                # Forward pass, assuming model returns both forecast and attention
                # Modify your model's forward to return attention weights if needed
                forecast, attentions = self._predict_with_attention(input_seq)

                # Collect predictions and true for metrics
                all_predictions.append(forecast.cpu())
                all_targets.append(true_seq.cpu())

                # Compute metrics per horizon
                for t in range(self.model.forecast_horizon):
                    pred_t = forecast[:, t, ...]  # (B,)
                    true_t = true_seq[:, t, ...]  # (B,)
                    mse_sum[t] += torch.mean((pred_t - true_t) ** 2).item()
                    mae_sum[t] += torch.mean(torch.abs(pred_t - true_t)).item()

                # Save attentions for visualization if available
                if attentions is not None:
                    all_attention_maps.append(attentions)

        # Compute overall metrics
        mse_per_horizon = mse_sum / total_samples
        mae_per_horizon = mae_sum / total_samples
        print("Evaluation Results:")
        for t in range(self.model.forecast_horizon):
            print(f"Horizon {t+1}: MSE={mse_per_horizon[t]:.4f}, MAE={mae_per_horizon[t]:.4f}")

        # Optionally, plot some attention maps for selected examples
        self._visualize_attention_maps(all_attention_maps, all_predictions, all_targets)

    def _predict_with_attention(self, input_seq: torch.Tensor):
        """
        Inference that also captures attention maps if model provides them
        Assumes model returns tuple of (forecast, attention_weights)
        """
        # To enable attention extraction, modify your model's forward to output attention if not
        # For this implementation, we assume it's adapted
        self.model.eval()
        # Monkey patch or modify model to output attentions
        # Alternatively, if your model already outputs them:
        forecast, attentions = None, None
        if hasattr(self.model, 'forward'):
            # Use a custom method if necessary
            forecast, attentions = self.model.forward(input_seq)
        else:
            forecast = self.model(input_seq)
        return forecast, attentions

    def _visualize_attention_maps(
        self,
        attentions: List,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        num_examples: int = 3
    ):
        """
        Plot attention heatmaps for a few examples, showing periodicity or shocks.
        """
        # Flatten lists
        attentions = attentions  # list of attention tensors with shape (layers, heads, N_q, N_k)
        preds = torch.cat(predictions, dim=0)
        trues = torch.cat(targets, dim=0)

        num_examples = min(num_examples, preds.shape[0])
        for i in range(num_examples):
            plt.figure(figsize=(15, 8))
            # For each example in batch
            pred_series = preds[i].squeeze()  # shape (T,)
            true_series = trues[i].squeeze()
            # Plot forecast and true
            plt.subplot(2, 1, 1)
            plt.plot(true_series.cpu(), label='Ground Truth')
            plt.plot(pred_series.cpu(), label='Forecast')
            plt.title(f'Forecast vs True for sample {i}')
            plt.legend()

            # Plot attention heatmap if available
            if attentions and len(attentions) > 0:
                # For illustration, pick last layer/ head
                last_layer_attn = attentions[-1]  # shape (layers, heads, N_q, N_k)
                # Select last layer
                last_layer = last_layer_attn[-1]  # shape (heads, N_q, N_k)
                for head_idx in range(last_layer.shape[0]):
                    attn_map = last_layer[head_idx]  # shape (N_q, N_k)
                    plt.subplot(2, 2, head_idx+2)
                    plot_attention_map(attn_map, title=f"Attention Map - Head {head_idx+1}")
            plt.show()

    def _load_model(self):
        """
        Load the saved model weights
        """
        state_dict = torch.load(self.model_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

# Usage example (assuming your main script will call):
# evaler = Evaluation('path/to/checkpoint.pth', 'ETTm1', './datasets/ETTm1', config)
# evaler.evaluate()
