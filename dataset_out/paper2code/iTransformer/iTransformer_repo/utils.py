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
