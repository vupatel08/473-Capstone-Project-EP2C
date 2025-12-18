## utils.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_attention_heatmaps(attention_matrices, dataset_name='Dataset', horizon='Horizon', save_path=None, show=False):
    """
    Visualizes a collection of attention matrices as heatmaps.
    Args:
        attention_matrices (list or np.ndarray): List or array of attention matrices of shape (N, D, D),
            where N is number of samples, D is feature dimension.
        dataset_name (str): Dataset identifier for plot titles.
        horizon (str): Horizon identifier for plot titles.
        save_path (str, optional): If provided, saves plots to the specified path.
        show (bool): Whether to display plots inline.
    """
    import math
    import os
    if isinstance(attention_matrices, torch.Tensor):
        attentions = attention_matrices.detach().cpu().numpy()
    elif isinstance(attention_matrices, np.ndarray):
        attentions = attention_matrices
    else:
        attentions = np.array(attention_matrices)

    num_matrices = attentions.shape[0]
    cols = min(4, num_matrices)
    rows = math.ceil(num_matrices / cols)

    plt.figure(figsize=(4 * cols, 4 * rows))
    for i in range(num_matrices):
        plt.subplot(rows, cols, i + 1)
        sns.heatmap(attentions[i], annot=False, cmap='viridis', cbar=True)
        plt.title(f'Sample {i+1}')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Dimension')
    plt.suptitle(f'Attention Matrices - {dataset_name} - Horizon {horizon}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_loss_landscape(loss_grid, x_vals, y_vals, save_path=None, show=False):
    """
    Plots a 2D loss landscape given evaluated loss over a grid of parameters.
    Args:
        loss_grid (np.ndarray): 2D array of shape (len(x_vals), len(y_vals)).
        x_vals (np.ndarray): 1D array of grid points along x-direction.
        y_vals (np.ndarray): 1D array along y-direction.
        save_path (str, optional): Path to save the plot.
        show (bool): Whether to display plot inline.
    """
    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(x_vals, y_vals)
    cp = plt.contourf(X, Y, loss_grid.T, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')
    plt.title('Loss Landscape')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def denormalize_sequence(seq: torch.Tensor, means: np.ndarray, stds: np.ndarray,
                         betas: np.ndarray = None, gammas: np.ndarray = None, epsilon: float=1e-8) -> torch.Tensor:
    """
    Denormalize a sequence normalized via RevIN.
    Args:
        seq (torch.Tensor): shape (B, L, D) or (B, D, L)
        means (np.ndarray): per-feature means (D,)
        stds (np.ndarray): per-feature stds (D,)
        betas (np.ndarray, optional): learned beta parameters (D,)
        gammas (np.ndarray, optional): learned gamma parameters (D,)
        epsilon (float): small constant for numerical stability
    Returns:
        torch.Tensor: denormalized sequence with same shape as input
    """
    # Convert stats to tensors
    means_tensor = torch.tensor(means, dtype=seq.dtype, device=seq.device)
    stds_tensor = torch.tensor(stds, dtype=seq.dtype, device=seq.device)
    if betas is not None:
        betas_tensor = torch.tensor(betas, dtype=seq.dtype, device=seq.device)
    else:
        betas_tensor = torch.zeros_like(means_tensor)
    if gammas is not None:
        gammas_tensor = torch.tensor(gammas, dtype=seq.dtype, device=seq.device)
    else:
        gammas_tensor = torch.ones_like(means_tensor)

    # Ensure shape (D,)
    # Reshape tensors for broadcasting
    mu = means_tensor.view(1, 1, -1)
    sigma = (stds_tensor + epsilon).view(1, 1, -1)
    beta = betas_tensor.view(1, 1, -1)
    gamma = gammas_tensor.view(1, 1, -1)

    # Denormalize
    denorm_seq = (seq - beta) / gamma
    denorm_seq = denorm_seq * sigma + mu
    return denorm_seq


def perform_ttest(performance_a: np.ndarray, performance_b: np.ndarray, alpha: float=0.05) -> Tuple[float, float, bool]:
    """
    Performs paired t-test between two performance arrays over multiple runs.
    Args:
        performance_a (np.ndarray): array of shape (num_runs,)
        performance_b (np.ndarray): array of shape (num_runs,)
        alpha (float): significance level, default=0.05
    Returns:
        Tuple of (t_statistic, p_value, is_significant)
    """
    t_stat, p_value = stats.ttest_rel(performance_a, performance_b)
    is_significant = p_value < alpha
    return t_stat, p_value, is_significant


def plot_performance_comparison(datasets: List[str], horizons: List[int], metrics: Dict, title='Model Performance Comparison', save_path=None, show=False):
    """
    Plots comparison of metrics over datasets and horizons.
    Args:
        datasets (list): List of dataset names
        horizons (list): List of horizon values
        metrics (dict): Nested dict with structure {dataset: {horizon: {'model_name': metric_value, ...}}}
        title (str): Plot title
        save_path (str, optional): Path to save plot
        show (bool): Whether to display
    """
    import itertools
    plt.figure(figsize=(12, 8))
    for dataset in datasets:
        for horizon in horizons:
            for model_name, value in metrics.get(dataset, {}).get(horizon, {}).items():
                plt.plot(horizon, value, 'o', label=f'{dataset}-{model_name}')
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Performance Metric')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
