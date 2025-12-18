## utils.py

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import random
import os

def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to zero mean and unit variance."""
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + 1e-8)

# ---------------------------
# Data Augmentation Functions
# ---------------------------

def erase_patch(patch: np.ndarray, probability: float = 0.2) -> np.ndarray:
    """Randomly set the entire patch to zero with given probability."""
    if random.random() < probability:
        return np.zeros_like(patch)
    return patch

def flip_patch(patch: np.ndarray, probability: float = 0.2) -> np.ndarray:
    """Invert the sign of the patch with given probability."""
    if random.random() < probability:
        return -patch
    return patch

def drop_patch(patch: np.ndarray, probability: float = 0.2) -> np.ndarray:
    """Zero out the patch with given probability."""
    if random.random() < probability:
        return np.zeros_like(patch)
    return patch

def add_sine_noise(patch: np.ndarray, freq_range: list = [0.67, 40]) -> np.ndarray:
    """Add a sine wave with random frequency to the patch."""
    freq = np.random.uniform(freq_range[0], freq_range[1])
    t = np.linspace(0, len(patch)/250, len(patch))
    sine_wave = 0.1 * np.sin(2 * np.pi * freq * t)
    return patch + sine_wave

def add_partial_sine(patch: np.ndarray, freq_range: list = [0.67, 40], ratio: float = 0.5) -> np.ndarray:
    """Add sine wave to a portion of the patch."""
    length = len(patch)
    start_idx = int(np.random.uniform(0, length * (1 - ratio)))
    end_idx = int(start_idx + length * ratio)
    t = np.linspace(0, (end_idx - start_idx)/250, end_idx - start_idx)
    freq = np.random.uniform(freq_range[0], freq_range[1])
    sine_wave = 0.1 * np.sin(2 * np.pi * freq * t)
    patch_copy = np.array(patch)
    patch_copy[start_idx:end_idx] += sine_wave
    return patch_copy

def add_white_noise(patch: np.ndarray, std: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to the patch."""
    noise = np.random.normal(0, std, size=patch.shape)
    return patch + noise

def apply_augmentation(patch: np.ndarray, augmentation_type: str, params: dict = {}) -> np.ndarray:
    """Apply specified augmentation to a patch."""
    if augmentation_type == 'erase':
        return erase_patch(patch, probability=params.get('probability', 0.2))
    elif augmentation_type == 'flip':
        return flip_patch(patch, probability=params.get('probability', 0.2))
    elif augmentation_type == 'drop':
        return drop_patch(patch, probability=params.get('probability', 0.2))
    elif augmentation_type == 'sine_wave':
        return add_sine_noise(patch, freq_range=params.get('frequency_range', [0.67, 40]))
    elif augmentation_type == 'partial_sine':
        return add_partial_sine(patch, freq_range=params.get('frequency_range', [0.67, 40]),
                                ratio=params.get('ratio', 0.5))
    elif augmentation_type == 'white_noise':
        return add_white_noise(patch, std=params.get('noise_std', 0.05))
    else:
        return patch

# ------------------------------
# Visualization: Attention Map
# ------------------------------

def plot_attention_map(attention_weights: np.ndarray, query_patch_idx: int,
                       lead_labels: list = None, save_path: str = None) -> None:
    """
    Plot attention map for a specific query patch.
    Args:
        attention_weights: shape [layers, heads, seq_len, seq_len], numpy array or tensor.
        query_patch_idx: int, index of the query patch.
        lead_labels: list of lead names for x/y labels (optional).
        save_path: if provided, save the figure.
    """
    import matplotlib.pyplot as plt

    # Averaging over layers and heads for visualization
    if isinstance(attention_weights, torch.Tensor):
        attn = attention_weights.detach().cpu().numpy()
    else:
        attn = attention_weights
    # shape: [layers, heads, seq_len, seq_len]
    attn_mean = attn.mean(axis=(0,1))
    # Get attention scores for the query patch to all key patches
    query_attention = attn_mean[query_patch_idx]  # shape [seq_len]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(query_attention)), query_attention)
    plt.xlabel('Patch Index')
    plt.ylabel('Attention Score')
    plt.title(f'Attention Map for Query Patch {query_patch_idx}')
    if lead_labels:
        plt.xticks(ticks=range(len(query_attention)), labels=lead_labels, rotation=90)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def generate_attention_maps_for_sample(sample: torch.Tensor, model: nn.Module,
                                       layer_indices: list, head_indices: list,
                                       save_dir: str) -> None:
    """
    Generate and save attention maps for a sample input.
    Args:
        sample: input tensor, shape [batch_size, channels, seq_len]
        model: the transformer model with accessible attention weights
        layer_indices: list of layer indices to extract attention from
        head_indices: list of head indices per layer
        save_dir: directory to save attention map images
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Hook to extract attention weights
    attention_outputs = []

    def get_attention_hook(module, input, output):
        # output is typically a tuple containing attention weights at specific points
        attention_outputs.append(output)

    hooks = []
    for layer_idx in layer_indices:
        handle = model.encoder.layers[layer_idx].self_attn.register_forward_hook(get_attention_hook)
        hooks.append(handle)
    model.eval()
    with torch.no_grad():
        _ = model(sample)
    # Remove hooks
    for h in hooks:
        h.remove()

    # attention_outputs now contains attention weights from specified layers
    for idx, attn in enumerate(attention_outputs):
        attn_array = attn[1]  # shape: [batch, heads, seq_len, seq_len]
        for head_idx in head_indices:
            attn_map = attn_array[0, head_idx].cpu().numpy()
            plot_attention_map(attn_map, query_patch_idx=0, save_path=os.path.join(save_dir, f'layer{layer_indices[idx]}_head{head_idx}.png'))

# ---------------------------
# Embedding Visualization
# ---------------------------

def plot_embeddings(embeddings: np.ndarray, labels: list = None,
                    title: str = 'Embedding T-SNE', save_path: str = None) -> None:
    """
    Reduce embeddings to 2D via t-SNE and plot.
    Args:
        embeddings: numpy array shape [num_samples, embedding_dim]
        labels: list or array for coloring (optional)
        title: plot title
        save_path: path to save figure
    """
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    if labels is not None:
        unique_labels = list(set(labels))
        for lbl in unique_labels:
            idxs = [i for i, l in enumerate(labels) if l == lbl]
            plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], label=str(lbl), alpha=0.6)
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def cluster_and_evaluate_embeddings(embeddings: np.ndarray, true_labels: list) -> float:
    """
    Cluster embeddings using GMM and compare to true labels.
    Args:
        embeddings: numpy array [n_samples, embedding_dim]
        true_labels: list of true label integers
    Returns:
        clustering accuracy
    """
    # Fit GMM with number of clusters = number of unique true labels or predefined
    n_clusters = len(set(true_labels))
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    predicted_labels = gmm.fit_predict(embeddings)

    # Map predicted clusters to true labels (unsupervised matching)
    from scipy.optimize import linear_sum_assignment
    contingency_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(n_clusters):
        for j in range(n_clusters):
            contingency_matrix[i, j] = np.sum((predicted_labels == i) & (np.array(true_labels) == j))
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    label_mapping = {row: col for row, col in zip(row_ind, col_ind)}
    predicted_mapped = [label_mapping[clust] for clust in predicted_labels]
    accuracy = np.mean([pred == true for pred, true in zip(predicted_mapped, true_labels)])
    return accuracy

# ---------------------------
# Helper Functions: Model Internal
# ---------------------------

def get_layerwise_attention_maps(model: nn.Module, sample_input: torch.Tensor,
                                 layer_idx: int, head_idx: int) -> np.ndarray:
    """
    Register a hook to extract attention weights from a specific layer and head.
    """
    attention_weights = []

    def hook(module, input, output):
        # output contains attention weights
        attn = output[1]  # shape: [batch, heads, seq_len, seq_len]
        attention_weights.append(attn.detach())

    handle = model.encoder.layers[layer_idx].self_attn.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        _ = model(sample_input)
    handle.remove()

    if attention_weights:
        attn = attention_weights[0]
        return attn[0, head_idx].cpu().numpy()
    else:
        return None

def generate_attention_overlay_on_ecg(input_ecg: np.ndarray, attention_scores: np.ndarray,
                                      lead_labels: list = None, save_path: str = None) -> None:
    """
    Overlay attention scores on the ECG waveform.
    """
    plt.figure(figsize=(12, 4))
    time = np.linspace(0, len(input_ecg)/250, len(input_ecg))
    plt.plot(time, input_ecg, label='ECG Signal')
    
    # Normalize attention scores for visualization
    scores_normalized = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
    for idx, score in enumerate(scores_normalized):
        if score > 0.6:
            plt.axvspan(time[idx*int(len(time)/len(attention_scores))],
                        time[(idx+1)*int(len(time)/len(attention_scores))],
                        color='red', alpha=score, label='High Attention' if idx == 0 else "")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Attention Overlay on ECG')
    if lead_labels:
        plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()
