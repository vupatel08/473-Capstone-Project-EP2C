## utils.py
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def get_positional_encoding(
    seq_len: int, 
    d_model: int, 
    learnable: bool = True
) -> torch.Tensor:
    """
    Generate positional encodings.
    Args:
        seq_len (int): Length of the sequence.
        d_model (int): Dimensionality of each position embedding.
        learnable (bool): If True, returns a learnable Parameter, else sinusoidal.
    Returns:
        torch.Tensor: Positional encoding tensor of shape (seq_len, d_model).
    """
    if learnable:
        # Initialize learnable positional embeddings
        pe = torch.nn.Parameter(torch.randn(seq_len, d_model))
        return pe
    else:
        # Sinusoidal positional encoding
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

def generate_mask(
    shape: Tuple[int, ...], 
    probability: float
) -> torch.Tensor:
    """
    Generate a binary mask tensor based on Bernoulli sampling.
    Args:
        shape (Tuple[int, ...]): Shape of the mask tensor.
        probability (float): Probability to set each element as 0 (mask).
    Returns:
        torch.Tensor: Binary mask tensor with 1 in unmasked, 0 in masked positions.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = torch.bernoulli(torch.full(shape, 1 - probability, device=device))
    # Convert to binary (0/1)
    mask = mask.clamp(0, 1)
    return mask

def apply_mask(
    tensor: torch.Tensor, 
    mask: torch.Tensor, 
    mask_value: float = 0.0
) -> torch.Tensor:
    """
    Apply a binary mask to a tensor element-wise.
    Args:
        tensor (torch.Tensor): Input tensor to mask.
        mask (torch.Tensor): Binary mask tensor (same shape as tensor).
        mask_value (float): Value to fill in masked positions.
    Returns:
        torch.Tensor: Masked tensor.
    """
    return tensor * mask + mask_value * (1 - mask)

def plot_attention_map(
    attention_scores: torch.Tensor, 
    title: str = ""
) -> None:
    """
    Plot attention heatmap for multi-head attention scores.
    Args:
        attention_scores (torch.Tensor): Attention scores shape (n_heads, seq_len_q, seq_len_k).
        title (str): Plot title.
    """
    import matplotlib.pyplot as plt
    n_heads, seq_q, seq_k = attention_scores.shape
    for h in range(n_heads):
        plt.figure(figsize=(6, 5))
        plt.imshow(attention_scores[h].detach().cpu(), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        plt.title(f"{title} - Head {h+1}")
        plt.show()

def plot_forecast_and_attention(
    forecast: np.ndarray, 
    input_seq: np.ndarray, 
    attention_map: np.ndarray, 
    title: str = ""
) -> None:
    """
    Visualize forecasted time series, input sequence, and attention map.
    Args:
        forecast (np.ndarray): Forecasted series (T,).
        input_seq (np.ndarray): Input historical series (L,).
        attention_map (np.ndarray): Attention weights (N_heads, T, L).
        title (str): Plot title.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(input_seq)), input_seq, label='Input Sequence')
    plt.plot(range(len(input_seq), len(input_seq)+len(forecast)), forecast, label='Forecast')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # Plot average attention scores if provided
    if attention_map is not None:
        avg_attention = np.mean(attention_map, axis=0)  # average over heads
        plt.figure(figsize=(8, 6))
        plt.imshow(avg_attention, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.xlabel('Input Sequence Positions')
        plt.ylabel('Forecast Steps')
        plt.title(f"{title} - Averaged Attention Map")
        plt.show()

def normalize_tensor(
    tensor: torch.Tensor, 
    method: str = "standard"
) -> torch.Tensor:
    """
    Normalize tensor via standardization or min-max scaling.
    Args:
        tensor (torch.Tensor): Tensor to normalize, shape (...).
        method (str): "standard" or "minmax".
    Returns:
        torch.Tensor: Normalized tensor.
    """
    if method == "standard":
        mean = tensor.mean()
        std = tensor.std()
        std = std if std > 0 else 1.0
        return (tensor - mean) / std
    elif method == "minmax":
        min_val = tensor.min()
        max_val = tensor.max()
        denom = max_val - min_val
        denom = denom if denom > 0 else 1.0
        return (tensor - min_val) / denom
    else:
        return tensor

def split_into_patches(
    sequence: torch.Tensor, 
    patch_size: int, 
    overlap: int = 0
) -> torch.Tensor:
    """
    Segment a sequence tensor into patches.
    Args:
        sequence (torch.Tensor): Input sequence of shape (L, D) or (B, L, D).
        patch_size (int): Length of each patch.
        overlap (int): Overlap length between patches.
    Returns:
        torch.Tensor: Tensor of shape (N_patches, patch_size, D).
    """
    seq_len = sequence.shape[0]
    stride = patch_size - overlap
    patches = []
    for start in range(0, seq_len - patch_size + 1, stride):
        patches.append(sequence[start:start+patch_size])
    return torch.stack(patches, dim=0)

def combine_patches(
    patches: torch.Tensor, 
    overlap: int = 0
) -> torch.Tensor:
    """
    Reconstruct sequence from patches by overlap-adding.
    Args:
        patches (torch.Tensor): Shape (N_patches, patch_size, D).
        overlap (int): Overlap length.
    Returns:
        torch.Tensor: Reconstructed sequence (L, D).
    """
    patch_size = patches.shape[1]
    stride = patch_size - overlap
    total_length = stride * (patches.shape[0] - 1) + patch_size
    D = patches.shape[2]
    sequence = torch.zeros((total_length, D), device=patches.device)
    count = torch.zeros((total_length, D), device=patches.device)
    for i, patch in enumerate(patches):
        start = i * stride
        sequence[start:start+patch_size] += patch
        count[start:start+patch_size] += 1
    return sequence / count

def create_horizon_queries(
    num_horizons: int, 
    embed_dim: int, 
    learnable: bool = True
) -> torch.Tensor:
    """
    Generate horizon-dependent query embeddings.
    Args:
        num_horizons (int): Number of forecast steps or horizons.
        embed_dim (int): Size of each query embedding.
        learnable (bool): Whether params are learnable.
    Returns:
        torch.Tensor: Tensor (num_horizons, embed_dim). If learnable, as Parameter.
    """
    if learnable:
        return torch.nn.Parameter(torch.randn(num_horizons, embed_dim))
    else:
        # Fixed or random initialization
        return torch.randn(num_horizons, embed_dim)

def init_parameters(
    tensor: torch.Tensor, 
    method: str = "xavier"
) -> None:
    """
    Initialize tensor parameters.
    Args:
        tensor (torch.Tensor): The tensor to initialize.
        method (str): Initialization method.
    """
    if method == "xavier":
        torch.nn.init.xavier_uniform_(tensor)
    elif method == "kaiming":
        torch.nn.init.kaiming_uniform_(tensor)
    elif method == "normal":
        torch.nn.init.normal_(tensor, mean=0, std=0.02)
    else:
        # default: do nothing
        pass
