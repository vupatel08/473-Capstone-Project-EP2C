## utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def bilinear_sample(feature_map: torch.Tensor, sampling_points: torch.Tensor) -> torch.Tensor:
    """
    Sample features from a feature map at given normalized sampling points using bilinear interpolation.
    
    Args:
        feature_map: Tensor of shape [B, C, H, W]
        sampling_points: Tensor of shape [N, P, 2], with values in [0,1], representing normalized (x, y) locations.
    
    Returns:
        sampled_features: Tensor of shape [N, P, C]
    """
    B, C, H, W = feature_map.shape
    N, P, _ = sampling_points.shape
    
    # Convert normalized coords to absolute pixel coordinates
    x = sampling_points[:,:,0] * (W - 1)
    y = sampling_points[:,:,1] * (H - 1)
    
    # Compute coordinates for the four neighbors
    x0 = x.floor().clamp(0, W - 1)
    y0 = y.floor().clamp(0, H - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    y1 = (y0 + 1).clamp(0, H - 1)
    
    # Gather pixel values at four corners
    # Expand dims for broadcasting
    B_idx = torch.arange(B, device=feature_map.device).view(-1, 1, 1)
    # For batch index, sample the same batch for all points
    # Reshape for gather
    def gather_pixel(x_idx, y_idx):
        """
        Gather pixel values at (x_idx, y_idx) locations for each batch.
        """
        # shape: [N, P]
        grid = torch.stack([y_idx, x_idx], dim=2)  # [N, P, 2]
        # Normalize to [-1,1] for grid_sample: not directly used here, so do direct indexing
        # Instead, do pixel gather:
        # flatten index
        flatten_idx = (grid[:,:,0] * W + grid[:,:,1]).long()  # [N, P]
        # Expand for batch dims
        pixel_vals = []
        for b in range(B):
            fmap = feature_map[b]
            fmap_flat = fmap.view(C, -1)  # [C, H*W]
            vals = fmap_flat[:, flatten_idx.view(-1)]  # [C, N*P]
            vals = vals.view(C, N, P).permute(1, 2, 0)  # [N, P, C]
            pixel_vals.append(vals)
        pixel_vals = torch.stack(pixel_vals, dim=0)  # [B, N, P, C]
        return pixel_vals
    
    # Gather pixel values for each neighbor
    Ia = gather_pixel(x0, y0)
    Ib = gather_pixel(x1, y0)
    Ic = gather_pixel(x0, y1)
    Id = gather_pixel(x1, y1)
    
    # Compute interpolation weights
    wx = (x - x0)
    wy = (y - y0)
    
    wx = wx.unsqueeze(-1)  # [N, P, 1]
    wy = wy.unsqueeze(-1)
    
    # Interpolate
    # shape: [B, N, P, C]
    sampled = (Ia * (1 - wx) * (1 - wy) +
               Ib * wx * (1 - wy) +
               Ic * (1 - wx) * wy +
               Id * wx * wy)
    return sampled

def generate_sampling_offsets(token_embedding: torch.Tensor, P: int, device: torch.device) -> torch.Tensor:
    """
    Generate relative sampling offsets conditioned on token embedding.
    
    Args:
        token_embedding: [N, D]
        P: int, number of sampling points
        device: torch device
    
    Returns:
        offsets: [N, P, 2], relative offsets
    """
    linear_layer = nn.Linear(token_embedding.shape[1], 2 * P).to(device)
    offsets = linear_layer(token_embedding)  # [N, 2*P]
    offsets = offsets.view(-1, P, 2)  # [N, P, 2]
    # Normalize offsets to roughly 3 std deviations as in paper
    # Here, just return raw; normalization can be applied externally
    return offsets

def convert_offsets_to_points(rois: torch.Tensor, rel_offsets: torch.Tensor) -> torch.Tensor:
    """
    Convert relative offsets conditioned on RoIs to absolute sampling points in normalized [0,1] coords.
    
    Args:
        rois: [N, 4], (x, y, w, h), normalized
        rel_offsets: [N, P, 2], (delta_x, delta_y), possibly normalized
    
    Returns:
        sampling_points: [N, P, 2], absolute locations in [0,1]
    """
    x, y, w, h = rois[:,0], rois[:,1], rois[:,2], rois[:,3]
    # Expand to shape [N, P]
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    w = w.unsqueeze(1)
    h = h.unsqueeze(1)
    delta_x = rel_offsets[:,:,0]
    delta_y = rel_offsets[:,:,1]
    # Compute absolute locations
    abs_x = x + 0.5 * delta_x * w
    abs_y = y + 0.5 * delta_y * h
    # Clamp to [0,1]
    abs_x = abs_x.clamp(0,1)
    abs_y = abs_y.clamp(0,1)
    sampling_points = torch.stack([abs_x, abs_y], dim=2)  # [N, P, 2]
    return sampling_points

def generate_roi_deltas(token_embedding: torch.Tensor) -> torch.Tensor:
    """
    Generate RoI adjustment deltas from token embedding.
    Following equation:
    (Δx, Δy, Δw, Δh) = Linear(t)
    """
    linear_layer = nn.Linear(token_embedding.shape[1], 4).to(token_embedding.device)
    delta = linear_layer(token_embedding)  # [N, 4]
    return delta

def update_rois(rois: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Update RoIs based on deltas using:
    x' = x + Δt_x * w
    y' = y + Δt_y * h
    w' = w * exp(Δt_w)
    h' = h * exp(Δt_h)
    
    Args:
        rois: [N, 4]
        delta: [N,4]
    Returns:
        new_rois: [N,4], clamped to [0,1]
    """
    x, y, w, h = rois[:,0], rois[:,1], rois[:,2], rois[:,3]
    Δx, Δy, Δw, Δh = delta[:,0], delta[:,1], delta[:,2], delta[:,3]
    x_new = x + Δx * w
    y_new = y + Δy * h
    w_new = w * torch.exp(Δw)
    h_new = h * torch.exp(Δh)
    # Clamp to [0,1]
    x_new = x_new.clamp(0,1)
    y_new = y_new.clamp(0,1)
    w_new = w_new.clamp(0,1)
    h_new = h_new.clamp(0,1)
    new_rois = torch.stack([x_new, y_new, w_new, h_new], dim=1)
    return new_rois

def initialize_weights(module: nn.Module, std: float = 0.02) -> None:
    """
    Initialize weights of linear, conv, and other layers with Xavier or normal.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'weight') and hasattr(module, 'bias'):
        if module.weight is not None:
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)

def get_lr_scheduler(optimizer: torch.optim.Optimizer, config: dict, total_steps: int) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build cosine warmup scheduler as per config.
    """
    warmup_epochs = config['training'].get('warmup_epochs', 5)
    def lr_lambda(current_step):
        warmup_steps = warmup_epochs * total_steps // config['training'].get('epochs', 1)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

def setup_logging():
    """
    Setup logging for training, including TensorBoard.
    """
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    return writer

def save_checkpoint(state: dict, filename: str):
    """
    Save model checkpoint.
    """
    torch.save(state, filename)

def load_checkpoint(filename: str) -> dict:
    """
    Load model checkpoint.
    """
    return torch.load(filename, map_location='cpu')
