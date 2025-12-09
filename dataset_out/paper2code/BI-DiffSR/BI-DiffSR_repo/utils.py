## utils.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
try:
    import lpips
except ImportError:
    lpips = None

from typing import Tuple, List
import math

# 1. Binarization Helpers

def sign_bin(x: torch.Tensor) -> torch.Tensor:
    """
    Binarize input tensor with sign function.
    Input:
        x (torch.Tensor): any real-valued tensor
    Output:
        torch.Tensor: +1 where x >= 0, -1 where x < 0
    """
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

class STESign(torch.autograd.Function):
    """
    Straight-Through Estimator for sign function.
    Forward pass: sign
    Backward pass: passes gradients unchanged (identity).
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return sign_bin(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        # Gradient is pass-through: just return grad_output
        return grad_output

def binarize_weight(weight: torch.Tensor, scale: bool = True) -> torch.Tensor:
    """
    Binarize weights with optional scaling.
    Args:
        weight (torch.Tensor): full-precision weight tensor
        scale (bool): whether to scale by mean absolute value
    Returns:
        torch.Tensor: binarized weight tensor
    """
    if scale:
        mean_abs = weight.abs().mean()
        w_binarized = sign_bin(weight) * mean_abs
    else:
        w_binarized = sign_bin(weight)
    return w_binarized

# 2. Activation Distribution Visualization

def plot_activation_distributions(activations: List[np.ndarray], timestep_indices: List[int], save_path: str = None):
    """
    Plot the activation distributions at various diffusion timesteps.
    Args:
        activations (List[np.ndarray]): list of activation arrays collected at different timesteps
        timestep_indices (List[int]): list of timestep indices corresponding to activations
        save_path (str): optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    for idx, act in zip(timestep_indices, activations):
        plt.hist(act.flatten(), bins=50, alpha=0.5, label=f'Timestep {idx}')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.title('Activation Distributions Across Timesteps')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# 3. Dataset Augmentation Methods

import torchvision.transforms as T

def random_crop(image: np.ndarray, crop_size: int = 64) -> np.ndarray:
    """
    Randomly crop a patch of size crop_size x crop_size from image.
    Args:
        image (np.ndarray): input image of shape (H, W, C)
        crop_size (int): size of the crop
    Returns:
        np.ndarray: cropped image patch
    """
    H, W, C = image.shape
    if H < crop_size or W < crop_size:
        raise ValueError("Image size smaller than crop size.")
    top = np.random.randint(0, H - crop_size + 1)
    left = np.random.randint(0, W - crop_size + 1)
    return image[top:top+crop_size, left:left+crop_size, :]

def random_flip(image: np.ndarray, flip_prob: float = 0.5) -> np.ndarray:
    """
    Random horizontal flip.
    Args:
        image (np.ndarray): input image
        flip_prob (float): probability to flip
    Returns:
        np.ndarray: flipped or original image
    """
    if np.random.rand() < flip_prob:
        return np.flip(image, axis=1)
    return image

def random_rotation(image: np.ndarray, rotations: List[int] = [90, 180, 270]) -> np.ndarray:
    """
    Randomly rotate image by one of the specified angles.
    Args:
        image (np.ndarray): input image
        rotations (List[int]): list of angles
    Returns:
        np.ndarray: rotated image
    """
    angle = np.random.choice(rotations)
    k = angle // 90
    return np.rot90(image, k)

def dataset_transform(image: np.ndarray, crop_size: int=64, flip: bool=True) -> np.ndarray:
    """
    Compose augmentation: random crop, flip, and rotation.
    Args:
        image (np.ndarray): input image
        crop_size (int): crop size
        flip (bool): whether to apply flip
    Returns:
        np.ndarray: augmented image
    """
    img = random_crop(image, crop_size)
    if flip:
        img = random_flip(img)
    img = random_rotation(img)
    return img

# 4. Metric Calculation Wrappers

def calculate_psnr(im_pred: np.ndarray, im_true: np.ndarray, border: int=0) -> float:
    """
    Compute PSNR between two images.
    Args:
        im_pred (np.ndarray): predicted image
        im_true (np.ndarray): ground truth image
        border (int): border width to exclude from computation
    Returns:
        float: PSNR value
    """
    if border > 0:
        im_pred = im_pred[border:-border, border:-border]
        im_true = im_true[border:-border, border:-border]
    mse = np.mean((im_pred - im_true) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / math.sqrt(mse))

def calculate_ssim(im_pred: np.ndarray, im_true: np.ndarray, border: int=0) -> float:
    """
    Compute SSIM between two images.
    Args:
        im_pred (np.ndarray): predicted image
        im_true (np.ndarray): ground truth image
        border (int): border width to exclude
    Returns:
        float: SSIM value
    """
    if border > 0:
        im_pred = im_pred[border:-border, border:-border]
        im_true = im_true[border:-border, border:-border]
    ssim_index = ssim(im_true, im_pred, data_range=1.0, gaussian=True)
    return ssim_index

def calculate_lpips(im_pred: np.ndarray, im_true: np.ndarray, device: str='cuda') -> float:
    """
    Calculate LPIPS perceptual metric.
    Args:
        im_pred (np.ndarray): predicted image in [0,1]
        im_true (np.ndarray): ground truth image in [0,1]
        device (str): 'cpu' or 'cuda'
    Returns:
        float: LPIPS score
    """
    if lpips is None:
        raise ImportError("LPIPS library is not installed.")
    # Convert to tensor [C, H, W], 3-channel normalized
    to_tensor = lambda img: torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device) * 2 - 1
    pred_tensor = to_tensor(im_pred)
    true_tensor = to_tensor(im_true)
    criterion = lpips.LPIPS(net='alex').to(device)
    with torch.no_grad():
        dist = criterion(pred_tensor, true_tensor)
    return dist.item()

# 5. Additional Utilities

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int=128) -> torch.Tensor:
    """
    Generate sinusoidal positional embeddings for diffusion timesteps.
    Args:
        timesteps (torch.Tensor): 1D tensor of shape (batch,)
        embedding_dim (int): dimension of the embedding
    Returns:
        torch.Tensor: tensor of shape (batch, embedding_dim)
    """
    device = timesteps.device
    half_dim = embedding_dim // 2
    exponent = torch.arange(half_dim, dtype=torch.float32, device=device) / float(half_dim)
    freqs = torch.exp(-np.log(10000) * exponent)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # shape (batch, half_dim)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if embedding_dim % 2 != 0:
        embedding = torch.cat([embedding, torch.zeros((embedding.shape[0],1), device=device)], dim=1)
    return embedding

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor [C, H, W] in [-1,1] to numpy image in [0,1].
    Args:
        tensor (torch.Tensor): image tensor
    Returns:
        np.ndarray: image in [0,1], shape (H, W, C)
    """
    tensor_clamped = torch.clamp(tensor, -1, 1)
    img = ((tensor_clamped + 1) / 2).cpu().numpy()
    return np.transpose(img, (1, 2, 0))

def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert numpy image [H, W, C] in [0,1] or [0,255] to tensor [-1,1].
    Args:
        image (np.ndarray): input image
    Returns:
        torch.Tensor: tensor in [-1,1], shape (C, H, W)
    """
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0)
    return tensor * 2 - 1

# 6. Gradient-compatible Binarization with STE (already defined as STESign class)

# 7. Error-Handling helper functions (added for robustness)

def check_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int]) -> None:
    """
    Utility to check if tensor shape matches expected.
    """
    if tensor.shape != expected_shape:
        raise ValueError(f"Tensor shape {tensor.shape} does not match expected {expected_shape}")

