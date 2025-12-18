## utils.py
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

# Utility for setting random seeds for reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------
# 1. Diffusion Schedule Constructors
# -----------------------------------

def get_cosine_schedule(total_steps, schedule_type='cosine', schedule_params=None):
    """
    Create schedules of alpha, beta, and sigma based on a cosine schedule.
    Args:
        total_steps (int): total number of diffusion steps
        schedule_type (str): 'cosine' (default)
        schedule_params (dict): dict of parameters (e.g., 's' for cosine schedule)
    Returns:
        dict: containing tensors 'alpha', 'beta', 'alphabar', 'alphabar_sqrt', 'one_minus_alphabar_sqrt', 'sigmas'
    """
    if schedule_params is None:
        schedule_params = {'s': 0.008}
    s = schedule_params.get('s', 0.008)

    step_indices = torch.arange(0, total_steps + 1).float()
    t_vals = step_indices / total_steps  # in [0,1]

    # Cosine schedule for alpha_bar
    alphas_bar = torch.cos(((t_vals + s) / (1 + s)) * (math.pi / 2)) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]  # normalize to 1 at t=0

    # Compute alphas_t
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.clamp(alphas, max=1.0)  # avoid numerical issues
    betas = 1 - alphas
    betas = torch.clamp(betas, min=0.0001, max=0.9999)

    alpha = 1 - betas
    alpha_clipped = torch.clamp(alpha, min=0.0, max=1.0)

    alphas_cumprod = torch.cumprod(alpha_clipped, dim=0)

    # Schedule tensors
    alpha_tensor = alpha_clipped
    alphabar = alphas_cumprod
    alphabar_sqrt = torch.sqrt(alphabar)
    one_minus_alphabar_sqrt = torch.sqrt(1 - alphabar)
    
    # σ_t for reverse diffusion noise
    sigmas = torch.sqrt(betas)

    return {
        'alpha': alpha_tensor,
        'beta': betas,
        'alphabar': alphabar,
        'alphabar_sqrt': alphabar_sqrt,
        'one_minus_alphabar_sqrt': one_minus_alphabar_sqrt,
        'sigmas': sigmas
    }

# ----------------------------------------
# 2. Noise addition during training
# ----------------------------------------

def q_sample(x, t, noise, alpha_schedule):
    """
    Add noise to x at timestep t according to schedule.
    Args:
        x: Tensor (B, D), clean data (vector or token embedding)
        t: LongTensor (B,), diffusion timestep indices
        noise: Tensor (B, D), standard normal noise
        alpha_schedule: dict with 'alphabar', 'alphabar_sqrt', 'one_minus_alphabar_sqrt'
    Returns:
        x_t: noisy version of x
    """
    batch_size = x.shape[0]
    # Index schedule tensors for each t
    alphabar_sqrt = alpha_schedule['alphabar_sqrt']
    one_minus_alphabar_sqrt = alpha_schedule['one_minus_alphabar_sqrt']
    # Gather per sample
    sqrt_alphabar_t = alphabar_sqrt[t].to(x.device)  # shape [B]
    sqrt_one_minus_alphabar_t = one_minus_alphabar_sqrt[t].to(x.device)

    # Reshape to enable broadcasting
    sqrt_alphabar_t = sqrt_alphabar_t.unsqueeze(-1)  # (B,1)
    sqrt_one_minus_alphabar_t = sqrt_one_minus_alphabar_t.unsqueeze(-1)

    x_t = sqrt_alphabar_t * x + sqrt_one_minus_alphabar_t * noise
    return x_t

# ---------------------------------------------
# 3. Reverse diffusion step for sampling
# ---------------------------------------------

def p_sample(x_t, t, z, denoiser, alpha_schedule, temperature=1.0):
    """
    Perform one reverse diffusion step.
    Args:
        x_t: Tensor (B, D), noisy input at step t
        t: int, current timestep
        z: conditioning vector (B, L, z_dim)
        denoiser: neural network, ε_θ
        alpha_schedule: dict with 'alpha', 'sigmas'
        temperature: float, scaling the diffusion noise
    Returns:
        x_prev: Tensor (B, D), denoised tensor at step t-1
    """
    device = x_t.device
    alpha = alpha_schedule['alpha'][t].to(device)
    sigma = alpha_schedule['sigmas'][t].to(device)

    t_tensor = torch.tensor([t], device=device).float()
    epsilon_theta = denoiser(x_t, t_tensor, z)  # shape (B, D)

    # Compute predicted x_0
    denom = torch.sqrt(alpha)
    x0_pred = (x_t - torch.sqrt(1 - alpha) * epsilon_theta) / denom

    # Mean of the posterior p(x_{t-1}|x_t)
    mean_x_prev = (x_t - (1 - alpha) / torch.sqrt(1 - alpha) * epsilon_theta) / torch.sqrt(alpha)

    # Add scaled noise
    # scaled by temperature to control diversity
    sigma_step = sigma * temperature
    noise = torch.randn_like(x_t) * sigma_step
    x_prev = mean_x_prev + noise
    return x_prev

# -------------------------------------------------
# 4. Complete sequence reverse diffusion routine
# -------------------------------------------------

def run_reverse_diffusion(z_seq, denoiser, alpha_schedule, total_steps, inference_steps, temperature=1.0, seed=None):
    """
    Generate a sequence of tokens/vectors starting from Gaussian noise.
    Args:
        z_seq: Conditioning tensor (B, L, z_dim)
        denoiser: trained ε_θ network
        alpha_schedule: dict with schedule arrays
        total_steps: total number of timesteps for the schedule
        inference_steps: number of steps for inference (less than total)
        temperature: control randomness in sampling
        seed: optional, for reproducibility
    Returns:
        final_x: generated tensor (B, L, D)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    device = z_seq.device
    B, L, D = z_seq.shape
    
    # Initialize x with pure noise
    x = torch.randn(B, L, D, device=device)

    # Gather inference steps
    inference_t = torch.linspace(total_steps - 1, 0, steps=inference_steps).long()

    for t_idx in inference_t:
        t = int(t_idx.item())
        x = p_sample(x, t, z_seq, denoiser, alpha_schedule, temperature=temperature)

    return x

# -------------------------------------------
# 5. Helper functions for timestep embedding
# -------------------------------------------

def get_timestep_embedding(t, dim):
    """
    Create sinusoidal embeddings for timesteps.
    Args:
        t: tensor (B,) with timestep indices
        dim: int, embedding dimension
    Returns:
        embeddings: tensor (B, dim)
    """
    half_dim = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(0, half_dim, device=t.device, dtype=torch.float) / half_dim)
    args = t[:, None].float() * freq[None, :]  # (B, half_dim)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0,1))
    return embedding

# -------------------------------------------
# 6. Save/load checkpoint functions
# -------------------------------------------

def save_checkpoint(checkpoint_dir, epoch, autoregressive, denoising, optimizer, additional_state=None):
    """
    Save models and optimizer state to disk.
    Args:
        checkpoint_dir (str): directory to save checkpoints
        epoch (int): current epoch
        autoregressive: model
        denoising: model
        optimizer: optimizer
        additional_state: dict of extra info (optional)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'epoch': epoch,
        'autoregressive_state_dict': autoregressive.state_dict(),
        'denoising_state_dict': denoising.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    if additional_state is not None:
        state.update(additional_state)
    torch.save(state, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

def load_checkpoint(filepath, autoregressive=None, denoising=None, optimizer=None):
    """
    Load checkpoint and restore model & optimizer states.
    Args:
        filepath (str): path to checkpoint
        autoregressive: model to load weights into (optional)
        denoising: model to load weights into (optional)
        optimizer: optimizer to load state into (optional)
    Returns:
        dict: containing 'epoch' and any other info stored
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    if autoregressive is not None:
        autoregressive.load_state_dict(checkpoint['autoregressive_state_dict'])
    if denoising is not None:
        denoising.load_state_dict(checkpoint['denoising_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

# -------------------------------------------
# 7. Normalization functions
# -------------------------------------------

def normalize(x: torch.Tensor, eps: float=1e-8):
    """
    Normalize vectors per sample to unit L2 norm.
    """
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=eps)
    return x / norm

# -------------------------------------------
# 8. Miscellaneous numpy/timer utilities
# -------------------------------------------

# Placeholder for evaluation metrics: FID, Inception score, PR
# These typically rely on external libraries or torchvision models.
# Assume external functions are used, e.g., in evaluation.py
# The following can be simple wrappers or even pass-through if importing from elsewhere.

# Example:
# def compute_fid(real_path, gen_path, device, num_workers=4): ...
# def get_inception_score(image_dir, device, batch_size=..., num_samples=...)

# For clarity, provide no-op for now
def compute_fid(real_path, gen_path, device, num_workers=4):
    """
    Placeholder, assume external invocation (e.g., torch-fid).
    """
    raise NotImplementedError("Use external library `torch-fid` for FID computation.")

def get_inception_score(image_dir, device, batch_size=512, num_samples=10000):
    """
    Placeholder for IS calculation.
    """
    raise NotImplementedError("Use external implementation or torchmetrics for Inception Score.")

def extract_features(images, model_name='inception_v3', device='cpu'):
    """
    Extract features from images for metrics. Placeholder for actual feature extraction.
    """
    # Implement using torchvision models or external libraries
    raise NotImplementedError("Implement feature extraction with torchvision models.")

def save_images(images, filename):
    """
    Save tensor images to file. Placeholder using torchvision.utils
    """
    from torchvision.utils import save_image
    save_image(images, filename)

