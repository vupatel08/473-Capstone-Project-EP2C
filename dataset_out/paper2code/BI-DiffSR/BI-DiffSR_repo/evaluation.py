## evaluation.py
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import torchvision.transforms.functional as TF

try:
    import lpips
except ImportError:
    lpips = None

from dataset_loader import DatasetLoader
from model import UNet
from utils import (
    get_timestep_embedding,
    tensor_to_image,
)

# Load configuration
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Extract evaluation parameters
scale_factor = config['evaluation'].get('scale', 2)
datasets_names = config['evaluation']['datasets']
metrics_list = config['evaluation'].get('metrics', ['PSNR', 'SSIM', 'LPIPS'])
inference_T = config['diffusion'].get('inference_timesteps', 50)
T_total = config['diffusion'].get('total_timesteps', 2000)

# Initialize test data loaders
dataset_paths = {}
# Assuming dataset_paths mapping is known; modify as needed
# e.g., dataset_paths = {'Set5':'path_to_Set5/', ...}
# Here, we set placeholders; replace with actual paths
for name in datasets_names:
    dataset_paths[name] = f'path_to_{name}'

loader_obj = DatasetLoader(dataset_paths, batch_size=1)
test_loaders = {}
for name in datasets_names:
    test_loaders[name] = loader_obj.get_test_loader(name)

# Initialize LPIPS if needed
if 'LPIPS' in metrics_list and lpips is not None:
    lpips_model = lpips.LPIPS(net='alex').to(device)
else:
    lpips_model = None

# Load model
model_config = {
    'channels': config['model'].get('channels', 64),
    'encoder_levels': config['model'].get('encoder_levels', 4),
    'res_blocks_per_level': config['model'].get('res_blocks_per_level', 2),
    'decoder_res_blocks': config['model'].get('decoder_res_blocks', 3),
    'total_timesteps': config['diffusion'].get('total_timesteps', 2000),
    'timestep_encoding_K': config['model'].get('timestep_encoding_K', 5),
}
model = UNet(model_config).to(device)
model_ckpt = 'checkpoints/model_final.pth'  # modify as needed
model.load_state_dict(torch.load(model_ckpt, map_location='cpu'))
model.eval()

# Diffusion schedule function (cosine schedule or linear as in training)
def get_diffusion_schedule(T):
    beta_start, beta_end = 0.0001, 0.02
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return torch.tensor(betas, dtype=torch.float32).to(device), torch.tensor(alpha_bars, dtype=torch.float32).to(device)

betas, alpha_bars = get_diffusion_schedule(T_total)

# Sampling function using DDIM
def ddim_sample(condition_lr, condition_hr_size, batch_size=1, seed=None):
    """
    condition_lr: tensor [B, 3, H, W], in [0,1]
    condition_hr_size: tuple (H, W)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    H, W = condition_hr_size
    # Initialize noisy x_T
    x_t = torch.randn(batch_size, 3, H, W, device=device)

    # Prepare scheduling
    total_T = T_total
    schedule = get_diffusion_schedule(total_T)

    # Create array of t_indices for inference steps
    t_steps = np.linspace(T_total-1, 0, inference_T, dtype=int)

    for t_idx in tqdm(t_steps, desc='Sampling'):
        t_b = torch.tensor([t_idx]*batch_size, device=device)
        t_emb = get_timestep_embedding(t_b, model_config['channels']).to(device)

        # Prepare model input: concatenate LR with current noisy HR
        model_input = torch.cat([condition_lr, x_t], dim=1)

        with torch.no_grad():
            epsilon_pred = model(model_input, t_b.float())

        alpha_t = schedule['alphas_cumprod'][t_idx]
        alpha_prev = schedule['alphas_cumprod'][t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=device)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        sqrt_alpha_prev = torch.sqrt(alpha_prev)

        # Compute predicted x_0
        x0_pred = (x_t - sqrt_one_minus_alpha_t * epsilon_pred) / sqrt_alpha_t

        # Compute mean of x_{t-1}
        mean_x_prev = x0_pred * torch.sqrt(alpha_prev) + torch.sqrt(1 - alpha_prev) * epsilon_pred

        # For stochastic sampling with eta>0 (not used here, for deterministic set eta=0)
        eta = 0.0
        if eta > 0:
            sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            noise = torch.randn_like(x_t)
            x_prev = mean_x_prev + sigma_t * noise
        else:
            x_prev = mean_x_prev

        x_t = x_prev

    return x_t

# Standardize data range for evaluation: images in [0,1]
def to_01(image_tensor):
    return torch.clamp((image_tensor + 1) / 2, 0, 1)

# Metrics functions
def compute_psnr(im_pred, im_true):
    im_pred_np = im_pred.astype(np.float32)
    im_true_np = im_true.astype(np.float32)
    return psnr_fn(im_true_np, im_pred_np, data_range=1.0)

def compute_ssim(im_pred, im_true):
    im_pred_np = im_pred.astype(np.float32)
    im_true_np = im_true.astype(np.float32)
    return ssim_fn(im_true_np, im_pred_np, data_range=1.0)

def compute_lpips(im_pred, im_true):
    if lpips_model is None:
        return None
    # expec images in [0,1], shape (H,W,C)
    im_pred_pt = torch.from_numpy(im_pred).permute(2,0,1).unsqueeze(0).to(device)*2 - 1
    im_true_pt = torch.from_numpy(im_true).permute(2,0,1).unsqueeze(0).to(device)*2 - 1
    with torch.no_grad():
        dist = lpips_model(im_pred_pt, im_true_pt)
    return dist.item()

# Main evaluation function
def evaluate_dataset(dataset_loader, dataset_name, save_dir='evaluation_results'):
    os.makedirs(save_dir, exist_ok=True)
    psnr_list = []
    ssim_list = []
    lpips_list = []

    for batch_idx, batch in enumerate(tqdm(dataset_loader, desc=f'Evaluating {dataset_name}')):
        with torch.no_grad():
            # batch['LR']: [1, 3, H, W], [0,1]
            # batch['HR']: [1, 3, H, W], [0,1]
            lr_img = batch['LR'].to(device)
            hr_img = batch['HR'].to(device)

            _, _, H, W = hr_img.shape

            # Sample SR image
            sr_img = ddim_sample(lr_img, (H, W), batch_size=1, seed=42+batch_idx)

            # Convert tensors to numpy images
            hr_np = tensor_to_image(hr_img.squeeze(0).cpu())
            sr_np = tensor_to_image(sr_img.squeeze(0).cpu())

            # Also get LR image for visualization
            lr_np = tensor_to_image(lr_img.squeeze(0).cpu())

            # Save images
            base_name = f"{dataset_name}_img{batch_idx}"
            TF.to_pil_image(hr_np).save(os.path.join(save_dir, f"{base_name}_HR.png"))
            TF.to_pil_image(sr_np).save(os.path.join(save_dir, f"{base_name}_SR.png"))
            TF.to_pil_image(lr_np).save(os.path.join(save_dir, f"{base_name}_LR.png"))

            # Prepare images in [0,1] for metric calculation
            hr_for_metrics = hr_np
            sr_for_metrics = sr_np

            # Extract luminance channel
            hr_y = rgb2y(hr_for_metrics)
            sr_y = rgb2y(sr_for_metrics)

            # Compute metrics
            p_val = compute_psnr(sr_y, hr_y)
            s_val = compute_ssim(sr_y, hr_y)
            if 'LPIPS' in metrics_list and lpips_model is not None:
                lp_val = compute_lpips(sr_for_metrics, hr_for_metrics)
            else:
                lp_val = None

            psnr_list.append(p_val)
            ssim_list.append(s_val)
            if lp_val is not None:
                lpips_list.append(lp_val)

    # Report averages
    print(f"Results for {dataset_name}:")
    print(f" PSNR: {np.mean(psnr_list):.4f} dB (+/- {np.std(psnr_list):.4f})")
    print(f" SSIM: {np.mean(ssim_list):.4f} (+/- {np.std(ssim_list):.4f})")
    if 'LPIPS' in metrics_list and lpips_model is not None:
        print(f" LPIPS: {np.mean(lpips_list):.4f} (+/- {np.std(lpips_list):.4f})")
    print("-"*50)

# Helper: RGB to Y (luminance) in [0,1]
def rgb2y(rgb_img):
    # rgb_img shape (H, W, 3), dtype float in [0,1]
    R = rgb_img[:,:,0]
    G = rgb_img[:,:,1]
    B = rgb_img[:,:,2]
    Y = 0.299*R + 0.587*G + 0.114*B
    return Y

# Run evaluation for all datasets
for dataset_name, loader in test_loaders.items():
    evaluate_dataset(loader, dataset_name)
