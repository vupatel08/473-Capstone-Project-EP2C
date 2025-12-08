## trainer.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset_loader import DatasetLoader
from model import UNet
from utils import (
    get_timestep_embedding,
    binarize_weight,
    STESign,
    calculate_psnr,
    calculate_ssim,
    calculate_lpips
)

# Load configuration from yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Extract training params
lr = config['training'].get('learning_rate', 1e-4)
batch_size = config['training'].get('batch_size', 16)
total_iters = config['training'].get('total_iterations', 1_000_000)
iter_per_epoch = config['training'].get('iterations_per_epoch', 5000)
crop_size = config['training'].get('image_crop_size', 64)
aug_flip = config['training'].get('augmentation', {}).get('flip', True)
rotations = config['training'].get('augmentation', {}).get('rotations', [90,180,270])

# Diffusion parameters
T = config['diffusion'].get('total_timesteps', 2000)
inference_T = config['diffusion'].get('inference_timesteps', 50)

# Model params
channels = config['model'].get('channels', 64)
encoder_levels = config['model'].get('encoder_levels',4)
res_blocks_per_level = config['model'].get('res_blocks_per_level',2)
decoder_res_blocks = config['model'].get('decoder_res_blocks',3)
timesteps_K = config['model'].get('timestep_encoding_K',5)

# Binarization configs
bias_pairs_num = config['model'].get('binarization', {}).get('bias_pairs',5)
scale_weights = config['model'].get('binarization', {}).get('scale_weights', True)

# Initialize datasets and dataloaders
dataset_paths = {
    'DIV2K': 'path_to_DIV2K/',  # update path as needed
    'Flickr2K': 'path_to_Flickr2K/'
}
# Load data
data_loader_obj = DatasetLoader(dataset_paths, batch_size)
train_loader = data_loader_obj.get_train_loader()

# Placeholder for test loaders: load on-demand
test_loaders = {}
for test_name in ['Set5', 'B100', 'Urban100', 'Manga109']:
    test_loaders[test_name] = data_loader_obj.get_test_loader(test_name)

# Initialize model
model = UNet({
    'channels': channels,
    'encoder_levels': encoder_levels,
    'res_blocks_per_level': res_blocks_per_level,
    'decoder_res_blocks': decoder_res_blocks,
    'total_timesteps': T,
    'timestep_encoding_K': timesteps_K
}).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
# Use L1 loss for diffusion noise prediction
criterion = nn.L1Loss()

# Function for cosine schedule for beta (diffusion)
def get_diffusion_coeffs(T):
    # Use cosine schedule for alpha_bar for simplicity
    betas = np.linspace(0.0001, 0.02, T)  # or implement cosine schedule if desired
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    return torch.tensor(betas, dtype=torch.float32), torch.tensor(alpha_bars, dtype=torch.float32)

betas, alpha_bars = get_diffusion_coeffs(T)
betas = betas.to(device)
alpha_bars = alpha_bars.to(device)

# Training loop
global_iter = 0
pbar = tqdm(total=total_iters, desc='Training')
while global_iter < total_iters:
    for batch in train_loader:
        # Get high-res images (normalized to [0,1])
        hr_images: torch.Tensor = batch['HR'].to(device)  # shape: (B,C,H,W)
        lr_images: torch.Tensor = batch['LR'].to(device)
        B = hr_images.shape[0]

        # Sample random timestep t for each sample in batch
        t = torch.randint(1, T+1, (B,), device=device).long()
        t_emb = get_timestep_embedding(t, channels)  # shape: (B, channels)

        # Add noise according to diffusion schedule
        # Sample epsilon ~ N(0,1)
        epsilon = torch.randn_like(hr_images)
        # Gather alpha_bar for t
        alpha_bar_t = alpha_bars[t-1].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # shape: (B,1,1,1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
        x_t = sqrt_alpha_bar * hr_images + sqrt_one_minus_alpha_bar * epsilon

        # Prepare model input: concatenate LR condition + noisy HR
        # Concatenate along channel: [LR, x_t] with 3 channels each -> total 6 channels
        model_input = torch.cat([lr_images, x_t], dim=1)

        # Forward pass
        pred_noise = model(model_input, t.float())  # model expects float t (or int, but converted to float inside)
        # Compute loss
        loss = criterion(pred_noise, epsilon)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        pbar.update(1)
        if global_iter % 1000 == 0:
            # Optional: evaluate on a small batch or subset
            # For simplicity, skip validation during training here
            pass

        # Save checkpoints periodically
        if global_iter % 10000 == 0:
            save_path = f'checkpoints/model_{global_iter}.pth'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        global_iter +=1
        if global_iter >= total_iters:
            break
pbar.close()

# Save final model
torch.save(model.state_dict(), 'checkpoints/model_final.pth')

# --- Optional: Save activation distribution visualization, or validation style evaluations ---

print("Training complete. Model saved.")

# If desired, add inference functions or validation scripts below
# For example, run inferences on test datasets with DDIM sampling, compute PSNR, SSIM, LPIPS, etc.
# This would be implemented in separate scripts or a test/evaluate function.
