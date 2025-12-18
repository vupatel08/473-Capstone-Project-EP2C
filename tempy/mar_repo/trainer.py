## trainer.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from model import TransformerModel, DiffusionHead
from dataset_loader import ImagenetDataset
from tokenizer_utils import get_tokenizer
from evaluation import compute_fid, compute_inception_score

import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_mixed_precision = config.get('misc', {}).get('use_mixed_precision', False)

# Set random seed
seed = config.get('misc', {}).get('seed', 42)
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize tokenizer
tokenizer_type = config['data'].get('tokenizer_type', 'VQ-16')
tokenizer_path = config['data'].get('tokenizer_path', '')
tokenizer = get_tokenizer(tokenizer_type, tokenizer_path, device=device)
tokenizer_mode = getattr(tokenizer, 'mode', 'discrete')  # 'discrete' or 'continuous'

# Prepare dataset loader
train_dataset = ImagenetDataset(
    image_dir=config['data']['dataset_path'],
    tokenizer_type=tokenizer_type,
    tokenizer_path=tokenizer_path,
    image_size=config['data']['image_size'],
    seq_length=1024,
    mode=tokenizer_mode,
    encode_on_the_fly=True,
    seed=seed,
    max_samples=None
)
train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                          shuffle=True, num_workers=4, pin_memory=True)

# Initialize model components
# For simplicity, assume we use 'causal' mode by default; switch to 'bidirectional' for MAR
attention_mode = 'causal'  # or 'bidirectional', as needed
transformer = TransformerModel(
    num_layers=32,
    model_width=1024,
    attention_mode=attention_mode,
    positional_embedding='sine'
).to(device)

diffusion_head = DiffusionHead(
    input_dim=1024,
    mlp_depth=3,
    mlp_width=1024
).to(device)

# Diffusion schedule parameters
num_timesteps = 1000
# Use cosine schedule parameters (from DDPM)
t_schedule = np.arange(0, 1.0, 1.0 / num_timesteps)
alphas_cumprod = np.cos((t_schedule + 0.5 * math.pi) / 2.0) ** 2  # cosine schedule
alphas_cumprod = np.clip(alphas_cumprod, 0.0, 1.0)
alphas_schedule = torch.tensor(alphas_cumprod, dtype=torch.float32).to(device)

# Function to get alpha_t and sigma_t for a given timestep
def get_alpha_sigma(t):
    # t in [0, 1], normalized
    # Find closest index
    index = min(int(t * (num_timesteps - 1)), num_timesteps - 1)
    alpha_t = alphas_schedule[index]
    sigma_t = torch.sqrt(1 - alpha_t)
    return alpha_t, sigma_t

# Optimizer
optimizer = optim.AdamW(
    list(transformer.parameters()) + list(diffusion_head.parameters()),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay'],
    betas=tuple(config['training']['optimizer_params']['betas'])
)

# EMA for stable inference
ema_decay = 0.9999
ema_params = {}
for name, param in transformer.named_parameters():
    ema_params[name] = param.data.clone()

# Utility function for updating EMA
def update_ema():
    for name, param in transformer.named_parameters():
        ema_params[name] = ema_params[name] * ema_decay + param.data * (1 - ema_decay)

# Initialize mixed precision if needed
scaler = None
if use_mixed_precision:
    scaler = torch.cuda.amp.GradScaler()

# Training loop
epochs = config['training']['epochs']
batch_size = config['training']['batch_size']
steps_per_epoch = len(train_loader)
total_steps = epochs * steps_per_epoch

print("Starting training...")
global_step = 0

for epoch in range(epochs):
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch_idx, (images, tokens) in enumerate(pbar):
        images = images.to(device)  # [B,3,H,W]
        batch_size_curr = images.shape[0]

        # Encode images into tokens
        # tokens: shape [B, seq_len], dtype int or float depending on mode
        with torch.no_grad():
            encoded = []
            if tokenizer_mode == 'discrete':
                # output: [B, seq_len] of ints
                tokens_seq = tokens.to(device).long()
            else:
                # continuous features
                tokens_seq = tokens.to(device)  # shape [B, seq_len, feature_dim]

        # Generate conditioning vectors z for each token
        # For transformer, process sequence up to current token
        # Here, for simplicity, process entire sequence at once
        # Auto-regressive mode: causal attention
        transformer.set_attention_mode('causal')
        # Forward pass
        # tokens_seq shape: [B, seq_len]
        # For causal, input tokens are previous tokens, target is tokens at position i
        # But for diffusion, we need to predict x_i conditioned on previous
        # We process the whole sequence; loss computed on all
        # To simulate autoregression, process the sequence as is
        # Get sequence representations (z) for each token
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            out_features = transformer(tokens_seq)
            # out_features: [B, seq_len, D]
            # Use for diffusion head
            # For each token position i, produce z_i
            z = out_features  # [B, seq_len, D], conditioning per token
            # For simplicity, reshape z to [B * seq_len, D]
            z_flat = z.reshape(-1, z.shape[-1])

        # Diffusion process: for each token, sample t and corrupt
        # Sample t uniformly in [1, 1000]
        t_vals = torch.randint(1, num_timesteps+1, (batch_size_curr * tokens_seq.shape[1],), device=device).float()
        # Normalize t to [0, 1]
        t_norm = t_vals / num_timesteps

        # Get alpha_t and sigma_t for each t
        alpha_t_list = []
        sigma_t_list = []
        for t in t_norm:
            alpha_t, sigma_t = get_alpha_sigma(t.item()), torch.sqrt(1 - get_alpha_sigma(t.item()))
            alpha_t_list.append(alpha_t)
            sigma_t_list.append(sigma_t)
        alpha_t_tensor = torch.tensor(alpha_t_list, device=device)
        sigma_t_tensor = torch.tensor(sigma_t_list, device=device)

        # Prepare true tokens for corruption
        if tokenizer_mode == 'discrete':
            x = tokens_seq.reshape(-1).float()  # [B * seq_len]
        else:
            x = tokens_seq.reshape(-1, tokens_seq.shape[-1])  # [B * seq_len, feat_dim]

        # Add Gaussian noise
        noise = torch.randn_like(x)
        # x_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * epsilon
        x_t = torch.empty_like(x)
        for idx in range(x.shape[0]):
            a = alpha_t_tensor[idx]
            s = sigma_t_tensor[idx]
            x_t[idx] = torch.sqrt(a) * x[idx] + s * noise[idx]

        # Predict noise with diffusion head
        # For each token, input: x_t, t, z_i
        # Extract z_i: for token i, z_i is z at position i in sequence
        # reshape z appropriately
        z_cond = z.reshape(-1, z.shape[-1])  # already flattened
        # Run diffusion head
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            noise_pred = diffusion_head(x_t, t_vals, z_cond)

        # Compute diffusion loss (MSE between predicted noise and true noise)
        loss_diffusion = F.mse_loss(noise_pred, noise)

        # Total loss (could include var-lb term if desired)
        loss = loss_diffusion

        # Backpropagation
        optimizer.zero_grad()
        if use_mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient clipping
        clip_grad_norm = config['training'].get('clip_grad_norm', 0)
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(transformer.parameters()) + list(diffusion_head.parameters()),
                max_norm=clip_grad_norm
            )

        # Optimizer step
        if use_mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Update EMA
        update_ema()

        epoch_loss += loss.item()

        # Logging
        pbar.set_postfix(loss=loss.item(), diffusion_timestep=t_vals.mean().item())

        # Increment step
        global_step += 1

    # End of epoch: log average loss
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Avg diffusion loss: {avg_loss:.4f}")

    # Save checkpoint periodically
    if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
        save_path = f'checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'transformer': transformer.state_dict(),
            'diffusion_head': diffusion_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ema_params': ema_params,
            'epoch': epoch + 1
        }, save_path)
        print(f"Checkpoint saved at {save_path}")

# After training completes, save final models
torch.save({
    'transformer': transformer.state_dict(),
    'diffusion_head': diffusion_head.state_dict(),
    'ema_params': ema_params
}, 'final_model.pt')
print("Training completed and model saved.")

# --- Optional: Validation or Sample Generation can be added after training ---
