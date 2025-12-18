## trainer.py
import os
import math
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from dataset_loader import ImageTokensDataset
from model import TransformerAutoRegressive, DiffusionMLP
from utils import get_cosine_schedule, q_sample, save_checkpoint, load_checkpoint

# Load configuration
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fix seed for reproducibility
seed = cfg.get('seed', 42)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Dataset and DataLoader
dataset_params = cfg['dataset']
dataset_path = dataset_params['path']
tokenizer_type = dataset_params.get('tokenizer_type', 'vq-gan')
sequence_length = dataset_params.get('sequence_length', 1024)
normalization = dataset_params.get('normalization', True)

dataset = ImageTokensDataset(
    dataset_path=dataset_path,
    tokenizer_type=tokenizer_type,
    sequence_length=sequence_length,
    normalization=normalization
)
dataloader = DataLoader(
    dataset,
    batch_size=cfg['training'].get('batch_size', 2048),
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

# Model hyperparameters
transformer_cfg = cfg['model']['transformer']
denoiser_cfg = cfg['model']['diffusion_denoiser']

# Instantiate models
autoregressive_model = TransformerAutoRegressive(transformer_cfg).to(device)
denoising_model = DiffusionMLP(
    residual_blocks=denoiser_cfg['residual_blocks'],
    residual_width=denoiser_cfg['residual_width']
).to(device)

# Optimizer parameters
lr = cfg['training'].get('learning_rate', 8e-4)
optimizer_params = {
    'lr': lr,
    'weight_decay': cfg['training'].get('optimizer_params', {}).get('weight_decay', 0.02),
    'betas': cfg['training'].get('optimizer_params', {}).get('betas', [0.9, 0.999])
}
optimizer = optim.AdamW(
    list(autoregressive_model.parameters()) + list(denoising_model.parameters()),
    **optimizer_params
)

# Scheduler setup: linear warmup + cosine decay
warmup_epochs = cfg['training'].get('warmup_epochs', 10)
epochs = cfg['training'].get('epochs', 400)
steps_per_epoch = cfg['training'].get('steps_per_epoch', 2500)
total_training_steps = steps_per_epoch * epochs

lr_schedule = get_cosine_schedule(
    optimizer=optimizer,
    warmup_steps=warmup_epochs * steps_per_epoch,
    total_steps=total_training_steps
)

# Diffusion schedule parameters
diff_schedule_type = cfg['diffusion'].get('schedule_type', 'cosine')
total_diffusion_steps = cfg['diffusion'].get('total_steps', 1000)
inference_steps = cfg['diffusion'].get('inference_steps', 100)
temperature = cfg['diffusion'].get('temperature', 1.0)

# For simplicity, assume get_cosine_schedule returns a step-dependent learning rate
# Set to optimizer: optimizer.step(), lr_scheduler.step()

# Optional: load checkpoint
start_epoch = 0
checkpoint_path = cfg.get('checkpoint_path', None)
if checkpoint_path and os.path.exists(checkpoint_path):
    load_checkpoint(checkpoint_path, autoregressive_model, denoising_model, optimizer)
    # Might also load epoch info

# Training Loop
global_step = 0
for epoch in range(start_epoch, epochs):
    autoregressive_model.train()
    denoising_model.train()

    epoch_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        # Prepare input: batch shape: [B, L, D] or [B, L]
        x = batch.to(device)  # tokens or vectors

        B, L = x.shape[0], x.shape[1]
        # Generate conditioning vectors z^i using transformer
        # Input previous tokens: for autoregression, teacher-forcing uses ground truth
        # Compute z^i for each sequence: shape [B, L, z_dim]
        z_seq = autoregressive_model(x)

        # Initialize diffusion variables
        # For each token in batch, sample t uniformly
        t_indices = torch.randint(0, total_diffusion_steps, (B, L), device=device).long()
        # For each token, get alpha_bar (cumulative noise) from schedule
        alpha_bar = get_cosine_schedule(t_indices, total_diffusion_steps)

        # Add Gaussian noise to tokens to get x_t
        # For discrete tokens, treat as such; for vectors, treat as continuous
        # Here, we implement for continuous vectors; for discrete, one might embed
        epsilon = torch.randn_like(x)
        x_t = q_sample(x, t_indices, epsilon, alpha_schedule=alpha_bar)

        # Prepare inputs for denoising model
        # Conditioned on z^i and timestep t
        t_current = t_indices.float()
        # For time embedding
        t_emb = SinusoidalPosEmb(cfg['model'].get('diffusion_denoiser', {}).get('time_emb_dim', 1024))(t_current)
        # Predict noise ε_θ
        epsilon_pred = denoising_model(x_t, t_current, z_seq)

        # Compute diffusion loss (MSE)
        # As in paper, loss scaled per timestep
        loss = torch.mean((epsilon_pred - epsilon) ** 2)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(list(autoregressive_model.parameters()) + list(denoising_model.parameters()), max_norm=cfg['training'].get('gradient_clip_norm', 1.0))
        optimizer.step()
        lr_schedule.step()
        global_step += 1

        epoch_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f} LR: {optimizer.param_groups[0]['lr']:.6f}")

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")

    # Save checkpoint periodically
    if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
        save_checkpoint(
            checkpoint_dir=cfg['output_paths']['checkpoints_dir'],
            epoch=epoch+1,
            autoregressive_model=autoregressive_model,
            denoising_model=denoising_model,
            optimizer=optimizer,
            global_step=global_step
        )

    # Optionally run validation, sample generation, or evaluation here

# End of training
print("Training completed.")
