# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import yaml

# Load configuration from 'config.yaml'
# Assuming this script is run in the same directory as the config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract dataset parameters from config
DATASET_PATH = config['dataset'].get('path', './')  # default fallback
TOKENIZER_TYPE = config['dataset'].get('tokenizer_type', 'vq-gan')  # default
TOKENIZER_NAME = config['dataset'].get('tokenizer_name', 'vgg-16')  # default
SEQ_LEN = config['dataset'].get('sequence_length', 1024)
NORMALIZATION = config['dataset'].get('normalization', True)

# Define the filename pattern based on the tokenizer
# This depends on how the dataset files are stored.
# For example, assume they are numpy files: '*.npy'
# and stored with consistent naming.
if TOKENIZER_TYPE == 'vq-gan':
    FILE_PATTERN = os.path.join(DATASET_PATH, '*.npy')  # discrete token IDs
elif TOKENIZER_TYPE == 'continuous':
    FILE_PATTERN = os.path.join(DATASET_PATH, '*.npy')  # latent vectors
else:
    raise ValueError(f"Unknown tokenizer_type: {TOKENIZER_TYPE}")

# Optional: Define padding token for discrete tokens
PAD_TOKEN_ID = 0

class ImageTokensDataset(Dataset):
    """
    Dataset class for loading tokenized images for autoregressive diffusion training.
    Supports both discrete token IDs (from VQ-GAN) and continuous latent vectors.
    """
    def __init__(self, dataset_path, tokenizer_type='vq-gan', sequence_length=1024,
                 normalization=True, shuffle_buffer_size=65536):
        """
        Args:
            dataset_path (str): Path to directory containing token files.
            tokenizer_type (str): Type of tokenizer ('vq-gan' or 'continuous').
            sequence_length (int): Fixed length of token sequences.
            normalization (bool): Whether to normalize continuous tokens.
            shuffle_buffer_size (int): Size for shuffling buffer.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.tokenizer_type = tokenizer_type
        self.sequence_length = sequence_length
        self.normalization = normalization
        self.shuffle_buffer_size = shuffle_buffer_size

        # List all data sample files
        self.samples = sorted(glob.glob(os.path.join(self.dataset_path, '*.npy')))
        assert len(self.samples) > 0, f"No data files found in {self.dataset_path}"

        # For shuffling, create a random permutation of indices
        self.indices = np.arange(len(self.samples))
        np.random.seed(42)
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load a single sample, process it, and return as tensor.
        """
        # Map idx through permutation for shuffling
        real_idx = self.indices[idx]
        filename = self.samples[real_idx]
        data = np.load(filename)

        # data shape depends on tokenizer
        # For 'vq-gan': data shape = (sequence_length,) of int IDs
        # For 'continuous': data shape = (sequence_length, D) of floats
        if self.tokenizer_type == 'vq-gan':
            tokens = data.astype(np.int64)  # ensure integer dtype
            # Pad if necessary
            if tokens.shape[0] < self.sequence_length:
                pad_length = self.sequence_length - tokens.shape[0]
                tokens = np.pad(tokens, (0, pad_length), constant_values=PAD_TOKEN_ID)
            elif tokens.shape[0] > self.sequence_length:
                tokens = tokens[:self.sequence_length]
            tokens_tensor = torch.from_numpy(tokens)
            # No normalization needed for discrete tokens
            return tokens_tensor

        elif self.tokenizer_type == 'continuous':
            # Assumed data shape = (sequence_length, D)
            # Possibly stored as float32
            if data.shape[0] < self.sequence_length:
                pad_length = self.sequence_length - data.shape[0]
                pad_vals = np.zeros((pad_length, data.shape[1]), dtype=data.dtype)
                data = np.concatenate([data, pad_vals], axis=0)
            elif data.shape[0] > self.sequence_length:
                data = data[:self.sequence_length]
            tokens_tensor = torch.from_numpy(data).float()

            # Normalize if flag is set
            if self.normalization:
                # Normalize to zero mean, unit variance
                mean = tokens_tensor.mean()
                std = tokens_tensor.std()
                if std > 0:
                    tokens_tensor = (tokens_tensor - mean) / std
                else:
                    tokens_tensor = tokens_tensor - mean
            return tokens_tensor
        else:
            raise ValueError(f"Unknown tokenizer_type: {self.tokenizer_type}")

# Optional: create a DataLoader for batching, shuffling
# You can instantiate in your training script as:
# dataset = ImageTokensDataset(DATASET_PATH, TOKENIZER_TYPE, SEQ_LEN, NORMALIZATION)
# dataloader = DataLoader(dataset, batch_size=..., shuffle=True, num_workers=4, drop_last=True)

# Example of how to instantiate:
# dataset = ImageTokensDataset(
#     dataset_path=DATASET_PATH,
#     tokenizer_type=TOKENIZER_TYPE,
#     sequence_length=SEQ_LEN,
#     normalization=NORMALIZATION
# )

# For external use, export the class
if __name__ == "__main__":
    # Test loading a sample
    dataset = ImageTokensDataset(DATASET_PATH, TOKENIZER_TYPE, SEQ_LEN, NORMALIZATION)
    print(f"Loaded {len(dataset)} samples.")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
```

## evaluation.py

```python
## evaluation.py
import os
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from torchvision.utils import save_image

from dataset_loader import ImageTokensDataset
from model import TransformerAutoRegressive, DiffusionMLP
from utils import (
    load_model,
    get_cosine_schedule,
    get_inception_score,
    compute_fid,
    extract_features,
    save_images,
    set_seed
)

# Load configuration from 'config.yaml'
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Set device and seed for reproducibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = cfg.get('seed', 42)
set_seed(seed)

# Load paths and parameters
dataset_params = cfg['dataset']
eval_batch_size = cfg['evaluation'].get('eval_batch_size', 512)
eval_steps = cfg['evaluation'].get('eval_steps', 10000)
fid_real_path = cfg['evaluation'].get('fid_real_dataset_path', None)
fid_lib = cfg['evaluation'].get('fid_scoring_lib', 'torch-fid')
sequence_length = dataset_params.get('sequence_length', 1024)
tokenizer_type = dataset_params.get('tokenizer_type', 'vq-gan')
checkpoint_dir = cfg['output_paths']['checkpoints_dir']
sample_dir = cfg['output_paths'].get('samples_dir', './samples')
results_dir = cfg['output_paths'].get('evaluation_results_dir', './eval')

# Create directories if needed
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Load trained models
# Assume checkpoint files are named 'autoregressive.pth' and 'denoising.pth'
ar_ckpt = os.path.join(checkpoint_dir, 'autoregressive.pth')
denoiser_ckpt = os.path.join(checkpoint_dir, 'denoising.pth')

# Instantiate models
# Load autoregressive transformer
ar_cfg = cfg['model']['transformer']
ar_model = load_model(TransformerAutoRegressive, ar_ckpt).to(device)
ar_model.eval()

# Load diffusion denoising network
denoiser_cfg = cfg['model']['diffusion_denoiser']
denoising_model = load_model(DiffusionMLP, denoiser_ckpt).to(device)
denoising_model.eval()

# Load evaluation dataset
dataset = ImageTokensDataset(
    dataset_path=dataset_params['path'],
    tokenizer_type=tokenizer_type,
    sequence_length=sequence_length,
    normalization=dataset_params.get('normalization', True)
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=eval_batch_size,
    shuffle=False,  # for reproducibility, shuffle can be False here
    num_workers=4,
    pin_memory=True,
    drop_last=False
)

# Load evaluation dataset images for real references (for FID)
# Assuming a function to load real images from tokens for FID
# and a separate folder to save generated images
# These can be assigned as needed; in practice, we should prepare real images for FID.
# For this code, we assume it's handled elsewhere or by torch-fid
# Similarly, for computing FID, real datasets are used directly.

# Load the pre-trained classifier (e.g., Inception) for IS and feature extraction
# For simplicity, we use utils.extract_features (assuming it's implemented)
# For FID, assuming torch-fid's functions are used

# Function to generate conditioning vectors for a batch
@torch.no_grad()
def generate_conditioning(x_batch):
    # x_batch: shape [B, L], LongTensor
    z_seq = ar_model(x_batch)
    return z_seq

# Function to run reverse diffusion conditioned on z^i
@torch.no_grad()
def run_reverse_diffusion(z_seq, init_noise=None):
    B, L, D = z_seq.shape
    # Initialize x with Gaussian noise
    if init_noise is None:
        x = torch.randn(B, L, D, device=device)
    else:
        x = init_noise.to(device)

    total_inference_steps = cfg['diffusion'].get('inference_steps', 100)
    schedule_type = cfg['diffusion'].get('schedule_type', 'cosine')
    schedule_params = cfg['diffusion'].get('noise_schedule_params', {})
    # Get schedule for t
    alpha_t, beta_t, sigma_t = get_cosine_schedule(total_inference_steps, schedule_type, **schedule_params)

    # Get inference timesteps: from T-1 down to 0
    inference_t = torch.linspace(total_inference_steps - 1, 0, steps=total_inference_steps).long()

    for t_idx in inference_t:
        t = torch.tensor([t_idx], device=device).float()
        t_batch = t.expand(B)
        # Predict noise
        epsilon_theta = denoising_model(x, t_batch, z_seq)
        alpha = alpha_t[t_idx].to(device)
        sigma = sigma_t[t_idx].to(device)
        # Calculate coefficient
        denom = alpha.sqrt()
        # Reverse diffusion
        x0_pred = (x - (1 - alpha).sqrt() * epsilon_theta) / denom
        mean_x_prev = (x - (1 - alpha) / (1 - alpha).sqrt() * epsilon_theta) / denom
        # Add noise scaled by sigma and temperature
        sigma_scaled = sigma * cfg['diffusion'].get('temperature', 1.0)
        delta = torch.randn_like(x) * sigma_scaled
        x = mean_x_prev + delta
    return x

# Main evaluation loop
generated_images = []
all_fid_preds = []
all_real_preds = []

for batch_idx, batch in enumerate(dataloader):
    if batch_idx * eval_batch_size >= eval_steps:
        break
    x_tokens = batch.to(device)  # shape [B, L], LongTensor or float
    B = x_tokens.shape[0]

    # Generate conditioning vectors
    z_seq = generate_conditioning(x_tokens)

    # Run reverse diffusion conditioned on z^i
    generated_vectors = run_reverse_diffusion(z_seq)

    # Decode vectors into images
    # Assuming a decoder function: decode_tokens
    # For placeholders, we use simple normalization or identity
    # If vectors are image features, decode accordingly
    # For this, suppose there exists a 'decode_tokens' function
    # For illustration, we treat vectors as features directly
    # Replace with actual decode if available
    generated_images_batch = None
    try:
        from utils import decode_tokens
        generated_images_batch = decode_tokens(generated_vectors)
    except ImportError:
        # fallback: if no decoder, just save raw vectors
        generated_images_batch = generated_vectors

    # Save images for FID
    save_image(
        (generated_images_batch + 1) / 2,  # normalize to [0,1] assuming input in [-1,1]
        os.path.join(sample_dir, f"generated_{batch_idx}.png"),
        normalize=True
    )
    generated_images.append(generated_images_batch)

# Concatenate all generated images
all_g_images = torch.cat(generated_images, dim=0)
# For real dataset images, load a subset for FID
# Let's assume a folder with real images is available and use torch-fid for FID
# If not, you need to extract real images from the dataset:
# For simplicity, user should provide real image folder path matching tokens
real_image_folder = fid_real_path
print("Computing FID...")
fid_score = compute_fid(real_image_folder, sample_dir, device=device, num_workers=4)

# Compute Inception Score
print("Computing Inception Score...")
is_mean, is_std = get_inception_score(sample_dir, device=device, batch_size=eval_batch_size)
# Compute other metrics: Precision, Recall
# Assume `extract_features` function computes features from images
print("Extracting features for metric analysis...")
gen_features = extract_features(all_g_images, model_name='inception_v3', device=device)
# For real features
# Assuming real images could be loaded similarly
real_images_for_metrics = None
try:
    real_images_for_metrics = torch.load(os.path.join(real_image_folder, 'images.pt')).to(device)
except:
    # skip or user provides precomputed features
    pass

# Placeholders for real features, replace with actual feature extraction
real_features = None
if real_images_for_metrics is not None:
    real_features = extract_features(real_images_for_metrics, model_name='inception_v3', device=device)

# Calculate precision and recall based on features
# For simplicity, assume functions compute_precision_recall exist
from utils import compute_precision_recall
precision, recall = compute_precision_recall(gen_features, real_features)

# Save metrics
results_path = os.path.join(results_dir, 'evaluation_metrics.txt')
with open(results_path, 'w') as f:
    f.write(f"FID: {fid_score:.4f}\n")
    f.write(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")

print(f"Evaluation completed. Results saved to {results_path}")
print(f"FID: {fid_score:.4f}")
print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
```

## main.py

```python
# main.py
import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np

from dataset_loader import ImageTokensDataset
from model import TransformerAutoRegressive, DiffusionMLP
from trainer import Trainer
from sampling import sample
from evaluation import evaluate
from utils import (
    get_cosine_schedule,
    load_model,
    save_checkpoint,
    set_seed
)

def main():
    # ==========================
    # 1. Argument parsing & Config
    # ==========================
    parser = argparse.ArgumentParser(description="Autoregressive Image Generation without VQ")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--mode', type=str, choices=['train', 'sample', 'eval'], default='train', help='Operation mode')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for loading models')
    parser.add_argument('--sample_num', type=int, default=4, help='Number of samples to generate in sample mode')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override seed if provided
    seed = args.seed if args.seed is not None else cfg.get('seed', 42)
    set_seed(seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================
    # 2. Setup output directories
    # ==========================
    os.makedirs(cfg['output_paths']['checkpoints_dir'], exist_ok=True)
    os.makedirs(cfg['output_paths']['sample_results_dir'], exist_ok=True)
    os.makedirs(cfg['output_paths']['evaluation_results_dir'], exist_ok=True)

    # ==========================
    # 3. Data loading
    # ==========================
    dataset_params = cfg['dataset']
    dataset = ImageTokensDataset(
        dataset_path=dataset_params['path'],
        tokenizer_type=dataset_params.get('tokenizer_type', 'vq-gan'),
        sequence_length=dataset_params.get('sequence_length', 1024),
        normalization=dataset_params.get('normalization', True),
        shuffle_buffer_size=dataset_params.get('shuffle_buffer_size', 65536)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg['training'].get('batch_size', 2048),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # ==========================
    # 4. Initialize models
    # ==========================
    transformer_cfg = cfg['model']['transformer']
    denoiser_cfg = cfg['model']['diffusion_denoiser']

    # Instantiate autoregressive transformer
    ar_model = TransformerAutoRegressive(transformer_cfg).to(device)

    # Instantiate diffusion denoising MLP
    denoising_model = DiffusionMLP(
        residual_blocks=denoiser_cfg['residual_blocks'],
        residual_width=denoiser_cfg['residual_width']
    ).to(device)

    # Optionally load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        load_model(ar_model, args.checkpoint + '_ar.pth')
        load_model(denoising_model, args.checkpoint + '_diff.pth')
    else:
        # Save initial weight for checkpoint
        pass

    # ==========================
    # 5. Diffusion schedule setup
    # ==========================
    diff_schedule_type = cfg['diffusion'].get('schedule_type', 'cosine')
    total_diffusion_steps = cfg['diffusion'].get('total_steps', 1000)
    inference_steps = cfg['diffusion'].get('inference_steps', 100)
    schedule_params = cfg['diffusion'].get('noise_schedule_params', {})
    temperature = cfg['diffusion'].get('temperature', 1.0)

    alpha_t, beta_t, sigma_t = get_cosine_schedule(
        total_diffusion_steps,
        schedule_type=diff_schedule_type,
        **schedule_params
    )

    # ==========================
    # 6. Setup optimizer & scheduler
    # ==========================
    optimizer = torch.optim.AdamW(
        list(ar_model.parameters()) + list(denoising_model.parameters()),
        lr=cfg['training'].get('learning_rate', 8e-4),
        betas=cfg['training']['optimizer_params'].get('betas', [0.9, 0.999]),
        weight_decay=cfg['training'].get('optimizer_params', {}).get('weight_decay', 0.02)
    )
    total_epochs = cfg['training'].get('epochs', 400)
    steps_per_epoch = cfg['training'].get('steps_per_epoch', len(dataloader))
    total_steps = total_epochs * steps_per_epoch
    lr_scheduler = get_cosine_schedule(optimizer, warmup_epochs=cfg['training'].get('warmup_epochs', 10),
                                         total_steps=total_steps)

    # ==========================
    # 7. Training, evaluation, sampling
    # ==========================
    if args.mode == 'train':
        print("Starting training...")
        trainer = Trainer(
            autoregressive_model=ar_model,
            denoising_model=denoising_model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            alpha_schedule=alpha_t,
            sigma_schedule=sigma_t,
            total_diffusion_steps=total_diffusion_steps,
            diffusion_schedule_type=diff_schedule_type,
            diffusion_schedule_params=schedule_params,
            device=device,
            save_dir=cfg['output_paths']['checkpoints_dir'],
            log_interval=100,
            max_epochs=total_epochs
        )
        trainer.train()
        print("Training finished.")
    elif args.mode == 'sample':
        print("Starting sampling...")
        # Load latest checkpoint if provided else assume training has been done
        load_model(ar_model, args.checkpoint + '_ar.pth' if args.checkpoint else None)
        load_model(denoising_model, args.checkpoint + '_diff.pth' if args.checkpoint else None)

        for i in range(args.sample_num):
            # Generate sample sequence
            decoded_vectors = sample(
                batch_size=1,
                sequence_length=dataset_params.get('sequence_length',1024),
                device=device,
                seed=seed,
                mode='auto',  # or 'masked' based on strategy
                inference_steps=inference_steps,
                temperature=temperature
            )
            # Decode to image
            # Assuming a decode_tokens method: replace as needed
            # For now, just save vectors or call custom decode
            # Example placeholder:
            # image = decode_tokens(decoded_vectors.squeeze(0))
            # save_image(image, os.path.join(cfg['output_paths']['sample_results_dir'], f'sample_{i}.png'))
            # Since decode_tokens is not provided, save vectors as numpy for now
            np.save(os.path.join(cfg['output_paths']['sample_results_dir'], f"sample_{i}.npy"), decoded_vectors.cpu().numpy())
        print("Sampling completed.")
    elif args.mode == 'eval':
        print("Starting evaluation...")
        # Load checkpoints
        load_model(ar_model, args.checkpoint + '_ar.pth' if args.checkpoint else None)
        load_model(denoising_model, args.checkpoint + '_diff.pth' if args.checkpoint else None)

        # Evaluate model with metrics: FID, IS, Precision, Recall
        results = evaluate(
            autoregressive_model=ar_model,
            denoising_model=denoising_model,
            dataset=dataset,
            device=device,
            eval_batch_size=cfg['evaluation'].get('eval_batch_size', 512),
            eval_steps=cfg['evaluation'].get('eval_steps', 10000),
            sample_results_dir=cfg['output_paths']['sample_results_dir'],
            real_dataset_path=cfg['evaluation'].get('fid_real_dataset_path', None)
        )
        # results is a dict with metrics
        # Save to file
        result_path = os.path.join(cfg['output_paths']['evaluation_results_dir'], 'evaluation_results.txt')
        with open(result_path, 'w') as f:
            for key, val in results.items():
                f.write(f"{key}: {val}\n")
        print(f"Evaluation results saved to {result_path}")
    else:
        print(f"Mode {args.mode} not recognized!")

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm

# Import from transformers for positional encodings if desired, or define custom
# Here, we implement sinusoidal positional embeddings manually
# No external dependencies besides torch

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timestep encoding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: [batch_size]
        device = t.device
        half_dim = self.dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, device=device).float() / half_dim
        )
        # shape: [batch_size, half_dim]
        args = t[:, None].float() * freq[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            # pad to match dimension
            embedding = F.pad(embedding, (0,1))
        return embedding  # shape: [batch_size, dim]

class TransformerAutoRegressive(nn.Module):
    """
    Transformer backbone for autoregressive modeling with support for causal and bidirectional attention.
    Generates conditioning vectors z^i for each token position.
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration with transformer hyperparameters.
                - num_layers
                - hidden_dim
                - num_heads
                - dropout_rate
                - max_sequence_length
        """
        super().__init__()
        # Extract parameters from config with defaults
        num_layers = config.get('num_layers', 32)
        hidden_dim = config.get('hidden_dim', 1024)
        num_heads = config.get('num_heads', 16)
        dropout = config.get('dropout_rate', 0.1)
        max_seq_len = config.get('max_sequence_length', 1024)
        # Embedding layer for token inputs if discrete
        # For continuous tokens, they can be passed directly as input features
        # For flexibility, assume input tokens are embedded in input tensor directly
        self.input_dim = config.get('input_dim', hidden_dim)
        # For token embedding: either identity or an embedding for discrete tokens
        self.is_discrete = config.get('is_discrete', True)
        vocab_size = config.get('vocab_size', 256)
        embed_dim = self.input_dim

        if self.is_discrete:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        else:
            # For continuous tokens, input is already in embedded form
            self.token_embedding = nn.Identity()

        # Positional embeddings (learned)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim)
        )

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)

        # For producing conditioning vector z^i from each token embedding
        # Here, simply use the transformer output at each position
        # Optionally, can define an MLP or linear projection
        self.z_proj = nn.Linear(embed_dim, config.get('z_dim', 1024))

        # Store attention mode: 'causal' or 'bidir'
        self.attention_mode = config.get('attention_mode', 'causal')  # default
        self.max_seq_len = max_seq_len

    def forward(self, input_tokens, attention_mask=None):
        """
        Args:
            input_tokens: LongTensor or FloatTensor of shape [batch_size, seq_len]
                - LongTensor if discrete tokens
                - FloatTensor if continuous tokens
            attention_mask: optional mask tensor [batch_size, seq_len], boolean
        Returns:
            z_seq: Tensor of shape [batch_size, seq_len, z_dim], conditioning vectors for each token
        """
        # Embed tokens
        if self.is_discrete:
            x = self.token_embedding(input_tokens)  # shape: [B, L, D]
        else:
            x = input_tokens  # assume already float embeddings

        # Add positional embedding
        seq_len = x.shape[1]
        pos_emb = self.pos_embedding[:, :seq_len, :]
        h = x + pos_emb

        # Create transformer mask based on mode
        # For causal: causal mask, for bidir: no mask or full attention
        if self.attention_mode == 'causal':
            # Generate causal mask
            device = h.device
            mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        elif self.attention_mode == 'bidir':
            mask = None  # no mask, full attention
        else:
            raise ValueError(f"Unknown attention_mode: {self.attention_mode}")

        # Run through transformer encoder
        # Note: transformer accepts attn_mask of shape [seq_len, seq_len]
        h_encoded = self.transformer(h.permute(1,0,2), mask=mask)  # [L, B, D]
        h_encoded = h_encoded.permute(1,0,2)  # [B, L, D]

        # Produce conditioning vectors z^i for each token
        z_seq = self.z_proj(h_encoded)  # shape: [B, L, z_dim]

        return z_seq  # conditioning vectors per token

class DiffusionMLP(nn.Module):
    """
    Small residual MLP for predicting noise in diffusion process.
    Incorporates timestep embedding and conditioning vector z^i.
    """
    def __init__(self, residual_blocks: int=3, residual_width: int=1024, input_dim: int=1024, z_dim: int=1024, time_emb_dim: int=1024):
        """
        Args:
            residual_blocks (int): Number of residual blocks.
            residual_width (int): Width of residual blocks.
            input_dim (int): Dim of input tokens (x_t).
            z_dim (int): Dim of conditioning vector z^i.
            time_emb_dim (int): Dim of timestep embedding.
        """
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.time_emb_dim = time_emb_dim

        # Map timestep embedding to initial residual input
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, residual_width),
            nn.SiLU()
        )

        # Build residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(residual_blocks):
            self.res_blocks.append(
                ResidualMLPBlock(residual_width)
            )

        # Final linear layer to predict noise
        self.output_layer = nn.Linear(residual_width, input_dim)

    def forward(self, x, t, z):
        """
        Args:
            x: Tensor [B, L, D], noisy tokens
            t: Tensor [B], timestep indices
            z: Tensor [B, L, z_dim], conditioning vectors per token
        Returns:
            epsilon_pred: Tensor [B, L, D], predicted noise
        """
        # Generate timestep embedding
        t_emb = SinusoidalPosEmb(self.time_emb_dim)(t)  # [B, time_emb_dim]
        t_emb = self.time_mlp(t_emb)  # [B, residual_width]
        t_emb = t_emb[:, None, :].expand(-1, x.shape[1], -1)  # [B, L, residual_width]

        # Concatenate or add t_emb and z to input
        # Here, we do addition after a linear projection
        # Alternatively, concatenate
        # Let's add t_emb and z (projected) for simplicity
        z_proj = nn.Linear(self.z_dim, x.shape[-1], bias=False).to(x.device)
        z_cond = z_proj(z)  # [B, L, D]
        # Option 1: Add
        h = x + z_cond + t_emb

        # Pass through residual blocks
        for res_block in self.res_blocks:
            h = res_block(h)

        # Final prediction
        epsilon_pred = self.output_layer(h)
        return epsilon_pred

class ResidualMLPBlock(nn.Module):
    """
    Residual block with LayerNorm, Linear, SiLU activation.
    """
    def __init__(self, width: int):
        super().__init__()
        self.norm1 = LayerNorm(width)
        self.linear1 = nn.Linear(width, width)
        self.norm2 = LayerNorm(width)
        self.linear2 = nn.Linear(width, width)
        self.activation = nn.SiLU()

        # Initialize weights with Xavier
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.norm2(out)
        out = self.linear2(out)
        return residual + out

class TimestepEmbedding(nn.Module):
    """
    Generate fixed sinusoidal embeddings for timestep t.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t shape: [B]
        device = t.device
        half_dim = self.dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, device=device).float() / half_dim
        )
        args = t[:, None].float() * freq[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding  # shape: [B, dim]

# Utility functions or classes for scheduling, positional encodings,
# and other helpers can be added here as needed.
```

## sampling.py

```python
"""
sampling.py

Implements reverse diffusion sampling for autoregressive image generation conditioned on previous tokens.
Supports both sequential and masked (parallel) generation modes, with temperature scaling.
Utilizes trained diffusion denoiser (`ε_θ`) and autoregressive transformer (`f`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import math
import os

from model import DiffusionMLP
from utils import load_model, get_cosine_schedule

# Load configuration
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config parameters
diffusion_cfg = cfg['diffusion']
inference_steps = diffusion_cfg.get('inference_steps', 100)
total_steps = diffusion_cfg.get('total_steps', 1000)
schedule_type = diffusion_cfg.get('schedule_type', 'cosine')
schedule_params = diffusion_cfg.get('noise_schedule_params', {})
temperature = diffusion_cfg.get('temperature', 1.0)

# Load trained models
# Assumes checkpoint paths are specified or set defaults
checkpoint_dir = cfg['output_paths']['checkpoints_dir']
denoising_ckpt = os.path.join(checkpoint_dir, 'denoising.pth')
model_ckpt = os.path.join(checkpoint_dir, 'autoregressive.pth')

# Load diffusion denoiser (ε_θ)
denoiser = load_model(DiffusionMLP, denoising_ckpt).to(device).eval()
# Load autoregressive transformer model (assumed to be in model.py)
# For inference, we only need to instantiate; load trained weights too
# For simplicity, assume the AR model is also loaded from checkpoint
# Here, we mock a generic function; replace as appropriate
from model import TransformerAutoRegressive
ar_model = load_model(TransformerAutoRegressive, model_ckpt).to(device).eval()

# Helper: diffusion schedule (αt, βt, σt) based on schedule_type
def get_schedule(total_steps, schedule_type='cosine', **kwargs):
    # For cosine schedule, we adopt similar to paper
    if schedule_type == 'cosine':
        s = kwargs.get('s', 0.008)
        t_vals = torch.linspace(0, 1, steps=total_steps + 1)
        alphas_bar = torch.cos((t_vals + s) / (1 + s) * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar.max()  # normalize to [0,1]
        alpha_bar_prev = alphas_bar[:-1]
        alpha_bar_next = alphas_bar[1:]
        alpha_t = alpha_bar_next / alpha_bar_prev
        beta_t = 1 - alpha_t
        # Compute standard deviations for reverse process
        sigmas = ((1 - alpha_t) / alpha_t).sqrt()
        return alpha_t, beta_t, sigmas
    else:
        # Implement other schedule types if needed
        raise NotImplementedError(f"Schedule type {schedule_type} not implemented.")

# Prepare schedule for inference steps
alpha_t, beta_t, sigma_t = get_schedule(
    total_steps,
    schedule_type=schedule_type,
    **schedule_params
)

# Select inference timesteps: evenly spaced from T to 0
def get_inference_timesteps(total_steps, inference_steps):
    return torch.linspace(total_steps - 1, 0, steps=inference_steps).long()

# Generate conditioning vectors for tokens
def generate_conditioning(x, ar_model, batch_size, seq_len, mode='auto'):
    """
    Generate conditioning vectors z^i for each token in sequence.
    Input:
      - x: current token sequence, shape [B, L], LongTensor
      - ar_model: autoregressive model to generate conditioning vectors
      - mode: 'auto' (sequential) or 'masked' (parallel masked)
    Output:
      - z_seq: conditioning tensors, shape [B, L, z_dim]
    """
    # For simplicity, assume ar_model can handle masked/sequences
    with torch.no_grad():
        z_seq = ar_model(x)  # shape [B, L, z_dim]
    return z_seq

# Main sampling function
def sample(
    batch_size=1,
    sequence_length=cfg['dataset'].get('sequence_length', 1024),
    device=device,
    seed=None,
    initial_x=None,                # Optional: starting noise, [B, L, D]
    mode='auto',                   # 'auto' (sequential) or 'masked'
    mask_ratio_start=1.0,          # start mask ratio for masked MAR
    mask_ratio_end=0.0,            # end mask ratio
    num_inference_steps=inference_steps,
    temperature=temperature,
    save_path=None                 # Optional: save generated image
):
    """
    Generates tokens and images using reverse diffusion conditioned on autoregressive model.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Initialize x_T
    if initial_x is not None:
        x = initial_x.to(device)
    else:
        # Start from Gaussian noise
        D = denoiser.input_dim if hasattr(denoiser, 'input_dim') else 1024
        x = torch.randn(batch_size, sequence_length, D, device=device)

    # Generate conditioning vectors z^i for the sequence
    # For masked MAR, generate once; for sequential, generate per token
    # For simplicity, here assume fully masked/multi-token predicted at once
    # It can be adapted into sequential generation with per-token ar_model calls
    # Example: for sequential, generate conditioning per timestep
    # Here, assuming full parallel generation for efficiency
    # Placeholder for sequence of known tokens: for initial step, no known tokens
    # Input tokens: set to zeros or special mask tokens.
    input_tokens = torch.zeros(batch_size, sequence_length, dtype=torch.long, device=device)
    # Generate conditioning vectors once
    z_seq = generate_conditioning(input_tokens, ar_model, batch_size, sequence_length, mode=mode)

    # Get inference timesteps
    inference_t = get_inference_timesteps(total_steps, num_inference_steps)

    # Run reverse diffusion
    for t_idx in inference_t:
        t = torch.tensor([t_idx], device=device).float()  # shape [1]
        # Expand to batch size
        t_batch = t.expand(batch_size)
        
        # Compute timestep embedding
        t_emb = get_timestep_embedding(t_batch, denoiser.time_emb_dim).to(device)  # [B, T_dim]
        
        # Predict noise
        epsilon_theta = denoiser(x, t_batch, z_seq)  # shape [B, L, D]
        
        # Get schedule values for current t
        alpha = alpha_t[t_idx].to(device)
        sigma = sigma_t[t_idx].to(device)
        
        # Compute coefficient terms
        alpha_sqrt = alpha.sqrt()
        one_minus_alpha_sqrt = (1 - alpha).sqrt()
        
        # Calculate the model mean x_{t-1}
        denom = alpha.sqrt()
        # Reverse diffusion formula
        pred_x0 = (x - (1 - alpha).sqrt() * epsilon_theta) / alpha.sqrt()
        
        # Compute mean of p(x_{t-1}|x_t, z)
        mean_x_prev = (x - (1 - alpha) / one_minus_alpha_sqrt * epsilon_theta) / denom
        
        # Add noise scaled by sigma and temperature
        # To incorporate temperature, scale the variance (sigma)
        scaled_sigma = sigma * temperature
        delta = torch.randn_like(x) * scaled_sigma
        
        # Sample next x
        x = mean_x_prev + delta

    # After final step, denoise to get final tokens
    # For discrete tokens: softmax / argmax
    # For continuous tokens: interpret directly
    # Here, assuming modeling of continuous tokens
    final_x = x  # shape [B, L, D]

    # Optional: decode tokens into images
    # Depending on tokenizer, either apply a decoder or interpret as latent vectors
    # Placeholder: if using vectors, decode via some decoder (not provided here)
    # For demonstration, convert to tokens by argmin of some similarity (if applicable)
    # or directly interpret as image features.
    # Assume we have a decoder function: decode_tokens(final_x) -> images
    # For now, just return final_x, or implement decode as needed.

    return final_x

def get_timestep_embedding(t, dim):
    """
    Create sinusoidal timestep embeddings, similar to positional encoding.
    t: tensor of shape [B]
    """
    half_dim = dim // 2
    freq = torch.exp(
        -math.log(10000) * torch.arange(0, half_dim, device=t.device).float() / half_dim
    )
    args = t[:, None] * freq[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding  # shape [B, dim]


# Example usage:
# final_tokens_or_vectors = sample(
#     batch_size=4,
#     sequence_length=cfg['dataset'].get('sequence_length', 1024),
#     seed=42,
#     mode='masked',  # or 'auto' for sequential
#     num_inference_steps=100,
#     temperature=1.0
# )
# # Post-process final tokens or vectors as needed, e.g., decode images
```

## trainer.py

```python
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
```

## utils.py

```python
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

```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\mar\mar_repo`
