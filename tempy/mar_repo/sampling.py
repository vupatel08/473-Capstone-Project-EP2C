## sampling.py
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.transforms import functional as TF
import base64
from io import BytesIO
from PIL import Image

import yaml
from model import TransformerModel, DiffusionHead
from tokenizer_utils import get_tokenizer
from dataset_loader import get_tokenizer  # if needed, or use from above
from evaluation import save_image, compute_fid  # Optional, for saving images and metrics

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Diffusion sampling parameters
diff_steps = config['sampling'].get('diffusion_steps', 100)  # default 100
temperature = config['sampling'].get('temperature', 1.0)
guidance_scale = config['sampling'].get('guidance_scale', 4.0)

# Model paths
checkpoint_path = 'final_model.pt'
tokenizer_type = config['data'].get('tokenizer_type', 'VQ-16')
tokenizer_path = config['data'].get('tokenizer_path', '')

# Load trained models
# Load the Transformer model
attention_mode_infer = 'causal'  # or 'bidirectional' for MAR
transformer = TransformerModel(
    num_layers=32,
    model_width=1024,
    attention_mode=attention_mode_infer,
    positional_embedding='sine'
).to(device)
# Load diffusion head
diffusion_head = DiffusionHead(
    input_dim=1024,
    mlp_depth=3,
    mlp_width=1024
).to(device)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
transformer.load_state_dict(checkpoint['transformer'])
diffusion_head.load_state_dict(checkpoint['diffusion_head'])

# Set models in eval mode
transformer.eval()
diffusion_head.eval()

# Load tokenizer
tokenizer = get_tokenizer(tokenizer_type, tokenizer_path, device=device)

# Diffusion schedule (cosine schedule)
# Precompute alpha_t, sqrt_alpha_t, sigma_t for diffusion steps
num_total_steps = 1000  # training steps
t_schedule = np.linspace(0, 1, num_total_steps)
alphas_cumprod = np.cos((t_schedule + 0.5 * math.pi) / 2) ** 2
alphas_cumprod = np.clip(alphas_cumprod, 0, 1)
alphas = torch.tensor(alphas_cumprod, dtype=torch.float32).to(device)

# Function to get alpha_t and sigma_t at timestep t in [0,1]
def get_alpha_sigma(t):
    idx = min(int(t * (num_total_steps - 1)), num_total_steps - 1)
    alpha_t = alphas[idx]
    sigma_t = torch.sqrt(1 - alpha_t)
    return alpha_t, sigma_t

# Reverse diffusion step (predict x_{t-1} given x_t)
def p_sample(x_t, t, z, guidance_scale=1.0):
    """
    Perform one reverse diffusion step.
    Args:
        x_t: current noisy sample (tensor)
        t: scalar timestep in [0,1]
        z: conditioning vector [batch, feature_dim]
        guidance_scale: scale for guidance (>=1)
    Returns:
        x_{t-1}
    """
    alpha_t, sigma_t = get_alpha_sigma(t)
    # Predict noise using diffusion head
    epsilon_theta = diffusion_head(x_t, torch.tensor([t], device=x_t.device), z)  # shape: same as x_t
    # Guidance scaling (if class-conditional guidance is used, not here)
    # In this implementation, guidance_scale is for rescaling epsilon_theta
    # But since there's no class guidance in sampling, skip and apply guidance to epsilon if needed
    # For simplicity, use guidance_scale to scale epsilon
    epsilon = epsilon_theta * guidance_scale
    # Compute mean of p(x_{t-1} | x_t)
    mean = (x_t - (1 - alpha_t).sqrt() * epsilon) / alpha_t.sqrt()
    # Sample noise for stochasticity
    noise = torch.randn_like(x_t)
    x_prev = mean + sigma_t * noise
    return x_prev

# Main sampling function
def generate_image(prompt=None):
    """
    Generate an image conditioned on prompt (if class label or seed tokens).
    Args:
        prompt: optional class label or seed tokens.
    Returns:
        PIL Image
    """
    # Initialization: start from pure noise
    seq_length = 1024  # as per dataset
    feature_dim = 1024  # model's feature dimension
    # Initialize x_T
    x_t = torch.randn(1, seq_length, feature_dim, device=device)
    # For guided condition, create condition vectors
    with torch.no_grad():
        # Generate the condition vector z from prompt if provided
        # For unconditional, we can set z to zeros
        # For class-conditional, implement if class info available
        # Here, assuming unconditional (no class conditioning)
        z = torch.zeros(1, feature_dim, device=device)

        # Reverse diffusion process
        for step in tqdm(reversed(range(diff_steps)), desc='Sampling', leave=False):
            t_norm = (step + 0.5) / diff_steps  # normalized t
            t_scalar = t_norm  # float in [0,1]
            # Run one step
            x_t = p_sample(x_t, t_scalar, z, guidance_scale=guidance_scale)
        
        # After diffusion steps, x_t should approximate a sample from p(x|z)
        # Map x_t (batch=1) into tokens using decoding
        # For input features, decode directly
        token_sequence = decode_features_to_tokens(x_t.squeeze(0))
        # Decode tokens into image
        img = tokenizer.decode(token_sequence)
        return img

def decode_features_to_tokens(features):
    """
    Given the continuous features at the final diffusion step, decode into tokens.
    Args:
        features: tensor [seq_len, feature_dim]
    Returns:
        list of token indices
    """
    # For simplicity, assume the features are already scaled appropriately
    # For discrete tokens: find nearest codebook indices
    # For the purpose of this code, we assume model was trained with discrete tokens
    # and the features are from the codebook lookup.
    # So, perform nearest neighbor search in codebook embedding space
    # Here, we mock this: assume features are codebook embeddings
    # and perform nearest neighbor search.
    # In practice, implement FAISS or similar library for efficiency.
    # For now, return a dummy sequence (e.g., zeros)
    # Replace with actual nearest neighbor search as needed.
    # Example:
    # codebook_embeddings = ... (from tokenizer's codebook)
    # distances = torch.cdist(features.unsqueeze(0), codebook_embeddings.weight)
    # indices = torch.argmin(distances, dim=2).squeeze(0)
    # return indices.tolist()

    # Placeholder:
    seq_len = features.shape[0]
    return [0] * seq_len

# For multiple images, add loop here:
def generate_multiple_images(n=1):
    images = []
    for _ in range(n):
        img = generate_image()
        images.append(img)
    return images

# Optional: Save result
def save_image(image, filename='generated.png'):
    image.save(filename)

# Main execution
if __name__ == "__main__":
    # Generate a sample image
    img = generate_image()
    save_image(img, 'sample_generated.png')
    print('Image saved to sample_generated.png')
