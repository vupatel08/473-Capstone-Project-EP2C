## main.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import (
    get_timestep_embedding,
    sign_bin,
)
from dataset_loader import DatasetLoader
from model import UNet
from evaluation import evaluate_dataset
from sampling import ddim_sample

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract training parameters with defaults
    learning_rate = float(config['training'].get('learning_rate', 1e-4))
    batch_size = int(config['training'].get('batch_size', 16))
    total_iterations = int(config['training'].get('total_iterations', 10**6))
    iter_per_epoch = int(config['training'].get('iterations_per_epoch', 5000))
    crop_size = int(config['training'].get('image_crop_size', 64))
    augment_flip = bool(config['training'].get('augmentation', {}).get('flip', True))
    rotations = list(config['training'].get('augmentation', {}).get('rotations', [90,180,270]))

    # Diffusion parameters
    T = int(config['diffusion'].get('total_timesteps', 2000))
    inference_T = int(config['diffusion'].get('inference_timesteps', 50))
    eta = 0.0  # deterministic DDIM

    # Model hyperparameters
    channels = int(config['model'].get('channels', 64))
    encoder_levels = int(config['model'].get('encoder_levels', 4))
    res_blocks_per_level = int(config['model'].get('res_blocks_per_level', 2))
    decoder_res_blocks = int(config['model'].get('decoder_res_blocks',3))
    timestep_K = int(config['model'].get('timestep_encoding_K', 5))
    bias_pairs_num = int(config['model'].get('binarization', {}).get('bias_pairs', 5))
    scale_weights = bool(config['model'].get('binarization', {}).get('scale_weights', True))

    # Dataset paths: update these paths to your dataset locations
    dataset_paths = {
        'DIV2K': 'path_to_DIV2K/',     # <-- replace with actual paths
        'Flickr2K': 'path_to_Flickr2K/'
    }
    test_dataset_names = ['Set5', 'B100', 'Urban100', 'Manga109']

    # Instantiate DataLoader for training
    loader_obj = DatasetLoader(dataset_paths, batch_size)
    train_loader = loader_obj.get_train_loader()

    # Instantiate DataLoaders for test datasets
    test_loaders = {}
    for name in test_dataset_names:
        test_loaders[name] = loader_obj.get_test_loader(name)

    # Initialize the UNet model
    model_kwargs = {
        'channels': channels,
        'encoder_levels': encoder_levels,
        'res_blocks_per_level': res_blocks_per_level,
        'decoder_res_blocks': decoder_res_blocks,
        'total_timesteps': T,
        'timestep_encoding_K': timestep_K,
    }
    model = UNet(model_kwargs).to(device)
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    criterion = nn.L1Loss()

    # Diffusion schedule: betas, alphas, alpha_bars
    def get_diffusion_schedule(T):
        betas = np.linspace(0.0001, 0.02, T)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        return torch.tensor(betas, dtype=torch.float32).to(device), torch.tensor(alpha_bars, dtype=torch.float32).to(device)

    betas, alpha_bars = get_diffusion_schedule(T)

    # Optional: load checkpoint if resuming
    # ckpt_path = 'checkpoints/model_final.pth'
    # if os.path.exists(ckpt_path):
    #     model.load_state_dict(torch.load(ckpt_path))
    #     print(f"Loaded checkpoint {ckpt_path}")

    total_iters = total_iterations
    pbar = tqdm(total=total_iters)
    global_iter = 0

    # Training loop
    while global_iter < total_iters:
        for batch in train_loader:
            # Get data: HR and LR tensors
            hr_images = batch['HR'].to(device)    # shape: (B,C,H,W), [0,1]
            lr_images = batch['LR'].to(device)    # shape: (B,C,H,W), [0,1]

            B = hr_images.shape[0]
            # Sample random integer timestep t for each sample
            t = torch.randint(1, T+1, (B,), device=device).long()
            t_emb = get_timestep_embedding(t, channels).to(device)  # shape: (B, channels)

            # Add noise according to diffusion schedule
            epsilon = torch.randn_like(hr_images)
            alpha_t = alpha_bars[t-1].view(-1,1,1,1)  # shape: (B,1,1,1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            x_t = sqrt_alpha_t * hr_images + sqrt_one_minus_alpha_t * epsilon

            # Prepare model input: concatenate LR + noisy HR
            model_input = torch.cat([lr_images, x_t], dim=1)  # 6 channels

            # Forward pass
            pred_epsilon = model(model_input, t.float())  # model expects float t (same shape as t_emb)
            loss = criterion(pred_epsilon, epsilon)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)
            if global_iter % 1000 == 0:
                # Optional validation or logging
                pass
            # Save checkpoint periodically
            if global_iter > 0 and global_iter % 10000 == 0:
                save_dir = 'checkpoints'
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'model_{global_iter}.pth')
                torch.save(model.state_dict(), save_path)
            global_iter += 1
            if global_iter >= total_iters:
                break
    pbar.close()

    # Save final model (if not already saved)
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/model_final.pth')
    print("Training complete. Model saved.")

    # After training, optionally run inference/evaluation:
    # for name, loader in test_loaders.items():
    #     evaluate_dataset(loader, name, save_dir='results')  # or implement accordingly

if __name__ == "__main__":
    main()
