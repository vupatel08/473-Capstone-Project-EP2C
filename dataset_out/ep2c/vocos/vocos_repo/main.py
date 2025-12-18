## main.py
import os
import yaml
import torch
import random
import numpy as np

from spectral_utils import SpectralUtils
from dataset_loader import DatasetLoader
from model import SpectralPredictor
from discriminator import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from trainer import Trainer
from evaluation import run_inference_and_evaluate

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create necessary directories for checkpoints and eval outputs
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
    eval_output_dir = config.get('evaluation', {}).get('save_dir', './evaluation_outputs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(eval_output_dir, exist_ok=True)

    # Initialize SpectralUtils with parameters from config
    spect_utils = SpectralUtils(
        sample_rate = config['dataset'].get('sample_rate', 24000),
        n_fft = config['model'].get('fft_size', 1024),
        hop_length = config['model'].get('hop_length', 256),
        n_mels = config['dataset']['mel_params'].get('n_mels', 100),
        window_type='hann'
    )

    # Initialize DatasetLoader
    dataset = DatasetLoader(config)

    # Initialize generator (SpectralPredictor)
    gen_params = config['model']
    generator = SpectralPredictor(gen_params).to(device)

    # Initialize discriminators
    D_mpd = MultiPeriodDiscriminator().to(device)
    D_mrd = MultiResolutionDiscriminator().to(device)

    # Setup optimizers
    lr = config['training'].get('learning_rate', 2e-4)
    betas = tuple(config['training'].get('optimizer_betas', [0.9, 0.999]))
    weight_decay = config['training'].get('AdamW_weight_decay', 0.01)

    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    d_optimizer_mpd = torch.optim.AdamW(D_mpd.parameters(), lr=lr, betas=betas)
    d_optimizer_mrd = torch.optim.AdamW(D_mrd.parameters(), lr=lr, betas=betas)

    # Training parameters
    total_iterations = int(config['training'].get('total_iterations', 2000000))
    save_interval = int(config['training'].get('save_interval', 100000))
    log_interval = int(config['training'].get('log_interval', 1000))
    # Loss weights
    lambda_adv = 1.0
    lambda_mel = 1.0
    lambda_feat = 10.0

    # Initialize Trainer
    trainer = Trainer({
        'generator': generator,
        'D_mpd': D_mpd,
        'D_mrd': D_mrd,
        'g_optimizer': g_optimizer,
        'd_optimizer_mpd': d_optimizer_mpd,
        'd_optimizer_mrd': d_optimizer_mrd,
        'spect_utils': spect_utils,
        'dataset': dataset,
        'config': config,
        'checkpoint_dir': checkpoint_dir,
        'total_iterations': total_iterations,
        'save_interval': save_interval,
        'log_interval': log_interval,
        'lambda_adv': lambda_adv,
        'lambda_mel': lambda_mel,
        'lambda_feat': lambda_feat,
        'device': device
    })

    # Check for existing checkpoints (optional, here start fresh)
    # If desired, add code to load latest checkpoint

    # Run training
    trainer.train()

    # After training, run evaluation (optional, here at the end)
    # We assume validation mel spectrograms and GT waveforms are prepared
    run_inference_and_evaluate(
        config.get('evaluation', {}).get('eval_mel_files', []),
        config.get('dataset', {}).get('ground_truth_waveforms', [])
    )

if __name__ == "__main__":
    main()
