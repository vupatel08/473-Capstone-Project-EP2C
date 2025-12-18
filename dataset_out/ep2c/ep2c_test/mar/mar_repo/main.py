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
