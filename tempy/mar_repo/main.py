## main.py
import os
import sys
import yaml
import torch
import numpy as np
import copy
from tqdm import tqdm

# Import modules from project
import tokenizer_utils
from dataset_loader import Dataset
from model import TransformerModel, DiffusionHead
from trainer import Trainer
from sampling import Sampler
from evaluation import save_image, compute_fid, compute_inception_score

def main():
    # Load configuration from YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed for reproducibility
    seed = config.get('misc', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = config.get('misc', {}).get('use_mixed_precision', False)

    # Handle paths
    dataset_path = config['data']['dataset_path']
    val_dataset_path = config['data']['val_dataset_path']
    tokenizer_type = config['data'].get('tokenizer_type', 'VQ-16')
    tokenizer_path = config['data'].get('tokenizer_path', '')

    # Initialize tokenizer
    tokenizer = tokenizer_utils.get_tokenizer(tokenizer_type, tokenizer_path, device=device)
    tokenizer_mode = getattr(tokenizer, 'mode', 'discrete')  # 'discrete' or 'continuous'

    # Create Dataset and DataLoaders
    train_dataset = Dataset(
        image_dir=dataset_path,
        tokenizer_type=tokenizer_type,
        tokenizer_path=tokenizer_path,
        image_size=256,
        seq_length=1024,
        mode=tokenizer_mode,
        encode_on_the_fly=True,
        seed=seed
    )
    val_dataset = Dataset(
        image_dir=val_dataset_path,
        tokenizer_type=tokenizer_type,
        tokenizer_path=tokenizer_path,
        image_size=256,
        seq_length=1024,
        mode=tokenizer_mode,
        encode_on_the_fly=True,
        seed=seed+1
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['evaluation'].get('evaluation_batch_size', 128),
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model components
    # Decide on attention mode: causal for autoregressive training
    attention_mode = 'causal'  # or 'bidirectional' for MAR training
    transformer_cfg = config['model']['transformer']
    transformer = TransformerModel(
        num_layers=transformer_cfg['num_layers'],
        model_width=transformer_cfg['model_width'],
        attention_mode=attention_mode,
        positional_embedding=transformer_cfg.get('positional_embedding', 'sine')
    ).to(device)

    diffusion_cfg = config['model']['diffusion_head']
    diffusion_head = DiffusionHead(
        input_dim=diffusion_cfg['input_dim'],
        mlp_depth=diffusion_cfg.get('mlp_depth', 3),
        mlp_width=diffusion_cfg.get('mlp_width', 1024)
    ).to(device)

    # Diffusion schedule parameters
    num_timesteps = 1000  # training steps
    # Compute cosine schedule for alphas
    t_schedule = np.linspace(0, 1, num_timesteps)
    alphas_cumprod = np.cos((t_schedule + 0.5 * np.pi) / 2) ** 2
    alphas_cumprod = np.clip(alphas_cumprod, 1e-4, 1.0)
    alphas_tensor = torch.tensor(alphas_cumprod, dtype=torch.float32, device=device)

    # Function to get alpha_t and sigma_t given normalized t
    def get_alpha_sigma(t_norm):
        idx = min(int(t_norm * (num_timesteps - 1)), num_timesteps - 1)
        alpha_t = alphas_tensor[idx]
        sigma_t = torch.sqrt(1 - alpha_t)
        return alpha_t, sigma_t

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        list(transformer.parameters()) + list(diffusion_head.parameters()),
        lr=config['training']['learning_rate'],
        betas=tuple(config['training']['optimizer_params']['betas']),
        weight_decay=config['training']['weight_decay']
    )

    # EMA setup
    ema_decay = 0.9999
    ema_params = {}
    for name, param in transformer.named_parameters():
        ema_params[name] = param.data.clone()

    def update_ema():
        for name, param in transformer.named_parameters():
            ema_params[name] = ema_params[name] * ema_decay + param.data * (1 - ema_decay)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Check for existing checkpoint
    checkpoint_path = 'checkpoint.pt'
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        transformer.load_state_dict(checkpoint['transformer'])
        diffusion_head.load_state_dict(checkpoint['diffusion_head'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for name, param in transformer.named_parameters():
            param.data.copy_(checkpoint['ema_params'][name])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    # Decide mode: train or sample
    mode = 'train'  # change to 'sample' for inference after training
    total_epochs = config['training']['epochs']

    if mode == 'train':
        print("Starting training...")
        for epoch in range(start_epoch, total_epochs):
            epoch_loss = 0.0
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}") as pbar:
                for batch_idx, (images, tokens) in enumerate(pbar):
                    images = images.to(device)
                    B = images.shape[0]
                    # Tokens: shape [B, seq_len]
                    tokens = tokens.to(device)

                    # Get sequence conditions (z) from transformer outputs
                    transformer.train()
                    if attention_mode == 'causal':
                        transformer.set_attention_mode('causal')
                    else:
                        transformer.set_attention_mode('bidirectional')

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        out_feat = transformer(tokens)  # [B, seq_len, D]
                        z = out_feat  # conditioning vector per token

                    # Prepare for diffusion: sample t uniformly
                    t_vals = torch.randint(1, num_timesteps+1, (B * tokens.shape[1],), device=device).float()
                    t_norms = t_vals / num_timesteps

                    # Get alpha_t, sigma_t
                    alpha_ts = []
                    sigma_ts = []
                    for t in t_norms:
                        alpha_t, sigma_t = get_alpha_sigma(t.item())
                        alpha_ts.append(alpha_t)
                        sigma_ts.append(sigma_t)
                    alpha_ts = torch.stack(alpha_ts)
                    sigma_ts = torch.stack(sigma_ts)

                    # Reshape z to match tokens
                    z_flat = z.reshape(-1, z.shape[-1])  # [B*seq_len, D]
                    # Prepare x (ground truth tokens)
                    if tokenizer_mode == 'discrete':
                        x = tokens.reshape(-1).float()
                    else:
                        x = tokens.reshape(-1, tokens.shape[-1])  # continuous features

                    # Add noise
                    noise = torch.randn_like(x)
                    x_t = torch.empty_like(x)
                    for i in range(x.shape[0]):
                        a = alpha_ts[i]
                        s = sigma_ts[i]
                        x_t[i] = torch.sqrt(a) * x[i] + s * noise[i]

                    # Predict noise using diffusion head
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        noise_pred = diffusion_head(x_t, t_vals, z_flat)

                    # Compute diffusion loss (MSE)
                    loss_diff = F.mse_loss(noise_pred, noise)
                    loss = loss_diff

                    # Backpropagation
                    optimizer.zero_grad()
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    # EMA update
                    update_ema()

                    epoch_loss += loss.item()
                    pbar.set_postfix(loss=loss.item(), t_mean=t_vals.float().mean().item())

            print(f"Epoch {epoch+1} completed. Avg loss: {epoch_loss / len(train_loader):.4f}")

            # Save checkpoint
            if (epoch + 1) % 50 == 0 or (epoch + 1) == total_epochs:
                torch.save({
                    'transformer': transformer.state_dict(),
                    'diffusion_head': diffusion_head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ema_params': ema_params,
                    'epoch': epoch
                }, 'checkpoint.pt')
                print(f"Checkpoint saved at epoch {epoch+1}")

        # Save final model
        torch.save({
            'transformer': transformer.state_dict(),
            'diffusion_head': diffusion_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ema_params': ema_params
        }, 'final_model.pt')
        print("Training completed and saved as final_model.pt")
    else:
        # Inference mode
        # Load trained model weights
        if os.path.exists('final_model.pt'):
            checkpoint = torch.load('final_model.pt', map_location=device)
            transformer.load_state_dict(checkpoint['transformer'])
            diffusion_head.load_state_dict(checkpoint['diffusion_head'])
            for name, param in transformer.named_parameters():
                param.data.copy_(checkpoint['ema_params'][name])
            print("Loaded trained model for sampling.")
        else:
            print("No trained checkpoint found. Exiting.")
            sys.exit(1)

        # Generate images
        sampler = Sampler(transformer, diffusion_head, None, config['sampling'])

        # Generate single or multiple images
        generated_images = []
        num_samples = 1  # can be changed
        for _ in range(num_samples):
            img = sampler.generate_sequence()
            generated_images.append(img)

        # Save or display generated images
        output_dir = 'generated_images'
        os.makedirs(output_dir, exist_ok=True)
        for idx, img in enumerate(generated_images):
            save_path = os.path.join(output_dir, f'generated_{idx+1}.png')
            save_image(img, save_path)
            print(f"Saved generated image {save_path}")

        # Optional: evaluate generated images
        # Here, you can load images and compute FID, IS etc., using evaluation.py
        # For brevity, skip evaluation here

if __name__ == '__main__':
    main()

