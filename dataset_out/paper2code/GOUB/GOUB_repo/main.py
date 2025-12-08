## main.py
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

# Import custom modules
import schedule_utils
from dataset_loader import DatasetLoader
from model import ScoreUNet
from trainer import DiffusionTrainer
from sampling import Sampler
from evaluation import Evaluation

def main():
    # --- 1. Load configuration from YAML ---
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- 2. Set device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 3. Generate schedule arrays ---
    schedule_type = config.get('schedule', {}).get('schedule_type', 'cosine')
    schedule_steps = config.get('schedule', {}).get('steps', 100)
    t_array = schedule_utils.get_time_schedule(T=1.0, N=schedule_steps)  # times from 0 to 1
    theta_array = schedule_utils.compute_theta(t_array, schedule_type)
    cum_theta = schedule_utils.compute_cum_theta(theta_array, t_array)
    g_array = schedule_utils.compute_g(t_array, theta_array, lambda_sq=30)
    sigma_sq = schedule_utils.compute_sigma_t(t_array, theta_array, cum_theta, lambda_sq=30)
    sigma_t_T = schedule_utils.compute_sigma_t_T(t_array, cum_theta, T=1.0, lambda_sq=30)

    # Convert schedule arrays to tensors for batch indexing
    schedule_params = {
        'theta': torch.tensor(theta_array, dtype=torch.float32),
        'cum_theta': torch.tensor(cum_theta, dtype=torch.float32),
        'g': torch.tensor(g_array, dtype=torch.float32),
        'sigma': torch.tensor(sigma_sq, dtype=torch.float32),
        'sigma_t_T': torch.tensor(sigma_t_T, dtype=torch.float32),
    }

    # --- 4. Load Dataset ---
    dataset_path = config.get('dataset', {}).get('root_path', './dataset')
    dataset_type = config.get('dataset', {}).get('type', 'inpainting')  # default to inpainting
    dataset_mode = 'train'  # training mode
    dataset = DatasetLoader(dataset_path, batch_size=config['training'].get('batch_size',8),
                            mode=dataset_mode, dataset_type=dataset_type,
                            image_size=128)
    dataloader = torch.utils.data.DataLoader(dataset,
                                           batch_size=config['training'].get('batch_size',8),
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=True)

    # --- 5. Initialize Model ---
    model_params = {
        'in_channels':3,
        'base_channels':64,
        'depth':4,
        'use_self_attention':False
    }
    model = ScoreUNet(**model_params).to(device)

    # --- 6. Set optimizer and scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training'].get('learning_rate',1e-4))
    total_steps = config['training'].get('total_steps',900_000)
    lr_decay_steps = config['training'].get('lr_decay_steps',[300_000, 500_000, 600_000, 700_000])

    def lr_lambda(step):
        factor = 1.0
        for decay in lr_decay_steps:
            if step >= decay:
                factor *= 0.5
        return factor
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- 7. Initialize Trainer ---
    trainer = DiffusionTrainer({
        'model': model,
        'optimizer': optimizer,
        'schedule_params': schedule_params,
        'dataloader': dataloader,
        'total_steps': total_steps,
        'lr_scheduler': scheduler,
        'device': device,
        'schedule_type': schedule_type,
        'schedule_steps': schedule_steps,
        'use_mean_ode': True,  # Always use deterministic Mean-ODE for inference
        'lambda_sq': 30
    })

    # --- 8. Training Loop ---
    print("Starting training...")
    pbar = tqdm(total=total_steps)
    for step in range(total_steps):
        # Get batch:
        try:
            batch = next(trainer.data_iter)
        except AttributeError:
            # Initialize data iterator if not started
            trainer.data_iter = iter(dataloader)
            batch = next(trainer.data_iter)
        except StopIteration:
            trainer.data_iter = iter(dataloader)
            batch = next(trainer.data_iter)

        x0, xT, mask = batch
        x0 = x0.to(device)
        if xT is not None:
            xT = xT.to(device)
        else:
            xT = torch.zeros_like(x0)

        batch_size = x0.shape[0]
        # Sample random t for ELBO
        t_idx = np.random.randint(0, schedule_steps+1, size=batch_size)
        t_norm = torch.tensor(t_idx / schedule_steps, dtype=torch.float32, device=device)
        # Collect schedule parameters at sampled t
        theta_t = schedule_params['theta'][t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        cum_theta_t = schedule_params['cum_theta'][t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        g_t = schedule_params['g'][t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        sigma_t = schedule_params['sigma'][t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        sigma_t_T_curr = schedule_params['sigma_t_T'][t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # Generate x_t conditioned on x0 (sampling from closed form)
        # For training, we approximate x_t using the equation:
        # x_t = (x0 - (1 - exp(-cum_theta)) * xT) * sqrt(sigma_t) + epsilon * sqrt(sigma_t)
        epsilon = torch.randn_like(x0)
        exp_cum_theta = torch.exp(-cum_theta_t)
        denom = torch.sqrt(1 - torch.exp(-2 * cum_theta_t))
        x_t = ((x0 - (1 - exp_cum_theta) * xT) * torch.sqrt(sigma_t)) + epsilon * denom

        # Forward pass of network
        epsilon_theta = trainer.model(x_t, xT, t_norm)
        # Compute true scaled epsilon from the sampled x_t
        target_epsilon = (x_t - ((x0 - (1 - exp_cum_theta) * xT) * torch.sqrt(sigma_t))) / denom
        # Loss: L1 between predicted epsilon and true epsilon
        loss = torch.nn.functional.l1_loss(epsilon_theta, target_epsilon)

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.update(1)

        # Optional Logging
        if (step+1) % 1000 == 0:
            print(f"Step {step+1}/{total_steps} - Loss: {loss.item():.4f}")
    pbar.close()
    print("Training finished.")

    # --- 9. Save trained model ---
    save_path = 'checkpoint.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # --- 10. Inference / Restoration ---
    # Load best model (or current)
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # For inference: restore from a conditioned low-quality image x_T
    # Example: take a random batch from dataset or test set
    # Note: For demo, pick the first batch from dataloader
    for batch in dataloader:
        x_input, x_gt, _ = batch
        x_input = x_input.to(device)
        # Take the first image from batch
        x_cond = x_input[0].unsqueeze(0)  # shape: (1,3,H,W)
        break

    # Perform restoration
    sampler = Sampler(model, schedule_params, {
        'steps': config.get('inference', {}).get('steps', 100),
        'use_mean_ode': True
    })
    restored_x = sampler.restore(x_cond)

    # Save or display result
    output_dir = './restored_results'
    os.makedirs(output_dir, exist_ok=True)
    save_image((restored_x + 1.0)/2.0, os.path.join(output_dir, 'restored.png'))
    print(f"Restored image saved to {os.path.join(output_dir, 'restored.png')}")

    # --- 11. (Optional) Evaluate restored image ---
    # (can be called separately with Evaluation class)

if __name__ == "__main__":
    main()
