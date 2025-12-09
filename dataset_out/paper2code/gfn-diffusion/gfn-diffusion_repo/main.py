## main.py
import os
import yaml
import torch
import numpy as np
import random
from tqdm import tqdm

# Import core modules (assuming they are in the same directory)
from dataset_loader import get_dataset_loader
from model import NeuralSDE
from sampler import EulerSampler, MHLocalSearch
from buffer import ReplayBuffer
from evaluation import Estimator, visualize_results

# ------------------------ 1. Load Config ------------------------- #
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set seed for reproducibility
seed = config.get('hyperparameters', {}).get('seed', 42)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Set device
device_str = config.get('hyperparameters', {}).get('device', 'cuda:0')
device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

# ------------------------ 2. Data & Energy Setup ------------------------ #
dataset_cfg = config.get('dataset', {})
dataset_name = dataset_cfg.get('dataset_name', 'Manywell')
energy_fn = None

# For synthetic tasks, define energy functions
if dataset_cfg.get('type', 'synthetic_energy') == 'synthetic_energy':
    energy_fn = get_dataset_loader(dataset_cfg).energy_fn

# ------------------------ 3. Instantiate Models ------------------------ #
# Neural SDE for drift and diffusion
input_dim = config['model']['neural_sde'].get('input_dim', 2)
hidden_dim = config['model']['neural_sde'].get('hidden_dim', 400)
network_type = config['model']['neural_sde'].get('network_type', 'MLP')
learn_diffusion = False  # Fixed diffusion as per the experimental setting

model = NeuralSDE(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    network_type=network_type,
    learn_diffusion=learn_diffusion
).to(device)

# Log-parameter for partition function Z
logZ = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=True)

# Parameters for optimizer
params = list(model.get_parameters()) + [logZ]
optimizer = torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

# ------------------------ 4. Buffer & Exploration ------------------------ #
buffer_capacity = config['training'].get('buffer_capacity', 600000)
buffer = ReplayBuffer(capacity=buffer_capacity, prioritize=True, priority_k=0.01)

# Local search MH setup
mh_sampler = MHLocalSearch(
    energy_fn=energy_fn,
    initial_eta=0.01,
    target_accept=0.574,
    increase_factor=1.1,
    decrease_factor=0.9,
    max_steps=200,
    burn_in=100
)

# For visualization and evaluation
eval_freq = config['evaluation'].get('evaluation_freq', 1000)
save_every = config['hyperparameters'].get('save_checkpoint_every', 5000)
total_iterations = config['training'].get('total_iterations', 25000)

# ------------------------ 5. Training Loop ------------------------------ #
for it in tqdm(range(1, total_iterations + 1), desc='Training'):
    optimizer.zero_grad()

    # Decide sampling mode: 50% on-policy trajectory, 50% off-policy
    if random.random() < 0.5:
        # On-policy: sample trajectory from initial 'delta' (zero)
        x0 = torch.zeros((config['training'].get('batch_size', 300), input_dim), device=device)
        sampler = EulerSampler(model, T=100, delta_t=0.01)
        traj = sampler.sample(x0, energy_fn)
        final_state = traj[-1].detach()
        traj_states = traj  # list of states
        energy_value = energy_fn(final_state).detach()
        traj_type = 'on_policy'
    else:
        # Off-policy: sample from buffer + local MH
        batch_buffer_samples = buffer.sample(config['training'].get('batch_size', 300))
        buffer_states = torch.stack([s for s, e, c in batch_buffer_samples], dim=0).to(device)
        # Run MH in parallel to improve samples
        improved_states = mh_sampler.run(buffer_states, steps=mh_sampler.max_steps)
        final_state = improved_states.detach()
        traj_states = [final_state]
        energy_value = energy_fn(final_state).detach()
        traj_type = 'off_policy'

    # Generate trajectory with current model from final state
    x_start = final_state
    sampler = EulerSampler(model, T=100, delta_t=0.01)
    traj = sampler.sample(x_start, energy_fn)
    x_final = traj[-1]
    # Approximate train loss using Trajectory Balance
    # Here, for simplicity, sum log transition densities assuming Gaussian transitions
    log_P_fwd = 0.0
    for i in range(len(traj)-1):
        x_curr = traj[i]
        x_next = traj[i+1]
        t_curr = i/100
        drift, g = model.forward(x_curr, t_curr)
        mean = x_curr + drift * 0.01
        var = (g ** 2) * 0.01
        dist = torch.distributions.Normal(mean.squeeze(), torch.sqrt(var.squeeze()+1e-12))
        log_P_fwd += dist.log_prob(x_next.squeeze()).sum().item()
    # For simplicity, assume backward process same as forward (placeholders)
    log_P_bwd = log_P_fwd
    # Compute TB loss
    loss_TB = (logZ + log_P_fwd - energy_fn(x_final).log().mean() - log_P_bwd).pow(2)

    # Variance estimator loss (VarGrad), placeholder
    if 'use_VarGrad' in config['training'] and config['training']['use_VarGrad']:
        # For simplicity, replicate loss
        loss_VarGrad = torch.var(torch.tensor([ -energy_fn(s).item() for s, e, c in buffer.sample(100)]))
    else:
        loss_VarGrad = torch.tensor(0.0, device=device)

    total_loss = loss_TB
    if 'use_VarGrad' in config['training'] and config['training']['use_VarGrad']:
        total_loss += loss_VarGrad

    # Backpropagate
    total_loss.backward()
    # Gradient clipping if specified
    if 'gradient_clip_norm' in config['training']:
        torch.nn.utils.clip_grad_norm_(params, max_norm=config['training']['gradient_clip_norm'])
    optimizer.step()

    # ---------------- Buffer Update ---------------- #
    # Add current sample to buffer
    buffer.add(x_final.cpu(), energy_fn(x_final).item())

    # --------- Step size adaptation (not detailed) --------- #
    # Here, we could adapt eta if running MH, but for simplicity, skip as placeholder

    # ---------------- Evaluation ---------------- #
    if it % eval_freq == 0 or it == total_iterations:
        # Estimate logZ with importance sampling and VarGrad
        estimator = Estimator(model, energy_fn, logZ=logZ)
        logZ_estimate = estimator.estimate_logZ()
        # Compute Wasserstein distance (if applicable)
        # Generate samples for W2
        samples = sampler.sample(torch.zeros((1000, input_dim), device=device), energy_fn, steps=100)
        # For high-dimensional tasks, this is illustrative
        # Alternatively, target samples could be loaded if available
        # Here, just generate random samples for illustration
        target_samples = torch.randn((1000, input_dim))
        W2 = estimator.compute_wasserstein(samples, target_samples)
        print(f'Iter {it}: LogZ={logZ_estimate:.3f}, W2={W2:.3f}')

# ------------------------ 6. Save Final Model ------------------------ #
torch.save({'model_state_dict': model.state_dict(),
            'logZ': logZ.detach().cpu().numpy()},
           'final_model.pt')
# Save buffer if needed
# Save any logs or plots if desired

# ----------------------- 7. Final Visualization ------------------------ #
# Use 'visualize_results' to produce plots
# e.g.,
# visualize_results(energy_fn, samples=samples, true_samples=target_samples, task_name='Manywell')

