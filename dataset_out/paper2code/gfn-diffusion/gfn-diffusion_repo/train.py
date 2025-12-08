## train.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random

# Import custom modules
from dataset_loader import get_dataset_loader
from model import NeuralSDE
from sampler import EulerSampler
from buffer import ReplayBuffer
from evaluation import Estimator, visualize_samples
from sampler import MHLocalSearch

# ================== Load configuration ================== #
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set seeds for reproducibility
SEED = config.get('hyperparameters', {}).get('seed', 42)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Device configuration
device = torch.device(config.get('hyperparameters', {}).get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# ================== Hyperparameters & Setup ================== #
# Training parameters
lr_policy = config['training'].get('learning_rate', 1e-3)
batch_size = config['training'].get('batch_size', 300)
total_iters = config['training'].get('total_iterations', 25000)
grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)
clip_norm = config['training'].get('gradient_clip_norm', 1.0)

# Model parameters
input_dim = config['model']['neural_sde'].get('input_dim', 2)
hidden_dim = config['model']['neural_sde'].get('hidden_dim', 400)
network_type = config['model']['neural_sde'].get('network_type', 'MLP')
learn_diffusion = False  # For simplicity, fix diffusion; can be lazy learned if needed

# Diffusion schedule
T = config['diffusion_process'].get('T', 100)
delta_t = config['diffusion_process'].get('delta_t', 0.01)
beta_min = config['diffusion_process'].get('beta_min', 0.01)
beta_max = config['diffusion_process'].get('beta_max', 4.0)

# Training objectives flags
use_TB_loss = config['training_objectives'].get('trajectory_balance_loss', True)
use_VarGrad = config['training_objectives'].get('var_grad_loss', True)
exploration_loss_weight = config['training'].get('exploration_loss_weight', 1.0)

# Buffer parameters
buffer_capacity = config['training'].get('buffer_capacity', 600000)
buffer_strategy = config['training'].get('buffer_sampling_strategy', 'FIFO')
priority_k = config['training'].get('buffer_priority_k', 0.01)

# Exploration parameters for local MH
K_mh = config['exploration']['local_search'].get('steps_per_update', 200)
burn_in = config['exploration']['local_search'].get('burn_in_steps', 100)
initial_eta = config['exploration']['local_search'].get('initial_step_size', 0.01)
target_accept_rate = config['exploration']['local_search'].get('target_acceptance', 0.574)
increase_factor = config['exploration']['local_search'].get('step_size_increase_factor', 1.1)
decrease_factor = config['exploration']['local_search'].get('step_size_decrease_factor', 0.9)
max_mh_steps = config['exploration']['local_search'].get('max_steps', 200)

# Evaluation and log parameters
eval_freq = config['evaluation'].get('evaluation_freq', 1000)
save_every = config['hyperparameters'].get('save_checkpoint_every', 5000)

# ================== Instantiate dataset & energy fn ================== #
dataset_cfg = config['dataset']
dataset_obj = get_dataset_loader(dataset_cfg)

# For synthetic energy functions, define inline or import as needed
energy_fn = dataset_obj.energy_fn if hasattr(dataset_obj, 'energy_fn') else None

# If real data (e.g., MNIST), for conditional setting, prepping data loader
# For simplicity, assume synthetic energy target in this script

# ================== Initialize models and optimizer ================== #
# Neural SDE model
model = NeuralSDE(input_dim=input_dim, hidden_dim=hidden_dim,
                  network_type=network_type, learn_diffusion=learn_diffusion).to(device)

# LogZ parameter (scalar), initialized to 0
logZ_param = torch.tensor([0.0], requires_grad=True, device=device)

# Optimizer
optimizer = optim.Adam(
    list(model.get_parameters()) + [logZ_param],
    lr=lr_policy,
    betas=(0.9, 0.999),
    eps=1e-8
)

# Optional: different LR for flow params, as needed
# Here, set uniformly for simplicity

# Step size for MH local search
eta = torch.tensor([initial_eta], device=device)
eta.requires_grad = False

# ================== Initialize Buffer ================== #
buffer = ReplayBuffer(capacity=buffer_capacity, prioritize=True, priority_k=priority_k)
# For conditional tasks, initialize buffer of conditions, skipped here

# ================== Initialize MH Local Search ================== #
mh_sampler = MHLocalSearch(
    energy_fn=energy_fn,
    initial_eta=initial_eta,
    target_accept=target_accept_rate,
    increase_factor=increase_factor,
    decrease_factor=decrease_factor,
    max_steps=max_mh_steps,
    burn_in=burn_in
)

# ================== Main Training Loop ================== #
for it in tqdm(range(1, total_iters + 1), desc='Training'):
    optimizer.zero_grad()

    # Decide sampling type (on-policy or off-policy)
    # 50% on-policy, 50% off-policy exploration
    if random.random() < 0.5:
        # On-policy trajectory sampling
        # Initialize x from prior: zero or simple hypothesis
        x0 = torch.zeros((batch_size, input_dim), device=device)
        sampler = EulerSampler(model, T=T, delta_t=delta_t)
        traj_states = sampler.sample(x0, energy_fn)
        # Trajectory collected: list of states at each step; for TB, need full trajectory
        # For simplicity, store only final state and trajectory for loss
        final_state = traj_states[-1].detach()
        # Save trajectory path for loss
        trajectory_states = traj_states
        trajectory_type = 'on_policy'
        # Compute energies at final step
        energy_values = energy_fn(final_state).detach()
    else:
        # Off-policy exploration from buffer
        batch_buffer_samples = buffer.sample(batch_size, prioritized=True)
        # Extract samples and energies
        buffer_samples = []
        for s, e, c in batch_buffer_samples:
            buffer_samples.append(s.to(device))
        # Convert to tensor
        buffer_states = torch.stack(buffer_samples, dim=0)
        # Run local MH to improve samples
        # Run MH in parallel with adapt step size
        # mev: Optionally, we can run MH for each sample individually or batch
        improved_states = mh_sampler.run(buffer_states, steps=K_mh)
        final_state = improved_states.detach()
        # For buffer-based samples, need to generate trajectories as well
        # For simplicity, assume trajectory from buffer sample is a single step or minimal
        trajectory_states = [final_state]
        energy_values = energy_fn(final_state).detach()
        trajectory_type = 'off_policy'

    # Generate trajectories for gradient computation
    # For simplicity, here we simulate trajectory sampling for 100 steps with current model
    # Note: In practice, should reconstruct full trajectories from states
    # For an illustrative example, re-initialize from final state
    # Using Euler sampler
    x_start = final_state
    sampler = EulerSampler(model, T=T, delta_t=delta_t)

    # Sample trajectory starting from x_start
    traj = sampler.sample(x_start, energy_fn)
    # Store final state and trajectory for loss
    x_final = traj[-1]
    # Compute transition log probabilities
    # Define a function for transition density log probability if needed, placeholder here
    # For TB loss, assume symmetric Gaussian approximation, so focus on trajectory density ratio
    # For simplicity, estimate via approximation
    # Here, just treat trajectories as sequences for TB loss

    # Compute model's distribution log probability over trajectory:
    # For approximation, sum log transition densities (Gaussian)
    log_P_fwd = 0.0
    for i in range(len(traj) - 1):
        x_curr = traj[i]
        x_next = traj[i + 1]
        t_curr = i / T
        drift, g = model.forward(x_curr, t_curr)
        mean = x_curr + drift * delta_t
        var = (g ** 2) * delta_t
        # Compute log prob of x_next under Gaussian mean, var
        dist = torch.distributions.Normal(mean, torch.sqrt(var))
        log_prob = dist.log_prob(x_next).sum(dim=1).mean()
        log_P_fwd += log_prob

    # Transition density for backward process: assumed fixed or learned, here skipped
    # For simplicity, assume backward process approximated as same as forward
    # Otherwise, need to implement p_B and compute log-prob there

    # Compute trajectory probability ratio: approximate as sum of log transition densities
    # Here, as a placeholder, set log_P_bwd as same as log_P_fwd
    log_P_bwd = log_P_fwd

    # Compute TB loss: (log(Z) + log P_F - log R - log P_B)^2
    # Approximate log(Z) as learnable parameter logZ_param
    logZ = logZ_param
    # To prevent issues, avoid numerical instability
    loss_TB = (logZ + log_P_fwd - energy_fn(x_final).log().mean() - log_P_bwd).pow(2)

    # Variance estimator (VarGrad), optionally, not implemented in detail here
    if use_VarGrad:
        # Compute logs ratios for a minibatch
        log_ratios = []
        for s, e, c in buffer_buffer_samples:
            # approximate log ratio
            # R(s) = exp(-E(s))
            log_R = -e
            # P_F trajectory log prob: same as above, approximate
            # For simplicity, reuse log_P_fwd
            log_ratio = log_R - log_P_fwd
            log_ratios.append(log_ratio)
        log_ratios = torch.stack(log_ratios)
        loss_VarGrad = torch.var(log_ratios)
    else:
        loss_VarGrad = torch.tensor(0.0, device=device)

    # Exploration/loss terms: e.g., include buffer size regularization or other exploration
    # Here, just combine
    total_loss = 0.0
    if use_TB_loss:
        total_loss += loss_TB
    if use_VarGrad:
        total_loss += loss_VarGrad

    # Add exploration loss if needed (e.g., entropy regularization) - skipped here for simplicity
    # For example, could add a small entropy bonus on the policy

    # Backpropagate
    total_loss.backward()

    # Gradient clipping
    if clip_norm > 0:
        nn.utils.clip_grad_norm_(list(model.get_parameters()) + [logZ_param], max_norm=clip_norm)

    # Optimizer step
    optimizer.step()

    # ======= Buffer Update ======== #
    # Add low-energy states to buffer
    # Here, uniformly add current final state
    buffer.add(final_state.cpu(), energy_fn(final_state).item())

    # ======= Step Size Adaptation ======== #
    # During burn-in period or periodically, adjust eta based on MH acceptance rate
    # For the example, do it every 10 batches
    if it % 10 == 0:
        # compute acceptance rate over last batch
        # For simplicity, assume acceptance rate is available from mh_sampler
        # in real code, track acceptance during MH run
        # Here, simulate acceptance rate estimation
        acceptance_rate = float(mh_sampler.acceptance_rate.numpy()) if hasattr(mh_sampler, 'acceptance_rate') else 0.574
        if acceptance_rate > target_accept_rate:
            eta = min(eta * increase_factor, 1.0)
        elif acceptance_rate < target_accept_rate:
            eta = max(eta * decrease_factor, 1e-6)

    # ================== Logging & Saving ================== #
    if it % eval_freq == 0 or it == total_iters:
        # Run evaluation: estimate logZ, Wasserstein, visualization
        estimator = Estimator(model, energy_fn, logZ=logZ_param)
        logZ_estimate, logZ_std, W2 = estimator.estimate_all()
        print(f"Iteration {it}: LogZ estimate: {logZ_estimate:.3f} ± {logZ_std:.3f}, W2: {W2:.3f}, eta: {eta.item():.4f}")
        # Save checkpoint periodically
        if it % save_every == 0:
            save_path = f"checkpoint_iter_{it}.pt"
            torch.save({'model_state_dict': model.state_dict(),
                        'logZ': logZ_param.detach().cpu().numpy()},
                       save_path)

# ================== End of training ================== #

