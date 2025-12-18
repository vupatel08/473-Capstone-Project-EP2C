## evaluation.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List

import jax
import jax.numpy as jnp

from dataset_loader import DatasetLoader
from model import GNS, SEGNN
from utils import (
    compute_velocity_std,
    convolve_with_neighbors,
    neighbor_search,
    compute_dirichlet_energy,
    visualize_particle_field,
)
import force_smoothing
import sph_relaxation

import pickle
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set random seed for reproducibility
RNG_SEED = config.get('misc', {}).get('random_seed', 42)
key = jax.random.PRNGKey(RNG_SEED)

# Dataset and evaluation parameters
data_path = config['dataset'].get('path', 'datasets/lagrangian_fluid')
sequence_length = config['dataset'].get('sequence_length', 400)
eval_steps = config['evaluation'].get('rollout_steps', 400)
n_eval_trials = config['evaluation'].get('evaluation_trials', 12)
visualize = config['evaluation'].get('visualization', True)

# Load dataset
dataset_loader = DatasetLoader(data_path, config)
# Pick initial state from a specific trajectory, e.g., first one
initial_idx = 0
initial_seq = dataset_loader.get_sequence(initial_idx)
initial_positions = jnp.array(initial_seq['positions'][0])  # shape: (N, dim)
initial_velocities = jnp.array(initial_seq['velocities'][0])  # shape: (N, dim)
particle_types = jnp.array(initial_seq['types'])  # shape: (N,)
# External forces: optional
external_forces = (
    jnp.array(initial_seq['external_forces'][0]) if initial_seq['external_forces'] is not None else None
)

# Load trained model
model_type = config['model'].get('type', 'GNS').upper()
model_params = config['model'].get('params', {})

# Assume number of particle types from data
num_types = np.max(np.array(particle_types)) + 1

# Initialize model
if model_type == 'GNS':
    model = GNS(hyperparams=model_params, num_types=num_types, embedding_dim=16)
elif model_type == 'SEGNN':
    model = SEGNN(hyperparams=model_params, num_types=num_types, embedding_dim=16)
else:
    raise ValueError(f"Unsupported model type: {model_type}")

# Load model parameters; here we placeholder load from checkpoint, path can be configured
# For demonstration, suppose we have a loaded params dict
# with open('path_to_checkpoint.pkl', 'rb') as f:
#     saved = pickle.load(f)
#     params = saved['params']
# Replace with actual loading code
params = None  # Placeholder; assume model is properly loaded before evaluation

# For illustration purpose, initialize fake params (replace with real in practice)
variables = model.init(key, initial_positions, initial_velocities, jnp.zeros_like(initial_positions), particle_types, predict_forces=True)
params = variables['params']

# Hyperparameters
relax_hp = {
    'alpha': config.get('hyperparameters', {}).get('relaxation', {}).get('alpha', 0.03),
    'beta': config.get('hyperparameters', {}).get('relaxation', {}).get('beta', 0.0),
    'relaxation_steps': config.get('hyperparameters', {}).get('relaxation', {}).get('relaxation_steps', 3),
}
force_smooth_sigma_scale = config.get('hyperparameters', {}).get('external_force_smoothing', {}).get('sigma_scale', 0.025)

# Simulation parameters
dt = 1.0  # Assumed dataset timestep; adapt as needed
use_force_field = (
    config.get('hyperparameters', {}).get('force_field', {}).get('external_force_field', True)
)

# Metrics container
metrics_all = {
    'position_mse': [],
    'sinkhorn_divergence': [],
    'kinetic_energy_mse': [],
    'density_mae': [],
    'dirichlet_energy': [],
    'chamfer_distance': [],
}

# Prepare for visualization
save_dir = 'evaluation_results'
os.makedirs(save_dir, exist_ok=True)

for trial in range(n_eval_trials):
    print(f"\nStarting trial {trial+1}/{n_eval_trials}")
    # State initialization for each trial
    positions = initial_positions
    velocities = initial_velocities
    # Keep track of trajectory for metrics
    traj_positions = [positions]
    traj_velocities = [velocities]
    # For density/energy metrics, store per step
    density_list = []
    energy_list = []

    # For computing reference densities or forces, precompute if needed
    # For simplicity, assume reference density = 1.0
    rho_ref = 1.0

    for step in range(eval_steps):
        # ----------- Prepare input features -----------
        # Find neighbors
        neighbor_indices = neighbor_search(positions, cutoff=config['hyperparameters']['neighbor_search'].get('cutoff_radius', 1.5))
        # External forces: for current positions, generate or get from dataset if available
        if external_forces is None:
            ext_force = np.zeros_like(positions)
        else:
            # For evaluation, external forces could be constant or extracted from dataset
            ext_force = external_forces  # shape: (N, dim)

        # Apply force smoothing if enabled
        if use_force_field:
            # Compute velocity std for smoothing
            # Use recent velocities to estimate
            # For first step, no history: fall back to current velocity
            vel_for_sigma = velocities
            sigma_u = compute_velocity_std(vel_for_sigma, window_size=5)
            smoothed_force = force_smoothing.convolve_with_neighbors(
                positions, ext_force, config['hyperparameters']['external_force_smoothing']['sigma_scale'] * sigma_u
            )
        else:
            smoothed_force = ext_force

        # ----------- Model prediction -----------
        # Pass current state through model
        def model_apply(p, pos, vel, extf, types):
            return model.apply(p, pos, vel, extf, types, predict_forces=True)
        pred = model_apply(params, positions, velocities, smoothed_force, particle_types)
        predicted_acc = np.array(pred['acceleration'])  # shape: (N, dim)
        ext_forces_pred = np.array(pred['external_forces'])

        # ----------- Compute explicit external force component -----------
        # For models trained with external forces, subtract or add as per Eq.2
        # Here, we assume the model predicted total acceleration, including external g
        # We remove external component so model learns internal dynamics
        total_acc = predicted_acc + ext_forces_pred  # total acceleration
        # For evaluation, add external force explicitly
        total_acc_with_g = total_acc + smoothed_force

        # ----------- Integrate positions & velocities (semi-implicit Euler) -----------
        new_velocities = velocities + total_acc_with_g * dt
        new_positions = positions + new_velocities * dt

        # ----------- Optional: SPH Relaxation (if configured) -----------
        relax_steps = relax_hp['relaxation_steps']
        if relax_steps > 0:
            relaxation = sph_relaxation.SPHRelaxation(
                positions=new_positions,
                velocities=None,
                hyperparameters={
                    'alpha': relax_hp['alpha'],
                    'beta': relax_hp['beta'],
                    'relaxation_steps': relax_hp['relaxation_steps'],
                    'kernel_radius': config['hyperparameters']['neighbor_search'].get('relaxation_cutoff_radius', 3.0),
                    'p_ref': 1.0,
                    'rho_ref': rho_ref,
                }
            )
            new_positions = relaxation.relax(positions=new_positions, n_steps=relax_hp['relaxation_steps'])

        # ----------- Save trajectory data -----------
        traj_positions.append(new_positions)
        traj_velocities.append(new_velocities)

        # ----------- Update for next step -----------
        positions = new_positions
        velocities = new_velocities

        # ----------- Compute metrics at current step (or final) if needed -----------
        if step >= eval_steps - 1:
            # For metrics at the end of trajectory for this trial
            pass

    # --- After rollout: convert lists to arrays ---
    predicted_traj = np.stack(traj_positions, axis=0)  # shape: (steps+1, N, dim)
    true_positions_seq = np.array(dataset_loader.get_sequence(initial_idx)['positions'][:eval_steps+1])
    # Note: get_sequence returns full sequence; here, ensure matching length

    # ----------- Compute metrics -----------

    # 1. Position MSE (over entire trajectory)
    # For each particle, compute MSE over sequence
    pos_error = np.mean((predicted_traj - true_positions_seq)**2)
    # For a more detailed per-particle, per-step, average or visualization
    metrics_all['position_mse'].append(pos_error)

    # 2. Sinkhorn divergence
    # Compare last position point clouds
    pred_points = predicted_traj[-1]  # shape: (N, dim)
    true_points = true_positions_seq[-1]
    # For Sinkhorn, use SciPy or custom; here, placeholder
    sinkhorn = compute_sinkhorn_divergence(pred_points, true_points)
    metrics_all['sinkhorn_divergence'].append(sinkhorn)

    # 3. Kinetic energy MSE
    KE_pred = 0.5 * np.sum(traj_velocities[-1] ** 2, axis=-1)
    KE_true = 0.5 * np.sum(true_positions_seq[-1] - true_positions_seq[-2], axis=-1) ** 2 / (dt**2)
    KE_error = np.mean((KE_pred - KE_true) ** 2)
    metrics_all['kinetic_energy_mse'].append(KE_error)

    # 4. Density MAE
    # Compute density at last step: for simplicity, use particle positions and kernel sum
    neighbor_list = neighbor_search(predicted_traj[-1], cutoff=config['hyperparameters']['neighbor_search'].get('cutoff_radius', 1.5))
    density_pred = []
    for i in range(predicted_traj[-1].shape[0]):
        nbrs = neighbor_list[i]
        r_ij = np.linalg.norm(predicted_traj[-1][nbrs] - predicted_traj[-1][i], axis=1)
        W_vals = sph_kernel_quintic(r_ij, config['hyperparameters']['neighbor_search'].get('cutoff_radius', 1.5))
        density_pred.append(np.sum(W_vals))
    density_pred = np.array(density_pred)
    density_mae = np.mean(np.abs(density_pred - rho_ref))
    metrics_all['density_mae'].append(density_mae)

    # 5. Dirichlet energy
    dir_energy = compute_dirichlet_energy(density_pred, predicted_traj[-1], h=1.5)
    metrics_all['dirichlet_energy'].append(dir_energy)

    # 6. Chamfer distance
    cd = compute_chamfer_distance(predicted_traj[-1], true_positions_seq[-1])
    metrics_all['chamfer_distance'].append(cd)

    # ----------- Visualization if enabled -----------
    if visualize:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].scatter(true_positions_seq[-1][:,0], true_positions_seq[-1][:,1], c='blue', label='Truth', s=20)
        axs[0].scatter(predicted_traj[-1][:,0], predicted_traj[-1][:,1], c='red', label='Prediction', s=20, alpha=0.5)
        axs[0].set_title('Particle positions at final step')
        axs[0].legend()
        plot_density = density_pred
        axs[1].bar(range(len(plot_density)), plot_density)
        axs[1].set_title('Predicted density at final step')
        plt.savefig(f"{save_dir}/trial_{trial}_final.png")
        plt.close()

# --- Summary of metrics ---
print("\nEvaluation Summary (averaged over trials):")
for key, values in metrics_all.items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{key}: {mean_val:.4e} ± {std_val:.4e}")

# Save metrics to file
with open(os.path.join(save_dir, 'evaluation_metrics.pkl'), 'wb') as f:
    pickle.dump(metrics_all, f)

# Optional: save detailed trajectories, errors over time, etc.

# --- Utility functions (must exist or be implemented) ---

def compute_sinkhorn_divergence(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    Computes approximate Sinkhorn divergence between two point clouds.
    Placeholder: in practice, use a library like geomloss, POT, or implement.
    """
    # For simplicity, return Euclidean distance sum as placeholder
    return np.linalg.norm(pc1 - pc2)

def compute_chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    Compute symmetric Chamfer distance between two point clouds.
    """
    from scipy.spatial import cKDTree
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    dist1, _ = tree1.query(pc2)
    dist2, _ = tree2.query(pc1)
    return np.mean(dist1 ** 2) + np.mean(dist2 ** 2)
