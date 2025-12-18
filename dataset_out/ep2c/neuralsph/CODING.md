# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import numpy as np
from typing import Optional, Tuple, List, Dict


class Dataset:
    """
    Simple dataset object to store sequences of positions, velocities,
    particle types, and optional external forces.
    """
    def __init__(self,
                 positions: np.ndarray,
                 velocities: np.ndarray,
                 types: np.ndarray,
                 external_forces: Optional[np.ndarray] = None):
        self.positions = positions  # shape: (N_seq, seq_len, N_particles, dim)
        self.velocities = velocities  # shape: (N_seq, seq_len, N_particles, dim)
        self.types = types  # shape: (N_seq, N_particles)
        self.external_forces = external_forces  # shape: same as positions


class DatasetLoader:
    def __init__(self, dataset_path: str, config: Dict):
        """
        Loads dataset from the specified path, processes it, and stores in memory.

        Args:
            dataset_path (str): Path to dataset directory or files.
            config (Dict): Configuration dictionary with keys:
                - sequence_length (int): length of sequences for evaluation.
                - training_subsequence_interval (int): interval for training sampling.
        """
        self.dataset_path = dataset_path
        self.sequence_length = config.get('dataset', {}).get('sequence_length', 400)
        self.subsample_interval = config.get('dataset', {}).get('training_subsequence_interval', 100)

        # Internal storage
        self.positions_all = None  # shape: (total_samples, max_seq_len, N_particles, dim)
        self.velocities_all = None
        self.types_all = None
        self.forces_all = None

        # Load data
        self._load_data()

        # After loading, determine number of sequences
        self.total_sequences = self.positions_all.shape[0]

    def _load_data(self):
        """
        Loads dataset files from the path, supports npz or npy, or custom format.
        Assumes data stored with keys: 'positions', optionally 'forces', 'types'.
        """
        # Find data files in directory
        files = []
        if os.path.isdir(self.dataset_path):
            for fname in os.listdir(self.dataset_path):
                if fname.endswith('.npz') or fname.endswith('.npy'):
                    files.append(os.path.join(self.dataset_path, fname))
        elif os.path.isfile(self.dataset_path):
            files = [self.dataset_path]
        else:
            raise FileNotFoundError(f"Dataset path {self.dataset_path} not found.")

        # For simplicity, if multiple files, load and concatenate
        pos_list = []
        force_list = []
        type_list = []

        for fpath in files:
            if fpath.endswith('.npz'):
                data = np.load(fpath)
                positions = data['positions']  # shape: (N_seq, seq_len, N_particles, dim)
                if 'forces' in data:
                    forces = data['forces']
                else:
                    forces = None
                if 'types' in data:
                    types = data['types']
                else:
                    types = None
            elif fpath.endswith('.npy'):
                # Assume npy stores sequence of positions
                positions = np.load(fpath)  # shape: (N_seq, seq_len, N_particles, dim)
                forces = None
                types = None
            else:
                continue  # unsupported format

            pos_list.append(positions)
            if forces is not None:
                force_list.append(forces)
            if types is not None:
                type_list.append(types)

        if len(pos_list) == 0:
            raise ValueError("No valid dataset files found.")

        # Concatenate data from all files
        self.positions_all = np.concatenate(pos_list, axis=0)
        if force_list:
            self.forces_all = np.concatenate(force_list, axis=0)
        else:
            self.forces_all = None
        if type_list:
            self.types_all = np.concatenate(type_list, axis=0)
        else:
            # If no types provided, assign default type 0
            num_particles = self.positions_all.shape[2]
            self.types_all = np.zeros((self.positions_all.shape[0], num_particles), dtype=np.int32)

        # Validate data shapes
        N_seq, seq_len, N_particles, dim = self.positions_all.shape
        assert self.types_all.shape[0] == N_seq
        assert self.types_all.shape[1] == N_particles
        if self.forces_all is not None:
            assert self.forces_all.shape == (N_seq, seq_len, N_particles, dim)
        # Velocities can be derived or stored; here, we'll compute during getitem

    def get_sequence(self, index: int) -> Dict:
        """
        Returns a full sequence sample at index, including positions, velocities,
        types, and external forces if available.

        Args:
            index (int): index of the sequence.
        Returns:
            dict with keys:
                - 'positions': (seq_len, N_particles, dim)
                - 'velocities': (seq_len, N_particles, dim)
                - 'types': (N_particles,)
                - 'external_forces': (seq_len, N_particles, dim) or None
        """
        if index < 0 or index >= self.total_sequences:
            raise IndexError("Sequence index out of bounds.")

        pos_seq = self.positions_all[index]  # shape: (seq_len, N_particles, dim)
        # Derive velocities via finite differences, shape: (seq_len, N_particles, dim)
        velocities = np.zeros_like(pos_seq)
        velocities[1:] = pos_seq[1:] - pos_seq[:-1]
        velocities[0] = velocities[1]  # assign first timestep same as second for consistency

        types_seq = self.types_all[index]  # shape: (N_particles,)

        if self.forces_all is not None:
            forces_seq = self.forces_all[index]
        else:
            forces_seq = None

        sample = {
            'positions': pos_seq.astype(np.float32),
            'velocities': velocities.astype(np.float32),
            'types': types_seq.astype(np.int32),
            'external_forces': forces_seq.astype(np.float32) if forces_seq is not None else None
        }
        return sample

    def get_subsequence(self, index: int, start_time: int) -> Dict:
        """
        Get a subsequence of length self.sequence_length starting from start_time.

        Args:
            index (int): sequence index.
            start_time (int): starting timestep.

        Returns:
            dict with same keys as get_sequence, but truncated to subsequence length.
        """
        seq = self.get_sequence(index)
        end_time = start_time + self.sequence_length
        # Clip if necessary
        if end_time > seq['positions'].shape[0]:
            raise ValueError("Subsequence end exceeds sequence length.")
        subseq = {
            'positions': seq['positions'][start_time:end_time],
            'velocities': seq['velocities'][start_time:end_time],
            'types': seq['types'],
            'external_forces': None
        }
        if seq['external_forces'] is not None:
            subseq['external_forces'] = seq['external_forces'][start_time:end_time]
        return subseq

    def get_random_batch(self, batch_size: int) -> List[Dict]:
        """
        Randomly sample a list of sequences for batch training.

        Args:
            batch_size (int): number of sequences to sample.

        Returns:
            list of sample dictionaries.
        """
        indices = np.random.choice(self.total_sequences, size=batch_size, replace=False)
        return [self.get_sequence(i) for i in indices]
```

## evaluation.py

```python
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
```

## force_smoothing.py

```python
## force_smoothing.py

import numpy as np
from scipy.special import erf
from scipy.spatial import KDTree
from typing import Optional, Tuple

def compute_velocity_std(
    velocities: np.ndarray,
    window_size: int = 5
) -> float:
    """
    Calculate the isotropic standard deviation (sigma_u) of particle velocities
    over the most recent 'window_size' timesteps.

    Args:
        velocities (np.ndarray): Shape (seq_len, N_particles, dim),
            velocity sequences over time.
        window_size (int): Number of recent timesteps to consider.

    Returns:
        float: Scalar sigma_u representing the overall velocity std deviation.
    """
    if velocities.ndim != 3:
        raise ValueError(f"Velocities array must be shape (seq_len, N_particles, dim), but got {velocities.shape}")
    seq_len = velocities.shape[0]
    start_idx = max(0, seq_len - window_size)
    recent_vels = velocities[start_idx:, ...]  # shape: (window_size, N_particles, dim)
    # Compute std for each component across time and particles
    std_per_component = np.std(recent_vels, axis=(0,1))  # shape: (dim,)
    # Compute isotropic std via quadratic mean
    sigma_u = np.sqrt(np.mean(std_per_component ** 2))
    return float(sigma_u)

def gaussian_convolve_force(
    force_field: np.ndarray,
    sigma: float
) -> np.ndarray:
    """
    Approximate the convolution of a force field with a Gaussian kernel
    analytically using the error function (erf).

    Args:
        force_field (np.ndarray): Shape (N_particles, dim), the raw force values.
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        np.ndarray: Smoothed force field of shape (N_particles, dim).
    """
    # For force components modeled as step functions or similar,
    # the convolution results in an erf-based form per component.
    # Here, we assume force_field is known at particle locations.
    force_smoothed = np.zeros_like(force_field)
    for d in range(force_field.shape[1]):
        # erf-based smoothing for each component
        # To improve stability, clamp force component to avoid very large values
        force_smoothed[:, d] = erf(force_field[:, d] / (np.sqrt(2) * sigma))
    return force_smoothed

def effective_force_approximation(
    raw_force: np.ndarray,
    velocities: np.ndarray,
    method: str = 'gaussian',
    sigma_scale: float = 0.025
) -> np.ndarray:
    """
    Compute the smoothed external force field based on particle velocities
    and a chosen convolution method.

    Args:
        raw_force (np.ndarray): Shape (N_particles, dim), the instantaneous external force.
        velocities (np.ndarray): Shape (N_particles, dim), latest velocities for std estimation.
        method (str): 'gaussian' or 'erf', determines the smoothing approach.
        sigma_scale (float): Scaling factor for sigma derived from velocity std.

    Returns:
        np.ndarray: Smoothed force field with shape (N_particles, dim).
    """
    # Compute velocity std deviation vector (per component)
    std_vel = np.std(velocities, axis=0)  # shape: (dim,)
    # Overall scalar sigma: quadratic mean of component stds
    sigma_u = np.sqrt(np.mean(std_vel ** 2))
    sigma = sigma_u * sigma_scale

    if method == 'erf':
        # Analytic erf-based smoothing
        smoothed_force = np.zeros_like(raw_force)
        for d in range(raw_force.shape[1]):
            # Eq. (D.3): erf of (force component / (sqrt(2)*sigma))
            smoothed_force[:, d] = erf(raw_force[:, d] / (np.sqrt(2) * sigma))
        return smoothed_force
    elif method == 'gaussian':
        # Numerical convolution via kernel sum
        return gaussian_convolve_force(raw_force, sigma)
    else:
        raise ValueError(f"Unsupported force smoothing method: {method}")

def gaussian_convolve_force(
    force_field: np.ndarray,
    h: float
) -> np.ndarray:
    """
    Perform numerical convolution of the force field with Gaussian kernel
    using neighbor search.

    Args:
        force_field (np.ndarray): Shape (N_particles, dim), input force at particles.
        h (float): The kernel bandwidth (standard deviation).

    Returns:
        np.ndarray: Smoothed force field (N_particles, dim).
    """
    N, dim = force_field.shape
    smoothed = np.zeros_like(force_field)
    # Build neighbor search tree for particles
    # For efficiency, in practice, neighbor search should be cached per snapshot
    # For this module, assume force_field aligns with positions
    # For demonstration, perform neighbor search once per call
    # Note: Position data is needed; assuming external positions are available
    # Since only force and velocities are inputs, positions should be passed as argument
    # Here, we need to adapt: assuming positions are available externally when called.
    # For safer design, you should pass positions explicitly.
    raise NotImplementedError(
        "Neighbor search with positions is required for convolution. "
        "Please pass 'positions' array to this function for full implementation."
    )

def convolve_with_neighbors(
    positions: np.ndarray,
    force_field: np.ndarray,
    h: float
) -> np.ndarray:
    """
    Convolve force field with Gaussian kernel using neighbors identified via KDTree.

    Args:
        positions (np.ndarray): Particle positions (N_particles, dim).
        force_field (np.ndarray): Force at each particle (N_particles, dim).
        h (float): Kernel support radius.

    Returns:
        np.ndarray: Smoothed force field (N_particles, dim).
    """
    N, dim = positions.shape
    tree = KDTree(positions)
    smoothed_force = np.zeros_like(force_field)
    for i in range(N):
        neighbor_idx = tree.query_ball_point(positions[i], r=h)
        # compute kernel weights
        r_j = positions[neighbor_idx] - positions[i]  # shape: (num_neighbors, dim)
        r_norm = np.linalg.norm(r_j, axis=1) + 1e-8
        W_vals = sph_kernel_quintic(r_norm, h)
        # sum contributions
        weights = W_vals
        total_weight = np.sum(weights) + 1e-8
        # weighted sum of neighbor forces
        neighbor_forces = force_field[neighbor_idx]
        smoothed_force[i] = np.sum(neighbor_forces * weights[:, np.newaxis], axis=0) / total_weight
    return smoothed_force

def sph_kernel_quintic(r: np.ndarray, h: float) -> np.ndarray:
    """
    Evaluate the quintic spline kernel (support: 0 <= q < 3).

    Args:
        r (np.ndarray): Distance array.
        h (float): Smoothing length.

    Returns:
        np.ndarray: Kernel values at each r.
    """
    q = r / h
    W = np.zeros_like(r)
    c = 7.0 / (478.0 * np.pi * h ** 3)  # 3D normalization constant; adapt for 2D if needed
    for i, qi in enumerate(q):
        if 0 <= qi < 1:
            W[i] = c * ((3 - qi) ** 5 - 6 * (2 - qi) ** 5 + 15 * (1 - qi) ** 5)
        elif 1 <= qi < 2:
            W[i] = c * ((3 - qi) ** 5 - 6 * (2 - qi) ** 5)
        elif 2 <= qi < 3:
            W[i] = c * (3 - qi) ** 5
        else:
            W[i] = 0.0
    return W

# Note:
# Since neighbor search relies on particle positions, the main API for force smoothing
# should accept positions explicitly.
# Therefore, the function 'effective_force_approximation' should be called with positions,
# raw_force, velocities, and smoothing method as arguments.
# For modularity, you may define an additional wrapper or utility function at call-time.


```

## main.py

```python
```python
## main.py

import os
import time
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import pickle
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

# ----------------------------------------------
# 1. Load configuration
# ----------------------------------------------
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set seed for reproducibility
seed = config.get('misc', {}).get('random_seed', 42)
rng = jax.random.PRNGKey(seed)

# Dataset parameters
dataset_path = config['dataset'].get('path', 'datasets/lagrangian_fluid')
sequence_length = config['dataset'].get('sequence_length', 400)
sampling_interval = config['dataset'].get('training_subsequence_interval', 100)

# Model parameters
model_type = config['model'].get('type', 'GNS').upper()
model_params = config['model'].get('params', {})
# Determine number of types for embedding
# Load first sequence to infer max type
dataset_loader = DatasetLoader(dataset_path, config)
sample_seq = dataset_loader.get_sequence(0)
num_types = int(np.max(sample_seq['types'])) + 1

# Initialize model
if model_type == 'GNS':
    model = GNS(hyperparams=model_params, num_types=num_types, embedding_dim=16)
elif model_type == 'SEGNN':
    model = SEGNN(hyperparams=model_params, num_types=num_types, embedding_dim=16)
else:
    raise ValueError(f"Unsupported model type: {model_type}")

# ----------------------------------------------
# 2. Initialize model parameters
# ----------------------------------------------
import flax
from flax.training import train_state

@jax.jit
def init_model_params(rng, model, sample_input):
    variables = model.init(rng, *sample_input)
    return variables

# Sample input for initialization
sample_data = dataset_loader.get_sequence(0)
sample_positions = jnp.array(sample_data['positions'][0])  # shape: (N, dim)
sample_velocities = jnp.array(sample_data['velocities'][0]) # shape: (N, dim)
sample_types = jnp.array(sample_data['types'])  # shape: (N,)
sample_zero_force = jnp.zeros_like(sample_positions)

params_ = init_model_params(rng, model, (sample_positions, sample_velocities, sample_zero_force, sample_types, False))
params = params_['params']

# ----------------------------------------------
# 3. Setup optimizer for training
# ----------------------------------------------
import optax

learning_rate = config['training'].get('learning_rate', 0.001)
optimizer = optax.chain(
    optax.clip_by_global_norm(config['training'].get('gradient_clip_norm', 1.0)),
    optax.adam(learning_rate=learning_rate, weight_decay=1e-6)
)

from flax.training import train_state as flax_train_state

class TrainState(flax_train_state.TrainState):
    pass

state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

# Extract loss weights
loss_weights = config['training'].get('loss_weights', {
    'position_mse': 1.0,
    'velocity_mse': 0.1,
    'density_mae': 0.1
})

# Hyperparameters for physics corrections
sigma_scale = config['hyperparameters'].get('external_force_smoothing', {}).get('sigma_scale', 0.025)
relax_hp = config['hyperparameters'].get('relaxation', {
    'alpha': 0.03,
    'beta': 0.0,
    'relaxation_steps': 3
})

# Checkpoint config
save_dir = config.get('checkpoint', {}).get('save_dir', 'checkpoints')
save_freq = config.get('checkpoint', {}).get('save_frequency', 10)
resume_training = config.get('checkpoint', {}).get('resume', False)
os.makedirs(save_dir, exist_ok=True)

# ----------------------------------------------
# 4. Define loss and training step
# ----------------------------------------------
import functools

def compute_loss(params, batch, model, external_forces):
    """
    Compute supervised loss for a batch of sequences.
    batch contains: positions, velocities, types, external_forces
    """
    positions = batch['positions']  # shape: (seq_len, N, dim)
    velocities = batch['velocities']
    types_ = batch['types']
    ext_forces = batch['external_forces']
    seq_len = positions.shape[0]
    H = batch.get('history_len', 5)

    total_loss = 0.0
    count = 0

    for t in range(H, seq_len - 1):
        # Prepare input features
        past_pos = positions[max(0, t - H):t, :, :]
        past_vel = velocities[max(0, t - H):t, :, :]
        curr_pos = positions[t]
        curr_vel = velocities[t]
        curr_type = types_
        curr_ext_force = ext_forces[t]

        # Model prediction (note: for JAX, need to define a jit-compiled function)
        def model_apply(p, pos, vel, extf, types, predict_forces=True):
            return model.apply(p, pos, vel, extf, types, predict_forces=predict_forces)
        pred = model_apply(params, curr_pos, curr_vel, curr_ext_force, curr_type, predict_forces=True)
        pred_acc = np.array(pred['acceleration'])  # (N, dim)
        ext_forces_pred = np.array(pred['external_forces'])

        # Derive target acceleration (excluding external forces)
        delta_vel = np.array(velocities[t]) - np.array(velocities[t - 1])
        delta_pos = np.array(positions[t]) - np.array(positions[t - 1])
        delta_time = 1.0  # assuming normalized dataset timestep
        target_acc = (delta_vel - curr_ext_force) / delta_time

        total_acc = pred_acc + ext_forces_pred  # predicted total acceleration
        pred_delta_pos = 0.5 * total_acc * delta_time ** 2  # integrate to position change

        # Losses
        position_loss = np.mean((pred_delta_pos - delta_pos) ** 2) * loss_weights['position_mse']
        velocity_loss = np.mean((delta_vel - (total_acc * delta_time)) ** 2) * loss_weights['velocity_mse']

        # Density calculation (approximate, for illustration)
        # Using neighbor search
        # For simplicity, skipping density regularization here, but include if needed
        # For now, omit or implement as placeholder
        density_loss_val = 0.0

        total_loss += position_loss + velocity_loss + density_loss_val
        count += 1

    if count > 0:
        total_loss /= count
    return total_loss

@jax.jit
def train_step(state, batch, model, ext_forces):
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params, batch, model, ext_forces)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

# ----------------------------------------------
# 5. Load or resume training if specified
# ----------------------------------------------
if resume_training:
    # Implement loading checkpoint logic here
    pass

# ----------------------------------------------
# 6. Training loop
# ----------------------------------------------
total_sequences = dataset_loader.total_sequences
for epoch in range(1, config['training'].get('epochs', 100) + 1):
    start_time = time.time()
    # Shuffle dataset indices
    indices = np.arange(total_sequences)
    np.random.shuffle(indices)

    epoch_losses = []

    # Process batches
    for batch_start in range(0, total_sequences, config['training'].get('batch_size', 64)):
        batch_indices = indices[batch_start:batch_start + config['training'].get('batch_size',64)]
        batch_samples = [dataset_loader.get_sequence(i) for i in batch_indices]

        # Collate batch data
        positions_batch = []
        velocities_batch = []
        types_batch = []
        forces_batch = []
        for s in batch_samples:
            positions_batch.append(jnp.array(s['positions']))  # (seq_len, N, dim)
            velocities_batch.append(jnp.array(s['velocities']))
            types_batch.append(jnp.array(s['types']))  # (N,)
            forces_batch.append(jnp.array(s['external_forces']) if s['external_forces'] is not None else jnp.zeros_like(positions_batch[-1]))
        batch_data = {
            'positions': jnp.stack(positions_batch),
            'velocities': jnp.stack(velocities_batch),
            'types': jnp.stack(types_batch),
            'external_forces': jnp.stack(forces_batch),
            'history_len': 5
        }

        # Run optimizer step
        state, curr_loss = train_step(state, batch_data, model, batch_data['external_forces'])
        epoch_losses.append(curr_loss)

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} completed in {epoch_time:.2f} s, Loss: {np.mean(epoch_losses):.6f}")

    # Save checkpoint periodically
    if epoch % save_freq == 0:
        checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({'params': jax.device_get(state.params)}, f)

# Save final model
final_path = os.path.join(save_dir, 'final_model.pkl')
with open(final_path, 'wb') as f:
    pickle.dump({'params': jax.device_get(state.params)}, f)

# ----------------------------------------------
# 7. Long-Horizon Rollout with Physics Corrections
# ----------------------------------------------
# Select a starting trajectory, e.g., first sequence in dataset
initial_seq = dataset_loader.get_sequence(0)
initial_positions = jnp.array(initial_seq['positions'][0])
initial_velocities = jnp.array(initial_seq['velocities'][0])
initial_types = jnp.array(initial_seq['types'])

# Optional: external forces for initial state
external_forces = (
    jnp.array(initial_seq['external_forces'][0]) if initial_seq['external_forces'] is not None else None
)

# Compute dataset velocity std for force smoothing
# Here, use the first few velocities as a proxy
if initial_seq['velocities'].shape[0] >= 5:
    recent_vels = np.array(initial_seq['velocities'][0:5])
else:
    recent_vels = np.array(initial_seq['velocities'])
sigma_u = compute_velocity_std(recent_vels, window_size=5)

# Prepare initial state
positions = initial_positions
velocities = initial_velocities
particle_types = initial_types

# Initialize for rollout
predicted_positions = [positions]
predicted_velocities = [velocities]

# Instantiate SPH relaxation routine
sph_params = {
    'alpha': relax_hp['alpha'],
    'beta': relax_hp['beta'],
    'relaxation_steps': relax_hp['relaxation_steps'],
    'kernel_radius': config['hyperparameters'].get('neighbor_search', {}).get('relaxation_cutoff_radius', 3.0),
    'p_ref': 1.0,
    'rho_ref': 1.0,
}
sph_relax = sph_relaxation.SPHRelaxation(positions, velocities=None, hyperparameters=sph_params)

# Run simulation for specified steps
num_steps = config['evaluation'].get('rollout_steps', 400)
for step_idx in range(num_steps):
    # 1. Compute neighbors
    neighbors = neighbor_search(positions, cutoff=some_cutoff_radius)
    # 2. External force: set or use precomputed smoothed forces
    if external_forces is None:
        ext_force = np.zeros_like(positions)
    else:
        ext_force = np.array(external_forces)

    # 3. Force smoothing (Gaussian or erf)
    smoothed_force = force_smoothing.convolve_with_neighbors(positions, ext_force, sigma_u * sigma_scale)

    # 4. Model prediction for acceleration (total, including external g)
    def model_apply(p, pos, vel, extf, types):
        return model.apply(state.params, pos, vel, extf, types, predict_forces=True)
    pred = model_apply(None, positions, velocities, smoothed_force, particle_types)
    pred_acc = np.array(pred['acceleration'])  # (N, dim)
    ext_pred = np.array(pred['external_forces'])

    # 5. Disentangle external force (as in Eq. 2)
    # Suppose model predicts total including external g, so:
    internal_acc = pred_acc
    # For adding explicit external g, if desired:
    total_acc = internal_acc + ext_pred

    # 6. Integrate position & velocity (semi-implicit Euler)
    delta_time = 1.0
    new_velocities = velocities + total_acc * delta_time
    new_positions = positions + new_velocities * delta_time

    # 7. Apply SPH relaxation to correct particle distribution
    if relax_hp['relaxation_steps'] > 0:
        relaxation = sph_relaxation.SPHRelaxation(
            new_positions,
            velocities=None,
            hyperparameters={
                'alpha': relax_hp['alpha'],
                'beta': relax_hp['beta'],
                'relaxation_steps': relax_hp['relaxation_steps'],
                'kernel_radius': sph_params['kernel_radius'],
                'p_ref': 1.0,
                'rho_ref': 1.0,
            }
        )
        new_positions = relaxation.relax(positions=new_positions, n_steps=relax_hp['relaxation_steps'])

    # 8. Store for next iteration
    predicted_positions.append(new_positions)
    predicted_velocities.append(new_velocities)
    positions = new_positions
    velocities = new_velocities

# 9. After rollout, evaluate metrics
# Load true trajectory from dataset (assuming continuous sequence)
true_seq = dataset_loader.get_sequence(0)['positions']
true_positions = jnp.array(true_seq[:num_steps + 1])  # include initial

# Convert predicted trajectory to array
pred_traj = np.stack(predicted_positions, axis=0)  # shape: (steps+1, N, dim)

# Compute and log metrics over the trajectory
# E.g.: position MSE
pos_mse = np.mean((pred_traj - true_positions) ** 2)
print(f"Long rollout position MSE: {pos_mse:.6e}")

# Generate visualizations if needed, save trajectories, etc.

# ------------------------------
# Note: For brevity, evaluation metrics and visualizations are omitted here.
# In practice, you should call the evaluation routines and save figures,
# similar to those in evaluation.py, feeding in pred_traj, true_positions, etc.
# ------------------------------
```

## model.py

```python
## model.py

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Dict, Tuple, Any
import functools

# If equivariance is enabled, use e3nn modules
try:
    import e3nn_jax as e3nn
except ImportError:
    e3nn = None

class ParticleTypeEmbedding(nn.Module):
    """
    Embedding layer for particle types if enabled.
    """
    num_types: int
    embedding_dim: int

    @nn.compact
    def __call__(self, particle_types: jnp.ndarray) -> jnp.ndarray:
        embed = nn.Embed(num_embeddings=self.num_types, features=self.embedding_dim)
        return embed(particle_types)

class MLP(nn.Module):
    """
    A simple multi-layer perceptron with specified features per layer.
    """
    features: list
    activation: Any = nn.relu
    final_activation: Optional[Any] = None

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation(x)
        x = nn.Dense(self.features[-1])(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x

class GNS(nn.Module):
    """
    Standard (non-equivariant) GNN-based model for acceleration prediction.
    """
    hyperparams: Dict
    num_types: int
    embedding_dim: int

    def setup(self):
        self.hidden_dim = self.hyperparams.get('hidden_dim', 128)
        self.num_layers = self.hyperparams.get('num_layers', 10)

        # Particle type embedding if enabled
        if self.hyperparams.get('particle_type_embedding', True):
            self.type_embedding = ParticleTypeEmbedding(self.num_types, self.embedding_dim)
        else:
            self.type_embedding = None

        # MLP encoder
        encoder_layers = [self.hidden_dim] * (self.num_layers // 2) + [self.hidden_dim]
        self.encoder = MLP(encoder_layers)

        # Message passing layers
        self.message_layers = []
        for _ in range(self.num_layers):
            self.message_layers.append(
                nn.Dense(self.hidden_dim)
            )

        # Decoder for acceleration
        self.decoder = MLP([self.hidden_dim, self.hidden_dim, 3])  # 3 for 3D, adjust for dims

    def encode_node_features(self, velocities, particle_types):
        """
        Encode features by concatenating velocities and particle type embeddings.
        """
        features = [velocities]
        if self.type_embedding:
            type_emb = self.type_embedding(particle_types)
            features.append(type_emb)
        node_feat = jnp.concatenate(features, axis=-1)
        return node_feat

    def __call__(self,
                 positions: jnp.ndarray,
                 velocities: jnp.ndarray,
                 external_force: Optional[jnp.ndarray],
                 particle_types: jnp.ndarray,
                 predict_forces: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Forward pass:
        Args:
            positions: (N, d)
            velocities: (N, d)
            external_force: (N, d) or None
            particle_types: (N,)
            predict_forces: if True, outputs raw accelerations including external g, else just internal
        Returns:
            dict:
                'acceleration': (N, d)
                'internal_acceleration': (N, d)
                'external_forces': (N, d)
        """
        N, d = positions.shape

        # Build input features
        node_features = self.encode_node_features(velocities, particle_types)

        # Start message passing
        h = node_features
        for layer in self.message_layers:
            h = layer(h)
            h = nn.relu(h)

        # Compute output acceleration
        acc = self.decoder(h)
        acc = acc.reshape((N, d))
        # By default, this is raw acceleration including external g if provided

        # Organize outputs
        output = {
            'acceleration': acc,
            'internal_acceleration': acc,
            'external_forces': external_force if external_force is not None else jnp.zeros_like(acc)
        }

        return output

# If equivariance is enabled, define SEGNN
class SEGNN(nn.Module):
    """
    E(3)-equivariant GNN for acceleration prediction using e3nn modules.
    """
    hyperparams: Dict
    num_types: int
    embedding_dim: int

    def setup(self):
        if e3nn is None:
            raise ImportError("e3nn is required for SEGNN but not installed.")

        self.hidden_dim = self.hyperparams.get('hidden_dim', 128)
        self.num_layers = self.hyperparams.get('num_layers', 10)

        # Particle type embedding if enabled
        if self.hyperparams.get('particle_type_embedding', True):
            self.type_embedding = ParticleTypeEmbedding(self.num_types, self.embedding_dim)
        else:
            self.type_embedding = None

        # Build E(3)-equivariant layers
        self.layers = []
        for _ in range(self.num_layers):
            layer = e3nn.nn.SequentialModule([
                e3nn.nn.Linear(self.hidden_dim),
                e3nn.nn.ReLU(),
                e3nn.nn.Linear(self.hidden_dim)
            ])
            self.layers.append(layer)

        # Final MLP to produce accelerations
        self.output_mlp = e3nn.nn.SequentialModule([
            e3nn.nn.Linear(self.hidden_dim),
            e3nn.nn.ReLU(),
            e3nn.nn.Linear(self.hidden_dim),
            e3nn.nn.ReLU(),
            e3nn.nn.Linear(self.hidden_dim, output_dim=3)  # 3D acceleration
        ])

    def encode_node_features(self, velocities, particle_types):
        """
        Encode node features with type embeddings if enabled.
        """
        features = [velocities]
        if self.type_embedding:
            type_emb = self.type_embedding(particle_types)
            features.append(type_emb)
        node_feat = jnp.concatenate(features, axis=-1)
        return node_feat

    def message_passing(self, h, edge_index, relative_positions):
        """
        Implement message passing with e3nn modules.
        Args:
            h: node features with e3nn types
            edge_index: (2, E) tensor of edges
            relative_positions: (E, d)
        Returns:
            Updated node features
        """
        # Build messages based on relative positions
        # For simplicity, implement a basic message function
        # e3nn edge models can consider geometric features
        messages = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            rel_pos = relative_positions[i]
            edge_feat = e3nn.SphericalTensor.tensor_from_tensor(rel_pos)
            msg = nn.Dense(self.hidden_dim)(h[src])
            messages.append(msg)
        messages = jnp.stack(messages, axis=0)
        # Aggregate messages per node
        # Here, for illustration, sum aggregation
        sum_messages = jnp.zeros_like(h)
        for i in range(edge_index.shape[1]):
            dst = edge_index[1, i]
            sum_messages = sum_messages.at[dst].add(messages[i])
        return sum_messages

    def __call__(self,
                 positions: jnp.ndarray,
                 velocities: jnp.ndarray,
                 external_force: Optional[jnp.ndarray],
                 particle_types: jnp.ndarray,
                 predict_forces: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Forward pass:
        Args same as GNS, returns:
            dict with keys:
                'acceleration': total acceleration
                'internal_acceleration': learned internal component
                'external_forces': external forcing component
        """
        N, d = positions.shape

        # Build input features
        node_features = self.encode_node_features(velocities, particle_types)

        # Construct edges based on neighbor relations
        # For this placeholder, assume edge_index and relative_positions are precomputed
        # In practice, include a neighbor search routine
        edge_index, relative_positions = self.build_graph(positions)

        h = node_features
        for layer in self.layers:
            h = layer(h)
            h = e3nn.nn.ReLU()(h)

        # Apply message passing
        messages = self.message_passing(h, edge_index, relative_positions)
        # Combine with node features
        h_updated = h + messages

        # Final acceleration prediction
        acc = self.output_mlp(h_updated)
        acc = acc.reshape((N, d))

        # Return as dictionary
        output = {
            'acceleration': acc,
            'internal_acceleration': acc,
            'external_forces': external_force if external_force is not None else jnp.zeros_like(acc)
        }
        return output

    def build_graph(self, positions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Placeholder for graph construction and relative position calculation.
        In practice, replace with neighbor search (e.g., via scipy KDTree in preprocessing).
        """
        # Dummy: connect each node to its K nearest neighbors
        # For code simplicity, assume fully connected (not efficient in practice)
        N = positions.shape[0]
        edge_index = jnp.array([[i for i in range(N) for _ in range(N)],
                                [j for i in range(N) for j in range(N)]])
        rel_positions = positions[edge_index[1]] - positions[edge_index[0]]
        return edge_index, rel_positions

# Additional utility functions for parameter save/load could be added if needed

```

## sph_relaxation.py

```python
## sph_relaxation.py
"""
SPHRelaxation: Implements particle distribution correction via SPH-based position relaxation.
This class performs iterative position updates to reduce clustering artifacts, enforce boundary
conditions, and stabilize long-term particle simulations. It supports configurable hyperparameters
and utilizes neighbor search, density, pressure, and viscous force computations based on classical SPH.
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Optional, Tuple

class SPHRelaxation:
    """
    Performs SPH-based position relaxation to improve particle distribution.
    Supports temperature-like pressure correction and viscosity forces.
    """
    def __init__(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        densities: Optional[np.ndarray] = None,
        particle_types: Optional[np.ndarray] = None,
        hyperparameters: dict = None,
        boundary_mask: Optional[np.ndarray] = None
    ):
        """
        Initialize the SPH relaxation routine.

        Args:
            positions (np.ndarray): Particle positions, shape (N_particles, dim).
            velocities (np.ndarray, optional): Velocities; used for viscous term. shape (N_particles, dim).
            densities (np.ndarray, optional): Particle densities; if None, will compute during relaxation.
            particle_types (np.ndarray, optional): Particle types; used for boundary conditions.
            hyperparameters (dict): Dictionary of relaxation hyperparameters:
                - alpha: float, force scale for pressure term.
                - beta: float, force scale for viscous term.
                - relaxation_steps: int, number of relaxation iterations.
                - kernel_radius: float, support radius for SPH kernels.
                - p_ref: float, reference pressure coefficient.
                - rho_ref: float, reference density (default: 1.0).
            boundary_mask (np.ndarray, optional): boolean array indicating boundary/wall particles.
        """
        self.positions = positions
        self.velocities = velocities if velocities is not None else np.zeros_like(positions)
        self.densities = densities
        self.types = particle_types
        self.boundary_mask = boundary_mask

        # Set default hyperparameters if not provided
        default_hp = {
            'alpha': 0.03,
            'beta': 0.0,
            'relaxation_steps': 3,
            'kernel_radius': 1.5,  # typical cutoff radius
            'p_ref': 1.0,
            'rho_ref': 1.0
        }
        if hyperparameters is None:
            hyperparameters = {}
        self.hyperparameters = {**default_hp, **hyperparameters}

        # Initialize neighbor search structures
        self.N, self.dim = self.positions.shape
        self.h = self.hyperparameters['kernel_radius']
        self._build_neighbor_structure()

    def _build_neighbor_structure(self):
        """
        Build neighbor list using KDTree for spatial searches within kernel support radius.
        """
        self.tree = KDTree(self.positions)

    def update_positions(self) -> np.ndarray:
        """
        Perform a single relaxation iteration, updating positions in-place or returning new positions.

        Returns:
            np.ndarray: Updated positions array (N_particles, dim).
        """
        # Step 1: Neighbor search
        neighbors_list = self._query_neighbors()

        # Step 2: Compute densities if not provided
        if self.densities is None:
            self.densities = self._compute_density(self.positions, neighbors_list)

        # Step 3: Compute pressure for each particle
        pressure = self._compute_pressure(self.densities)

        # Apply density correction/clipping for free surface stabilization
        density_clipped = self._density_clipping(self.densities)

        # Recompute pressure after density correction if needed
        pressure = self._compute_pressure(density_clipped)

        # Enforce boundary conditions on pressure if boundary mask provided
        if self.boundary_mask is not None:
            pressure = self._apply_boundary_conditions(pressure, neighbors_list)

        # Step 4: Compute forces (pressure + viscosity)
        pressure_force = self._compute_pressure_force(pressure, neighbors_list)
        viscous_force = self._compute_viscous_force(neighbors_list)
        # Scale forces with relaxation hyperparameters
        alpha = self.hyperparameters['alpha']
        beta = self.hyperparameters['beta']
        total_force = alpha * pressure_force + alpha * beta * viscous_force

        # Step 5: Position correction (relaxation)
        delta_positions = total_force / (self.densities[:, None] + 1e-8)  # avoid div by zero
        # Typically, relaxation updates positions based on force scaled by alpha
        # Here, total_force scaled outside, so directly add delta
        new_positions = self.positions + delta_positions

        return new_positions

    def relax(self, positions: np.ndarray, n_steps: int = 1, update_densities: bool = True) -> np.ndarray:
        """
        Perform multiple relaxation iterations.

        Args:
            positions (np.ndarray): Initial particle positions.
            n_steps (int): Number of relaxation iterations.
            update_densities (bool): Whether to recompute densities each iteration.

        Returns:
            np.ndarray: Relaxed particle positions.
        """
        current_positions = positions.copy()
        for _ in range(n_steps):
            self.positions = current_positions
            self._build_neighbor_structure()
            # Optionally update densities
            if update_densities:
                self.densities = self._compute_density(self.positions, self._query_neighbors())
            # Perform one relaxation step
            current_positions = self.update_positions()
        return current_positions

    def _query_neighbors(self) -> list:
        """
        Query neighbors within kernel support radius for current positions.

        Returns:
            list: list of neighbor indices per particle.
        """
        self.tree = KDTree(self.positions)
        neighbors_list = self.tree.query_ball_point(self.positions, r=self.h)
        return neighbors_list

    def _compute_density(self, positions: np.ndarray, neighbors_list: list) -> np.ndarray:
        """
        Compute density at each particle via kernel summation (Eq. 1).

        Args:
            positions (np.ndarray): (N, dim)
            neighbors_list (list): neighbor indices per particle

        Returns:
            np.ndarray: densities, shape (N,)
        """
        mass = 1.0  # assume unit mass per particle, or set accordingly
        densities = np.zeros(self.N)
        for i in range(self.N):
            nbrs = neighbors_list[i]
            if len(nbrs) == 0:
                continue
            r_ij = np.linalg.norm(positions[nbrs] - positions[i], axis=1)
            W_vals = sph_kernel_quintic(r_ij, self.h)
            densities[i] = np.sum(W_vals) * mass
        return densities

    def _compute_pressure(self, density: np.ndarray) -> np.ndarray:
        """
        Compute pressure using the equation of state p(rho).

        Args:
            density (np.ndarray): Densities (N,)

        Returns:
            np.ndarray: pressures (N,)
        """
        p_ref = self.hyperparameters['p_ref']
        rho_ref = self.hyperparameters['rho_ref']
        pressure = p_ref * (density / rho_ref - 1.0)
        return pressure

    def _density_clipping(self, density: np.ndarray, rho_ref: float = 1.0,
                          tol_lower: float = 0.98, tol_upper: float = 1.02) -> np.ndarray:
        """
        Clip densities to enforce bounds, reducing free surface inaccuracies.

        Args:
            density (np.ndarray): Raw densities.
            rho_ref (float): Reference density.
            tol_lower (float): lower threshold multiplier.
            tol_upper (float): upper threshold multiplier.

        Returns:
            np.ndarray: Corrected/clipped densities.
        """
        lower = rho_ref * tol_lower
        upper = rho_ref * tol_upper
        clamped_density = np.copy(density)
        clamped_density[clamped_density < lower] = rho_ref
        clamped_density[clamped_density > upper] = upper
        return clamped_density

    def _apply_boundary_conditions(self, pressure: np.ndarray, neighbors_list: list) -> np.ndarray:
        """
        Enforce boundary (wall) particles pressure based on neighbors to prevent penetration.

        Args:
            pressure (np.ndarray): Particle pressures.
            neighbors_list (list): neighbor indices per particle.

        Returns:
            np.ndarray: Pressure with boundary condition enforcement.
        """
        if self.boundary_mask is None:
            return pressure
        pressure_bc = np.copy(pressure)
        for i, is_boundary in enumerate(self.boundary_mask):
            if is_boundary:
                neighbor_indices = neighbors_list[i]
                # Only consider fluid neighbors (assuming boundary particles are flagged separately)
                neighbor_pressures = pressure[neighbor_indices]
                pressure_bc[i] = np.mean(neighbor_pressures)
        return pressure_bc

    def _compute_pressure_force(self, pressure: np.ndarray, neighbors_list: list) -> np.ndarray:
        """
        Calculate pressure gradient force using pairwise pressure differences (Eq. 3).

        Args:
            pressure (np.ndarray): Particle pressures.
            neighbors_list (list): Neighbor indices per particle.

        Returns:
            np.ndarray: Pressure force vectors (N, dim).
        """
        force = np.zeros_like(self.positions)
        mass = 1.0
        for i in range(self.N):
            nbrs = neighbors_list[i]
            if len(nbrs) == 0:
                continue
            r_i = self.positions[i]
            p_i = pressure[i]
            for j in nbrs:
                if i == j:
                    continue
                r_j = self.positions[j]
                r_ij = r_i - r_j
                r_norm = np.linalg.norm(r_ij) + 1e-8
                W_grad = sph_kernel_gradient(r_ij, r_norm, self.h)
                # pressure difference
                delta_p = p_i - pressure[j]
                force_contrib = -mass * delta_p * W_grad / (self.densities[j] + 1e-8)
                force[i] += force_contrib
        return force

    def _compute_viscous_force(self, neighbors_list: list) -> np.ndarray:
        """
        Compute viscous Laplacian force approximation (Eq. 4).

        Args:
            neighbors_list (list): Neighbor indices per particle.

        Returns:
            np.ndarray: Viscous force vectors (N, dim).
        """
        viscous_force = np.zeros_like(self.positions)
        nu = self.hyperparameters.get('nu', 0.0)  # optional, default 0
        h = self.h
        for i in range(self.N):
            nbrs = neighbors_list[i]
            if len(nbrs) == 0:
                continue
            r_i = self.positions[i]
            v_i = self.velocities[i]
            for j in nbrs:
                if i == j:
                    continue
                r_j = self.positions[j]
                v_j = self.velocities[j]
                r_ij = r_i - r_j
                r_norm = np.linalg.norm(r_ij) + 1e-8
                # Use Laplacian kernel derivative approximation
                lap = sph_kernel_laplacian(r_norm, h)
                visc_contrib = nu * (v_j - v_i) * lap / (self.densities[j] + 1e-8)
                viscous_force[i] += visc_contrib
        return viscous_force

    def sph_kernel_gradient(self, r: np.ndarray, r_norm: float, h: float) -> np.ndarray:
        """
        Evaluate gradient of the quintic kernel for distance r.

        Args:
            r (np.ndarray): R vector.
            r_norm (float): Norm of r.
            h (float): Support radius.

        Returns:
            np.ndarray: Gradient vector.
        """
        q = r_norm / h
        if q < 1:
            factor = -5 * (3 - q) ** 4 + 30 * (2 - q) ** 4 - 75 * (1 - q) ** 4
            W_grad = factor / (h * r_norm + 1e-8) * r
        elif 1 <= q < 2:
            factor = -5 * (3 - q) ** 4 + 30 * (2 - q) ** 4
            W_grad = factor / (h * r_norm + 1e-8) * r
        elif 2 <= q < 3:
            factor = 5 * (3 - q) ** 4
            W_grad = factor / (h * r_norm + 1e-8) * r
        else:
            W_grad = np.zeros_like(r)
        return W_grad

    def sph_kernel_laplacian(self, r_norm: float, h: float) -> float:
        """
        Evaluate the Laplacian of the kernel for particle positions.

        Args:
            r_norm (float): Distance between particles.
            h (float): Support radius.

        Returns:
            float: Laplacian value.
        """
        q = r_norm / h
        if q < 1:
            lap = -5 * (3 - q) ** 4 + 30 * (2 - q) ** 4 - 75 * (1 - q) ** 4
            lap /= (h ** 2)
        elif 1 <= q < 2:
            lap = -5 * (3 - q) ** 4 + 30 * (2 - q) ** 4
            lap /= (h ** 2)
        elif 2 <= q < 3:
            lap = 5 * (3 - q) ** 4
            lap /= (h ** 2)
        else:
            lap = 0.0
        return lap

```

## trainer.py

```python
## trainer.py

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pickle
from typing import Dict, Any, Tuple, List
from functools import partial

from dataset_loader import DatasetLoader, Dataset
from model import GNS, SEGNN
from utils import compute_velocity_std, effective_force_approximation
from sph_relaxation import SPHRelaxation

# Load configuration
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set random seed for reproducibility
rng_seed = config.get('misc', {}).get('random_seed', 42)
rng = jax.random.PRNGKey(rng_seed)

# Dataset loading
dataset_path = config['dataset'].get('path', 'datasets/lagrangian_fluid')
dataset_loader = DatasetLoader(dataset_path, config)

# Select and initialize model
model_type = config['model'].get('type', 'GNS').upper()
model_params = config['model'].get('params', {})

# Determine particle type count for embedding if needed
# For simplicity, assume types are integers starting from 0
# and infer max particle type label
sample_seq = dataset_loader.get_sequence(0)
num_types = int(np.max(sample_seq['types'])) + 1
particle_type_embedding = model_params.get('particle_type_embedding', True)

if model_type == 'GNS':
    model = GNS(hyperparams=model_params, num_types=num_types, embedding_dim=16)
elif model_type == 'SEGNN':
    model = SEGNN(hyperparams=model_params, num_types=num_types, embedding_dim=16)
else:
    raise ValueError(f"Unsupported model type: {model_type}")

# Initialize model parameters
import flax
import flax.linen as nn
from flax.training import train_state
import functools

@jax.jit
def init_model_params(rng, model, sample_input):
    variables = model.init(rng, *sample_input)
    return variables

sample_data = dataset_loader.get_sequence(0)
sample_positions = jnp.array(sample_data['positions'][0])  # shape: (N_particles, dim)
sample_velocities = jnp.array(sample_data['velocities'][0])  # shape: (N_particles, dim)
sample_type = jnp.array(sample_data['types'])  # shape: (N_particles,)
sample_external_force = jnp.zeros_like(sample_positions)  # placeholder

variables = init_model_params(rng, model, (sample_positions, sample_velocities, sample_external_force, sample_type, False))
params = variables

# Setup optimizer
learning_rate = config.get('training', {}).get('learning_rate', 0.001)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=learning_rate, weight_decay=1e-6)
)
# Create train state
import flax.training as train_module

class TrainState(train_module.TrainState):
    pass

state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

# Hyperparameters for training
loss_weights = config.get('training', {}).get('loss_weights', {
    'position_mse': 1.0,
    'velocity_mse': 0.1,
    'density_mae': 0.1
})

# External force smoothing parameters
sigma_scale = config.get('hyperparameters', {}).get('external_force_smoothing', {}).get('sigma_scale', 0.025)

# Relaxation hyperparameters
relax_params = config.get('hyperparameters', {}).get('relaxation', {
    'alpha': 0.03,
    'beta': 0.0,
    'relaxation_steps': 3
})

# Training hyperparameters
num_epochs = config.get('training', {}).get('epochs', 100)
batch_size = config.get('training', {}).get('batch_size', 64)
save_dir = config.get('checkpoint', {}).get('save_dir', 'checkpoints/')
save_freq = config.get('checkpoint', {}).get('save_frequency', 10)
resume = config.get('checkpoint', {}).get('resume', False)

# Create checkpoint directory if not exists
os.makedirs(save_dir, exist_ok=True)

# Utility functions for training
@jax.jit
def compute_loss(params, batch, model, external_force_features):
    """
    Compute total loss for a batch, including position, velocity, density, energy losses.
    """
    pos_seq = batch['positions']  # shape: (seq_len, N, dim)
    vel_seq = batch['velocities']
    types = batch['types']
    ext_forces = batch.get('external_forces', None)
    H = batch.get('history_len', 5)
    seq_len, N, dim = pos_seq.shape

    total_loss = 0.0
    total_reg_loss = 0.0
    total_density_loss = 0.0
    total_kin_energy_loss = 0.0
    total_dirichlet_loss = 0.0

    for t in range(H, seq_len-1):  # exclude initial steps if needed
        # Prepare model inputs
        past_positions = pos_seq[max(0, t - H):t, :, :]  # shape: (H, N, dim)
        past_velocities = vel_seq[max(0, t - H):t, :, :]
        # Use latest velocities for force smoothing
        velocities_for_sigma = vel_seq[max(0, t - 1, 0):t, :, :]  # shape: (1, N, dim)
        velocities_for_sigma = velocities_for_sigma[-1]  # shape: (N, dim)

        # Compute external force features if applicable
        if ext_forces is not None:
            ext_force_feat = ext_forces[t]  # shape: (N, dim)
        else:
            ext_force_feat = jnp.zeros_like(past_positions[-1, :, :])
        # Optional: smooth external force
        # For simplicity, assume forces are already smoothed if needed
        # TODO: integrate force smoothing here if required

        # Model prediction
        # Note: For JAX, need to define a function
        def model_apply_fn(params, positions, velocities, ext_force, types):
            return model.apply(params, positions, velocities, ext_force, types, predict_forces=True)
        # Prepare input features
        current_pos = pos_seq[t]
        current_vel = vel_seq[t]
        # For model, merge inputs as needed
        pred = model.apply(params, current_pos, current_vel, ext_force_feat, types, predict_forces=True)
        pred_acc = pred['acceleration']
        ext_force_pred = pred['external_forces']  # may contain learned external component

        # Compute target acceleration
        true_vel = vel_seq[t]
        prev_vel = vel_seq[t-1]
        delta_vel = true_vel - prev_vel
        delta_pos = pos_seq[t] - pos_seq[t-1]
        delta_time = 1.0  # assuming normalized timestep
        # Finite difference acceleration
        target_acc = (delta_vel - ext_force_feat) / delta_time  # Eq. 2, subtract external force to get internal acceleration

        # Predicted total acceleration: model + external force
        total_acc = pred_acc + ext_force_pred

        # Losses
        position_target = delta_pos
        position_pred = pred_acc + ext_force_pred  # acceleration leads to delta position via integration
        # For position loss, integrate acceleration (semi-implicit)
        pred_delta_pos = 0.5 * total_acc * delta_time ** 2
        position_loss = jnp.mean((pred_delta_pos - position_target) ** 2) * loss_weights['position_mse']

        velocity_loss = jnp.mean((delta_vel - total_acc * delta_time) ** 2) * loss_weights['velocity_mse']

        # Density loss via sampling and kernel sum
        # For simplicity, compute on a subset or the current state
        raw_density = compute_density(pos_seq[t], neighbor_search(pos_seq[t], hyperparameters.get('neighbor_search', {}).get('cutoff_radius', 1.5)), mass=1.0, h=dataset_loader.sequence_length)  # dummy
        density_mae = jnp.mean(jnp.abs(raw_density - 1.0)) * loss_weights['density_mae']

        # Optional: energy regularization (e.g., kinetic energy)
        kin_energy = 0.5 * jnp.sum(true_vel ** 2, axis=-1)
        pred_kin_energy = 0.5 * jnp.sum((true_vel + total_acc * delta_time) ** 2, axis=-1)
        kin_energy_loss = jnp.mean((pred_kin_energy - kin_energy) ** 2)

        # Optional: Dirichlet energy of density field
        dirichlet_energy_value = compute_dirichlet_energy(raw_density, pos_seq[t], h=dataset_loader.sequence_length)

        # Aggregate individual losses
        total_loss += position_loss + velocity_loss + density_mae + kin_energy_loss + dirichlet_energy_value
        total_reg_loss += density_mae
        total_density_loss += density_mae
        total_kin_energy_loss += kin_energy_loss
        total_dirichlet_loss += dirichlet_energy_value

    # Average over time steps
    n_steps = seq_len - 1 - H
    total_loss /= n_steps
    total_reg_loss /= n_steps
    total_density_loss /= n_steps
    total_kin_energy_loss /= n_steps
    total_dirichlet_loss /= n_steps

    # Combine losses
    total_loss_final = total_loss
    return total_loss_final, {
        'density_mae': total_density_loss,
        'kinetic_energy': total_kin_energy_loss,
        'dirichlet_energy': total_dirichlet_loss
    }

# Training loop
for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    # Shuffle dataset
    total_samples = dataset_loader.total_sequences
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    batch_losses = []

    for batch_start in range(0, total_samples, batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        batch_samples = [dataset_loader.get_sequence(i) for i in batch_indices]

        # Prepare batch data: shape (seq_len, N, dim)
        batch_positions = []
        batch_velocities = []
        batch_types = []
        batch_forces = []

        for sample in batch_samples:
            batch_positions.append(jnp.array(sample['positions']))  # shape: (seq_len, N, dim)
            batch_velocities.append(jnp.array(sample['velocities']))
            batch_types.append(jnp.array(sample['types']))
            if sample['external_forces'] is not None:
                batch_forces.append(jnp.array(sample['external_forces']))
            else:
                batch_forces.append(jnp.zeros_like(batch_positions[-1]))
        # Convert to array/dictionary
        batch_dict = {
            'positions': jnp.stack(batch_positions),  # (batch_size, seq_len, N, dim)
            'velocities': jnp.stack(batch_velocities),
            'types': jnp.stack(batch_types),
            'external_forces': jnp.stack(batch_forces)
        }
        # For simplicity, process one batch
        # TODO: vectorize over batch if needed

        # For each sample in batch, process
        def loss_fn(params):
            total_batch_loss = 0.0
            total_metrics = {}
            for i in range(batch_size):
                sample_batch = {
                    'positions': batch_dict['positions'][i],  # shape: (seq_len, N, dim)
                    'velocities': batch_dict['velocities'][i],
                    'types': batch_dict['types'][i],
                    'external_forces': batch_dict['external_forces'][i],
                    'history_len': 5
                }
                loss_value, metrics = compute_loss(params, sample_batch, model, batch['external_forces'][i])
                total_batch_loss += loss_value
                # Collect metrics if needed
            mean_loss = total_batch_loss / batch_size
            return mean_loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_value, metrics), grads = grad_fn(state.params)
        # Apply gradient clipping via optimizer chain
        state = state.apply_gradients(grads=grads)

        batch_losses.append(loss_value)

    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch} completed in {epoch_time:.2f} s, Avg Loss: {np.mean(batch_losses):.6f}")

    # Save checkpoint
    if epoch % save_freq == 0:
        checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({'params': jax.device_get(state.params), 'optimizer_state': jax.device_get(state.opt_state)}, f)

# Save final model
final_path = os.path.join(save_dir, 'final_model.pkl')
with open(final_path, 'wb') as f:
    pickle.dump({'params': jax.device_get(state.params), 'optimizer_state': jax.device_get(state.opt_state)}, f)

print("Training completed and model saved.")
```

## utils.py

```python
## utils.py
import numpy as np
from scipy.special import erf
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def compute_velocity_std(velocities: np.ndarray, window_size: int = 5) -> float:
    """
    Calculate the isotropic standard deviation (sigma_u) of velocity components
    over the most recent 'window_size' timesteps.

    Args:
        velocities (np.ndarray): Array of shape (N_particles, dim) representing velocities at latest timestep(s).
        window_size (int): Number of recent timesteps to consider for std calculation.

    Returns:
        float: Isotropic standard deviation across particle velocities.
    """
    # velocities shape: (sequence_length, N_particles, dim)
    if velocities.ndim != 3:
        raise ValueError(f"Expected velocities shape (seq_len, N_particles, dim), got {velocities.shape}")
    seq_len = velocities.shape[0]
    start_idx = max(0, seq_len - window_size)
    recent_vels = velocities[start_idx:, ...]  # shape: (window_size, N_particles, dim)
    # Compute std per component over the window and particles
    std_per_component = np.std(recent_vels, axis=0)  # shape: (N_particles, dim)
    # Average over particles
    std_per_component_mean = np.mean(std_per_component, axis=0)  # shape: (dim,)
    # Compute quadratic mean for isotropic sigma
    sigma_u = np.sqrt(np.mean(std_per_component_mean ** 2))
    return float(sigma_u)

def gaussian_convolve_force(force_field: np.ndarray, sigma: float) -> np.ndarray:
    """
    Approximate the convolution of a force field with a Gaussian kernel
    analytically using the error function (erf).

    Args:
        force_field (np.ndarray): Shape (N_particles, dim).
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        np.ndarray: Smoothed force field, same shape as input.
    """
    # For step-like force fields, analytical convolution along each component:
    # convolution with erf: f_smooth(y) = force_component * erf((y - y0)/ (sqrt(2)*sigma))
    # For general force fields, this simplifies to a weighted sum, but here we implement the analytical for step/constant forces.
    # Since force_field is per particle, and force varies per particle, we assume this applies component-wise.
    # For illustration, we smooth each component independently assuming a step function.
    # Alternatively, implement numerical convolution if force_field varies smoothly.
    force_smoothed = np.zeros_like(force_field)
    for d in range(force_field.shape[1]):
        force_smoothed[:, d] = erf(force_field[:, d] / (np.sqrt(2) * sigma))
    # Optional: scale back to force magnitude if force components are normalized
    return force_smoothed * np.max(np.abs(force_field), axis=0)

def effective_force_approximation(
    force_field: np.ndarray,
    velocities: np.ndarray,
    convolution_method: str = 'gaussian',
    sigma_scale: float = 0.025
) -> np.ndarray:
    """
    Compute the effective external force map based on velocity statistics
    and a smoothing kernel, either analytically or numerically.

    Args:
        force_field (np.ndarray): Shape (N_particles, dim), instantaneous external force.
        velocities (np.ndarray): Shape (N_particles, dim) velocity data for std calculation.
        convolution_method (str): 'gaussian' or 'erf'. Determines convolution approach.
        sigma_scale (float): Scaling factor for sigma derivation from velocity std.

    Returns:
        np.ndarray: Smoothed external force field, shape: (N_particles, dim).
    """
    # Compute velocity std deviation per particle
    # If multiple timesteps are available, compute over recent historical data
    sigma_u = np.std(velocities, axis=0)  # shape: (dim,)
    sigma = np.sqrt(np.mean(sigma_u ** 2)) * sigma_scale

    # For spatial smoothing, convolve force with Gaussian, analytical approximation:
    if convolution_method == 'erf':
        # Use erf of (force / (sqrt(2)*sigma))
        smoothed_force = np.zeros_like(force_field)
        for d in range(force_field.shape[1]):
            smoothed_force[:, d] = erf(force_field[:, d] / (np.sqrt(2) * sigma))
        return smoothed_force
    elif convolution_method == 'gaussian':
        # Numerical approximation via kernel sum
        # For simplicity, assume force_field is already spatially sampled
        return gaussian_convolve_force(force_field, sigma)
    else:
        raise ValueError(f"Unknown convolution method: {convolution_method}")

def neighbor_search(positions: np.ndarray, cutoff_radius: float) -> list:
    """
    Identify neighboring particles within cutoff radius using KDTree.

    Args:
        positions (np.ndarray): Shape (N_particles, dim).
        cutoff_radius (float): Radius for neighbor search.

    Returns:
        list: List of lists, where each sublist contains neighbor indices for the corresponding particle.
    """
    tree = KDTree(positions)
    neighbors_list = tree.query_ball_point(positions, r=cutoff_radius)
    return neighbors_list

def sph_kernel_quintic(r: np.ndarray, h: float) -> np.ndarray:
    """
    Evaluate the quintic spline kernel W(r|h) for each inter-particle distance r.

    Args:
        r (np.ndarray): Distances, shape: (num_neighbors,)
        h (float): Smoothing length.

    Returns:
        np.ndarray: Kernel evaluations at each r, shape: (num_neighbors,)
    """
    q = r / h
    W = np.zeros_like(q)

    # Coefficients for the quintic spline kernel (from Monaghan 1993)
    sigma = 7 / (478 * np.pi * h ** 3)  # normalization constant in 3D
    for i, qi in enumerate(q):
        if 0 <= qi < 1:
            W[i] = sigma * ((3 - qi) ** 5 - 6 * (2 - qi) ** 5 + 15 * (1 - qi) ** 5)
        elif 1 <= qi < 2:
            W[i] = sigma * ((3 - qi) ** 5 - 6 * (2 - qi) ** 5)
        elif 2 <= qi < 3:
            W[i] = sigma * (3 - qi) ** 5
        else:
            W[i] = 0.0
    return W

def compute_density(
    positions: np.ndarray,
    neighbor_list: list,
    mass: float,
    h: float,
    rho_min: float = 0.98,
    rho_max: float = 1.02,
    rho_ref: float = 1.0
) -> np.ndarray:
    """
    Calculate the density at each particle via kernel summation (Eq. 1).
    Apply density clipping for free surface correction.

    Args:
        positions (np.ndarray): (N_particles, dim)
        neighbor_list (list): List of neighbor indices per particle
        mass (float): Uniform mass for each particle
        h (float): Kernel support radius
        rho_min, rho_max (float): Clipping thresholds relative to rho_ref
        rho_ref (float): Reference density

    Returns:
        density (np.ndarray): Shape (N_particles,)
    """
    N = positions.shape[0]
    density = np.zeros(N)
    for i in range(N):
        neighbors = neighbor_list[i]
        r_ijs = np.linalg.norm(positions[neighbors] - positions[i], axis=1)  # shape: (num_neighbors,)
        W_vals = sph_kernel_quintic(r_ijs, h)
        density[i] = np.sum(W_vals) * mass
    # Density clipping: enforce minimum and maximum
    lower_bound = rho_ref * rho_min
    upper_bound = rho_ref * rho_max
    density = np.clip(density, lower_bound, upper_bound)
    return density

def compute_pressure(density: np.ndarray, p_ref: float, rho_ref: float = 1.0) -> np.ndarray:
    """
    Compute pressure using the equation of state p = p_ref * (rho / rho_ref - 1).

    Args:
        density (np.ndarray): Particle density (N_particles,)
        p_ref (float): Reference pressure coefficient
        rho_ref (float): Reference density

    Returns:
        pressure (np.ndarray): Shape (N_particles,)
    """
    pressure = p_ref * (density / rho_ref - 1.0)
    return pressure

def pressure_clamp(pressure: np.ndarray, rho_ref: float = 1.0, clip_min: float=0.98, clip_max: float=1.02) -> np.ndarray:
    """
    Clamp pressure values to prevent tensile instability, based on thresholds.

    Args:
        pressure (np.ndarray): Unclamped pressure values
        rho_ref (float): Reference density
        clip_min, clip_max (float): Clipping thresholds relative to rho_ref

    Returns:
        np.ndarray: Clamped pressure values
    """
    min_val = rho_ref * clip_min
    max_val = rho_ref * clip_max
    return np.clip(pressure, min_val, max_val)

def density_at_surface_correction(raw_density: np.ndarray, rho_ref: float=1.0, tol_lower=0.98, tol_upper=1.02) -> np.ndarray:
    """
    Correct densities at free surfaces by clipping and enforcing threshold bounds.

    Args:
        raw_density (np.ndarray): Raw density from summation
        rho_ref (float): Reference density
        tol_lower, tol_upper (float): Tolerance bounds for density clipping

    Returns:
        corrected_density (np.ndarray): Density after correction
    """
    corrected_density = np.copy(raw_density)
    # Set densities below threshold to rho_ref
    corrected_density[corrected_density < rho_ref * tol_lower] = rho_ref
    # Clip densities to upper threshold
    corrected_density[corrected_density > rho_ref * tol_upper] = rho_ref * tol_upper
    return corrected_density

def boundary_condition_wall(pressure: np.ndarray, neighbors: list, wall_mask: np.ndarray) -> np.ndarray:
    """
    Enforce wall boundary conditions, setting wall particle pressures to average of neighbors,
    avoiding penetration and modeling impermeability.

    Args:
        pressure (np.ndarray): Current pressures per particle
        neighbors (list): List of neighbor indices per particle
        wall_mask (np.ndarray): Boolean array: True for wall particles

    Returns:
        np.ndarray: Enforced pressure array
    """
    pressure_enforced = np.copy(pressure)
    for i, is_wall in enumerate(wall_mask):
        if is_wall:
            neighbor_indices = neighbors[i]
            # Only consider neighboring fluid particles for pressure averaging
            neighbor_pressures = pressure[neighbor_indices]
            # Set wall particle pressure to average neighbor pressure
            pressure_enforced[i] = np.mean(neighbor_pressures)
    return pressure_enforced

def compute_dirichlet_energy(
    density: np.ndarray,
    positions: np.ndarray,
    h: float
) -> float:
    """
    Calculate Dirichlet energy of the density field to quantify clustering and instability.

    Args:
        density (np.ndarray): Particle densities
        positions (np.ndarray): Particle positions, shape (N, dim)
        h (float): Kernel smoothing length used for gradient approximation

    Returns:
        float: Total Dirichlet energy
    """
    N, dim = positions.shape
    # Approximate gradient of density via kernel derivatives
    energy = 0.0
    for i in range(N):
        neighbors = neighbor_search(positions, h)[i]
        r_ij = positions[neighbors] - positions[i]
        r_norm = np.linalg.norm(r_ij, axis=1) + 1e-8
        # Derivative of kernel w.r.t. r
        grad_W = sph_kernel_gradient(r_ij, r_norm, h)
        # Sum over neighbors
        grad_density = np.sum(grad_W * (density[neighbors][:, np.newaxis] / density[i]), axis=0)
        energy += np.linalg.norm(grad_density) ** 2
    return energy / N

def sph_kernel_gradient(r_vectors: np.ndarray, r_norm: np.ndarray, h: float) -> np.ndarray:
    """
    Compute the gradient of the quintic spline kernel with respect to position.

    Args:
        r_vectors (np.ndarray): Vector differences, shape (num_neighbors, dim)
        r_norm (np.ndarray): Norms of r_vectors, shape: (num_neighbors,)
        h (float): Smoothing length

    Returns:
        np.ndarray: Gradient vectors, shape (num_neighbors, dim)
    """
    q = r_norm / h
    # Compute derivative of W with respect to q
    # Derivative of quintic spline
    dW_dq = np.zeros_like(q)
    # Implement piecewise derivatives similar to sph_kernel_quintic
    for i, qi in enumerate(q):
        if 0 <= qi < 1:
            dW_dq[i] = (1/h) * ( ( -5 * (3 - qi) ** 4 + 30 * (2 - qi) ** 4 - 75 * (1 - qi) ** 4))
        elif 1 <= qi < 2:
            dW_dq[i] = ( -5 * (3 - qi) ** 4 + 30 * (2 - qi) ** 4)
        elif 2 <= qi < 3:
            dW_dq[i] = (5 * (3 - qi) ** 4)
        else:
            dW_dq[i] = 0.0
    # Gradient: dW/dx = (dW/dq) * (r_vector / r_norm) / h
    grad_W = (dW_dq / (r_norm + 1e-8))[:, np.newaxis] * r_vectors
    return grad_W

def visualize_particle_field(
    positions: np.ndarray,
    density: Optional[np.ndarray] = None,
    title: str = "Particle Field"
):
    """
    Generate and display scatter plot of particles, optionally colored by density.

    Args:
        positions (np.ndarray): Particle positions, shape (N, dim)
        density (np.ndarray, optional): Particle densities for coloring.
        title (str): Plot title.
    """
    plt.figure(figsize=(6,6))
    if density is not None:
        plt.scatter(positions[:, 0], positions[:, 1], c=density, cmap='viridis', s=20)
        plt.colorbar(label='Density')
    else:
        plt.scatter(positions[:, 0], positions[:, 1], c='blue', s=20)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.show()
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\neuralsph\neuralsph_repo`
