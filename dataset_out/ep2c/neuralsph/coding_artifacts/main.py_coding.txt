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