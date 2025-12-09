# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py

import os
import glob
import numpy as np
import pickle
import json
import h5py
from typing import Tuple, List, Dict, Optional

import torch
from torch.utils.data import Dataset

# FEM simulation library (assuming fenics is installed)
# Because actual FEM implementation is complex, here we define an interface placeholder.
# Replace with actual FEM solver code as needed.
try:
    import fenics
except ImportError:
    fenics = None  # Placeholder if fenics is not installed

# Geometry processing libraries
import pyvista as pv

# Config parser to load configuration from YAML
import yaml

# Load the config.yaml to access dataset parameters
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Constants from config
DATASET_NAME = CONFIG['dataset']['dataset_name']
TRAIN_SIZE = CONFIG['dataset']['train_size']
TEST_SIZE = CONFIG['dataset']['test_size']
VALIDATION_SIZE = CONFIG['dataset']['validation_size']
FREQUENCY_RANGE = CONFIG['dataset']['frequency_range']
FREQUENCY_POINTS = CONFIG['dataset']['frequency_points']
DISCRETIZATION_MESH = CONFIG['dataset']['discretization_mesh']

# Data directory (assumed structure)
DATA_DIR = './data/'  # Update as per actual data location

# Response data (precomputed FEM results) directory
RESPONSE_DATA_DIR = os.path.join(DATA_DIR, DATASET_NAME, 'responses')

# Shape data directory
SHAPE_DATA_DIR = os.path.join(DATA_DIR, DATASET_NAME, 'shapes')

# Helper functions

def load_shape_file(shape_path: str):
    """Loads raw shape geometry file (mesh, STL, OBJ, etc)."""
    mesh = pv.read(shape_path)
    return mesh

def convert_mesh_to_sdf(mesh: pv.PolyData, grid_size: Tuple[int, int]) -> np.ndarray:
    """Convert mesh to signed distance function (SDF) grid representation."""
    # Create a uniform grid for sampling
    # Define domain bounds (assuming shape in normalized coordinate space)
    bounds = mesh.bounds
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    # For 2D plate, flatten to 2D bounds
    # For simplicity, assume planar shape in XY
    nx, ny = grid_size
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    xv, yv = np.meshgrid(xs, ys, indexing='ij')
    # Sample points on grid
    points = np.vstack([xv.ravel(), yv.ravel(), np.zeros_like(xv.ravel())]).T
    # Compute signed distances (inside/outside)
    signed_distances = mesh.compute_implicit_distance(points)
    sdf = signed_distances.reshape((nx, ny))
    return sdf

def normalize_properties(props: Dict) -> np.ndarray:
    """Normalize scalar properties as per dataset default ranges."""
    # Pull properties
    length = props.get('length', 1.0)  # meters
    width = props.get('width', 1.0)
    thickness = props.get('thickness', 1.0)
    density = props.get('density', 1.0)  # kg/m^3
    youngs_modulus = props.get('youngs_modulus', 1.0)  # Pa
    poisson_ratio = props.get('poisson_ratio', 0.3)
    loss_factor = props.get('loss_factor', 0.02)
    boundary_stiffness = props.get('boundary_stiffness', 0.0)
    load_x = props.get('load_position_x', 0.5)
    load_y = props.get('load_position_y', 0.5)

    # Normalize (assuming known ranges, else scale between 0-1)
    # Here, for simplicity, normalize to [0,1] based on dataset bounds
    # Replace with actual normalization if known
    prop_array = np.array([
        length, width, thickness, density, youngs_modulus,
        poisson_ratio, loss_factor, boundary_stiffness, load_x, load_y
    ], dtype=np.float32)

    return prop_array

def load_response_data(sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads FEM simulation results: velocity fields and response response over frequencies."""
    response_path = os.path.join(RESPONSE_DATA_DIR, f"{sample_id}_response.hdf5")
    with h5py.File(response_path, 'r') as f:
        # velocity data: shape (num_freq_points, nx, ny)
        velocity_data = f['velocity'][:]
        response_data = f['response'][:]
        # Convert to np.ndarray
        velocity = np.array(velocity_data)
        response = np.array(response_data)
    return velocity, response

# Define a data object class
class ShapeData:
    def __init__(self, mesh: pv.PolyData, sdf: np.ndarray):
        self.mesh = mesh
        self.sdf = sdf  # Signed distance function grid

class MaterialProperties:
    def __init__(self, array: np.ndarray):
        self.tensor = torch.tensor(array, dtype=torch.float32)

class ResponseFunction:
    def __init__(self, response: np.ndarray):
        self.response = torch.tensor(response, dtype=torch.float32)  # shape (num_freq_points,)

# Dataset class
class VibratingPlatesDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',  # options: 'train', 'val', 'test'
        shape_ids: Optional[List[str]] = None,
        load_precomputed: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.shape_ids = shape_ids
        self.load_precomputed = load_precomputed

        # Load all sample IDs
        if shape_ids is None:
            shape_files = glob.glob(os.path.join(SHAPE_DATA_DIR, '*.stl'))
            shape_ids_full = [os.path.splitext(os.path.basename(p))[0] for p in shape_files]
            # Shuffle and split
            np.random.seed(42)  # ensure reproducibility
            np.random.shuffle(shape_ids_full)
            total = len(shape_ids_full)
            if split == 'train':
                self.shape_ids = shape_ids_full[:int(0.8*total)]
            elif split == 'val':
                self.shape_ids = shape_ids_full[int(0.8*total):int(0.9*total)]
            elif split == 'test':
                self.shape_ids = shape_ids_full[int(0.9*total):]
        # Else use provided list
        else:
            # Assume provided IDs exist
            pass

        # Preload data or set to load on demand
        # For large dataset, loading on demand is preferable
        self.data_cache = {}  # Optional: cache preprocessed data if needed

        # Load all shape file paths
        self.shape_paths = [
            os.path.join(SHAPE_DATA_DIR, f"{sid}.stl") for sid in self.shape_ids
        ]

        # Response frequency array (assumed same for all)
        self.freqs = np.linspace(FREQUENCY_RANGE['min'], FREQUENCY_RANGE['max'], FREQUENCY_POINTS)

    def __len__(self):
        return len(self.shape_ids)

    def __getitem__(self, idx):
        shape_id = self.shape_ids[idx]

        # Load shape
        shape_path = self.shape_paths[idx]
        mesh = load_shape_file(shape_path)

        # Convert shape to sdf grid
        sdf = convert_mesh_to_sdf(mesh, grid_size=(int(DISCRETIZATION_MESH.split('x')[0]), int(DISCRETIZATION_MESH.split('x')[1])))

        shape_data = ShapeData(mesh=mesh, sdf=sdf)

        # Load or generate properties
        # Assuming properties stored in a JSON file
        props_path = os.path.join(DATA_DIR, 'properties', f"{shape_id}.json")
        with open(props_path, 'r') as f:
            props_dict = json.load(f)
        props_array = normalize_properties(props_dict)
        props_tensor = torch.tensor(props_array, dtype=torch.float32)

        # Load response data
        velocity, response = load_response_data(shape_id)

        # Convert velocity to tensor (shape: freq_points x nx x ny)
        velocity_tensor = torch.tensor(velocity, dtype=torch.float32)

        # Response spectrum tensor
        response_tensor = torch.tensor(response, dtype=torch.float32)

        sample = {
            'shape_enc': torch.tensor(sdf, dtype=torch.float32),  # shape encoding
            'props': props_tensor,
            'freqs': torch.tensor(self.freqs, dtype=torch.float32),
            'velocity': velocity_tensor,
            'response': response_tensor,
            'shape_id': shape_id
        }

        return sample

    def get_all_freqs(self):
        return torch.tensor(self.freqs, dtype=torch.float32)

    def load_data_cache(self):
        # Optional: preload all data into memory for fast access
        pass

# Note: For actual FEM simulation, replace load_response_data with a function
# that runs the FEM solver on-the-fly or precomputes/stores results as above.

# Additional helper functions for FEM simulation (placeholder)
def run_fem_simulation(shape_mesh: pv.PolyData, properties: Dict, frequencies: np.ndarray):
    """
    Runs FEM simulation over specified frequencies, returning velocity fields and response.
    This function must implement the PDE solving, boundary conditions, load application, etc.
    For this project, precomputed responses are loaded; implementation here is a placeholder.
    """
    # Placeholder: Return zeros
    num_freqs = len(frequencies)
    nx, ny = int(DISCRETIZATION_MESH.split('x')[0]), int(DISCRETIZATION_MESH.split('x')[1])
    velocity_fields = np.zeros((num_freqs, nx, ny))
    response = np.zeros(num_freqs)
    return velocity_fields, response

# To generate dataset from scratch, implement functions calling FEM solver, then store responses.
```


## evaluation.py

```python
## evaluation.py

import numpy as np
import scipy.stats
import scipy.signal
import scipy.optimize
import matplotlib.pyplot as plt

from typing import List, Tuple

# Load configuration
import yaml
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Extract frequency parameters and response grid from config
FREQ_MIN = CONFIG['dataset']['frequency_range'].get('min', 1.0)
FREQ_MAX = CONFIG['dataset']['frequency_range'].get('max', 300.0)
N_FREQ_POINTS = CONFIG['dataset'].get('frequency_points', 300)
FREQUENCIES = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ_POINTS)

# 1. Compute Earth Mover's Distance (Wasserstein)
def compute_emd(true_response: np.ndarray, pred_response: np.ndarray) -> float:
    """
    Compute Earth Mover's Distance (Wasserstein) between true and predicted response distributions.
    Assumes responses are on the same frequency grid.
    Responses should be positive; normalize to sum to 1.
    """
    # Normalize responses to probability distributions
    true_prob = true_response / np.sum(true_response) if np.sum(true_response) > 0 else np.ones_like(true_response) / len(true_response)
    pred_prob = pred_response / np.sum(pred_response) if np.sum(pred_response) > 0 else np.ones_like(pred_response) / len(pred_response)

    # Use scipy's wasserstein_distance; frequencies are coordinates
    emd = scipy.stats.wasserstein_distance(FREQUENCIES, FREQUENCIES, u_weights=true_prob, v_weights=pred_prob)
    return emd

# 2. Peak detection
def detect_peaks(response_vector: np.ndarray, prominence: float = 0.5) -> np.ndarray:
    """
    Detect peaks in response_vector using scipy.signal.find_peaks.
    Response vector: 1D array of response over FREQUENCIES.
    Prominence threshold as in dataset analysis.
    """
    peaks_idx, _ = scipy.signal.find_peaks(response_vector, prominence=prominence)
    peaks_freqs = FREQUENCIES[peaks_idx]
    return peaks_freqs

# 3. Peak matching using Hungarian algorithm
def match_peaks(gt_peaks: np.ndarray, pred_peaks: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Match ground-truth peaks to predicted peaks.
    Returns list of index pairs (gt_idx, pred_idx).
    """
    if len(gt_peaks) == 0 or len(pred_peaks) == 0:
        return [], np.array([])

    cost_matrix = np.abs(gt_peaks[:, None] - pred_peaks[None, :])  # shape (gt_peaks, pred_peaks)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    matches = list(zip(row_ind, col_ind))
    return matches, cost_matrix

# 4. Peak ratio and shift error
def compute_peak_errors(gt_peaks: np.ndarray, pred_peaks: np.ndarray, matches: List[Tuple[int, int]], gt_response: np.ndarray, pred_response: np.ndarray) -> Tuple[float, float]:
    """
    Compute peak ratio error and mean peak shift error.
    """
    num_gt = len(gt_peaks)
    num_pred = len(pred_peaks)
    if num_gt == 0 or num_pred == 0:
        # No peaks detected in either, define errors as zero or full (1)
        ratio_error = 1.0
        shift_error = np.nan
        return ratio_error, shift_error

    ratio = min(num_gt / num_pred, num_pred / num_gt)
    ratio_error = 1.0 - ratio  # minimal ratio-based error
    
    # Compute shift errors for matched peaks
    shift_errors = []
    for gt_idx, pred_idx in matches:
        shift = np.abs(gt_peaks[gt_idx] - pred_peaks[pred_idx])
        shift_errors.append(shift)
    if len(shift_errors) > 0:
        mean_shift = np.mean(shift_errors)
    else:
        mean_shift = np.nan
    return ratio_error, mean_shift

# 5. Plot responses and peak detection
def plot_responses(freqs: np.ndarray, true_response: np.ndarray, pred_response: np.ndarray,
                   gt_peaks: np.ndarray = None, pred_peaks: np.ndarray = None):
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, true_response, label='Ground Truth', linewidth=2)
    plt.plot(freqs, pred_response, label='Prediction', linewidth=2, linestyle='--')
    if gt_peaks is not None:
        plt.scatter(gt_peaks, np.interp(gt_peaks, freqs, true_response), marker='x', color='red', label='GT peaks')
    if pred_peaks is not None:
        plt.scatter(pred_peaks, np.interp(pred_peaks, freqs, pred_response), marker='o', color='blue', label='Pred peaks')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Normalized Response')
    plt.legend()
    plt.grid(True)
    plt.title('Frequency Response Comparison')
    plt.show()

# 6. Plot peak alignment with lines connecting matched peaks
def plot_peak_matches(freqs: np.ndarray,
                      true_response: np.ndarray, pred_response: np.ndarray,
                      gt_peaks: np.ndarray, pred_peaks: np.ndarray,
                      matches: List[Tuple[int, int]]):
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, true_response, label='Ground Truth', linewidth=2)
    plt.plot(freqs, pred_response, label='Prediction', linewidth=2, linestyle='--')
    # Plot unmatched peaks
    unmatched_gt = set(range(len(gt_peaks))) - set(i for i, _ in matches)
    unmatched_pred = set(range(len(pred_peaks))) - set(j for _, j in matches)
    plt.scatter(gt_peaks[list(unmatched_gt)], np.interp(gt_peaks[list(unmatched_gt)], freqs, true_response),
                marker='x', color='red', s=100, label='Unmatched GT')
    plt.scatter(pred_peaks[list(unmatched_pred)], np.interp(pred_peaks[list(unmatched_pred)], freqs, pred_response),
                marker='o', color='blue', s=100, label='Unmatched Pred')
    # Draw lines for matched peaks
    for gt_idx, pred_idx in matches:
        plt.plot([gt_peaks[gt_idx], pred_peaks[pred_idx]],
                 [np.interp(gt_peaks[gt_idx], freqs, true_response),
                  np.interp(pred_peaks[pred_idx], freqs, pred_response)],
                 'k--', linewidth=0.8)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Normalized Response')
    plt.legend()
    plt.title('Peak Matching')
    plt.grid(True)
    plt.show()

# 7. (Optional) Velocity field visualization placeholder
# Assuming velocity fields are 2D arrays over a spatial grid
def plot_velocity_field(velocity: np.ndarray, title: str = "Velocity Field"):
    import matplotlib.pyplot as plt
    plt.imshow(velocity, cmap='viridis', origin='lower')
    plt.colorbar(label='Velocity magnitude')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# 8. Utility to perform overall evaluation on a single sample
def evaluate_response(gt_response: np.ndarray, pred_response: np.ndarray,
                      gt_velocity: np.ndarray = None, pred_velocity: np.ndarray = None,
                      prominence: float = 0.5) -> dict:
    """
    Evaluate response shape similarity, peak detection, peak shift, and visualizations.
    Inputs:
        gt_response, pred_response: numpy arrays over FREQUENCIES
        gt_velocity, pred_velocity: optional 2D spatial arrays
    Returns:
        dict of metrics, scores, peak info
    """
    results = {}
    # Normalize responses
    gt_norm = gt_response / np.sum(gt_response) if np.sum(gt_response) > 0 else np.ones_like(gt_response) / len(gt_response)
    pred_norm = pred_response / np.sum(pred_response) if np.sum(pred_response) > 0 else np.ones_like(pred_response) / len(pred_response)

    # 1. Waterstein (EMD)
    results['EMD'] = compute_emd(gt_norm, pred_norm)

    # 2. Peak detection
    gt_peaks = detect_peaks(gt_response, prominence=prominence)
    pred_peaks = detect_peaks(pred_response, prominence=prominence)
    results['GroundTruthPeaks'] = gt_peaks
    results['PredictedPeaks'] = pred_peaks

    # 3. Peak matching
    matches, cost_matrix = match_peaks(gt_peaks, pred_peaks)
    # 4. Peak ratio and shift error
    ratio_error, shift_error = compute_peak_errors(gt_peaks, pred_peaks, matches, gt_response, pred_response)
    results['PeakRatioError'] = ratio_error
    results['PeakShift'] = shift_error

    # For visualization
    results['matches'] = matches
    results['gt_peaks'] = gt_peaks
    results['pred_peaks'] = pred_peaks

    return results

# 9. Optional: example visualization of response and peaks
def visualize_response_and_peaks(freqs, gt_response, pred_response, results):
    plot_responses(freqs, gt_response, pred_response,
                   gt_peaks=results['gt_peaks'],
                   pred_peaks=results['pred_peaks'])
    # Plot peak matches
    plot_peak_matches(freqs, gt_response, pred_response,
                      results['gt_peaks'], results['pred_peaks'], results['matches'])

# END of evaluation.py
```

## main.py

```python
## main.py
import os
import time
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import project modules
import dataset_loader
import model
import evaluation
import visualization

def main():
    # ---------------------------
    # 1. Load configuration
    # ---------------------------
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset parameters
    dataset_name = cfg['dataset']['dataset_name']
    train_size = cfg['dataset']['train_size']
    val_size = cfg['dataset']['validation_size']
    test_size = cfg['dataset']['test_size']
    freq_range = cfg['dataset']['frequency_range']
    freq_points = cfg['dataset']['frequency_points']
    discretization_mesh = cfg['dataset'].get('discretization_mesh', '181x121')
    mesh_dims = tuple(map(int, discretization_mesh.split('x')))
    freq_min = float(freq_range.get('min',1))
    freq_max = float(freq_range.get('max',300))
    freqs = np.linspace(freq_min, freq_max, freq_points)

    # Model parameters
    model_arch = cfg['model']['architecture']
    encoder_type = cfg['model'].get('encoder', {}).get('type', 'implicit_shape_encoder')
    shape_representation = cfg['model'].get('encoder', {}).get('shape_representation', 'signed_distance_function')
    scalar_dim = 7  # as indicated
    channels = cfg['model'].get('channels', 64)
    depth = cfg['model'].get('depth', 4)
    response_type = cfg['model'].get('response_decoder', {}).get('type', 'velocity_field')
    
    # Training parameters
    lr = cfg['training'].get('learning_rate', 1e-3)
    batch_size = cfg['training'].get('batch_size', 16)
    max_epochs = cfg['training'].get('epochs', 300)
    early_patience = cfg['training'].get('early_stopping_patience', 20)
    loss_weights = cfg['training']['loss_weights']
    v_loss_weight = loss_weights.get('velocity_loss_weight', 0.25)
    f_loss_weight = loss_weights.get('response_loss_weight', 0.75)
    save_checkpoints = cfg['training'].get('save_checkpoints', True)
    save_best_only = cfg['training'].get('save_best_only', True)

    # ---------------------------
    # 2. Dataset loading / generation
    # ---------------------------
    print("Loading datasets...")
    # Load or generate datasets
    # Here, we'll assume pre-generated datasets (more realistic).
    # If generating, implement a call to dataset_loader.generate_dataset()
    full_train_dataset = dataset_loader.VibratingPlatesDataset(
        data_dir='./data',
        split='train'
    )
    test_dataset = dataset_loader.VibratingPlatesDataset(
        data_dir='./data',
        split='test'
    )
    # For validation, take subset from train
    total_train_samples = len(full_train_dataset)
    indices = np.arange(total_train_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    val_idx = indices[:int(0.1*total_train_samples)]  # 10% for validation
    train_idx = indices[int(0.1*total_train_samples):]
    train_subset = torch.utils.data.Subset(full_train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_train_dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Datasets loaded.")

    # ---------------------------
    # 3. Model instantiation
    # ---------------------------
    print(f"Initializing model: {model_arch}")
    net = model.LearningVibrationModel()
    net.to(device)

    # Load checkpoint if available
    checkpoint_path = './checkpoint_best.pth'
    start_epoch = 1
    best_val_loss = np.inf
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        chkpt = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(chkpt['model_state_dict'])
        start_epoch = chkpt['epoch'] + 1
        best_val_loss = chkpt['best_val_loss']

    # ---------------------------
    # 4. Optimizer and scheduler
    # ---------------------------
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    # ---------------------------
    # 5. Training loop
    # ---------------------------
    print("Starting training...")
    for epoch in range(start_epoch, max_epochs+1):
        start_time = time.time()
        net.train()
        total_loss = 0.0
        total_vloss = 0.0
        total_floss = 0.0
        for batch in train_loader:
            shape_sdf = batch['shape_enc'].to(device)        # (B,H,W)
            properties = batch['props'].to(device)           # (B,7)
            freqs_np = batch['freqs'].to(device)             # (F,)
            velocities_gt = batch['velocity'].to(device)     # (B,F,H,W)
            responses_gt = batch['response'].to(device)      # (B,F)
            B, F, H, W = velocities_gt.shape

            optimizer.zero_grad()

            # Prepare expanded inputs for all frequencies in batch
            shape_input_exp = shape_sdf.unsqueeze(1).expand(-1, F, -1, -1).reshape(-1, H, W)  # (B*F,H,W)
            props_exp = properties.unsqueeze(1).expand(-1, F, -1).reshape(-1, 7)
            freq_flat = freqs_np.unsqueeze(0).expand(B, -1).reshape(-1)

            if net.response_type == 'velocity_field':
                pred_vel = net(shape_input_exp, props_exp, freq_flat)  # (B*F,2,H,W)
                # Log scale for loss
                pred_vel_log = torch.log(torch.clamp(pred_vel**2 + 1e-8, min=1e-8))
                gt_vel_flat = velocities_gt.reshape(-1, 2, H, W)
                gt_vel_log = torch.log(torch.clamp(gt_vel_flat**2 + 1e-8, min=1e-8))
                velocity_loss = nn.functional.mse_loss(pred_vel_log, gt_vel_log)

                # Derive frequency response (mean squared velocity)
                pred_resp = ((pred_vel_log**2).mean(dim=[2,3])) * 10  # scaled to dB
                resp_gt_flat = responses_gt.reshape(-1)
                resp_mean = resp_gt_flat.mean()
                resp_std = resp_gt_flat.std()
                resp_gt_norm = (resp_gt_flat - resp_mean)/resp_std
                resp_pred_norm = (pred_resp.squeeze() - resp_mean)/resp_std
                response_loss = nn.functional.mse_loss(resp_pred_norm, resp_gt_norm)
            else:
                # Direct scalar response prediction
                pred_resp = net(shape_input_exp, props_exp, freq_flat)  # (B*F,)
                resp_gt_flat = responses_gt.reshape(-1)
                mean_resp = resp_gt_flat.mean()
                std_resp = resp_gt_flat.std()
                resp_gt_norm = (resp_gt_flat - mean_resp)/std_resp
                resp_pred_norm = (pred_resp - mean_resp)/std_resp
                response_loss = nn.functional.mse_loss(resp_pred_norm, resp_gt_norm)
                velocity_loss = torch.tensor(0., device=device)

            total_batch_loss = v_loss_weight * velocity_loss + f_loss_weight * response_loss
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total_batch_loss.item()
            total_vloss += velocity_loss.item()
            total_floss += response_loss.item()
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch}/{max_epochs}] in {epoch_time:.1f}s - Loss: {total_loss/len(train_loader):.4f} (Vel: {total_vloss/len(train_loader):.4f}, Resp: {total_floss/len(train_loader):.4f})")
        
        # ---------------------------
        # 6. Validation
        # ---------------------------
        net.eval()
        val_emse = []
        val_emd = []
        val_peak_err = []
        with torch.no_grad():
            for v_batch in val_loader:
                shape_sdf = v_batch['shape_enc'].to(device)
                properties = v_batch['props'].to(device)
                freqs_v = v_batch['freqs'].to(device)
                velocities_gt = v_batch['velocity'].to(device)
                responses_gt = v_batch['response'].to(device)
                B, F, H, W = velocities_gt.shape

                shape_input_exp = shape_sdf.unsqueeze(1).expand(-1, F, -1, -1).reshape(-1, H, W)
                props_exp = properties.unsqueeze(1).expand(-1, F, -1).reshape(-1,7)
                freq_flat = freqs_v.unsqueeze(0).expand(B, -1).reshape(-1)

                if net.response_type == 'velocity_field':
                    pred_vel = net(shape_input_exp, props_exp, freq_flat)
                    pred_vel_log = torch.log(torch.clamp(pred_vel**2 + 1e-8, min=1e-8))
                    pred_resp = ((pred_vel_log**2).mean(dim=[2,3])) * 10
                    resp_gt_flat = responses_gt.reshape(-1)
                    resp_pred = pred_resp.squeeze().cpu()
                else:
                    pred_resp = net(shape_input_exp, props_exp, freq_flat)
                    resp_gt_flat = responses_gt.reshape(-1).cpu()
                    resp_pred = pred_resp.cpu()

                # Compute metrics
                emse_value = evaluation.compute_mse(resp_gt_flat, resp_pred)
                emd_value = evaluation.compute_emd(resp_gt_flat, resp_pred)
                peaks_true = evaluation.detect_peaks(resp_gt_flat)
                peaks_pred = evaluation.detect_peaks(resp_pred.numpy())
                pe_ratio, pe_shift = evaluation.compute_peak_errors(peaks_true, peaks_pred, *evaluation.match_peaks(peaks_true, peaks_pred), resp_gt_flat, resp_pred)
                val_emse.append(emse_value)
                val_emd.append(emd_value)
                val_peak_err.append(pe_ratio)  # For simplicity, just store ratio error
                
        # Average validation metrics
        val_emse_mean = np.mean(val_emse)
        val_emd_mean = np.mean(val_emd)
        val_peak_mean = np.mean(val_peak_err)
        print(f"Validation metrics -- EMSE: {val_emse_mean:.4f}, EMD: {val_emd_mean:.4f}, Peak Ratio Error: {val_peak_mean:.4f}")

        scheduler.step(val_emse_mean)

        # Save checkpoint if improved
        if save_checkpoints:
            is_better = val_emse_mean < best_val_loss
            if is_better:
                best_val_loss = val_emse_mean
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, checkpoint_path)
                print("Saved new best model.")

        # Early stopping
        if val_emse_mean >= best_val_loss:
            early_patience -= 1
            if early_patience == 0:
                print(f"No improvement for patience epochs. Stopping early.")
                break
        else:
            early_patience = cfg['training'].get('early_stopping_patience', 20)

    # ---------------------------
    # 7. Load best model and evaluate on test set
    # ---------------------------
    print("Loading best model for test evaluation...")
    if os.path.exists(checkpoint_path):
        chkpt = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(chkpt['model_state_dict'])
    net.eval()

    all_emse = []
    all_emd = []
    all_pek = []

    with torch.no_grad():
        for batch in test_loader:
            shape_sdf = batch['shape_enc'].to(device)  # (B, H, W)
            properties = batch['props'].to(device)
            freqs_t = batch['freqs'].to(device)
            velocities_gt = batch['velocity'].to(device)
            responses_gt = batch['response'].to(device)
            B, F, H, W = velocities_gt.shape

            shape_input_exp = shape_sdf.unsqueeze(1).expand(-1, F, -1, -1).reshape(-1, H, W)
            props_exp = properties.unsqueeze(1).expand(-1, F, -1).reshape(-1,7)
            freq_flat = freqs_t.unsqueeze(0).expand(B, -1).reshape(-1)

            if net.response_type == 'velocity_field':
                pred_vel = net(shape_input_exp, props_exp, freq_flat)
                pred_vel_log = torch.log(torch.clamp(pred_vel**2 + 1e-8, min=1e-8))
                pred_resp = ((pred_vel_log**2).mean(dim=[2,3]))*10
            else:
                pred_resp = net(shape_input_exp, props_exp, freq_flat)
            resp_gt_flat = responses_gt.reshape(-1).cpu()
            resp_pred = pred_resp.cpu()

            emse = evaluation.compute_mse(resp_gt_flat, resp_pred)
            emd = evaluation.compute_emd(resp_gt_flat, resp_pred)
            peaks_true = evaluation.detect_peaks(resp_gt_flat)
            peaks_pred = evaluation.detect_peaks(resp_pred.numpy())
            pe_ratio, pe_shift = evaluation.compute_peak_errors(peaks_true, peaks_pred, *evaluation.match_peaks(peaks_true, peaks_pred), resp_gt_flat, resp_pred)
            all_emse.append(emse)
            all_emd.append(emd)
            all_pek.append(pe_shift if not np.isnan(pe_shift) else 0.0)

    # Report test metrics
    print(f"\n=== Test Results ===")
    print(f"EMSE: {np.mean(all_emse):.4f} ± {np.std(all_emse):.4f}")
    print(f"EMD: {np.mean(all_emd):.4f} ± {np.std(all_emd):.4f}")
    print(f"Peak shift (mean): {np.mean(all_pek):.4f}")

    # Optionally, visualize example responses/velocity fields
    # For brevity, not included here

if __name__ == "__main__":
    main()
```

## model.py

```python
## model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Extract model parameters from config
ARCHITECTURE_TYPE = CONFIG['model'].get('architecture', 'UNet')  # 'UNet', 'FNO', etc.
ENCODER_TYPE = CONFIG['model'].get('encoder', {}).get('type', 'implicit_shape_encoder')
SHAPE_REPRESENTATION = CONFIG['model'].get('encoder', {}).get('shape_representation', 'signed_distance_function') # placeholder
RESPONDER_TYPE = CONFIG['model'].get('response_decoder', {}).get('type', 'velocity_field')  # or 'scalar_response'
CHANNELS = CONFIG['model'].get('channels', 64)
DEPTH = CONFIG['model'].get('depth', 4)
FREQ_EMBED_SIZE = 32  # Size of frequency embedding, can be adjusted or made configurable

# Helper functions
def get_fourier_features(f, num_features=FREQ_EMBED_SIZE):
    """
    Creates Fourier features for scalar frequency input.
    Args:
        f: scalar tensor, shape (batch_size,)
        num_features: int, number of frequency features
    Returns:
        feature tensor of shape (batch_size, 2 * num_features)
    """
    omega = torch.linspace(0., 1., steps=num_features, device=f.device)
    f = f.unsqueeze(-1)  # (batch_size, 1)
    f_scaled = f * omega * 2 * math.pi  # scale
    sin_feat = torch.sin(f_scaled)
    cos_feat = torch.cos(f_scaled)
    return torch.cat([sin_feat, cos_feat], dim=-1)  # (batch_size, 2 * num_features)

# Shape Encoder Modules
class ImplicitShapeEncoder(nn.Module):
    """
    Encodes shape expressed as a Signed Distance Function (SDF) grid
    via a few convolutional layers to produce a fixed-length embedding.
    """
    def __init__(self, input_channels=1, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2)
        self.norm1 = nn.LayerNorm([32, None, None])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.norm2 = nn.LayerNorm([64, None, None])
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.norm3 = nn.LayerNorm([128, None, None])
        self.fc = nn.Linear(128, embedding_dim)
        
    def forward(self, sdf: torch.Tensor):
        """
        Args:
            sdf: tensor shape (batch_size, height, width)
        Returns:
            embedding: tensor shape (batch_size, embedding_dim)
        """
        x = sdf.unsqueeze(1)  # (B, 1, H, W)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        # Global average pooling
        x = x.mean(dim=[2,3])  # (B, 128)
        embedding = self.fc(x)
        return embedding

# Example placeholder for ResNet18 encoder
# For brevity, use torchvision's ResNet if available:
import torchvision.models as models
class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained=False, embedding_dim=128):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        # Remove classification head
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embedding_dim)
    def forward(self, x):
        """
        Args:
            x: tensor (B, C, H, W)
        Returns:
            embedding: tensor (B, embedding_dim)
        """
        x = self.resnet(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)
        return self.fc(x)

# Vision Transformer encoder placeholder
# Implementing a minimal ViT encoder as a subclass
class ViTEncoder(nn.Module):
    def __init__(self, image_size=64, patch_size=16, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        # For simplicity, use nn.TransformerEncoder with patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.embedding = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: tensor (B, 1, H, W)
        Returns:
            embedding: tensor (B, embed_dim)
        """
        x = self.embedding(x)  # (B, embed_dim, Hh, Wh)
        B, C, Hh, Wh = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # (Hh*Wh, B, embed_dim)
        x = self.transformer(x)  # same shape
        # Pool over spatial patches
        x = x.mean(dim=0)  # (B, embed_dim)
        return x

# FiLM conditioning layer
class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, scalar_dim):
        """
        Args:
            feature_dim: dimensionality of the features to condition
            scalar_dim: number of scalar conditioning parameters
        """
        super().__init__()
        self.film_fc = nn.Linear(scalar_dim, 2 * feature_dim)  # gamma and beta

    def forward(self, features: torch.Tensor, scalar_params: torch.Tensor):
        """
        Args:
            features: (B, feature_dim)
            scalar_params: (B, scalar_dim)
        Returns:
            conditioned features: (B, feature_dim)
        """
        gamma_beta = self.film_fc(scalar_params)  # (B, 2*feature_dim)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return features * gamma + beta

# Response decoders
class VelocityFieldUNet(nn.Module):
    """
    UNet architecture for predicting velocity fields conditioned on shape + scalar params + frequency.
    Uses FiLM layers for conditioning.
    """
    def __init__(self, in_channels=1, base_channels=CHANNELS, depth=DEPTH):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth

        # Encoding path
        self.encoders = nn.ModuleList()
        for i in range(depth):
            in_ch = in_channels if i == 0 else base_channels * 2**(i-1)
            out_ch = base_channels * 2**i
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LayerNorm([out_ch, None, None]),
                    nn.ReLU()
                )
            )

        # Decoding path
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth-1)):
            in_ch = base_channels * 2**(i+1)
            out_ch = base_channels * 2**i
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LayerNorm([out_ch, None, None]),
                    nn.ReLU()
                )
            )

        # Final convolution to 2 velocity components
        self.final_conv = nn.Conv2d(base_channels, 2, kernel_size=1)

        # Self-attention layers in encoder and decoder
        self.encoder_self_attn = nn.MultiheadAttention(embed_dim=base_channels, num_heads=4)
        self.decoder_self_attn = nn.MultiheadAttention(embed_dim=base_channels, num_heads=4)

        # FiLM layers will be applied after each encoder block
        # For simplicity, create a list
        self.film_layers = nn.ModuleList([
            FiLMLayer(base_channels * 2 ** i, scalar_dim=7) for i in range(depth)
        ])

    def forward(self, shape_feat: torch.Tensor, scalar_params: torch.Tensor, f: torch.Tensor):
        """
        Args:
            shape_feat: (B, H, W, C) feature map or (B, C) shape embedding
            scalar_params: (B, 7)
            f: (B,) scalar, frequency value
        Returns:
            velocity_field: (B, 2, H, W)
        """
        # For simplicity, assume shape_feat is spatial for unet input or pooled for vector
        # First, expand shape_feat to spatial grid if needed
        # Let's assume input shape_feat is (B, C) and broadcast to spatial
        # Alternatively, shape_feat can be a feature map if coming from a CNN encoder
        # Here, treat shape_feat as a vector; expand spatially
        B = shape_feat.shape[0]
        H, W = 64, 64  # or set according to input; be consistent with dataset
        feat_map = shape_feat.unsqueeze(-1).unsqueeze(-1).expand(B, shape_feat.shape[-1], H, W)

        # Embed frequency
        freq_emb = get_fourier_features(f, num_features=FREQ_EMBED_SIZE)  # (B, 2*FREQ_EMBED_SIZE)
        # Expand freq_emb spatially for FiLM conditioning
        freq_emb_exp = freq_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # Initial input features
        x = feat_map  # (B, C, H, W)

        # Encoder path
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            # Apply FiLM for conditioning
            conditioned = self.film_layers[i](x.permute(0,2,3,1).mean(dim=[1,2]), scalar_params)  # pool spatial
            # Broadcast back
            gamma = conditioned.unsqueeze(1).unsqueeze(2)
            beta = conditioned.unsqueeze(1).unsqueeze(2)
            x = x * gamma + beta
            skips.append(x)
            # Downsample for next layer if needed? (In current implementation, strides are inside convs)

        # Bottleneck act (could add attentions here)
        # For simplicity, skip
        # Decoder path
        for i, decoder in enumerate(self.decoders):
            skip_feat = skips[-(i+2)]
            x = F.interpolate(x, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_feat], dim=1)
            x = decoder(x)

        velocity = self.final_conv(x)  # (B, 2, H, W)
        return velocity

class ResponseMLP(nn.Module):
    """
    Fully-connected network to predict scalar response F(f) given combined features.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, input_dim)
        Returns:
            scalar response: (B, 1)
        """
        return self.layers(x).squeeze(-1)

# Main model class supporting variants
class LearningVibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Instantiate shape encoder
        encoder_type = ARCHITECTURE_TYPE.lower()
        if encoder_type == 'implicit_shape_encoder':
            self.shape_encoder = ImplicitShapeEncoder(embedding_dim=128)
            shape_feat_dim = 128
        elif encoder_type == 'resnet18':
            self.shape_encoder = ResNet18Encoder(pretrained=False, embedding_dim=128)
            shape_feat_dim = 128
        elif encoder_type == 'vit':
            self.shape_encoder = ViTEncoder(embed_dim=128)
            shape_feat_dim = 128
        else:
            raise ValueError(f"Unsupported encoder type: {ARCHITECTURE_TYPE}")

        # Scalar properties embedding (can be normalized, so just linear layer)
        self.prop_fc = nn.Linear(7, 64)

        # FiLM layers to condition shape features
        self.film_layers = nn.ModuleList([
            FiLMLayer(shape_feat_dim, scalar_dim=7) for _ in range(DEPTH)
        ])

        # Response decoder
        decoder_type = RESPONDER_TYPE.lower()
        if decoder_type == 'velocity_field':
            self.response_decoder = VelocityFieldUNet(
                in_channels=1, base_channels=CHANNELS, depth=DEPTH
            )
            self.response_type = 'velocity_field'
        elif decoder_type == 'scalar_response':
            # Input_dim: shape feature + propagated scalar + freq embedding
            self.response_decoder = ResponseMLP(input_dim=shape_feat_dim + 7 + 2 * FREQ_EMBED_SIZE)
            self.response_type = 'scalar_response'
        else:
            raise ValueError(f"Unsupported response decoder type: {RESPONDER_TYPE}")

    def forward(self, shape_input: torch.Tensor, scalar_props: torch.Tensor, frequency: torch.Tensor):
        """
        Args:
            shape_input: shape data, e.g., sdf grid or images, shape depends on encoder
            scalar_props: (B, 7)
            frequency: (B,) frequency scalar values
        Returns:
            response: response prediction (scalar or velocity map)
        """
        # Encode shape
        shape_feat = self.shape_encoder(shape_input)  # (B, feature_dim)
        # Encode properties
        prop_feat = self.prop_fc(scalar_props)  # (B, 64) optionally
        # Conditioning shape features with scalar properties via FiLM
        conditioned_feat = shape_feat
        for i, film in enumerate(self.film_layers):
            conditioned_feat = film(conditioned_feat, scalar_props)  # (B, feature_dim)

        # Embed frequency
        freq_emb = get_fourier_features(frequency, num_features=FREQ_EMBED_SIZE)  # (B, 2*FREQ_EMBED_SIZE)

        if self.response_type == 'velocity_field':
            # Decode velocity field conditioned on shape + frequency + properties
            velocity = self.response_decoder(conditioned_feat, scalar_props, frequency)
            return velocity  # shape (B, 2, H, W)
        else:
            # Concatenate features for scalar response
            combined_input = torch.cat([conditioned_feat, scalar_props, freq_emb], dim=-1)
            response_scalar = self.response_decoder(combined_input)  # (B,)
            return response_scalar

```

## train.py

```python
## train.py

import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import modules from project
import dataset_loader
import model
import evaluation
import visualization

def main():
    # ---------------------------
    # 1. Load configuration
    # ---------------------------
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset parameters
    dataset_name = cfg['dataset']['dataset_name']
    train_size = cfg['dataset']['train_size']
    val_size = cfg['dataset']['validation_size']
    test_size = cfg['dataset']['test_size']
    freqs_min = cfg['dataset']['frequency_range']['min']
    freqs_max = cfg['dataset']['frequency_range']['max']
    freq_points = cfg['dataset']['frequency_points']
    mesh_dim = cfg['dataset']['discretization_mesh']
    # Model parameters
    model_arch = cfg['model']['architecture']
    encoder_type = cfg['model'].get('encoder', {}).get('type', 'implicit_shape_encoder')
    shape_rep = cfg['model'].get('encoder', {}).get('shape_representation', 'signed_distance_function')
    scalar_dim = 7  # as per description, adjust if needed
    channels = cfg['model'].get('channels', 64)
    depth = cfg['model'].get('depth', 4)
    response_type = cfg['model'].get('response_decoder', {}).get('type', 'velocity_field')
    # Training parameters
    lr = cfg['training'].get('learning_rate', 1e-3)
    batch_size = cfg['training'].get('batch_size', 16)
    max_epochs = cfg['training'].get('epochs', 300)
    early_patience = cfg['training'].get('early_stopping_patience', 20)
    v_loss_weight = cfg['training']['loss_weights'].get('velocity_loss_weight', 0.25)
    f_loss_weight = cfg['training']['loss_weights'].get('response_loss_weight', 0.75)
    # Other
    save_checkpoints = cfg['training'].get('save_checkpoints', True)
    save_best_only = cfg['training'].get('save_best_only', True)

    # ---------------------------
    # 2. Data loading
    # ---------------------------
    print("Loading datasets...")
    # Load datasets: assuming Dataset class supports train/val/test split
    full_dataset = dataset_loader.VibratingPlatesDataset(
        data_dir='./data',  # or adjust
        split='train'
    )
    # To get specific subset sizes, assume Dataset supports splitting
    # Or, if dataset_loader has train/val/test datasets, load accordingly
    # For simplicity, assume train/val/test DataLoaders are created here

    # Load full dataset and create splits
    # Note: in dataset_loader, __init__ has methods to get total samples, so:
    total_samples = len(full_dataset)
    indices = np.arange(total_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[:int(0.8*total_samples)]
    val_idx = indices[int(0.8*total_samples):int(0.9*total_samples)]
    test_idx = indices[int(0.9*total_samples):]

    # Create subset datasets
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)
    test_subset = torch.utils.data.Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print("Datasets loaded.")
    # ---------------------------
    # 3. Model initialization
    # ---------------------------
    print(f"Initializing model architecture: {model_arch}")
    # Instantiate model (assumed in model.py)
    net = model.LearningVibrationModel()
    net.to(device)

    # Optional: load checkpoint if resuming
    start_epoch = 1
    best_val_loss = np.inf
    checkpoint_path = './checkpoint_best.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

    # ---------------------------
    # 4. Optimizer and scheduler
    # ---------------------------
    optimizer = optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    # ---------------------------
    # 5. Loss function setup
    # ---------------------------
    # We define loss functions inline during training

    # ---------------------------
    # 6. Training loop
    # ---------------------------
    print("Starting training...")
    no_grad = torch.no_grad

    # For early stopping
    best_epoch = start_epoch
    no_improve_epochs = 0
    total_batches = len(train_loader)

    for epoch in range(start_epoch, max_epochs +1):
        start_time = time.time()
        net.train()
        total_loss = 0.0
        total_velocity_loss = 0.0
        total_response_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Load batch data and move to device
            shape_sdf = batch['shape_enc'].to(device)               # shape (B, H, W)
            properties = batch['props'].to(device)                  # shape (B, scalar_dim)
            freqs = batch['freqs'].to(device)                       # shape (F,)
            velocity_gt = batch['velocity'].to(device)              # shape (B, F, H, W)
            response_gt = batch['response'].to(device)              # shape (B, F)
            shape_ids = batch['shape_id']                            # list of ids if needed

            # Zero grads
            optimizer.zero_grad()

            # For each sample, evaluate at multiple frequencies:
            # To vectorize, process all (B, F) pairs together
            B, F, H, W = velocity_gt.shape

            # Expand shape encoding and properties for F frequencies
            shape_input = shape_sdf  # shape (B, H, W)
            props = properties       # shape (B, scalar_dim)
            freq_batch = freqs.unsqueeze(0).expand(B, F)   # (B, F)
            freq_flat = freq_batch.reshape(-1)             # (B*F,)

            shape_input_exp = shape_input.unsqueeze(1).expand(-1, F, -1, -1).reshape(-1, H, W)   # (B*F, H, W)
            props_exp = props.unsqueeze(1).expand(-1, F, -1).reshape(-1, scalar_dim)             # (B*F, scalar_dim)

            # Forward pass via model
            # Depending on architecture, response is velocity field or direct scalar
            if net.response_type == 'velocity_field':
                velocity_pred = net(shape_input_exp, props_exp, freq_flat)
                # velocity_pred: (B*F, 2, H, W)
                # For loss, compute MSE on log velocities
                velocity_pred_log = torch.log(torch.clamp(velocity_pred**2 + 1e-8, min=1e-8))
                velocity_gt_flat = velocity_gt.reshape(-1, 2, H, W)
                velocity_gt_log = torch.log(torch.clamp(velocity_gt_flat**2 + 1e-8, min=1e-8))
                velocity_loss = nn.functional.mse_loss(velocity_pred_log, velocity_gt_log)
                # Derive response: measure the mean of squared velocities over the domain
                pred_response = ((velocity_pred_log**2).mean(dim=[2,3])) * 10  # scaled for decibel scale
                # Normalize true response as per description (already scaled/log)
                response_gt_exp = response_gt.reshape(-1)
                response_gt_mean = response_gt_exp.mean()
                response_gt_std = response_gt_exp.std()
                response_gt_norm = (response_gt_exp - response_gt_mean)/response_gt_std
                pred_response_norm = (pred_response.squeeze() - response_gt_mean)/response_gt_std
                response_loss = nn.functional.mse_loss(pred_response_norm, response_gt_norm)
            else:
                # Response directly predicted (scalar), shape (B*F,)
                pred_response = net(shape_input_exp, props_exp, freq_flat)
                response_gt_exp = response_gt.reshape(-1)
                # Normalize
                mean_resp = response_gt_exp.mean()
                std_resp = response_gt_exp.std()
                response_gt_norm = (response_gt_exp - mean_resp)/std_resp
                pred_response_norm = (pred_response - mean_resp)/std_resp
                response_loss = nn.functional.mse_loss(pred_response_norm, response_gt_norm)
                velocity_loss = torch.tensor(0.0, device=device)

            # Combine losses
            total_batch_loss = v_loss_weight * velocity_loss + f_loss_weight * response_loss
            total_batch_loss.backward()
            # Optional: clip gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total_batch_loss.item()
            total_velocity_loss += velocity_loss.item()
            total_response_loss += response_loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_batches
        avg_v_loss = total_velocity_loss / total_batches
        avg_f_loss = total_response_loss / total_batches
        print(f"Epoch [{epoch}/{max_epochs}] - Time: {epoch_time:.2f}s - Loss: {avg_loss:.4f} (Vel: {avg_v_loss:.4f}, Resp: {avg_f_loss:.4f})")

        # ---------------------------
        # 7. Validation
        # ---------------------------
        net.eval()
        val_metrics = {'EMSE': [], 'EMD': [], 'PEAKS': []}
        with torch.no_grad():
            for v_batch in val_loader:
                shape_sdf = v_batch['shape_enc'].to(device)
                properties = v_batch['props'].to(device)
                freqs = v_batch['freqs'].to(device)
                velocity_gt = v_batch['velocity'].to(device)
                response_gt = v_batch['response'].to(device)

                B, F, H, W = velocity_gt.shape
                shape_input = shape_sdf
                props = properties
                freq_batch = freqs.unsqueeze(0).expand(B, F)
                freq_flat = freq_batch.reshape(-1)

                shape_input_exp = shape_input.unsqueeze(1).expand(-1, F, -1, -1).reshape(-1, H, W)
                props_exp = props.unsqueeze(1).expand(-1, F, -1).reshape(-1, scalar_dim)

                if net.response_type == 'velocity_field':
                    velocity_pred = net(shape_input_exp, props_exp, freq_flat)
                    velocity_pred_log = torch.log(torch.clamp(velocity_pred**2 + 1e-8, min=1e-8))
                    # Convert to response function for metric
                    pred_response = ((velocity_pred_log**2).mean(dim=[2,3])) * 10
                    # Compute metrics
                    # Retrieve ground-truth responses
                    resp_gt = response_gt.reshape(-1)
                    resp_pred = pred_response.squeeze().cpu()
                    # Compute metrics
                    emse_val = evaluation.compute_mse(resp_gt, resp_pred)
                    emd_val = evaluation.compute_emd(resp_gt, resp_pred)
                    peaks_true = evaluation.detect_peaks(resp_gt)
                    peaks_pred = evaluation.detect_peaks(resp_pred.numpy())
                    pe = evaluation.peak_error(peaks_true, peaks_pred, resp_gt, resp_pred)
                else:
                    pred_response = net(shape_input_exp, props_exp, freq_flat)
                    resp_gt = response_gt.reshape(-1).cpu()
                    resp_pred = pred_response.cpu()
                    emse_val = evaluation.compute_mse(resp_gt, resp_pred)
                    emd_val = evaluation.compute_emd(resp_gt, resp_pred)
                    peaks_true = evaluation.detect_peaks(resp_gt)
                    peaks_pred = evaluation.detect_peaks(resp_pred.numpy())
                    pe = evaluation.peak_error(peaks_true, peaks_pred, resp_gt, resp_pred)

                val_metrics['EMSE'].append(emse_val)
                val_metrics['EMD'].append(emd_val)
                val_metrics['PEAKS'].append(pe)

        mean_emse = np.mean(val_metrics['EMSE'])
        mean_emd = np.mean(val_metrics['EMD'])
        mean_pe = np.mean(val_metrics['PEAKS'])
        print(f"Validation - EMSE: {mean_emse:.4f}, EMD: {mean_emd:.4f}, PeakError: {mean_pe:.4f}")

        # Step learning rate scheduler
        scheduler.step(mean_emse)

        # Save checkpoint
        if save_checkpoints:
            is_best = (mean_emse < best_val_loss)
            if is_best:
                best_val_loss = mean_emse
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, checkpoint_path)
                print(f"Saved best model at epoch {epoch}")
            elif not save_best_only:
                # Save every epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, f'./checkpoint_epoch_{epoch}.pth')

        # Early stopping
        if mean_emse >= best_val_loss:
            no_improve_epochs += 1
            if no_improve_epochs >= early_patience:
                print(f"No improvement for {early_patience} epochs. Early stopping.")
                break
        else:
            no_improve_epochs = 0

    # ---------------------------
    # 8. Final evaluation on test set
    # ---------------------------
    print("Testing best model...")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    all_EMSE = []
    all_EMD = []
    all_PE = []

    with torch.no_grad():
        for batch in test_loader:
            shape_sdf = batch['shape_enc'].to(device)
            properties = batch['props'].to(device)
            freqs = batch['freqs'].to(device)
            velocity_gt = batch['velocity'].to(device)
            response_gt = batch['response'].to(device)

            B, F, H, W = velocity_gt.shape
            shape_input = shape_sdf
            props = properties
            freq_batch = freqs.unsqueeze(0).expand(B, F)
            freq_flat = freq_batch.reshape(-1)

            shape_input_exp = shape_input.unsqueeze(1).expand(-1, F, -1, -1).reshape(-1, H, W)
            props_exp = props.unsqueeze(1).expand(-1, F, -1).reshape(-1, scalar_dim)

            if net.response_type == 'velocity_field':
                velocity_pred = net(shape_input_exp, props_exp, freq_flat)
                velocity_pred_log = torch.log(torch.clamp(velocity_pred**2 + 1e-8, min=1e-8))
                resp_pred = ((velocity_pred_log**2).mean(dim=[2,3])) * 10
            else:
                resp_pred = net(shape_input_exp, props_exp, freq_flat)

            # Retrieve ground truth responses
            resp_gt = response_gt.reshape(-1).cpu()
            resp_pred = resp_pred.cpu()

            # Compute metrics
            emse_val = evaluation.compute_mse(resp_gt, resp_pred)
            emd_val = evaluation.compute_emd(resp_gt, resp_pred)
            peaks_true = evaluation.detect_peaks(resp_gt)
            peaks_pred = evaluation.detect_peaks(resp_pred.numpy())
            pe = evaluation.peak_error(peaks_true, peaks_pred, resp_gt, resp_pred)

            all_EMSE.append(emse_val)
            all_EMD.append(emd_val)
            all_PE.append(pe)

    print(f"Test Results:\nEMSE: {np.mean(all_EMSE):.4f} ± {np.std(all_EMSE):.4f}")
    print(f"EMD: {np.mean(all_EMD):.4f} ± {np.std(all_EMD):.4f}")
    print(f"Peak Error: {np.mean(all_PE):.4f} ± {np.std(all_PE):.4f}")

    # Optional: generate visualization of example responses
    # For illustration, pick one sample
    # Here, could call visualization functions
    # For brevity, omitted

if __name__ == '__main__':
    main()
```

## visualization.py

```python
## visualization.py

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple

# Load configuration to match dataset response range
import yaml
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

FREQ_MIN = CONFIG['dataset']['frequency_range'].get('min', 1)
FREQ_MAX = CONFIG['dataset']['frequency_range'].get('max', 300)
FREQUENCY_POINTS = CONFIG['dataset']['frequency_points']
FREQUENCIES = np.linspace(FREQ_MIN, FREQ_MAX, FREQUENCY_POINTS)

def plot_response(frequencies: np.ndarray,
                  F_true: np.ndarray,
                  F_pred: Optional[np.ndarray] = None,
                  peaks_true: Optional[np.ndarray] = None,
                  peaks_pred: Optional[np.ndarray] = None,
                  match_indices: Optional[List[Tuple[int, int]]] = None,
                  title: str = '',
                  save_path: Optional[str] = None):
    """
    Plot true and predicted frequency response over frequency.
    Optionally mark peaks and matched peaks.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(frequencies, F_true, label='Ground Truth', color='blue', linewidth=2)
    if F_pred is not None:
        plt.plot(frequencies, F_pred, label='Prediction', color='orange', linestyle='--', linewidth=2)

    # Mark ground truth peaks
    if peaks_true is not None and len(peaks_true) > 0:
        plt.scatter(peaks_true, np.interp(peaks_true, frequencies, F_true),
                    marker='x', color='blue', s=100, label='GT peaks')
    # Mark predicted peaks
    if peaks_pred is not None and len(peaks_pred) > 0:
        plt.scatter(peaks_pred, np.interp(peaks_pred, frequencies, F_pred if F_pred is not None else F_true),
                    marker='o', color='orange', s=100, label='Pred peaks')

    # Draw lines for matched peaks (if provided)
    if match_indices is not None:
        for gt_idx, pred_idx in match_indices:
            freq_gt = peaks_true[gt_idx] if peaks_true is not None and len(peaks_true) > gt_idx else None
            freq_pred = peaks_pred[pred_idx] if peaks_pred is not None and len(peaks_pred) > pred_idx else None
            if freq_gt is not None and freq_pred is not None:
                plt.plot([freq_gt, freq_pred],
                         [np.interp(freq_gt, frequencies, F_true),
                          np.interp(freq_pred, frequencies, F_pred if F_pred is not None else F_true)],
                         'r--', linewidth=0.8)

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Normalized Response')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_velocity_field(grid_x: np.ndarray,
                        grid_y: np.ndarray,
                        V_true: np.ndarray,
                        V_pred: Optional[np.ndarray] = None,
                        title: str = '',
                        velocity_scale: float = 1.0,
                        save_path: Optional[str] = None) -> None:
    """
    Visualize 2D velocity fields (component-wise or magnitude).
    Inputs:
        grid_x, grid_y: 2D arrays defining spatial coordinates.
        V_true: 2D array, velocity magnitude or vector component (if vector, visualize magnitude).
        V_pred: optional, same shape as V_true.
        velocity_scale: scale factor for arrow length.
    """
    plt.figure(figsize=(10, 4))
    # If velocity is vector, plot quiver; if scalar magnitude, plot imshow
    # For generality, plot magnitude
    true_magnitude = np.linalg.norm(V_true, axis=0) if V_true.ndim == 3 else V_true
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(true_magnitude, origin='lower', extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()), cmap='viridis')
    plt.colorbar(im1, label='Velocity magnitude')
    plt.title('True velocity field')
    plt.xlabel('X')
    plt.ylabel('Y')

    if V_pred is not None:
        pred_magnitude = np.linalg.norm(V_pred, axis=0) if V_pred.ndim == 3 else V_pred
        plt.subplot(1, 2, 2)
        im2 = plt.imshow(pred_magnitude, origin='lower', extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()), cmap='viridis')
        plt.colorbar(im2, label='Velocity magnitude')
        plt.title('Predicted velocity field')
        plt.xlabel('X')
        plt.ylabel('Y')

    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_peak_matching(
        frequencies: np.ndarray,
        F_true: np.ndarray,
        F_pred: np.ndarray,
        peaks_true: np.ndarray,
        peaks_pred: np.ndarray,
        match_indices: List[Tuple[int, int]],
        title: str = '',
        save_path: Optional[str] = None):
    """
    Visualize response curves with peaks and peak-matching lines.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(frequencies, F_true, label='Ground Truth', color='blue', linewidth=2)
    plt.plot(frequencies, F_pred, label='Prediction', color='orange', linestyle='--', linewidth=2)

    # Plot all peaks
    plt.scatter(peaks_true, np.interp(peaks_true, frequencies, F_true),
                marker='x', color='blue', s=100, label='GT peaks')
    plt.scatter(peaks_pred, np.interp(peaks_pred, frequencies, F_pred),
                marker='o', color='orange', s=100, label='Pred peaks')

    # Draw lines for matched peaks
    for gt_idx, pred_idx in match_indices:
        freq_gt = peaks_true[gt_idx] if len(peaks_true) > gt_idx else None
        freq_pred = peaks_pred[pred_idx] if len(peaks_pred) > pred_idx else None
        if freq_gt is not None and freq_pred is not None:
            plt.plot([freq_gt, freq_pred],
                     [np.interp(freq_gt, frequencies, F_true),
                      np.interp(freq_pred, frequencies, F_pred)],
                     'r--', linewidth=0.8)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Normalized Response')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_comparison_at_frequency(frequency: float,
                                 V_true: np.ndarray,
                                 V_pred: np.ndarray,
                                 grid_x: np.ndarray,
                                 grid_y: np.ndarray,
                                 velocity_scale: float = 1.0,
                                 title: str = '',
                                 save_path: Optional[str] = None):
    """
    Plot velocity vector fields at a specific frequency for true and predicted.
    Inputs:
        grid_x, grid_y: coordinate meshgrid arrays.
        V_true, V_pred: vector component arrays with shape (2, H, W).
    """
    plt.figure(figsize=(12, 5))
    # Plot ground truth velocity vectors
    plt.subplot(1, 2, 1)
    plt.quiver(grid_x, grid_y,
               V_true[0], V_true[1],
               scale=velocity_scale, color='blue')
    plt.title(f'True velocity at {frequency:.1f} Hz')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)

    # Plot predicted velocity vectors
    plt.subplot(1, 2, 2)
    plt.quiver(grid_x, grid_y,
               V_pred[0], V_pred[1],
               scale=velocity_scale, color='orange')
    plt.title(f'Predicted velocity at {frequency:.1f} Hz')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)

    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\Learning_Vibrating_Plates\Learning_Vibrating_Plates_repo`
