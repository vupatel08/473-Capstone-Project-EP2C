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
