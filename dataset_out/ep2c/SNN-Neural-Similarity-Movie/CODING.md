# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import numpy as np
import h5py
import glob
import cv2
from typing import Tuple, Dict, List, Optional
import pandas as pd
from scipy.io import loadmat

from numpy.random import Generator, PCG64
import json

# For reproducibility
import torch

# Load configuration from a passed config dictionary or a loaded YAML
import yaml

# Load config.yaml if needed (assuming preloaded externally)
# For demonstration, assume `config` is passed or loaded outside this file

class NeuralDataLoader:
    def __init__(self,
                 neural_data_dir: str,
                 selected_regions: Optional[List[str]] = None,
                 firing_rate_threshold: float = 0.5,
                 trial_dim: str = 'trials'):
        """
        Initialize the data loader with directory containing neural responses.
        Args:
            neural_data_dir: Path to the directory containing neural response data files.
            selected_regions: List of cortical regions to load; defaults to all.
            firing_rate_threshold: Minimum firing rate (spikes/sec) to include neuron.
            trial_dim: String indicating the dimension in data representing trials ('trials' or 'neurons')
        """
        self.neural_data_dir = neural_data_dir
        self.selected_regions = selected_regions
        self.firing_rate_threshold = firing_rate_threshold
        # Store responses per region
        self.region_responses = dict()
        # Store the number of neurons per region
        self.region_neuron_counts = dict()
        # Load data
        self.load_neural_responses()

    def load_neural_responses(self):
        """
        Load neural response data from files, preprocess, and filter neurons.
        """
        # Assuming files named as 'region_name_responses.npy' or similar
        # Or, a JSON/HDF5 file containing responses for all regions
        # For the code, handle multiple typical formats

        # Example: Load from a directory containing .npy files per region
        # Or load a single data file containing dataset
        # Placeholder: Loading a JSON or .mat file per region

        # You should customize this path based on actual data format
        for filename in os.listdir(self.neural_data_dir):
            if filename.endswith('.npy'):
                region_name = filename[:-4]
                if self.selected_regions and region_name not in self.selected_regions:
                    continue
                data_path = os.path.join(self.neural_data_dir, filename)
                responses = np.load(data_path)  # shape: (neurons, trials, frames) or (neurons, frames, trials)
                # For simplicity, assume shape: (neurons, trials, frames)
                # Load associated trials info if necessary
                # todo: adjust based on real data format
                # For now, assume responses shape: (neurons, trials, frames)
                # Summarize to PSTH: sum spikes in each frame, then average over trials
                # Here, response is already spike count per trial and frame, so sum over trials
                response_mean = np.mean(responses, axis=1)  # shape: (neurons, frames)
                # Compute mean firing rate
                total_time_sec = response_mean.shape[1] / 30.0  # assuming frame rate 30 Hz
                mean_rate = np.mean(response_mean, axis=1) / (1.0)  # per frame, raw spike count
                # Since response_mean is sum per frame per neuron, total spikes per neuron = sum per neuron
                total_spikes_per_neuron = np.sum(response_mean, axis=1)
                firing_rates = total_spikes_per_neuron / total_time_sec  # spikes/sec

                # Filter neurons
                neuron_mask = firing_rates >= firing_rate_threshold
                response_filtered = response_mean[neuron_mask, :]  # shape: (filtered_neurons, frames)

                self.region_responses[region_name] = response_filtered
                self.region_neuron_counts[region_name] = response_filtered.shape[0]

        # Additional: If responses are stored differently, implement loading accordingly

    def get_responses(self, region_name: str) -> np.ndarray:
        """
        Return the neural responses matrix for a given region.
        Args:
            region_name: Name of cortical region.
        Returns:
            ndarray: response matrix of shape (neurons, frames)
        """
        return self.region_responses.get(region_name, np.array([]))

    def get_all_responses(self) -> Dict[str, np.ndarray]:
        """
        Return dictionary of all region responses.
        """
        return self.region_responses

    def load_neural_responses_from_mat(self, filepath: str, region_name: str) -> np.ndarray:
        """
        Load responses stored in a .mat file for a specific region.
        Customize based on data format.
        """
        mat = loadmat(filepath)
        # Assuming 'responses' key with shape: (neurons, trials, frames)
        responses = mat['responses']
        # Compute PSTH: mean spikes per frame over trials
        response_mean = np.mean(responses, axis=1)  # shape: (neurons, frames)
        # Filter neurons based on firing rate
        total_time_sec = response_mean.shape[1] / 30.0
        total_spikes = np.sum(response_mean, axis=1)
        firing_rates = total_spikes / total_time_sec
        neuron_mask = firing_rates >= self.firing_rate_threshold
        return response_mean[neuron_mask, :]

    def load_neural_responses_from_hdf5(self, filepath: str, region_name: str) -> np.ndarray:
        """
        Load responses stored in an HDF5 file.
        """
        with h5py.File(filepath, 'r') as f:
            # Structure depends on dataset
            # For example:
            data = f['responses'][...]  # shape: (neurons, trials, frames)
            response_mean = np.mean(data, axis=1)
            total_time_sec = response_mean.shape[1] / 30.0
            total_spikes = np.sum(response_mean, axis=1)
            firing_rates = total_spikes / total_time_sec
            neuron_mask = firing_rates >= self.firing_rate_threshold
            return response_mean[neuron_mask, :]


class StimuliMovieLoader:
    def __init__(self,
                 movie_paths: Dict[str, str],
                 resize_dims: Tuple[int, int] = (224, 224),
                 frame_rate: int = 30,
                 device: str = 'cuda'):
        """
        Load and preprocess stimuli movies for the experiments.
        Args:
            movie_paths: Dict with keys like 'Movie1', 'Movie2' pointing to file paths.
            resize_dims: Target resize dimensions.
            frame_rate: Frames per second of stimulus movies.
            device: 'cpu' or 'cuda'
        """
        self.movie_paths = movie_paths
        self.resize_dims = resize_dims
        self.frame_rate = frame_rate
        self.device = device
        self.movies = dict()  # key: name, value: numpy array of shape (frames, H, W, C)
        self.load_movies()

    def load_movies(self):
        """
        Load all movies specified.
        """
        for name, path in self.movie_paths.items():
            frames = self.read_video_as_numpy(path)
            self.movies[name] = frames

    def read_video_as_numpy(self, filepath: str) -> np.ndarray:
        """
        Read a video file and convert to NumPy array with frames resized.
        """
        cap = cv2.VideoCapture(filepath)
        frames_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize
            frame_resized = cv2.resize(frame, self.resize_dims)
            # Convert BGR to RGB if needed
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames_list.append(frame_rgb)
        cap.release()
        return np.array(frames_list)  # shape: (num_frames, H, W, C)

    def get_movie(self, name: str) -> np.ndarray:
        """
        Return the numpy array of frames for a given movie.
        """
        return self.movies.get(name, np.array([]))

    def get_all_movies(self) -> Dict[str, np.ndarray]:
        return self.movies


def generate_gaussian_noise_image(size: Tuple[int, int], mean: float = 0.5, std: float = 0.5):
    """
    Generate a Gaussian noise image with specified size.
    """
    noise = np.random.normal(loc=mean, scale=std, size=(size[0], size[1], 3))
    noise_clipped = np.clip(noise, 0.0, 1.0)  # ensure pixel values in [0,1]
    return (noise_clipped * 255).astype(np.uint8)

def shuffle_frames_in_window(movie_array: np.ndarray, window_size: int, rng: Generator) -> np.ndarray:
    """
    Shuffle frames within non-overlapping windows.
    Args:
        movie_array: numpy array shape (num_frames, H, W, C)
        window_size: size of window for shuffling
        rng: numpy random Generator
    Returns:
        shuffed_movie: numpy array of same shape
    """
    num_frames = movie_array.shape[0]
    shuffled_movie = movie_array.copy()

    for start in range(0, num_frames, window_size):
        end = min(start + window_size, num_frames)
        frame_indices = np.arange(start, end)
        rng.shuffle(frame_indices)
        shuffled_movie[start:end] = movie_array[frame_indices]
    return shuffled_movie

def replace_frames_with_noise(movie_array: np.ndarray, ratio: float, rng: Generator, noise_type='gaussian'):
    """
    Replace proportion of frames in movie with noise images.
    Args:
        movie_array: numpy array (frames, H, W, C)
        ratio: fraction of total frames to replace
        rng: numpy random Generator
        noise_type: 'gaussian' (default)
    Returns:
        modified_movie: numpy array
    """
    num_frames = movie_array.shape[0]
    num_replace = int(ratio * num_frames)
    indices = np.arange(num_frames)
    rng.shuffle(indices)
    replace_indices = indices[:num_replace]

    modified_movie = movie_array.copy()
    for idx in replace_indices:
        noise_img = generate_gaussian_noise_image((movie_array.shape[1], movie_array.shape[2]))
        # Convert to float normalized [0,1]
        noise_img_float = noise_img.astype(np.float32) / 255.0
        modified_movie[idx] = noise_img_float
    return modified_movie

def load_static_noise_images(size: Tuple[int, int], num_images: int = 100):
    """
    Load or generate a set of static noise images for experiments.
    Could be randomized, predefined, or loaded from a dataset.
    Here, generate random Gaussian noise images.
    """
    images = []
    rng = np.random.RandomState(0)
    for _ in range(num_images):
        img = generate_gaussian_noise_image(size, mean=0.5, std=0.5)
        images.append(img.astype(np.float32) / 255.0)
    return images
```

## evaluation.py

```python
"""evaluation.py

This module provides tools to compute representational similarities between model responses
and neural responses, estimate neural response ceilings, perform regression analyses,
and handle stimulus manipulations, following the experimental protocol described in the paper.

Dependencies:
- numpy
- scipy
- scikit-learn
- matplotlib

Ensure all are installed as per the environment setup.
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------- Helper functions --------

def compute_similarity_vector(responses: np.ndarray) -> np.ndarray:
    """
    Compute the concatenated similarity vector for responses over time.
    For each time t, computes Pearson correlations between responses at t and t+p for p > 0.
    Args:
        responses: np.ndarray of shape (N_units, T_timepoints)
    Returns:
        full_similarity_vector: np.ndarray of concatenated correlations
    """
    N, T = responses.shape
    s_list = []
    for t in range(T - 1):
        r_t = responses[:, t]
        # For each p > 0
        for p in range(1, T - t):
            r_tp = responses[:, t + p]
            # Compute Pearson correlation between r_t and r_t+p
            if np.std(r_t) == 0 or np.std(r_tp) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(r_t, r_tp)[0,1]
            s_list.append(corr)
    return np.array(s_list)

def compute_spearman_score(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Spearman correlation coefficient between two vectors.
    Args:
        vec1, vec2: 1D numpy arrays
    Returns:
        spearman_corr: float
    """
    corr, _ = stats.spearmanr(vec1, vec2)
    return corr

def compute_TSRSA(responses_model: np.ndarray,
                 responses_neural: np.ndarray) -> float:
    """
    Compute the TSRSA score (Spearman correlation of similarity vectors)
    between model and neural responses.
    Args:
        responses_model: np.ndarray, shape (N_model_units, T)
        responses_neural: np.ndarray, shape (N_neurons, T)
    Returns:
        score: float
    """
    s_model = compute_similarity_vector(responses_model)
    s_neural = compute_similarity_vector(responses_neural)
    # Compute Spearman correlation between the vectors
    score = compute_spearman_score(s_model, s_neural)
    return score

def estimate_neural_ceiling(neural_responses: np.ndarray,
                            n_splits: int = 2,
                            seed: int = 0) -> float:
    """
    Estimate the neural response ceiling via split half reliability.
    Args:
        neural_responses: np.ndarray, shape (N_neurons, T, N_trials)
        n_splits: int, number of splits (default 2)
        seed: int, random seed for reproducibility
    Returns:
        ceiling_score: float
    """
    np.random.seed(seed)
    N_neurons, T, N_trials = neural_responses.shape
    indices = np.arange(N_trials)
    np.random.shuffle(indices)
    split = N_trials // 2
    if N_trials < 2:
        # Cannot split, return maximum possible (1.0)
        return 1.0
    half1_idx = indices[:split]
    half2_idx = indices[split:]
    responses_half1 = np.mean(neural_responses[:, :, half1_idx], axis=2)  # shape: (N_neurons, T)
    responses_half2 = np.mean(neural_responses[:, :, half2_idx], axis=2)
    # Now, compute TSRSA between the two halves for all regions + layers
    # Here, responses are in shape (N_neurons, T)
    ceiling_pairs = []
    try:
        corr = compute_TSRSA(responses_half1, responses_half2)
        return corr
    except Exception:
        # fallback if shapes are incompatible
        return 0.0

def fit_neuron_regression(neural_data: np.ndarray,
                          model_responses: np.ndarray) -> float:
    """
    Fit linear regression for each neuron to model responses, compute R^2.
    Args:
        neural_data: np.ndarray, shape (N_neurons, T)
        model_responses: np.ndarray, shape (N_model_units, T)
    Returns:
        mean R2 score across neurons
    """
    N_neurons = neural_data.shape[0]
    R2s = []
    for i in range(N_neurons):
        y = neural_data[i, :]  # neural response for neuron i
        X = model_responses.T  # shape (T, N_model_units)
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        R2 = reg.score(X, y)
        R2s.append(R2)
    return np.mean(R2s)

def shuffle_frames_in_window(movie: np.ndarray, window_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Shuffle frames within non-overlapping windows.
    Args:
        movie: np.ndarray, shape (num_frames, H, W, C)
        window_size: int
        rng: numpy.random.Generator
    Returns:
        shuffled_movie: np.ndarray, same shape
    """
    num_frames = movie.shape[0]
    shuffled = movie.copy()
    for start in range(0, num_frames, window_size):
        end = min(start + window_size, num_frames)
        indices = np.arange(start, end)
        rng.shuffle(indices)
        shuffled[start:end] = movie[indices]
    return shuffled

def replace_frames_with_noise(movie: np.ndarray,
                              ratio: float,
                              rng: np.random.Generator,
                              noise_type: str='gaussian') -> np.ndarray:
    """
    Replace a proportion of frames with noise images.
    Args:
        movie: np.ndarray, shape (num_frames, H, W, C)
        ratio: float in [0,1]
        rng: numpy.random.Generator
        noise_type: str, typically 'gaussian'
    Returns:
        modified_movie: np.ndarray
    """
    num_frames = movie.shape[0]
    n_replace = int(ratio * num_frames)
    indices = np.arange(num_frames)
    rng.shuffle(indices)
    replace_idx = indices[:n_replace]
    modified = movie.copy()
    H, W, C = movie.shape[1], movie.shape[2], movie.shape[3]
    for idx in replace_idx:
        if noise_type == 'gaussian':
            noise_img = np.random.normal(0.5, 0.5, size=(H, W, C))
            noise_img = np.clip(noise_img, 0, 1)
        else:
            # Default fallback
            noise_img = np.zeros((H, W, C))
        modified[idx] = noise_img
    return modified

def plot_scores(x_axis, scores_dict, title='', xlabel='', ylabel='', save_path=None):
    """
    Plot scores with error bars if provided.
    Args:
        x_axis: list or array of manipulation levels
        scores_dict: dict of {'label': (mean, std)} or list of scores
        title, xlabel, ylabel: plot labels
        save_path: str, optional, save figure
    """
    plt.figure()
    for label, scores in scores_dict.items():
        if isinstance(scores, tuple):
            mean, std = scores
            plt.errorbar(x_axis, mean, yerr=std, label=label, capsize=3)
        else:
            plt.plot(x_axis, scores, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
# -------- Main functions / classes --------

# The module exposes:
# - compute_TSRSA
# - estimate_neural_ceiling
# - regression_score
# - shuffle_frames_in_window
# - replace_frames_with_noise
# - plot_scores
# These functions can be used in study scripts or notebooks following the methodology.

```

## feedback_module.py

```python
## feedback_module.py
import torch
import torch.nn as nn

class FeedbackModule(nn.Module):
    """
    Feedback module modeling corticocortical long-range feedback with recurrence and delay.
    Processes higher-region responses to generate feedback signals to lower layers.
    """

    def __init__(self, in_channels: int = 64,            # Feature dimension of higher region responses
                 feedback_strength: float = 1.0,        # Scalar to scale feedback signal
                 feedback_delay: int = 2,                 # Delay in time steps (corresponds to ms/frames)
                 feedback_connection_type: str = "recurrent"  # Feedback type; default 'recurrent'
                ):
        """
        Initialize FeedbackModule with configuration parameters.

        Args:
            in_channels (int): Dimension of higher-region response features.
            feedback_strength (float): Feedback influence scale.
            feedback_delay (int): Number of steps of delay in feedback connection.
            feedback_connection_type (str): Type of feedback ('recurrent' supported).
        """
        super().__init__()
        self.in_channels = in_channels
        self.feedback_strength = feedback_strength
        self.feedback_delay = feedback_delay
        self.feedback_connection_type = feedback_connection_type

        # Learnable linear projection for feedback
        self.feedback_proj = nn.Linear(in_channels, in_channels)
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.feedback_proj.weight)

        # Buffer to store delayed higher-region responses
        # Size: feedback_delay + 1 to include current response
        self.register_buffer("response_buffer", None)

    def reset(self, batch_size: int, device: torch.device):
        """
        Reset the feedback buffer at the start of a new stimulus/trial.
        Args:
            batch_size (int): Batch size for buffer initialization.
            device (torch.device): Device to place buffer tensors.
        """
        # Initialize buffer with zeros for delay
        self.response_buffer = [torch.zeros(batch_size, self.in_channels, device=device) for _ in range(self.feedback_delay + 1)]

    def forward(self, higher_response: torch.Tensor, current_step: int):
        """
        Generate feedback signal from higher-region response, accounting for delay.

        Args:
            higher_response (torch.Tensor): shape [batch_size, in_channels]
            current_step (int): current time step index in sequence

        Returns:
            feedback_signal (torch.Tensor): shape [batch_size, in_channels]
        """
        # Append current higher response to buffer
        if self.response_buffer is None:
            # On first call, initialize buffer
            batch_size = higher_response.shape[0]
            self.response_buffer = [torch.zeros(batch_size, self.in_channels, device=higher_response.device)
                                    for _ in range(self.feedback_delay + 1)]

        # Update buffer with current response
        self.response_buffer.append(higher_response.detach())

        # Handle delay: get response from 'feedback_delay' steps ago
        if current_step >= self.feedback_delay:
            delayed_response = self.response_buffer[-(self.feedback_delay + 1)]
        else:
            # Not enough history; use zeros
            delayed_response = self.response_buffer[0]

        # Remove oldest entry to keep buffer size consistent
        self.response_buffer.pop(0)

        # Project delayed response via learned weights
        feedback_signal = self.feedback_proj(delayed_response)
        # Scale feedback with strength parameter
        feedback_signal = self.feedback_strength * feedback_signal

        return feedback_signal
```

## main.py

```python
# main.py
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import custom modules
from dataset_loader import NeuralDataLoader, StimuliMovieLoader
from model import ResidualConvSpikingNet
from evaluation import compute_TSRSA, estimate_neural_ceiling, regression_score
from manipulations import shuffle_frames, replace_frames_with_noise
from feedback_module import FeedbackModule

def main():
    # --- Load configuration ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device_type = config['system'].get('device', 'cuda')
    device = torch.device(device_type if torch.cuda.is_available() else 'cpu')
    seed = config['system'].get('seed', 42)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # --- Load neural responses ---
    neural_dir = 'path_to_neural_responses'  # User should specify actual path
    neural_loader = NeuralDataLoader(neural_dir,
                                     selected_regions=None,
                                     firing_rate_threshold=0.5)
    neural_responses_dict = neural_loader.get_all_responses()
    # Neural responses per region: dict {region_name: np.array (neurons, frames)}

    # --- Load stimulus movies ---
    stim_paths = { 'Movie1': 'path_to_movie1.mp4',
                   'Movie2': 'path_to_movie2.mp4' }
    stim_loader = StimuliMovieLoader(stim_paths,
                                     resize_dims=(224,224),
                                     frame_rate=30,
                                     device=device_type)
    all_movies = stim_loader.get_all_movies()

    # --- Initialize the model ---
    model_config = config['model']
    model = ResidualConvSpikingNet(config).to(device)

    # --- Load pretrained weights ---
    # Placeholders: Paths to pre-trained checkpoints
    pretrained_ucf_path = 'path_to_pretrained_ucf.pth'
    pretrained_imagenet_path = 'path_to_pretrained_imagenet.pth'
    # Load checkpoint based on training setting
    # For demonstration, load UCF pretrained if available
    if os.path.exists(pretrained_ucf_path):
        state_dict = torch.load(pretrained_ucf_path)
        model.load_state_dict(state_dict)
    elif os.path.exists(pretrained_imagenet_path):
        state_dict = torch.load(pretrained_imagenet_path)
        model.load_state_dict(state_dict)
    else:
        print("Warning: Pretrained weights not found. Proceeding with random init.")

    # --- Set up optimizer ---
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(),
                           lr=config['training'].get('learning_rate', 0.1),
                           weight_decay=config['training'].get('weight_decay', 1e-5))
    # Learning Rate Scheduler
    from torch.optim.lr_scheduler import StepLR
    max_epochs = config['system'].get('max_epochs', 320)
    lr_decay_step = config['training'].get('lr_decay_steps', 100)
    lr_decay_rate = config['training'].get('lr_decay_rate', 0.1)
    lr_scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_rate)

    # --- Define training hyperparameters ---
    simulation_time = config['training'].get('simulation_time', 16)

    # --- Prepare datasets and loaders (Training on UCF101) ---
    # Placeholder: need to implement/assume data loaders exist
    # Assuming their implementation: load_ucf101_dataset returns dataset objects
    # Here, for core logic, skip detailed dataset loader code
    # Instead just outline or mock:
    # train_dataset, val_dataset = load_ucf101_dataset(...)
    # For brevity, assume DataLoader 'train_loader' exists
    # Example:
    # train_loader = DataLoader(train_dataset, batch_size=..., shuffle=True)
    # For demonstration, assume 'train_loader' is given

    # For this code snippet, we will not implement full dataset loading, assuming it's available
    # The user must adapt accordingly.

    # --- Training loop ---
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        total_samples = 0

        # Assume 'train_loader' is a DataLoader yielding (inputs, labels)
        # Placeholders:
        # for inputs, labels in train_loader:
        # For demonstration, skip actual batch loop
        # In practice, load batches and do:
        # inputs: [batch, T, C, H, W]; labels: [batch]
        # Ensure reproducibility of shuffling etc. with seed if needed

        # -- Pseudo code (since dataset/loading is not detailed) --
        # for batch in train_loader:
        #     inputs, labels = batch
        #     inputs = inputs.to(device)
        #     labels = labels.to(device)
        #     model.reset()
        #     for t in range(inputs.shape[1]):
        #         frame_t = inputs[:, t, :, :, :]
        #         outputs = model.forward(frame_t)
        #     logits = model.classifier(outputs)
        #     loss = criterion(logits, labels)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        # Here, demonstrate response simulation with placeholders:
        # (In the real code, replace with actual dataset loop)

        # -- End pseudo code --

        # Since dataset loops are not fully specified, we omit actual training code
        # and proceed to evaluation stage after training if available.

        # For demonstration, after training, save model
        if epoch % 10 == 0 or epoch == max_epochs:
            torch.save(model.state_dict(), f'model_ckpt_epoch_{epoch}.pth')

        # Step learning rate
        lr_scheduler.step()

    # --- Load responses for stimulus movies ---
    def get_responses_for_movie(movie_array: np.ndarray):
        """
        Run the model over the input frames and extract responses.
        Assumption: model outputs feature vectors per timestep.
        """
        batch_size = 1
        T = movie_array.shape[0]
        responses_layer_region = {}
        # Placeholder: use model.forward to get responses at each timestep
        responses_over_time = {f'region_{i}': [] for i in range(model_config['model']['num_regions'])}
        # For each time step
        for t in range(T):
            frame = movie_array[t]
            # convert to tensor
            input_tensor = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).to(device)  # [1,C,H,W]
            # Run forward
            model.reset()
            # Run model for one frame (simulate time steps etc.)
            # Or, for simplicity, if model has a method to process sequence:
            # response = model.forward_sequence(sequence)
            # For code structure, assume model has method to get responses
            responses_dict = model.forward(input_tensor.unsqueeze(1))
            # responses_dict: dict {region_name: tensor [batch, channels, H, W]}
            for region_name, resp in responses_dict.items():
                # flatten resp over spatial dims
                flattened = resp.view(resp.shape[0], -1).cpu().numpy()
                responses_over_time[region_name].append(flattened.squeeze())
        # Convert list to numpy array: shape (T, features)
        responses_array = {}
        for region_name, resp_list in responses_over_time.items():
            responses_array[region_name] = np.stack(resp_list, axis=0).T  # shape: [features, time]
        return responses_array

    # --- Run responses for each movie ---
    responses_movie1 = get_responses_for_movie(all_movies['Movie1'])
    responses_movie2 = get_responses_for_movie(all_movies['Movie2'])

    # --- Compute TSRSA between model and neural responses ---
    # For each cortical region:
    # Extract neural response: neural_responses_dict[region] shape: neurons x frames
    # For simplicity, select responses for the same frames as stimuli
    # Using only common frames (assuming response frames match stimuli frames)
    # For each region:
    for region in neural_responses_dict.keys():
        neural_resp = neural_responses_dict[region]  # shape: neurons x frames
        # responses_movieX are dense feature vectors
        # We need to match dimensions or response subsets
        # For simplicity, assume responses are per neuron/region
        # When features are high-dim, consider responses_movieX as responses
        # For accurate measures, extract responses at the layer/region level
        # For now, assume their vectors are compatible
        # Compute similarity
        # responses_movieX: features x time, neural_resp: neurons x frames
        # Adjust to similar shape: Responses over time
        # For now, compute TSRSA
        # -- For illustration, just simulate scores --
        score_movie1 = compute_TSRSA(
            responses_movie1.get(region, np.zeros((1,0))),  # shape: features x T
            neural_resp
        )
        score_movie2 = compute_TSRSA(
            responses_movie2.get(region, np.zeros((1,0))),
            neural_resp
        )
        print(f'Region: {region} TSRSA Movie1: {score_movie1:.3f}')
        print(f'Region: {region} TSRSA Movie2: {score_movie2:.3f}')

    # --- Response manipulations for dynamic/static info ---
    # Example: static manipulation
    # For each manipulation ratio
    ratios = [0.25, 0.5, 0.75, 1.0]
    for ratio in ratios:
        # Shuffle frames
        shuffled_movie = shuffle_frames(all_movies['Movie1'], window_size=10, seed=seed)
        responses_shuffled = get_responses_for_movie(shuffled_movie)
        # Compute similarity with neural responses
        for region in neural_responses_dict:
            neural_resp = neural_responses_dict[region]
            sim_score = compute_TSRSA(
                responses_shuffled.get(region, np.zeros((1,0))),
                neural_resp
            )
            print(f'Shuffle Ratio {ratio} Region {region} TSRSA: {sim_score:.3f}')
        # Replace frames with noise
        noisy_movie = replace_frames(all_movies['Movie1'], ratio=ratio, seed=seed)
        responses_noise = get_responses_for_movie(noisy_movie)
        for region in neural_responses_dict:
            neural_resp = neural_responses_dict[region]
            sim_score = compute_TSRSA(
                responses_noise.get(region, np.zeros((1,0))),
                neural_resp
            )
            print(f'Noise Replacement Ratio {ratio} Region {region} TSRSA: {sim_score:.3f}')

    # --- Evaluate static natural scene stimuli (static image) ---
    # Assuming I have static images, repeat process
    # For brevity, omitted here, but follow same procedure
    
    # --- Regression analysis to compare model and neuron responses ---
    # For original responses
    for region in neural_responses_dict:
        neural_resp = neural_responses_dict[region]  # shape: neurons x frames
        model_response = responses_movie1.get(region, np.zeros((1,0)))  # features x frames
        r2_score = regression_score(neural_resp, model_response)
        print(f'Regression R^2 for region {region}: {r2_score:.3f}')

    # --- Visualization and saving results ---
    # For a real implementation, store scores and plot
    # For demonstration:
    # plt.figure()
    # plt.plot(score_list)
    # plt.xlabel('Manipulation Level')
    # plt.ylabel('TSRSA Score')
    # plt.title('Dynamic Variation Impact')
    # plt.savefig('tsrsa_dynamic.png')
    # plt.show()

if __name__ == "__main__":
    main()
```

## manipulations.py

```python
## manipulations.py
import numpy as np
from typing import Tuple

def shuffle_frames(movie: np.ndarray, window_size: int, seed: int = 0) -> np.ndarray:
    """
    Shuffle frames within non-overlapping windows of the movie.

    Args:
        movie (np.ndarray): Original movie array of shape [num_frames, H, W, C], values in [0,1].
        window_size (int): Number of frames per window to shuffle within.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Shuffled movie with same shape as input.
    """
    rng = np.random.default_rng(seed)
    num_frames = movie.shape[0]
    shuffled_movie = movie.copy()

    for start in range(0, num_frames, window_size):
        end = min(start + window_size, num_frames)
        indices = np.arange(start, end)
        rng.shuffle(indices)
        shuffled_movie[start:end] = movie[indices]
    return shuffled_movie


def replace_frames_with_noise(movie: np.ndarray, ratio: float, noise_type: str='Gaussian', seed: int=0) -> np.ndarray:
    """
    Replace a proportion of frames in the movie with noise images.

    Args:
        movie (np.ndarray): Original movie array [num_frames, H, W, C], values in [0,1].
        ratio (float): Fraction [0,1] of total frames to replace.
        noise_type (str): Type of noise, default 'Gaussian'.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Modified movie with selected frames replaced by noise images.
    """
    rng = np.random.default_rng(seed)
    num_frames = movie.shape[0]
    num_replace = int(ratio * num_frames)
    indices = np.arange(num_frames)
    rng.shuffle(indices)
    replace_indices = indices[:num_replace]

    H, W, C = movie.shape[1], movie.shape[2], movie.shape[3]
    modified_movie = movie.copy()

    for idx in replace_indices:
        if noise_type == 'Gaussian':
            # Generate a Gaussian noise image with mean=0.5, std=0.5
            noise_img = rng.normal(loc=0.5, scale=0.5, size=(H, W, C)).astype(np.float32)
            # Clip to [0,1]
            noise_img = np.clip(noise_img, 0.0, 1.0)
        else:
            # Default to zeros if unknown noise type
            noise_img = np.zeros((H, W, C), dtype=np.float32)
        # Replace the frame
        modified_movie[idx] = noise_img
    return modified_movie


def generate_static_noise_image(size: Tuple[int, int], channels: int=3, seed: int=0) -> np.ndarray:
    """
    Generate a static Gaussian noise image, to be used as a texture.

    Args:
        size (Tuple[int, int]): Height and Width of the image.
        channels (int): Number of color channels.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Noise image array [H, W, C], values in [0,1].
    """
    rng = np.random.default_rng(seed)
    noise_img = rng.normal(loc=0.5, scale=0.5, size=(size[0], size[1], channels)).astype(np.float32)
    noise_img = np.clip(noise_img, 0.0, 1.0)
    return noise_img
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_functions import SurrogateSpike
from spikingjelly.clock_driven import neuron, layer


class SurrogateTangent(torch.autograd.Function):
    """Surrogate gradient using inverse tangent approximation."""
    @staticmethod
    def forward(ctx, input, alpha=1.0):
        ctx.alpha = alpha
        return (torch.atan(input * alpha) / torch.atan(torch.tensor(torch.pi / 2)))  # scaled to [0,1]

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.alpha
        # Derivative of atan: 1/(1 + (alpha * x)^2)
        grad_input = grad_output.clone() * alpha / (1 + (alpha * ctx.saved_tensors[0]) ** 2)
        return grad_input, None


def surrogate_activation(x, alpha=1.0):
    return SurrogateTangent.apply(x, alpha)


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron model with surrogate gradient.
    """
    def __init__(self, tau=2.0, threshold=1.0, reset_voltage=0.0, surrogate_alpha=1.0):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.reset_voltage = reset_voltage
        self.surrogate_alpha = surrogate_alpha
        self.register_buffer('V', None)

    def forward(self, input: torch.Tensor):
        # input: [batch, *, ...] shape
        if self.V is None or self.V.shape != input.shape:
            # Initialize membrane potential
            self.V = torch.zeros_like(input)
        else:
            # Update membrane potential
            delta_V = (input - (self.V - self.reset_voltage)) / self.tau
            self.V = self.V + delta_V

        # Generate spikes with surrogate gradient
        spike = surrogate_activation(self.V - self.threshold, self.surrogate_alpha)
        # Binarize spike: thresholding
        spike_bin = (spike >= 0).float()

        # Reset voltage where spikes occurred
        self.V = self.V * (1 - spike_bin) + self.reset_voltage * spike_bin
        return spike_bin

    def reset(self):
        self.V.zero_()


class ResidualBlock(nn.Module):
    """
    Residual convolutional block with spiking neurons.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 tau=2.0, threshold=1.0, reset_voltage=0.0, surrogate_alpha=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.spike_neuron = LIFNeuron(tau=tau, threshold=threshold, reset_voltage=reset_voltage, surrogate_alpha=surrogate_alpha)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        # Apply spiking neuron element-wise in spatial dimension
        out_spike = self.spike_neuron(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = out_spike + residual
        return out


class FeedbackModule(nn.Module):
    """
    Feedback module: processes higher-region responses to generate feedback signals.
    Supports recurrent feedback with optional delay.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 feedback_strength=1.0, delay=2):
        super().__init__()
        self.feedback_strength = feedback_strength
        self.delay = delay
        # Use a simple convolutional layer for feedback projection
        self.feedback_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        # Initialize weights (could be learned or fixed)
        nn.init.xavier_uniform_(self.feedback_conv.weight)

        # Buffer to store delayed responses
        self.register_buffer('feedback_buffer', None)

    def forward(self, higher_response, current_step):
        """
        higher_response: Tensor of shape (batch, channels, H, W)
        current_step: int (current time step)
        """
        if self.feedback_buffer is None:
            # Initialize buffer with zeros
            self.feedback_buffer = torch.zeros_like(higher_response)

        # Apply delay: feedback comes from previous 'delay' steps
        if current_step >= self.delay:
            feedback_signal = self.feedback_buffer
        else:
            feedback_signal = torch.zeros_like(higher_response)

        # Update buffer with current response (simulate delay)
        self.feedback_buffer = higher_response.detach()

        # Modulate feedback signal
        feedback = self.feedback_conv(feedback_signal) * self.feedback_strength
        return feedback


class ResidualConvSpikingNet(nn.Module):
    """
    Main architecture: multiple regions with residual blocks, feedback modules, recurrent dynamics.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        cnn_channels = config.model.feature_channels  # base channels e.g., 64
        self.num_regions = config.model.num_regions  # e.g., 6 cortical regions
        self.layers_per_region = config.model.layers_per_region  # e.g., [2,2,2]
        self.tau = config.system.membrane_tau
        self.threshold = config.model.neuron_threshold
        self.reset_voltage = config.system.reset_voltage
        self.surrogate_alpha = 1.0  # As per config, can be parameterized
        self.feedback_strength = config.system.feedback_strength
        self.feedback_delay = config.system.feedback_delay

        # Create residual blocks per region
        self.regions = nn.ModuleDict()
        for i in range(self.num_regions):
            region_layers = nn.ModuleList()
            in_c = 3 if i == 0 else cnn_channels
            out_c = cnn_channels
            for j in range(self.layers_per_region[i]):
                region_layers.append(
                    ResidualBlock(
                        in_channels=in_c,
                        out_channels=out_c,
                        tau=self.tau,
                        threshold=self.threshold,
                        reset_voltage=self.reset_voltage,
                        surrogate_alpha=self.surrogate_alpha
                    )
                )
                in_c = out_c
            self.regions[f'region_{i}'] = region_layers

        # Create feedback modules connecting higher to lower regions
        self.feedback_modules = nn.ModuleDict()
        for i in range(1, self.num_regions):
            # Feedback from region_{i} to region_{i-1}
            self.feedback_modules[f'feedback_{i}_to_{i-1}'] = FeedbackModule(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                feedback_strength=self.feedback_strength,
                delay=self.feedback_delay
            )

        # Initialize membrane potentials
        self.reset()

    def reset(self):
        # Reset all neuron states
        for region_layers in self.regions.values():
            for layer in region_layers:
                if hasattr(layer, 'spike_neuron'):
                    layer.spike_neuron.reset()

    def forward(self, input_seq):
        """
        input_seq: Tensor [batch, T, C, H, W]
        Returns:
            responses_dict: dict of region responses over time
        """
        batch_size, T, C, H, W = input_seq.shape
        # Initialize membrane potentials per layer
        mem_list = {}
        for key, region_layers in self.regions.items():
            for idx, layer in enumerate(region_layers):
                mem_list[f"{key}_layer_{idx}"] = torch.zeros(
                    batch_size, layer.conv.out_channels, H, W, device=input_seq.device)

        # Store responses for TSRSA: per region, list over time
        responses_dict = {f'region_{i}': [] for i in range(self.num_regions)}

        # For each time step
        for t in range(T):
            current_inp = input_seq[:, t, :, :, :]  # [batch, C, H, W]
            # Start from lowest region (region_0)
            higher_response = None
            for region_idx in range(self.num_regions):
                region_name = f'region_{region_idx}'
                region_layers = self.regions[region_name]

                x = current_inp if region_idx == 0 else None  # Input only for first region
                # Pass through residual blocks
                for layer_idx, layer_block in enumerate(region_layers):
                    # Add feedback if applicable
                    if region_idx > 0:
                        feedback_signal = self.feedback_modules[f'feedback_{region_idx}_to_{region_idx -1}'](
                            higher_response, current_step=t
                        )
                        # Inject feedback into the input; here we add to the feature map
                        x = x + feedback_signal
                    # Update membrane potential
                    mem = mem_list[f"{region_name}_layer_{layer_idx}"]
                    # Forward through residual block with membrane integration
                    out = layer_block.conv(x)
                    out = layer_block.bn(out)
                    # Spiking neuron
                    spike_out = layer_block.spike_neuron(out)
                    # Residual connection
                    x = spike_out + (x if t == 0 else x)  # skip connection
                    # Save membrane voltages
                    mem.copy_(layer_block.spike_neuron.V)
                # Response of this region at time t
                responses_dict[region_name].append(x.detach())

                # For next higher region, use the response of this region
                higher_response = x

        # Convert responses to tensors: list over time -> [batch, time, channels, H, W]
        for key in responses_dict:
            responses_tensor = torch.stack(responses_dict[key], dim=1)
            responses_dict[key] = responses_tensor

        return responses_dict
```


## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import datetime
import numpy as np

from dataset_loader import load_ucf101_dataset, load_imagenet_dataset
from model import ResidualConvSpikingNet
from evaluation import compute_TSRSA, compute_neural_ceiling, regression_score
from manipulations import shuffle_frames_in_window, replace_frames_with_noise
from feedback_module import FeedbackModule

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Device setup
device = torch.device(config['system'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
torch.manual_seed(config['system'].get('seed', 42))
if device.type == 'cuda':
    torch.cuda.manual_seed_all(config['system'].get('seed', 42))

# Build datasets and dataloaders
train_ucf, val_ucf = load_ucf101_dataset(
    data_dir='path_to_ucf101',  # specify your dataset path
    batch_size=config['training']['batch_size'],
    split_ratio=0.8,
    mode='train'
)
train_ucf_loader = DataLoader(train_ucf, batch_size=config['training']['batch_size'], shuffle=True)
val_ucf_loader = DataLoader(val_ucf, batch_size=config['training']['batch_size'], shuffle=False)

# For ImageNet, you can similarly create loaders if needed, or omit if not used in this script.
# For now, focus on UCF101 pretraining as per the paper.

# Initialize model
model = ResidualConvSpikingNet(config).to(device)

# Instantiate feedback modules, if needed, as part of model or separately
# Placeholder: feedback modules are integrated into model

# Surrogate gradient function is implemented in model.py via custom autograd

# Set optimizer
optimizer = optim.Adam(model.parameters(), lr=config['training'].get('learning_rate', 0.1), weight_decay=config['training'].get('weight_decay', 1e-5))

# Learning rate scheduler
lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config['training'].get('lr_decay_steps', 100),
    gamma=config['training'].get('lr_decay_rate', 0.1)
)

# Training parameters
max_epochs = config['system'].get('max_epochs', 320)
simulation_time = config['training'].get('simulation_time', 16)  # for UCF101

# Function to initialize membrane potentials
def reset_neuron_states(model):
    model.reset()

# Main training loop
for epoch in range(1, max_epochs + 1):
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(train_ucf_loader):
        inputs = inputs.to(device)  # shape: [batch, T, C, H, W]
        labels = labels.to(device)

        # Reset neuron states at each batch
        reset_neuron_states(model)

        # Initialize response storage
        batch_responses = []

        # Forward pass over sequence
        for t in range(inputs.shape[1]):
            # Extract current frame batch
            frame_t = inputs[:, t, :, :, :]  # shape: [batch, C, H, W]
            # For ImageNet, frames are repeated same image; no special handling needed here
            # Feedback mechanisms are embedded in model.forward
            outputs = model.forward(frame_t.unsqueeze(1))  # shape: [batch, ...], process as sequence step
            # Collect responses from a designated layer or output layer
            # For simplicity, assume model returns response features at current timestep
            # e.g., model returns a tensor [batch, feature_dim]
            # Here, we simulate responses: for TSRSA, responses per timestep are accumulated
            # Let's assume model provides 'response' in some form:
            # For this code, suppose model outputs features after the last residual layer
            # Alternatively, adapt model.py to return features at each timestep
            # Here, we assume 'outputs' is the feature vector
            batch_responses.append(outputs)

        # Stack responses over time: shape [batch, T, feature_dim]
        responses = torch.stack(batch_responses, dim=1)

        # Classification output (assuming a linear classifier head inside model, or add here)
        # For simplicity, suppose model has a classifier
        logits = model.classifier(responses[:, -1, :])  # use last timestep features
        loss = nn.CrossEntropyLoss()(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        # Ideally, the surrogate gradient is applied inside model's backward pass
        # Here, it's handled implicitly via the custom autograd functions in model.py

        optimizer.step()

        epoch_loss += loss.item() * labels.size(0)
        _, preds = torch.max(logits, 1)
        epoch_correct += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

    # Step LR scheduler
    lr_scheduler.step()

    # Log epoch metrics
    epoch_loss_avg = epoch_loss / total_samples
    epoch_acc = epoch_correct / total_samples
    print(f"Epoch [{epoch}/{max_epochs}] Loss: {epoch_loss_avg:.4f} Acc: {epoch_acc:.4f} LR: {lr_scheduler.get_last_lr()[0]:.6f}")

    # Save checkpoint periodically
    if epoch % 10 == 0 or epoch == max_epochs:
        ckpt_path = f'checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
        }, ckpt_path)

# After training, save final model
torch.save(model.state_dict(), 'loraFB_SpikingNet_final.pth')
print("Training completed and model saved.")
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\SNN-Neural-Similarity-Movie\SNN-Neural-Similarity-Movie_repo`
