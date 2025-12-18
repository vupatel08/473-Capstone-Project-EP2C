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
