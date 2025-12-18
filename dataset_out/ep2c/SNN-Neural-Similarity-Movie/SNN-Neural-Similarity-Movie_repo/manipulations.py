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
