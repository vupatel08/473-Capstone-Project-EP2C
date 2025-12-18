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

