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

