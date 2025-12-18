## visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import normalize_tensor

class Visualization:
    """
    Provides visualization utilities for impulse responses, spectral spectra, and response comparisons
    in the context of analyzing neural network-based image super-resolution models.
    Utilizes matplotlib for plotting and supports saving figures to specified directories.
    """

    def __init__(self, config=None):
        """
        Initialize Visualization with optional configurations.
        Uses default visualization directories if not specified.

        Args:
            config (dict, optional): Configuration dictionary with keys:
                - viz_save_dir (str): Directory to save plots. Defaults to './visualizations'.
        """
        if config is None:
            self.viz_save_dir = './visualizations'
        else:
            self.viz_save_dir = config.get('viz_save_dir', './visualizations')

        # Create directory if it doesn't exist
        if not os.path.exists(self.viz_save_dir):
            os.makedirs(self.viz_save_dir)

    def plot_impulse_response(self, h_resp: torch.Tensor, save_path: str = None, title: str = None):
        """
        Visualizes the spatial impulse response, channel-wise.

        Args:
            h_resp (torch.Tensor): Tensor of shape [C, H, W], impulse response.
            save_path (str, optional): Path to save image. Defaults to None (show plot).
            title (str, optional): Title for the plot. Defaults to None.
        """
        # Convert tensor to numpy array
        resp_np = h_resp.cpu().numpy()
        C, H, W = resp_np.shape

        for c in range(C):
            plt.figure(figsize=(6,6))
            plt.imshow(resp_np[c], cmap='viridis')
            plt.colorbar()
            plt.title(f"Impulse Response Channel {c}" if not title else title)
            if save_path:
                filename = os.path.join(self.viz_save_dir, f"{save_path}_channel_{c}.png")
                plt.savefig(filename, dpi=300)
                plt.close()
            else:
                plt.show()

    def plot_spectra(self, mag1: np.ndarray, phase1: np.ndarray,
                     mag2: np.ndarray, phase2: np.ndarray,
                     save_path_prefix: str = None, title: str = None):
        """
        Plots magnitude and phase spectra of two spectral responses for comparison.

        Args:
            mag1, mag2 (np.ndarray): Magnitude spectra [C, H, W].
            phase1, phase2 (np.ndarray): Phase spectra [C, H, W].
            save_path_prefix (str, optional): Prefix for saving plots. If None, plots are shown.
            title (str, optional): Overall title for the plots.
        """
        num_channels = mag1.shape[0]
        for c in range(num_channels):
            # Magnitude spectrum plot
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(np.fft.fftshift(mag1[c]), cmap='inferno')
            plt.title(f"{title} - Magnitude Spectrum (Channel {c})" if title else f"Channel {c} - Magnitude Spectrum")
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(np.fft.fftshift(mag2[c]), cmap='inferno')
            plt.title(f"{title} - Magnitude Spectrum (Channel {c})" if title else f"Channel {c} - Magnitude Spectrum")
            plt.colorbar()

            if save_path_prefix:
                filename = os.path.join(self.viz_save_dir, f"{save_path_prefix}_mag_channel_{c}.png")
                plt.savefig(filename, dpi=300)
                plt.close()
            else:
                plt.show()

            # Phase spectrum plot
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(np.fft.fftshift(phase1[c]), cmap='twilight')
            plt.title(f"{title} - Phase Spectrum (HR) (Channel {c})" if title else f"Channel {c} - Phase Spectrum")
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(np.fft.fftshift(phase2[c]), cmap='twilight')
            plt.title(f"{title} - Phase Spectrum (SR) (Channel {c})" if title else f"Channel {c} - Phase Spectrum")
            plt.colorbar()

            if save_path_prefix:
                filename = os.path.join(self.viz_save_dir, f"{save_path_prefix}_phase_channel_{c}.png")
                plt.savefig(filename, dpi=300)
                plt.close()
            else:
                plt.show()

    def plot_responses(self, input_img: torch.Tensor,
                       linear_response: torch.Tensor,
                       nonlinear_response: torch.Tensor,
                       save_path: str = None,
                       titles: list = None):
        """
        Visualize input, linear, and nonlinear responses side-by-side per channel.

        Args:
            input_img (torch.Tensor): [C, H, W]
            linear_response (torch.Tensor): [C, H, W]
            nonlinear_response (torch.Tensor): [C, H, W]
            save_path (str, optional): Base filename to save figure. Defaults to None.
            titles (list, optional): List of titles for the plot rows. Defaults to None.
        """
        input_np = input_img.cpu().numpy()
        lin_np = linear_response.cpu().numpy()
        nonlin_np = nonlinear_response.cpu().numpy()

        C, H, W = input_np.shape
        for c in range(C):
            plt.figure(figsize=(15,5))
            plt.subplot(1,3,1)
            plt.imshow(input_np[c], cmap='gray')
            plt.title(f'Input Channel {c}')
            plt.axis('off')

            plt.subplot(1,3,2)
            plt.imshow(lin_np[c], cmap='gray')
            plt.title('Linear Response')
            plt.axis('off')

            plt.subplot(1,3,3)
            plt.imshow(nonlin_np[c], cmap='gray')
            plt.title('Nonlinear Response')
            plt.axis('off')

            if save_path:
                filename = os.path.join(self.viz_save_dir, f"{save_path}_channel_{c}.png")
                plt.savefig(filename, dpi=300)
                plt.close()
            else:
                plt.show()

    def save_fig(self, fig: plt.Figure, filename: str):
        """
        Utility to save a matplotlib figure to file, ensuring directory exists.

        Args:
            fig (matplotlib.pyplot.Figure): The figure object.
            filename (str): Path to save.
        """
        dir_path = os.path.dirname(filename)
        os.makedirs(dir_path, exist_ok=True)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
