## spectral_analysis.py
import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
import torch

from utils import normalize_tensor

class SpectralAnalysis:
    """
    Provides spectral analysis utilities for deep learning super-resolution models:
    - Computing the 2D FFT spectra of images or responses.
    - Calculating the FSDS spectral similarity metric.
    - Visualizing magnitude and phase spectra.
    """

    def __init__(self, config=None):
        """
        Initializes the SpectralAnalysis class with default or provided configuration.
        
        Args:
            config (dict, optional): Configuration dictionary.
                Keys:
                 - device (str): 'cuda' or 'cpu' (default: 'cuda' if available)
                 - spectrum_normalization (bool): whether to normalize tensor before FFT (default: True)
                 - fft_size (int or tuple): size for FFT zero-padding (default: None, uses input shape)
        """
        # Set defaults
        if config is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.spectrum_normalization = True
            self.fft_size = None  # Use input size directly
        else:
            self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            self.spectrum_normalization = config.get('spectrum_normalization', True)
            self.fft_size = config.get('fft_size', None)

    def fft_response(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Computes the 2D Fourier spectrum magnitude and phase of an image or response.

        Args:
            image_tensor (torch.Tensor): [C, H, W], tensor in CPU or GPU.

        Returns:
            np.ndarray: 2D complex spectrum (FFT response) array of shape (C, H, W).
        """
        # Move to CPU for numpy processing
        tensor_np = image_tensor.cpu().numpy()
        C, H, W = tensor_np.shape

        # Optional normalization
        if self.spectrum_normalization:
            for c in range(C):
                tensor_np[c] = normalize_tensor(tensor_np[c])

        # Determine FFT size
        if self.fft_size is None:
            fftH, fftW = H, W
        elif isinstance(self.fft_size, tuple):
            fftH, fftW = self.fft_size
        else:
            # Single size input
            fftH, fftW = self.fft_size, self.fft_size

        # Zero-pad input to FFT size if needed
        mag_response = np.empty((C, fftH, fftW), dtype=np.complex64)
        for c in range(C):
            # Zero-pad
            padH = fftH - H
            padW = fftW - W
            pad_top = padH // 2
            pad_bottom = padH - pad_top
            pad_left = padW // 2
            pad_right = padW - pad_left
            
            padded = np.pad(tensor_np[c],
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='constant', constant_values=0)
            fft_res = scipy.fft.fft2(padded)
            mag_response[c] = fft_res

        return mag_response

    def calculate_fsds(self, spectrum_hr: dict, spectrum_sr: dict) -> float:
        """
        Calculates the FSDS similarity between two spectral power distributions.

        Args:
            spectrum_hr (dict): {'magnitude': list of np.ndarray, 'phase': list of np.ndarray}
            spectrum_sr (dict): same structure as spectrum_hr

        Returns:
            float: FSDS value (higher means more similar).
        """
        # Compute average magnitude response over channels
        mag_hr = np.mean(np.array(spectrum_hr['magnitude']), axis=0)
        mag_sr = np.mean(np.array(spectrum_sr['magnitude']), axis=0)

        # Compute power maps
        D_hr = mag_hr ** 2
        D_sr = mag_sr ** 2

        # Difference map
        D_diff = D_hr - D_sr

        # Sum of squared differences
        numerator = np.sum(np.abs(D_diff) ** 2)
        # Total spectral power of HR
        denominator = np.sum(np.abs(D_hr) ** 2) + 1e-8  # prevent division by zero

        # Compute FSDS in dB scale as per paper
        fsds_value = -10.0 * np.log10(numerator / denominator + 1e-8)
        return fsds_value

    def visualize_spectra(self,
                          spectrum_hr: dict,
                          spectrum_sr: dict,
                          save_path_prefix: str = None):
        """
        Visualizes magnitude and phase spectra for HR and SR responses.

        Args:
            spectrum_hr (dict): spectral response dictionary for high-res (ground truth)
            spectrum_sr (dict): spectral response dictionary for simulated response
            save_path_prefix (str, optional): If provided, prefix for saved images.
        """
        # Extract spectra
        mag_hr = np.array(spectrum_hr['magnitude'])
        phase_hr = np.array(spectrum_hr['phase'])
        mag_sr = np.array(spectrum_sr['magnitude'])
        phase_sr = np.array(spectrum_sr['phase'])

        # For each channel, plot magnitude and phase spectra
        num_chans = mag_hr.shape[0]

        for c in range(num_chans):
            # Plot magnitude spectra
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(np.fft.fftshift(mag_hr[c]), cmap='inferno')
            plt.title(f'HR Magnitude Spectrum (Channel {c})')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(np.fft.fftshift(mag_sr[c]), cmap='inferno')
            plt.title(f'SR Magnitude Spectrum (Channel {c})')
            plt.colorbar()

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_mag_channel_{c}.png")
            else:
                plt.show()
            plt.close()

            # Plot phase spectra
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(np.fft.fftshift(phase_hr[c]), cmap='twilight')
            plt.title(f'HR Phase Spectrum (Channel {c})')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(np.fft.fftshift(phase_sr[c]), cmap='twilight')
            plt.title(f'SR Phase Spectrum (Channel {c})')
            plt.colorbar()

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_phase_channel_{c}.png")
            else:
                plt.show()
            plt.close()
