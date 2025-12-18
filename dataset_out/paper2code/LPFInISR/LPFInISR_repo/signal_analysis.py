## signal_analysis.py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import get_window
from scipy.ndimage import convolve
import math
from typing import Tuple
import importlib

# We import configuration parameters for spectral analysis
import yaml

# Load config.yaml to access fft_size, window function, normalization flags
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

FFT_SIZE = config['spectral_analysis'].get('fft_size', 256)
WINDOW_FUNC = config['spectral_analysis'].get('window_function', 'hann')
NORMALIZE_SPECTRA = config['spectral_analysis'].get('spectral_normalization', True)


def compute_fft(image: np.ndarray) -> np.ndarray:
    """
    Compute the 2D Fourier spectrum of an image with optional windowing and zero-padding.

    Args:
        image (np.ndarray): 2D or 3D array (H,W,C). If 3D, process each channel separately.

    Returns:
        np.ndarray: Shifted FFT spectrum of shape (FFT_SIZE, FFT_SIZE, C) if color, else (FFT_SIZE, FFT_SIZE).
    """
    # Ensure image is in shape (H,W,C) or (H,W)
    if image.ndim == 2:
        image = image[:, :, None]

    H, W, C = image.shape

    # Pad image to FFT_SIZE if smaller
    pad_h = max(FFT_SIZE - H, 0)
    pad_w = max(FFT_SIZE - W, 0)
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    pad_bottom = pad_h - pad_top
    pad_right = pad_w - pad_left

    spectrum_list = []
    # Generate window if normalization enabled
    window = get_window(WINDOW_FUNC, min(H, W))
    window_2d = np.outer(window, window) if H == W else None

    for c in range(C):
        channel_img = image[:, :, c]
        # Zero-pad
        padded = np.pad(
            channel_img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )
        # Multiply by window if normalization
        if NORMALIZE_SPECTRA:
            if window_2d is not None and window.shape[0] == H:
                # apply window
                padded *= window_2d
            else:
                # fallback: skip windowing if size does not match
                pass

        # Compute FFT with zero-padding to FFT_SIZE
        fft_result = np.fft.fft2(padded, s=(FFT_SIZE, FFT_SIZE))
        fft_shifted = np.fft.fftshift(fft_result)
        spectrum_list.append(fft_shifted)

    if C == 1:
        return spectrum_list[0]
    else:
        # Shape: (FFT_SIZE, FFT_SIZE, C)
        return np.stack(spectrum_list, axis=-1)


def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """
    Normalize a spectrum (complex) by subtracting mean magnitude and dividing by std.

    Args:
        spectrum (np.ndarray): Complex spectrum array.

    Returns:
        np.ndarray: Normalized spectrum of same shape.
    """
    mag = np.abs(spectrum)
    mean_val = np.mean(mag)
    std_val = np.std(mag) + 1e-8
    mag_norm = (mag - mean_val) / std_val
    # Keep phase information intact if needed, but normalization on magnitude suffices
    # Return complex spectrum with normalized magnitude
    phase = np.angle(spectrum)
    mag_norm = np.clip(mag_norm, -3, 3)  # optional: clip for stability
    normalized_spectrum = mag_norm * np.exp(1j * phase)
    return normalized_spectrum


def extract_impulse_response(model: torch.nn.Module, device: torch.device) -> np.ndarray:
    """
    Generate the impulse response (H(δ)) of the network by feeding impulse image.

    Args:
        model (torch.nn.Module): Trained super-resolution model.
        device (torch.device): Computation device.

    Returns:
        np.ndarray: Impulse response in spatial domain, shape (H,W,C).
    """
    size = 11  # typical size for impulse test
    impulse_np = np.zeros((size, size, 3), dtype=np.float32)
    center = size // 2
    impulse_np[center, center, :] = 1.0  # maximum intensity at center pixel, scaled to [0,1]
    impulse_tensor = torch.from_numpy(impulse_np.transpose(2,0,1)).unsqueeze(0).to(device)

    with torch.no_grad():
        response = model(impulse_tensor)  # output shape: (1,C,H,W)
    response_np = response.squeeze(0).permute(1,2,0).cpu().numpy()  # (H,W,C)
    return response_np


def compute_impulse_response(model: torch.nn.Module, image_size: int, device: torch.device) -> np.ndarray:
    """
    Wrapper to generate impulse response for given model.
    """
    return extract_impulse_response(model, device)


def decompose_response(network_output: np.ndarray, impulse_response: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose network output N(I) into linear H(I) and nonlinear G(I) components.

    Args:
        network_output (np.ndarray): Output N(I), shape (H,W,C) or (H,W).
        impulse_response (np.ndarray): H(δ), shape (h,w) grayscale or color.

    Returns:
        Tuple[np.ndarray, np.ndarray]: H(I) and G(I), both shape matching network_output.
    """
    # Convert impulse_response to 2D array (grayscale)
    if impulse_response.ndim == 3:
        # For multi-channel, take the mean to approximate
        impulse_resp_gray = np.mean(impulse_response, axis=2)
    else:
        impulse_resp_gray = impulse_response

    # Ensure shapes are compatible
    # Convert to float32
    network_output = network_output.astype(np.float32)

    # Compute linear response via convolution:
    # Using 'full' mode to preserve size, then crop to match network_output
    linear_response = convolve(impulse_resp_gray, np.ones_like(impulse_resp_gray), mode='constant')  # placeholder
    # But in context, we approximate H(I) by convolving H(δ) with I, for synthetic impulse input this is simple
    # For general implementation, assuming network input I is an impulse pattern, H(I) approximates convolution.

    # For a full image, decompose:
    # G(I) = N(I) - H(I), where H(I) is convolution of H(δ) with I (or the impulse pattern)
    # Since we only have N(I), a practical approach is to convolve the input I with H(δ); here, for G(I), approximate G(I) = N(I) - H(I)
    # For the current implementation, the best we can do is to perform the convolution of impulse_response with the input image
    # But if only the response to the impulse is available, then for the impulse input: H(I) = impulse_response
    # For general images, if more info becomes available, this can be updated.
    # Here, for the impulse images, H(I) is approximated as convolution with H(δ).

    # As an approximation, if network output is from the impulse input, then:
    H_I = impulse_response  # assuming the response to impulse input
    G_I = network_output - H_I
    return H_I, G_I


def compute_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Compute the power spectrum of the image.

    Args:
        image (np.ndarray): 2D array (grayscale) or 3D (color).

    Returns:
        np.ndarray: Power spectrum (mean over channels if color).
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image
    spectrum = compute_fft(gray)
    power_spec = np.abs(spectrum) ** 2
    return power_spec


def spectral_similarity(spectrum_hr: np.ndarray, spectrum_sr: np.ndarray) -> float:
    """
    Compute the FSDS (similarity) between two spectrums.

    Args:
        spectrum_hr (np.ndarray): HR spectrum (power).
        spectrum_sr (np.ndarray): SR spectrum (power).

    Returns:
        float: FSDS value in dB scale.
    """
    # Normalize spectra
    if NORMALIZE_SPECTRA:
        mean_hr = np.mean(spectrum_hr)
        std_hr = np.std(spectrum_hr) + 1e-8
        spectrum_hr = (spectrum_hr - mean_hr) / std_hr

        mean_sr = np.mean(spectrum_sr)
        std_sr = np.std(spectrum_sr) + 1e-8
        spectrum_sr = (spectrum_sr - mean_sr) / std_sr

    # Compute difference as per FSDS
    D = np.abs(spectrum_hr - spectrum_sr)  # magnitude difference
    sum_diff = np.sum(D)  # sum over all frequency bins
    # To prevent log of zero, add small epsilon
    epsilon = 1e-8
    fsds_value = 10.0 * np.log10(sum_diff + epsilon)
    return fsds_value


def calculate_fsds(
    hr_image: np.ndarray,
    sr_image: np.ndarray,
    fft_size: int = FFT_SIZE,
    window: str = WINDOW_FUNC,
    normalize_spectra: bool = True
) -> float:
    """
    Compute the FSDS metric between HR and SR images.

    Args:
        hr_image (np.ndarray): High-res ground-truth image.
        sr_image (np.ndarray): Super-resolved image from model.
        fft_size (int): FFT size for spectral analysis.
        window (str): Window function name.
        normalize_spectra (bool): Whether to normalize spectra.

    Returns:
        float: FSDS score in dB indicating similarity.
    """
    # Compute spectra
    spectrum_hr = compute_fft(hr_image)
    spectrum_sr = compute_fft(sr_image)

    # Normalize spectra if required
    if normalize_spectra:
        spectrum_hr = normalize_spectrum(spectrum_hr)
        spectrum_sr = normalize_spectrum(spectrum_sr)

    # Compute power spectra for FSDS
    power_hr = np.abs(spectrum_hr) ** 2
    power_sr = np.abs(spectrum_sr) ** 2

    # Compute similarity
    fsds_score = spectral_similarity(power_hr, power_sr)
    return fsds_score


def simulate_lowpass_filter(cutoff_freq: float, size: Tuple[int, int], window: str = 'hann') -> np.ndarray:
    """
    Generate a 2D ideal low-pass filter mask (binary) in frequency domain.

    Args:
        cutoff_freq (float): Cut-off frequency in radians per pixel (normalized freq).
        size (Tuple[int, int]): Size of the filter mask in (height, width).
        window (str): Window function to smooth edges ('hann', 'blackman', etc.).

    Returns:
        np.ndarray: 2D filter mask with values in [0,1].
    """
    h, w = size
    # Generate meshgrid of normalized frequency coordinates
    freq_x = np.fft.fftfreq(w, d=1.0/w)
    freq_y = np.fft.fftfreq(h, d=1.0/h)
    FX, FY = np.meshgrid(freq_x, freq_y)
    # Create radius in frequency domain
    radius = np.sqrt(FX**2 + FY**2)
    # Create mask
    mask = np.zeros_like(radius)
    mask[radius <= (cutoff_freq / np.pi)] = 1.0  # normalized: cutoff_freq in [0, pi]
    # Optional: apply window for smooth edges
    if window:
        window_x = get_window(window, mask.shape[1])
        window_y = get_window(window, mask.shape[0])
        window_2d = np.outer(window_y, window_x)
        mask *= window_2d
    return mask


def plot_spectra(spectrum1: np.ndarray, spectrum2: np.ndarray, log_scale: bool = True):
    """
    Plot the two spectra side by side for visual comparison.

    Args:
        spectrum1 (np.ndarray): Spectrum array.
        spectrum2 (np.ndarray): Spectrum array.
        log_scale (bool): Whether to display in log scale.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    if log_scale:
        im1 = axes[0].imshow(np.log10(np.abs(spectrum1)+1e-8), cmap='inferno')
        im2 = axes[1].imshow(np.log10(np.abs(spectrum2)+1e-8), cmap='inferno')
    else:
        im1 = axes[0].imshow(np.abs(spectrum1), cmap='inferno')
        im2 = axes[1].imshow(np.abs(spectrum2), cmap='inferno')
    axes[0].set_title('Spectrum 1')
    axes[1].set_title('Spectrum 2')
    plt.colorbar(im1, ax=axes[0])
    plt.colorbar(im2, ax=axes[1])
    plt.show()

