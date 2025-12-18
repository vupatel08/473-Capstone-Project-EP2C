## utils.py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import get_window
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import os
import yaml

# Load configuration parameters from 'config.yaml'
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

FFT_SIZE: int = cfg['spectral_analysis'].get('fft_size', 256)
WINDOW_FUNC: str = cfg['spectral_analysis'].get('window_function', 'hann')
NORMALIZE_SPECTRA: bool = cfg['spectral_analysis'].get('spectral_normalization', True)
OUTPUT_DIR: str = cfg['utilities'].get('output_dir', './results')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility across numpy, torch, and standard random.
    """
    import random
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """
    Normalize the magnitude of a complex spectrum to zero mean and unit std.
    """
    mag = np.abs(spectrum)
    mean_val = np.mean(mag)
    std_val = np.std(mag) + 1e-8
    mag_norm = (mag - mean_val) / std_val
    # Clip for numerical stability
    mag_norm = np.clip(mag_norm, -3, 3)
    phase = np.angle(spectrum)
    # Reconstruct normalized spectrum
    return mag_norm * np.exp(1j * phase)

def compute_fft(image: np.ndarray) -> np.ndarray:
    """
    Compute the centered 2D FFT of an image with windowing.
    Supports multichannel images; processes each channel separately.
    Applies zero-padding to size FFT_SIZE.
    """
    if image.ndim == 2:
        image = image[:, :, None]
    H, W, C = image.shape

    # Determine padding sizes
    pad_h = max(FFT_SIZE - H, 0)
    pad_w = max(FFT_SIZE - W, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    spectra = []
    # Generate window for spectral tapering
    window = get_window(WINDOW_FUNC, min(H, W))
    if H == W:
        window2d = np.outer(window, window)
    else:
        window2d = None

    for c in range(C):
        channel_img = image[:, :, c]
        # Zero-pad
        padded = np.pad(channel_img, ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='constant', constant_values=0)
        # Apply window if size matches
        if window2d is not None and padded.shape[0] == H and padded.shape[1] == W:
            padded = padded * window2d
        # FFT with size FFT_SIZE
        fft_res = fft2(padded, s=(FFT_SIZE, FFT_SIZE))
        fft_res = fftshift(fft_res)
        spectra.append(fft_res)

    if C == 1:
        return spectra[0]
    else:
        return np.stack(spectra, axis=-1)

def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """
    Normalize complex spectrum: scale magnitudes to zero mean, unit variance.
    """
    mag = np.abs(spectrum)
    mean_val = np.mean(mag)
    std_val = np.std(mag) + 1e-8
    mag_norm = (mag - mean_val) / std_val
    phase = np.angle(spectrum)
    return mag_norm * np.exp(1j * phase)

def extract_impulse_response(model: torch.nn.Module, device: torch.device) -> np.ndarray:
    """
    Generate the network's impulse response by passing an impulse image.
    Returns as a spatial array (H, W, C).
    """
    size = 11  # typical size for impulse test
    impulse_np = np.zeros((size, size, 3), dtype=np.float32)
    center = size // 2
    impulse_np[center, center, :] = 1.0  # Max intensity scaled to [0,1]
    input_tensor = torch.from_numpy(impulse_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        response = model(input_tensor)
    # Response shape: (1,C,H,W)
    response_np = response.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return response_np

def convolve_linear_response(impulse_response: np.ndarray, input_image: np.ndarray) -> np.ndarray:
    """
    Approximate linear response by convolving a 2D impulse response with an input image.
    Supports grayscale or color images.
    """
    # Convert to grayscale if color
    if input_image.ndim == 3:
        input_gray = np.mean(input_image, axis=2)
        resp_gray = np.mean(impulse_response, axis=2)
        response = convolve(input_gray, resp_gray, mode='constant', cval=0.0)
        # For color output, replicate the response
        response_c = np.repeat(response[:, :, None], 3, axis=2)
        return response_c
    else:
        response = convolve(input_image, impulse_response, mode='constant', cval=0.0)
        return response

def decompose_response(network_output: np.ndarray, impulse_response: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose the network output into linear H(I) and nonlinear G(I).
    Assumes that H(δ) (impulse_response) is the linear part's response to impulse input.
    """
    # Convert impulse_response to grayscale if needed
    if impulse_response.ndim == 3:
        impulse_gray = np.mean(impulse_response, axis=2)
    else:
        impulse_gray = impulse_response
    # For synthetic impulse input, H(I) ≈ convolution of impulse_response with the input
    # Here, as a simplification, assume H(I) = convolution of impulse_response with I
    # Let's perform the convolution
    H_I = convolve_linear_response(impulse_response, windowed_input=None)
    # But since we lack the actual input image, a practical approximation:
    # We'll consider H(I) as the response to impulse input
    # The full response N(I) is provided (network_output)
    G_I = network_output - H_I
    return H_I, G_I

def compute_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Compute the shift-invariant magnitude spectrum of an image.
    """
    spectrum = compute_fft(image)
    mag = np.abs(spectrum)
    if NORMALIZE_SPECTRA:
        mag = (mag - np.mean(mag)) / (np.std(mag) + 1e-8)
    return mag

def spectral_similarity(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Compute the similarity measure between two spectra (e.g., FSDS).
    """
    diff = np.abs(spectrum1 - spectrum2)
    sum_diff = np.sum(diff)
    epsilon = 1e-8
    return 10.0 * np.log10(sum_diff + epsilon)

def calculate_fsds(hr_image: np.ndarray, sr_image: np.ndarray) -> float:
    """
    Calculate Frequency Spectrum Distribution Similarity (FSDS) between HR and SR images.
    """
    # Compute spectra
    spectrum_hr = compute_fft(hr_image)
    spectrum_sr = compute_fft(sr_image)
    mag_hr = np.abs(spectrum_hr)
    mag_sr = np.abs(spectrum_sr)

    if NORMALIZE_SPECTRA:
        mag_hr = (mag_hr - np.mean(mag_hr)) / (np.std(mag_hr) + 1e-8)
        mag_sr = (mag_sr - np.mean(mag_sr)) / (np.std(mag_sr) + 1e-8)

    fsds_value = spectral_similarity(mag_hr, mag_sr)
    return fsds_value

def generate_lowpass_filter(cutoff_freq: float, size: Tuple[int, int], window_func: str = 'hann') -> np.ndarray:
    """
    Generate a 2D ideal low-pass filter mask in frequency domain with optional window smoothing.
    """
    h, w = size
    freq_x = np.fft.fftfreq(w)
    freq_y = np.fft.fftfreq(h)
    FX, FY = np.meshgrid(freq_x, freq_y)
    radius = np.sqrt(FX ** 2 + FY ** 2)
    norm_cutoff = cutoff_freq / np.pi  # normalized cutoff (0 to 0.5)
    mask = np.zeros_like(radius)
    mask[radius <= norm_cutoff] = 1.0
    # Apply window to taper edges
    if window_func:
        window_x = get_window(window_func, w)
        window_y = get_window(window_func, h)
        window2d = np.outer(window_y, window_x)
        mask *= window2d
    return mask

def apply_lowpass_filter(image: np.ndarray, cutoff_freq: float, size: Tuple[int, int], window_func: str = 'hann') -> np.ndarray:
    """
    Filter an image with a generated low-pass filter in frequency domain.
    Returns the real part of the inverse FFT result.
    """
    spectrum = compute_fft(image)
    filter_mask = generate_lowpass_filter(cutoff_freq, size, window_func)
    # Apply filter in frequency domain
    filtered_spectrum = spectrum * filter_mask
    # Inverse FFT
    img_filtered = ifft2(ifftshift(filtered_spectrum)).real
    return img_filtered

def plot_spectrum(spectrum: np.ndarray, title: str = 'Spectrum'):
    """
    Plot the magnitude spectrum log scale.
    """
    plt.figure()
    plt.imshow(np.log10(np.abs(spectrum) + 1e-8), cmap='inferno')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title.replace(' ', '_')}.png"))
    plt.close()

def plot_spectra_comparison(spectrum1: np.ndarray, spectrum2: np.ndarray, title1: str='Spectrum 1', title2: str='Spectrum 2'):
    """
    Plot two spectra side by side for comparison.
    """
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(np.log10(np.abs(spectrum1)+1e-8), cmap='inferno')
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(np.log10(np.abs(spectrum2)+1e-8), cmap='inferno')
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title1}_vs_{title2}.png"))
    plt.close()
