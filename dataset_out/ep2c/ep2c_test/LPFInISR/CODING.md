# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
from typing import Tuple, Optional
import torch
from PIL import Image
import torchvision.transforms as transforms

def create_impulse_image(size: Tuple[int, int], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate a 2D impulse (delta) image tensor with a single central pixel set to 1, others zero.
    Shape: (H, W)
    
    Args:
        size (Tuple[int, int]): (height, width) of the image.
        device (torch.device, optional): Device to place tensor on. Defaults to CUDA if available, else CPU.
        
    Returns:
        torch.Tensor: 2D tensor with impulse at center.
    """
    height, width = size
    impulse_img = torch.zeros((height, width), dtype=torch.float32)
    center_x = height // 2
    center_y = width // 2
    impulse_img[center_x, center_y] = 1.0
    if device is not None:
        impulse_img = impulse_img.to(device)
    return impulse_img

def load_image(path: str, normalize: bool = True, to_grayscale: bool = False) -> torch.Tensor:
    """
    Load an image from disk and convert to tensor [C, H, W], optionally normalize.
    Supports grayscale or RGB.
    
    Args:
        path (str): Path to the image file.
        normalize (bool): If True, normalize pixel values to [0,1].
        to_grayscale (bool): If True, convert image to grayscale.
        
    Returns:
        torch.Tensor: image tensor [C, H, W]
    """
    # Open image using PIL
    img = Image.open(path)
    if to_grayscale:
        img = img.convert('L')
        transform = transforms.ToTensor()  # shape: [1, H, W]
    else:
        img = img.convert('RGB')
        transform = transforms.ToTensor()  # shape: [3, H, W]
    img_tensor = transform(img)  # [C, H, W]
    if normalize:
        # To [0,1], which torchvision does by default on ToTensor()
        pass
    return img_tensor

class DatasetLoader:
    """
    Basic dataset loader utility.
    Can load images from specified directory paths, possibly apply cropping/resizing.
    For current experiments focusing on synthetic impulse images, utility is minimal.
    """

    def __init__(self,
                 dataset_dir: str,
                 image_size: Tuple[int, int],
                 crop: bool = False,
                 crop_size: Tuple[int, int] = (128, 128),
                 to_grayscale: bool = False):
        """
        Initialize DatasetLoader with dataset directory and parameters.
        
        Args:
            dataset_dir (str): Path to directory containing images.
            image_size (Tuple[int, int]): Size for resizing images.
            crop (bool): Whether to crop images for dataset sampling.
            crop_size (Tuple[int, int]): Crop size if cropping enabled.
            to_grayscale (bool): Whether to convert images to grayscale.
        """
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.crop = crop
        self.crop_size = crop_size
        self.to_grayscale = to_grayscale
        # Gather image file paths
        self.image_paths = self._collect_image_paths()

    def _collect_image_paths(self):
        """
        Collects image file paths from dataset directory.
        Supports common image formats.
        """
        valid_exts = ['.png', '.jpg', '.jpeg', '.bmp']
        paths = []
        for fname in os.listdir(self.dataset_dir):
            if any(fname.lower().endswith(ext) for ext in valid_exts):
                full_path = os.path.join(self.dataset_dir, fname)
                paths.append(full_path)
        return paths

    def load_image(self, path: str) -> torch.Tensor:
        """
        Load a single image, resize/crop as needed.
        """
        img = Image.open(path)
        if self.to_grayscale:
            img = img.convert('L')
            transform = transforms.ToTensor()
        else:
            img = img.convert('RGB')
            transform = transforms.ToTensor()

        img_tensor = transform(img)  # [C, H, W]
        # Resize to target size if necessary
        if (img_tensor.shape[1], img_tensor.shape[2]) != self.image_size:
            resize_transform = transforms.Resize(self.image_size)
            img = resize_transform(img)
            img_tensor = transforms.ToTensor()(img)
        # Crop if enabled
        if self.crop:
            crop_transform = transforms.CenterCrop(self.crop_size)
            img = crop_transform(img)
            img_tensor = transforms.ToTensor()(img)
        return img_tensor

    def get_dataset(self):
        """
        Return list of image tensors loaded from dataset directory.
        """
        dataset = []
        for path in self.image_paths:
            img_tensor = self.load_image(path)
            dataset.append(img_tensor)
        return dataset

    def get_random_sample(self):
        """
        Return a random image tensor from dataset.
        """
        import random
        idx = random.randint(0, len(self.image_paths) - 1)
        return self.load_image(self.image_paths[idx])
```

## impulse_generation.py

```python
## impulse_generation.py
import torch

def create_impulse_image(size, device=None):
    """
    Generate a 2D impulse image tensor with a single pixel set to 1 at the center, others zero.

    Args:
        size (tuple or list): Pair of integers (height, width) specifying image dimensions.
        device (str or torch.device, optional): Device on which tensor should be allocated.
            Defaults to None, which results in tensor on CPU.

    Returns:
        torch.Tensor: 2D tensor of shape (height, width), dtype=torch.float32, with a single impulse pixel.
    """
    # Parse size
    height, width = size

    # Initialize tensor of zeros with specified size
    impulse_img = torch.zeros((height, width), dtype=torch.float32)

    # Calculate center indices
    center_x = height // 2
    center_y = width // 2

    # Set impulse pixel to 1
    impulse_img[center_x, center_y] = 1.0

    # Move to specified device if provided
    if device is not None:
        impulse_img = impulse_img.to(device)

    return impulse_img
```

## main.py

```python
# main.py
import os
import yaml
import torch
import json

from dataset_loader import create_impulse_image
from model_loader import ModelLoader
from response_analysis import ResponseAnalyzer
from spectral_analysis import SpectralAnalysis
from visualization import Visualization
from utils import fft2d, ifft2d, normalize_tensor

def main():
    # 1. Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup device
    device_str = config.get('general', {}).get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # 3. Instantiate and load the model
    model_path = config.get('model', {}).get('pretrained_path', '')
    model_name = config.get('model', {}).get('name', '')
    model_loader = ModelLoader(model_path, model_name, device=device_str)
    model = model_loader.load_model()

    # 4. Instantiate utilities
    response_analyzer = ResponseAnalyzer()
    spect_analyzer = SpectralAnalysis()
    visualizer = Visualization(config.get('evaluation', {}))

    # 5. Generate impulse input image
    impulse_size = tuple(config.get('impulse_image', {}).get('size', [128, 128]))
    # Create impulse image tensor
    impulse_img = create_impulse_image(impulse_size, device=device)

    # 6. Compute the impulse response from the model
    impulse_response = response_analyzer.compute_impulse_response(model, impulse_img)
    # Visualize impulse response spatial pattern
    visualizer.plot_impulse_response(impulse_response, save_path='impulse_response')

    # 7. Calculate the linear response H(I) by convolving input with impulse response
    # Although input is impulse, H(I) is just the impulse response shifted/scaled
    # For analysis, convolve the impulse input with the impulse response
    # The response to the impulse is the impulse response itself; just visualize
    # For generality, we can also compute response to zero image plus impulse
    linear_response = response_analyzer.extract_linear_response(impulse_img, impulse_response)

    # 8. Obtain full network output N(I) for the impulse input
    model.eval()
    with torch.no_grad():
        n_output = model(impulse_img.unsqueeze(0)).squeeze(0)  # [C, H, W]
    # Compute G(I) = N(I) - H(I)
    # Make sure shapes are aligned:
    # H(I), N(I): [C, H, W], response_analyzer.extract_linear_response returns same shape
    G_I = n_output - linear_response

    # 9. Visualize the responses
    visualizer.plot_responses(impulse_img, linear_response, G_I, save_path='response_comparison')

    # 10. Spectral analysis of responses
    spectrum_N = spect_analyzer.fft_response(n_output)
    spectrum_H = spect_analyzer.fft_response(linear_response)
    spectrum_G = spect_analyzer.fft_response(G_I)

    # Visualize spectra for N, H, G
    visualizer.visualize_spectra(spectrum_N, spectrum_H,
                                 spectrum_G,
                                 save_path_prefix='spectra_response')

    # 11. Calculate and save FSDS metric comparing N(I) and H(I)
    fsds_value = spect_analyzer.calculate_fsds(spectrum_N, spectrum_H)
    print(f"FSDS (N vs H): {fsds_value:.2f} dB")
    
    # Save metrics to JSON
    metrics_output = {
        'model_name': model_name,
        'fsds': fsds_value,
        'resp_shapes': {
            'N': list(n_output.shape),
            'H': list(linear_response.shape),
            'G': list(G_I.shape)
        }
    }
    save_dir = config.get('evaluation', {}).get('metrics_save_path', './metrics')
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    with open(save_dir, 'w') as f:
        json.dump(metrics_output, f, indent=4)

if __name__ == '__main__':
    main()
```

## model_loader.py

```python
# model_loader.py
import os
import torch

# Import model architectures; assuming these classes are available in the codebase
# If not, placeholders or dummy classes should be replaced with actual implementations.
try:
    from models.swinir import SwinIR
except ImportError:
    # Placeholder class if actual class is unavailable.
    class SwinIR(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            # Dummy initialization
        def forward(self, x):
            return x

try:
    from models.rdn import RDN
except ImportError:
    # Placeholder class if actual class is unavailable.
    class RDN(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
        def forward(self, x):
            return x

class ModelLoader:
    """
    A class to load pretrained super-resolution models based on configuration.
    """

    def __init__(self, model_path: str, model_name: str, device: str = 'cuda'):
        """
        Initializes the ModelLoader with checkpoint path, model architecture name, and device.

        Args:
            model_path (str): Path to the pretrained model weights (.pth or .pt).
            model_name (str): Name of the model architecture ('SwinIR', 'RDN', etc.).
            device (str): Device to load the model onto ('cuda' or 'cpu').

        Raises:
            ValueError: If unsupported model_name provided.
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.model = None

    def build_model(self):
        """
        Builds the model architecture based on the model_name.

        Returns:
            torch.nn.Module: Instantiated model architecture.

        Raises:
            ValueError: If model_name is unsupported.
        """
        architecture_map = {
            'SwinIR': SwinIR,
            'RDN': RDN,
            # Extend this dictionary with other supported architectures
        }

        if self.model_name not in architecture_map:
            raise ValueError(f"Unsupported model architecture: {self.model_name}")

        # Instantiate the selected model with necessary parameters.
        # These parameters should match those used during training.
        # For simplicity, they are omitted here; adapt as needed.
        if self.model_name == 'SwinIR':
            # Example parameters; replace with actual ones as needed
            self.model = architecture_map[self.model_name](
                upscale=2,
                in_chans=3,
                img_size=128,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6],
                model_name='SwinIR',
                upsampler='pixelshuffle'
            )
        elif self.model_name == 'RDN':
            # Example parameters; replace as needed
            self.model = architecture_map[self.model_name](
                in_channels=3,
                out_channels=3,
                num_features=64,
                growth_rate=32,
                num_blocks=16
            )
        else:
            # Fallback: instantiate with empty init or raise error
            # Or extend the logic with other architectures
            self.model = architecture_map[self.model_name]()
        return self.model

    def load_model(self):
        """
        Loads the pretrained weights into the model.

        Returns:
            torch.nn.Module: Model with loaded weights in eval mode.

        Raises:
            FileNotFoundError: If the model checkpoint file does not exist.
            RuntimeError: If loading state_dict fails.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")

        # Build model architecture
        model = self.build_model()

        # Load checkpoint; support architectures with or without 'state_dict' key
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load parameters into model
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            raise RuntimeError(f"Error loading state_dict: {e}")

        # Move model to device
        model.to(self.device)

        # Set to evaluation mode
        model.eval()

        # Store in instance variable
        self.model = model

        return self.model
```

## response_analysis.py

```python
## response_analysis.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import fft2d, ifft2d, normalize_tensor

class ResponseAnalyzer:
    """
    A class to perform spectral and spatial analysis of neural network super-resolution models,
    including impulse response extraction, convolution with impulse response, spectral spectra visualization,
    and FSDS metric calculation.
    """
    def __init__(self, config=None):
        """
        Initialize the ResponseAnalyzer.

        Args:
            config (dict, optional): Configuration dictionary. If None, defaults are used.
                Expected keys:
                 - device (str): 'cuda' or 'cpu' (default: 'cuda' if available)
                 - spectrum_normalization (bool): whether to normalize spectra before FFT (default: True)
        """
        if config is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.spectrum_normalization = True
        else:
            self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            self.spectrum_normalization = config.get('spectrum_normalization', True)

    def compute_impulse_response(self, model, impulse_img):
        """
        Feed an impulse image through the model to obtain the impulse response.
        Assumes the model is in eval mode.

        Args:
            model (torch.nn.Module): Pretrained super-resolution network.
            impulse_img (torch.Tensor): 3D tensor [C, H, W], device-compatible tensor representing impulse image.

        Returns:
            torch.Tensor: Response image [C, H, W], model's response to the impulse.
        """
        model.eval()
        with torch.no_grad():
            # Ensure input is batch dimension
            batch_input = impulse_img.unsqueeze(0)  # [1, C, H, W]
            output = model(batch_input)  # [1, C, H, W]
            response = output.squeeze(0)  # [C, H, W]
        return response

    def extract_linear_response(self, input_img, impulse_response):
        """
        Compute the linear response of the input image by convolving with the impulse response
        via FFT in the frequency domain.

        Args:
            input_img (torch.Tensor): [C, H, W], input tensor
            impulse_response (torch.Tensor): [C, H, W], impulse response (assumed same shape as input_or_response)

        Returns:
            torch.Tensor: The linear response image [C, H, W]
        """
        return self.convolve_with_impulse_response(input_img, impulse_response)

    def convolve_with_impulse_response(self, input_img, impulse_response):
        """
        Convolve input image with impulse response using FFT for efficiency.

        Args:
            input_img (torch.Tensor): [C, H, W]
            impulse_response (torch.Tensor): [C, H, W], same shape as input_img

        Returns:
            torch.Tensor: convolved image [C, H, W]
        """
        # Check device consistency
        input_img = input_img.to(self.device)
        impulse_response = impulse_response.to(self.device)

        # FFT of both images
        fft_input = fft2d(input_img)
        fft_response = fft2d(impulse_response)

        # Element-wise multiplication in frequency domain
        fft_convolved = fft_input * fft_response

        # Inverse FFT to spatial domain
        convolved = ifft2d(fft_convolved)

        # Return magnitude (real part) tensor
        return torch.real(convolved)

    def compute_response(self, model, input_img):
        """
        Compute the model's response to an input image.

        Args:
            model (torch.nn.Module): Super-resolution model.
            input_img (torch.Tensor): [C, H, W].

        Returns:
            torch.Tensor: Response image [C, H, W].
        """
        model.eval()
        with torch.no_grad():
            out = model(input_img.unsqueeze(0))
            response = out.squeeze(0)
        return response

    def visualize_impulse_response(self, impulse_response, save_path=None):
        """
        Plot the impulse response in spatial domain.

        Args:
            impulse_response (torch.Tensor): [C, H, W]
            save_path (str, optional): If specified, save the plot image to path.
        """
        # Convert to numpy for plotting
        resp_np = impulse_response.cpu().numpy()
        num_channels = resp_np.shape[0]

        for c in range(num_channels):
            plt.figure(figsize=(6,6))
            plt.imshow(resp_np[c], cmap='jet')
            plt.title(f'Impulse Response Channel {c}')
            plt.colorbar()
            if save_path:
                plt.savefig(f"{save_path}_channel_{c}.png")
            else:
                plt.show()
            plt.close()

    def plot_response_comparison(self, input_img, linear_response, nonlinear_response, save_path=None):
        """
        Plot side-by-side comparison of input, linear, and nonlinear responses.

        Args:
            input_img (torch.Tensor): [C, H, W]
            linear_response (torch.Tensor): [C, H, W]
            nonlinear_response (torch.Tensor): [C, H, W]
            save_path (str, optional): Path to save the figure.
        """
        input_np = input_img.cpu().numpy()
        lin_np = linear_response.cpu().numpy()
        nonlin_np = nonlinear_response.cpu().numpy()

        num_channels = input_np.shape[0]
        for c in range(num_channels):
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
                plt.savefig(f"{save_path}_channel_{c}.png")
            else:
                plt.show()
            plt.close()

    def spectral_response_analysis(self, response_tensor, response_name='Response'):
        """
        Compute and plot the spectrum (magnitude and phase) of a response tensor.

        Args:
            response_tensor (torch.Tensor): [C, H, W]
            response_name (str): Name used in titles and labels.

        Returns:
            dict: {'magnitude': mag_spectrum, 'phase': phase_spectrum} in numpy arrays.
        """
        # Convert to numpy
        resp_np = response_tensor.cpu().numpy()

        # Normalize if activation is non-standard
        if self.spectrum_normalization:
            resp_np = normalize_tensor(resp_np)

        # Compute FFT: over each channel
        spectra = {'magnitude': [], 'phase': []}
        for c in range(resp_np.shape[0]):
            fft_res = np.fft.fft2(resp_np[c])
            mag = np.abs(fft_res)
            phase = np.angle(fft_res)
            # Log magnitude for visualization
            mag_log = np.log1p(mag)
            spectra['magnitude'].append(mag_log)
            spectra['phase'].append(phase)

        # Plot magnitude spectrum
        plt.figure(figsize=(12,5))
        for c in range(len(spectra['magnitude'])):
            plt.subplot(1,2,1)
            plt.imshow(np.fft.fftshift(spectra['magnitude'][c]), cmap='inferno')
            plt.title(f'{response_name} Magnitude Spectrum (Channel {c})')
            plt.colorbar()
        # Plot phase spectrum
        for c in range(len(spectra['phase'])):
            plt.subplot(1,2,2)
            plt.imshow(np.fft.fftshift(spectra['phase'][c]), cmap='twilight')
            plt.title(f'{response_name} Phase Spectrum (Channel {c})')
            plt.colorbar()
        plt.show()
        return spectra

    def compute_fsds(self, spectrum_hr, spectrum_sr):
        """
        Compute the Frequency Spectrum Distribution Similarity (FSDS) between two spectra.

        Args:
            spectrum_hr (dict): {'magnitude': list of np.ndarray, 'phase': list of np.ndarray}
            spectrum_sr (dict): same as above

        Returns:
            float: FSDS scalar value
        """
        # Average over channels
        mag_hr = np.mean(np.array(spectrum_hr['magnitude']), axis=0)
        mag_sr = np.mean(np.array(spectrum_sr['magnitude']), axis=0)

        # Spectrum difference map
        d_diff = mag_hr - mag_sr  # real-valued difference

        numerator = np.sum(d_diff ** 2)
        denominator = np.sum(mag_hr ** 2) + 1e-8  # prevent div zero

        fsds_value = -10.0 * np.log10(numerator / denominator + 1e-8)
        return fsds_value

```

## spectral_analysis.py

```python
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
```

## visualization.py

```python
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
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\LPFInISR\LPFInISR_repo`
