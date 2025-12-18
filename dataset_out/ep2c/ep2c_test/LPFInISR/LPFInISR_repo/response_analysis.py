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

