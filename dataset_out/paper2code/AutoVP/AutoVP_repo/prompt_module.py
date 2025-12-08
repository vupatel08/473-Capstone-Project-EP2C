## prompt_module.py
import torch
import torch.nn as nn
import torch.fft as fft
import matplotlib.pyplot as plt
import numpy as np

class PromptGenerator:
    def __init__(self, prompt_size: int = 16, prompt_type: str = 'pixel',
                 prompt_init_type: str = 'zeros', input_channels: int = 3,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initialize the prompt generator.
        Args:
            prompt_size (int): size of the prompt (pixels).
            prompt_type (str): 'pixel' or 'frequency'.
            prompt_init_type (str): 'zeros', 'random', 'learned'
            input_channels (int): e.g., 3 for RGB.
            device (torch.device): device to store parameters.
        """
        self.prompt_size = prompt_size
        self.prompt_type = prompt_type.lower()
        self.input_channels = input_channels
        self.device = device

        if self.prompt_type == 'pixel':
            # Pixel prompts: trainable tensor of shape (C, p, p)
            if prompt_init_type == 'zeros':
                init_tensor = torch.zeros((self.input_channels, self.prompt_size, self.prompt_size), device=self.device)
            elif prompt_init_type == 'random':
                init_tensor = 0.01 * torch.randn((self.input_channels, self.prompt_size, self.prompt_size), device=self.device)
            else:
                init_tensor = torch.zeros((self.input_channels, self.prompt_size, self.prompt_size), device=self.device)
            self.prompt_tensor = nn.Parameter(init_tensor)
        elif self.prompt_type == 'frequency':
            # Frequency prompts: trainable FFT coefficients (real)
            # Store real and imaginary parts separately for interpretability
            real_part = torch.zeros((self.input_channels, self.prompt_size, self.prompt_size), device=self.device)
            imag_part = torch.zeros_like(real_part)
            if prompt_init_type == 'random':
                real_part = 0.01 * torch.randn_like(real_part)
                imag_part = 0.01 * torch.randn_like(imag_part)
            elif prompt_init_type == 'zeros':
                real_part = torch.zeros_like(real_part)
                imag_part = torch.zeros_like(imag_part)
            # Parameters: real and imaginary parts
            self.real_coeffs = nn.Parameter(real_part)
            self.imag_coeffs = nn.Parameter(imag_part)
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")
        
    def get_prompt(self) -> torch.Tensor:
        """
        Return the current prompt tensor as a spatial image.
        For pixel prompts, return directly.
        For frequency prompts, perform inverse FFT.
        Output shape: (C, p, p)
        """
        if self.prompt_type == 'pixel':
            return self.prompt_tensor
        elif self.prompt_type == 'frequency':
            # Reconstruct complex FFT tensor
            complex_fft = torch.complex(self.real_coeffs, self.imag_coeffs)
            # Inverse FFT to spatial domain
            # Note: torch.fft.ifft2 output is complex, take real part
            spatial_prompt = fft.ifft2(complex_fft, norm='forward')
            spatial_prompt = spatial_prompt.real
            # Clamp or normalize for visualization if needed
            spatial_prompt = spatial_prompt.clamp(0, 1)
            return spatial_prompt
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")

    def update(self, grads: torch.Tensor, step_size: float=1.0):
        """
        Update the prompt parameters based on provided gradients.
        Args:
            grads (Tensor): gradient tensor of same shape as prompt tensor.
            step_size (float): learning rate for update.
        """
        if self.prompt_type == 'pixel':
            # manual update
            if self.prompt_tensor.grad is None:
                raise RuntimeError("Gradients must be computed before calling update.")
            # Use stored gradient
            self.prompt_tensor.data -= step_size * self.prompt_tensor.grad.data
            self.prompt_tensor.grad.zero_()
        elif self.prompt_type == 'frequency':
            # grads should be same shape as real_coeffs and imag_coeffs
            if hasattr(grads, 'real') and hasattr(grads, 'imag'):
                # grads as complex tensor
                self.real_coeffs.data -= step_size * grads.real
                self.imag_coeffs.data -= step_size * grads.imag
            else:
                # assume grads contains real and imaginary parts
                self.real_coeffs.data -= step_size * grads['real']
                self.imag_coeffs.data -= step_size * grads['imag']
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")

    def visualize(self):
        """
        Visualize the current prompt as an image.
        Returns:
            image (numpy array): shape (H, W, C), for plotting.
        """
        prompt_img = self.get_prompt()  # shape (C, p, p)
        # Convert to [0,255] image array for visualization
        np_img = prompt_img.detach().cpu().permute(1, 2, 0).numpy()
        # Normalize for visualization
        np_img = np.clip(np_img, 0, 1)
        return np_img

    def save_visualization(self, path: str):
        """
        Save the visualization as an image file.
        """
        import matplotlib.pyplot as plt
        np_img = self.visualize()
        plt.imshow(np_img)
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()
