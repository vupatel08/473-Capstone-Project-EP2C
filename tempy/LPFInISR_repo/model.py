# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import os

# We will implement a factory function to instantiate models based on architecture name.
# For demonstration, only EDSR is fully implemented with some typical residual blocks.
# Additional architectures like RCAN, SwinIR can be added similarly.

# Basic residual block used in EDSR
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Initialize weights
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out')

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + residual

# EDSR architecture
class EDSR(nn.Module):
    def __init__(self, num_channels: int = 3, num_features: int = 64,
                 num_blocks: int = 16, upscale: int = 2):
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.upscale = upscale

        # Head
        self.conv_input = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        # Tail: Upsampling
        if upscale == 2:
            self.up_sampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
            )
        elif upscale == 4:
            self.up_sampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
            )
        else:
            # For other upscale factors if needed
            self.up_sampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * (upscale ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale),
                nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
            )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_input(x)
        res = self.res_blocks(feat)
        out = feat + res
        out = self.up_sampler(out)
        return out

    def get_impulse_response(self, device: torch.device = torch.device('cpu')) -> np.ndarray:
        """
        Generate the linear response of the network by feeding an impulse image.

        Returns:
            np.ndarray: Impulse response as a 2D array.
        """
        import numpy as np
        # Create a small impulse image: e.g., 11x11 with center pixel = 1, others = 0
        size = 11
        impulse_np = np.zeros((size, size, 3), dtype=np.float32)
        center = size // 2
        impulse_np[center, center, :] = 1.0  # Max pixel intensity in scaled [0,1]
        # Convert to tensor with shape (1, 3, H, W)
        impulse_tensor = torch.from_numpy(impulse_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # Pass through network
        with torch.no_grad():
            output = self.forward(impulse_tensor)
        # Return as numpy array, shape (H, W, C)
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

        return output_np

# Future implementation of other architectures (RCAN, SwinIR) can be added here
# with similar interface and get_impulse_response method.

# Model factory function
def get_model(architecture: str = 'EDSR', config: Optional[Dict] = None) -> nn.Module:
    """
    Instantiate and optionally load pretrained weights based on configuration.

    Args:
        architecture (str): Model architecture name ('EDSR', 'RCAN', 'SwinIR').
        config (dict, optional): Configuration dictionary with model parameters,
                                 including 'pretrained' and 'checkpoint_path'.

    Returns:
        torch.nn.Module: Initialized model.
    """
    if config is None:
        config = {}

    # Support for multiple architectures
    if architecture.upper() == 'EDSR':
        model = EDSR(
            num_channels=3,
            num_features=config.get('num_features', 64),
            num_blocks=config.get('num_blocks', 16),
            upscale=config.get('upscale', 2)
        )
    elif architecture.upper() == 'RCAN':
        # Placeholder: Implement RCAN similarly
        raise NotImplementedError("RCAN architecture not yet implemented.")
    elif architecture.upper() == 'SWINIR':
        # Placeholder: Implement SwinIR
        raise NotImplementedError("SwinIR architecture not yet implemented.")
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Load weights if specified
    pretrained = config.get('pretrained', False)
    checkpoint_path = config.get('checkpoint_path', "")
    if pretrained and checkpoint_path:
        if os.path.isfile(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(state_dict)
                print(f"Loaded pretrained weights from {checkpoint_path}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Training from scratch.")
    return model
