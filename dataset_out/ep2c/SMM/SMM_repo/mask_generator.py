## mask_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MaskGenerator(nn.Module):
    """
    Lightweight CNN-based mask generator for sample-specific multi-channel masks.
    Generates a mask of reduced resolution from input images and performs patch-wise
    upsampling via pixel repetition to match input image size.
    """
    def __init__(self,
                 input_size: Tuple[int, int],
                 architecture_depth: int = 5,
                 kernel_size: int = 3,
                 filters: int = 64,
                 pooling_layers: int = 2,
                 output_ratio: float = 1/8):
        """
        Initialize the MaskGenerator.
        Args:
            input_size (Tuple[int, int]): Size of input images (H, W).
            architecture_depth (int): Number of convolutional layers.
            kernel_size (int): Kernel size for Conv layers.
            filters (int): Number of filters/channels in each conv layer.
            pooling_layers (int): Number of MaxPool layers.
            output_ratio (float): Ratio of output mask size to input size.
        """
        super(MaskGenerator, self).__init__()
        self.input_size = input_size  # (H, W)
        self.architecture_depth = architecture_depth
        self.kernel_size = kernel_size
        self.filters = filters
        self.pooling_layers = pooling_layers
        self.output_ratio = output_ratio

        # Compute the size of the intermediate feature map after pooling
        # Calculate number of pooling steps to determine output size
        H, W = self.input_size
        for _ in range(self.pooling_layers):
            H = H // 2
            W = W // 2
        self.reduced_size = (H, W)  # Size after pooling

        # Build convolutional layers
        layers = []

        in_channels = 3  # Input image has 3 channels
        for layer_idx in range(self.architecture_depth):
            conv_layer = nn.Conv2d(in_channels, self.filters, kernel_size=self.kernel_size, padding=1)
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
            nn.init.zeros_(conv_layer.bias)
            layers.append(conv_layer)
            # Optional: batch normalization for training stability
            layers.append(nn.BatchNorm2d(self.filters))
            layers.append(nn.ReLU(inplace=True))
            # Add pooling layers as specified
            if layer_idx < self.pooling_layers:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = self.filters

        # Final convolution to produce 3-channel mask
        final_conv = nn.Conv2d(self.filters, 3, kernel_size=3, padding=1)
        nn.init.kaiming_uniform_(final_conv.weight, nonlinearity='relu')
        nn.init.zeros_(final_conv.bias)
        layers.append(final_conv)

        self.net = nn.Sequential(*layers)

        # Determine output size of CNN: (batch_size, 3, H', W')
        self.output_size = self.reduced_size
        # The patch size for upsampling (integer)
        H_in, W_in = self.input_size
        H_out, W_out = self.output_size

        # Compute patch size (assumes integral division)
        self.patch_size_h = max(1, H_in // H_out)
        self.patch_size_w = max(1, W_in // W_out)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generate the sample-specific mask for input images.
        Args:
            image (torch.Tensor): Batch of images, shape (B, 3, H, W).

        Returns:
            torch.Tensor: Masks of shape (B, 3, H, W), upsampled via patch-wise repetition.
        """
        # Pass through CNN to get low-res masks
        mask_low_res = self.net(image)  # shape (B, 3, H', W')
        # Patch-wise upsampling via pixel repetition
        masks_upsampled = self._patch_upsample(mask_low_res)
        return masks_upsampled

    def _patch_upsample(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Upsample the mask from reduced resolution to original size
        by repeating each pixel in spatial dimensions.
        Args:
            mask (torch.Tensor): shape (B, 3, H', W').

        Returns:
            torch.Tensor: shape (B, 3, H, W), the upsampled mask.
        """
        B, C, H', W' = mask.shape
        # Repeat each pixel patch size times both vertically and horizontally
        upsampled = mask.repeat_interleave(self.patch_size_h, dim=2)
        upsampled = upsampled.repeat_interleave(self.patch_size_w, dim=3)
        # The resulting size may be larger than input due to integer division
        # Trim to match input size
        H_in, W_in = self.input_size
        return upsampled[:, :, :H_in, :W_in]

