# bn_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialBN(nn.Module):
    """
    Spatial Batch Normalization layer adapted for rate-based backpropagation.
    This BN normalizes across spatial dimensions per time step, with learnable parameters.
    
    Usage:
        - For single-step mode (rate_S), normalizing over spatial batch for each timestep.
        - During training, computes batch-wise mean and variance; during inference, uses stored parameters.
    """
    def __init__(self, num_features, epsilon=1e-5, affine=True, momentum=0.1):
        """
        Initializes the SpatialBN layer.
        Args:
            num_features (int): Number of input channels.
            epsilon (float): Small constant for numerical stability.
            affine (bool): If True, includes learnable affine parameters.
            momentum (float): Momentum for running stats (not used here, replaced with standard batch norm dynamics).
        """
        super(SpatialBN, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        # Register buffers for running mean and var for inference
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.training_mode = True  # control mode externally

    def forward(self, I_t):
        """
        Forward pass for spatial BN.
        Args:
            I_t (Tensor): Input tensor of shape [B, C, H, W]
        Returns:
            Tensor: Normalized tensor, shape same as input.
        """
        if self.training_mode:
            # Compute per batch mean and var over spatial dimensions
            mu = I_t.mean(dim=[0, 2, 3], keepdim=True)
            var = I_t.var(dim=[0, 2, 3], unbiased=False, keepdim=True)
            # Update running estimates (simulate online updates)
            self.running_mean = (1 - 0.1) * self.running_mean + 0.1 * mu.squeeze()
            self.running_var = (1 - 0.1) * self.running_var + 0.1 * var.squeeze()
        else:
            mu = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)

        # Normalize
        I_norm = (I_t - mu) / torch.sqrt(var + self.epsilon)
        if self.affine:
            # Apply learnable affine transformation
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
            I_norm = gamma * I_norm + beta
        return I_norm

    def set_training(self, mode=True):
        """
        Set training mode for BN.
        Args:
            mode (bool): True for training, False for inference.
        """
        self.training_mode = mode

    def backward(self, grad_output):
        """
        Backward pass for BN.
        Args:
            grad_output (Tensor): Gradient of loss w.r.t. output, shape same as input.
        Returns:
            grad_input (Tensor): Gradient of loss w.r.t. input.
        """
        # Manual backward to match the update rules and for clarity.
        # For the implementation, better to rely on autograd, but here we do explicit for transparency.
        # We provide gradient w.r.t. I_t, gamma, beta for parameter update.
        # Note: For simplicity, during training, batch stats are used, so autograd handles gradients.
        pass  # Implementation of backward is optional; rely on autograd for simplicity.

class TemporalBN(nn.Module):
    """
    Temporal Batch Normalization layer adapted for multi-step (rate_M) mode.
    This BN normalizes over entire sequence (time dimension + batch) for each feature.
    
    Usage:
        - During training, computes global mean and variance over all time steps and batch.
        - During inference, uses stored parameters for normalization.
    """
    def __init__(self, num_features, epsilon=1e-5, affine=True, momentum=0.1):
        """
        Initializes the TemporalBN layer.
        Args:
            num_features (int): Number of input channels.
            epsilon (float): Small constant for numerical stability.
            affine (bool): Enables learnable scale and shift.
            momentum (float): For running statistics.
        """
        super(TemporalBN, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        # Running global mean and var over entire sequences
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.training_mode = True

    def forward(self, I_seq):
        """
        Forward pass for temporal BN.
        Args:
            I_seq (Tensor): Input tensor of shape [B, T, C, H, W]
        Returns:
            Tensor: Normalized input, shape same as input.
        """
        if self.training_mode:
            # Compute mean and variance over batch and time (sequence)
            mu = I_seq.mean(dim=[0, 1, 3, 4], keepdim=True)
            var = I_seq.var(dim=[0, 1, 3, 4], unbiased=False, keepdim=True)
            # Update running estimates
            self.running_mean = (1 - 0.1) * self.running_mean + 0.1 * mu.squeeze()
            self.running_var = (1 - 0.1) * self.running_var + 0.1 * var.squeeze()
        else:
            mu = self.running_mean.view(1, 1, -1, 1, 1)
            var = self.running_var.view(1, 1, -1, 1, 1)
        # Normalize each sequence
        I_norm = (I_seq - mu) / torch.sqrt(var + self.epsilon)
        if self.affine:
            gamma = self.gamma.view(1, 1, -1, 1, 1)
            beta = self.beta.view(1, 1, -1, 1, 1)
            I_norm = gamma * I_norm + beta
        return I_norm

    def set_training(self, mode=True):
        """
        Set whether BN operates in training or inference mode.
        Args:
            mode (bool): True for training, False for inference.
        """
        self.training_mode = mode

    def backward(self, grad_output):
        """
        Backward pass for BN.
        Args:
            grad_output (Tensor): Gradient of loss w.r.t. output.
        Returns:
            grad_input (Tensor): Gradient of loss w.r.t. input.
        """
        # As with SpatialBN, rely on autograd for gradient calculations.
        pass  # Custom backward implementation optional. Rely on autograd.

# Note:
# - In practical training, we would set the models in training mode (set_training(True))
#   or eval mode (set_training(False)) depending on the phase.
# - For explicit gradient calculations (e.g., for stability bounds), additional code
#   can be added to extract and manipulate intermediate variables as needed.
# - For cross-mode switching, instantiate either SpatialBN or TemporalBN according to config.
# - The implementation simplifies the typical BN forward/backward, focusing on the core logic
#   suitable for rate code approximation and training. During actual backprop, rely on autograd.
```