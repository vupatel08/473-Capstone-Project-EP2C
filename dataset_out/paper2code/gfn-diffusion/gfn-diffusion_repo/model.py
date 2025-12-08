## model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# Load configuration parameters from the external config file if needed
# For this code snippet, parameters are passed explicitly during class initialization

class MLP(nn.Module):
    """
    Basic Multi-Layer Perceptron with configurable layers and activation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, activation: nn.Module = nn.ReLU):
        super().__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = activation()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {name: param for name, param in self.named_parameters()}


class NeuralSDE(nn.Module):
    """
    Neural network parameterization for drift u(x, t; θ) and diffusion g(x, t; θ).
    Accepts configuration for input dimension, hidden size, network type, and whether to learn g.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 400, network_type: str = 'MLP',
                 learn_diffusion: bool = False):
        """
        :param input_dim: Dimensionality of state x; default 2 for basic energy functions
        :param hidden_dim: Number of hidden units
        :param network_type: Architecture type, default 'MLP'
        :param learn_diffusion: If True, model diffusion g(x, t; θ) as a neural network
                                else, use fixed scalar diffusion
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.network_type = network_type
        self.learn_diffusion = learn_diffusion

        # Instantiate drift network u(x, t)
        # Input: x (batch, dim), t (scalar)
        # Optional: include t as additional feature
        self.u_network = self._build_network()
        # Instantiate diffusion g(x, t) if learn_diffusion else set as constant
        if self.learn_diffusion:
            # Neural network for diffusion coefficient, output scalar or vector
            self.g_network = self._build_network(output_dim=1)
        else:
            self.g_value = 1.0  # default fixed diffusion coefficient

    def _build_network(self, output_dim: int = 2) -> nn.Module:
        """
        Build MLP architecture based on configuration.
        :param output_dim: Output dimension
        """
        if self.network_type == 'MLP':
            return MLP(
                input_dim=self.input_dim + 1,  # append t as feature
                hidden_dim=self.hidden_dim,
                output_dim=output_dim,
                num_layers=2,
                activation=nn.ReLU
            )
        else:
            # Placeholder for other architectures if needed
            raise NotImplementedError(f"Network type {self.network_type} not implemented.")

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Forward pass for drift u(x, t; θ). Input x shape: [batch_size, input_dim]
        :param x: State tensor
        :param t: Time scalar
        :return: Tensor of shape [batch_size, input_dim], drift values
        """
        # Concatenate x and t (broadcasted) as input features
        t_tensor = torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)
        input_feat = torch.cat([x, t_tensor], dim=1)
        drift = self.u_network(input_feat)
        return drift

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve parameters from both drift and diffusion networks.
        """
        params = dict(self.u_network.named_parameters())
        if self.learn_diffusion:
            params.update(self.g_network.named_parameters())
        return params

    def diffusion_coeff(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute diffusion g(x, t; θ). If fixed, return scalar tensor.
        :param x: State tensor
        :param t: Time scalar
        """
        if self.learn_diffusion:
            g = self.g_network(torch.cat([x, torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)], dim=1))
            return g.squeeze(1)  # [batch_size]
        else:
            return torch.full((x.shape[0],), self.g_value, device=x.device, dtype=x.dtype)

    def initialize(self):
        """
        Initialize network weights.
        """
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class DiffusionCoefficient(nn.Module):
    """
    Encapsulates fixed or learned diffusion coefficient g(x, t)
    """
    def __init__(self, fixed: bool = True, value: float = 1.0):
        """
        :param fixed: if True, g is a fixed scalar; else modeled as small neural network
        :param value: scalar diffusion value if fixed
        """
        super().__init__()
        self.fixed = fixed
        if self.fixed:
            self.g = torch.tensor([value], dtype=torch.float32)
        else:
            # Model g as a neural network, e.g., a single-layer network
            self.g_network = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            self._initialize_weights()

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Return the diffusion coefficient g(x, t)
        :param x: state tensor
        :param t: current time (can be used for conditioning)
        """
        if self.fixed:
            return torch.full((x.shape[0],), self.g.item(), dtype=torch.float32, device=x.device)
        else:
            g = self.g_network(torch.cat([x, torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)], dim=1))
            return g.squeeze(1)

    def _initialize_weights(self):
        for m in self.g_network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
