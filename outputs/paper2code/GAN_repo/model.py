"""model.py

This module defines the architectures for the Generator and Discriminator according to
the GAN framework described in the paper "Generative Adversarial Networks". It also
provides the GANModel container class which offers methods to build the Generator and
Discriminator based on configuration parameters from config.yaml.
"""

from typing import Dict, Any
import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator network for the GAN.

    This class implements a four-layer fully connected network that maps a noise vector
    (default 100 dimensions) to an output vector of size 784 corresponding to a flattened
    MNIST image. Each hidden layer applies a Linear transformation followed by optional
    Batch Normalization and a ReLU activation. The final layer applies a Linear
    transformation followed by a Sigmoid activation to squash outputs to [0, 1].

    Attributes:
        model (nn.Sequential): The sequential container that holds the Generator layers.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        noise_dim: int = 100,
        output_dim: int = 784
    ) -> None:
        """
        Initializes the Generator with configuration parameters.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the generator.
            noise_dim (int, optional): Dimensionality of the input noise vector. Defaults to 100.
            output_dim (int, optional): Dimensionality of the generated output (flattened MNIST image). Defaults to 784.
        """
        super(Generator, self).__init__()

        # Retrieve generator configuration with defaults as defined in config.yaml
        default_hidden_layers = [1000, 500, 250, 100]
        hidden_layers = config.get("layers", default_hidden_layers)
        activation_name: str = config.get("activation", "ReLU")
        output_activation_name: str = config.get("output_activation", "sigmoid")
        batch_norm: bool = config.get("batch_normalization", True)

        # Map configuration activation strings to PyTorch activation classes
        activation_cls = self._get_activation(activation_name)
        output_activation_cls = self._get_activation(output_activation_name)

        layers_list = []
        current_dim: int = noise_dim

        # Build each hidden layer block
        for hidden_dim in hidden_layers:
            layers_list.append(nn.Linear(current_dim, hidden_dim))
            if batch_norm:
                layers_list.append(nn.BatchNorm1d(hidden_dim))
            layers_list.append(activation_cls())
            current_dim = hidden_dim

        # Final output layer mapping to output_dim followed by sigmoid activation
        layers_list.append(nn.Linear(current_dim, output_dim))
        layers_list.append(output_activation_cls())

        self.model = nn.Sequential(*layers_list)

    def _get_activation(self, act_name: str) -> type:
        """
        Converts an activation function name to its corresponding PyTorch class.

        Args:
            act_name (str): Name of the activation function (e.g., "ReLU", "sigmoid").

        Returns:
            type: A PyTorch activation class.
        """
        act_name_lower = act_name.lower()
        if act_name_lower == "relu":
            return nn.ReLU
        elif act_name_lower == "sigmoid":
            return nn.Sigmoid
        elif act_name_lower == "tanh":
            return nn.Tanh
        else:
            raise ValueError(f"Unsupported activation function: {act_name}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the Generator.

        Args:
            z (torch.Tensor): Input noise vector of shape (batch_size, noise_dim).

        Returns:
            torch.Tensor: Generated data of shape (batch_size, output_dim).
        """
        return self.model(z)


class Discriminator(nn.Module):
    """Discriminator network for the GAN.

    This class implements a four-layer fully connected network that takes a flattened MNIST
    image (784 dimensions) and outputs a probability (using Sigmoid) indicating whether the
    image is real or generated. Each hidden layer applies a Linear transformation followed by
    optional Batch Normalization and a ReLU activation.

    Attributes:
        model (nn.Sequential): The sequential container that holds the Discriminator layers.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        input_dim: int = 784
    ) -> None:
        """
        Initializes the Discriminator with configuration parameters.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the discriminator.
            input_dim (int, optional): Dimensionality of the input (flattened MNIST image). Defaults to 784.
        """
        super(Discriminator, self).__init__()

        # Retrieve discriminator configuration with defaults as defined in config.yaml
        default_layers = [1000, 500, 250, 1]
        layers_config = config.get("layers", default_layers)
        activation_name: str = config.get("activation", "ReLU")
        output_activation_name: str = config.get("output_activation", "sigmoid")
        batch_norm: bool = config.get("batch_normalization", True)

        activation_cls = self._get_activation(activation_name)
        output_activation_cls = self._get_activation(output_activation_name)

        layers_list = []
        current_dim: int = input_dim

        # Build all layers except the final one
        for hidden_dim in layers_config[:-1]:
            layers_list.append(nn.Linear(current_dim, hidden_dim))
            if batch_norm:
                layers_list.append(nn.BatchNorm1d(hidden_dim))
            layers_list.append(activation_cls())
            current_dim = hidden_dim

        # Final layer mapping to a single unit, followed by sigmoid activation
        final_dim: int = layers_config[-1]
        layers_list.append(nn.Linear(current_dim, final_dim))
        layers_list.append(output_activation_cls())

        self.model = nn.Sequential(*layers_list)

    def _get_activation(self, act_name: str) -> type:
        """
        Converts an activation function name to its corresponding PyTorch class.

        Args:
            act_name (str): Name of the activation function (e.g., "ReLU", "sigmoid").

        Returns:
            type: A PyTorch activation class.
        """
        act_name_lower = act_name.lower()
        if act_name_lower == "relu":
            return nn.ReLU
        elif act_name_lower == "sigmoid":
            return nn.Sigmoid
        elif act_name_lower == "tanh":
            return nn.Tanh
        else:
            raise ValueError(f"Unsupported activation function: {act_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the Discriminator.

        Args:
            x (torch.Tensor): Input tensor representing a flattened MNIST image.

        Returns:
            torch.Tensor: Output probability of the input being real.
        """
        return self.model(x)


class GANModel:
    """Container class for the GAN that holds both the Generator and Discriminator.

    Provides methods to build each network using configuration parameters.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initializes the GANModel with the provided configuration parameters.

        Args:
            params (Dict[str, Any]): Configuration dictionary (typically loaded from config.yaml).
        """
        self.params: Dict[str, Any] = params if params is not None else {}

    def build_generator(self) -> Generator:
        """
        Instantiates and returns the Generator network using configuration parameters.

        Returns:
            Generator: An instance of the Generator network.
        """
        generator_config: Dict[str, Any] = self.params.get("model", {}).get("generator", {})
        # Default noise dimension is 100 and output dimension is 784 (28x28 flattened)
        return Generator(generator_config, noise_dim=100, output_dim=784)

    def build_discriminator(self) -> Discriminator:
        """
        Instantiates and returns the Discriminator network using configuration parameters.

        Returns:
            Discriminator: An instance of the Discriminator network.
        """
        discriminator_config: Dict[str, Any] = self.params.get("model", {}).get("discriminator", {})
        # Input dimension for Discriminator is 784 (flattened MNIST image)
        return Discriminator(discriminator_config, input_dim=784)
