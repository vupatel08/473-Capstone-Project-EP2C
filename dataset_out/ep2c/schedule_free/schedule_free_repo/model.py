## model.py
import torch
import torch.nn as nn
import torchvision.models as models
import math

class Model(nn.Module):
    """
    This class encapsulates various neural network architectures used in experiments,
    including ResNet50, WideResNet, DenseNet, and others.
    It provides a unified interface for initialization, forward pass, and weight setup.
    """
    def __init__(self, model_class: str = "ResNet50", hyperparams: dict = None):
        """
        Initialize the model based on model_class and hyperparameters.
        Args:
            model_class (str): Type of architecture ('ResNet50', 'WideResNet', 'DenseNet', etc.)
            hyperparams (dict): Architecture-specific hyperparameters, e.g.,
                For WideResNet: {'depth': 16, 'width_multiplier': 8, 'dropout': 0.3}
                For DenseNet: {'growth_rate': 12, 'block_config': [6,12,24,16]}
        """
        super().__init__()
        self.model_class = model_class
        self.hyperparams = hyperparams if hyperparams is not None else {}

        # Select and instantiate architecture
        self.model = self.select_architecture()

        # Initialize weights explicitly to ensure reproducibility
        self.initialize_weights()

    def select_architecture(self) -> nn.Module:
        """
        Instantiate the neural network architecture based on self.model_class.
        Returns:
            nn.Module: the constructed model
        """
        if self.model_class == "ResNet50":
            # Standard ResNet50 from torchvision
            model = models.resnet50(pretrained=False)
        elif self.model_class == "WideResNet":
            # Build WideResNet with specified depth and width
            depth = self.hyperparams.get("depth", 16)
            widen_factor = self.hyperparams.get("width_multiplier",8)
            dropout = self.hyperparams.get("dropout", 0.3)
            model = self.build_wideresnet(depth, widen_factor, dropout)
        elif self.model_class == "DenseNet":
            # Use torchvision DenseNet121 as default; customize if needed
            growth_rate = self.hyperparams.get("growth_rate", 12)
            block_config = self.hyperparams.get("block_config", [6,12,24,16])
            model = models.densenet121(pretrained=False)
            # For customization, you can replace classifier or features accordingly
            # But for simplicity, we'll use the default as placeholder
        elif self.model_class == "ResNet":
            # General ResNet, e.g., for other depths if needed
            # default to ResNet50 for now
            model = models.resnet50(pretrained=False)
        else:
            raise ValueError(f"Unrecognized model class: {self.model_class}")
        return model

    def build_wideresnet(self, depth: int, widen_factor: int, dropout: float) -> nn.Module:
        """
        Build a WideResNet architecture.
        Args:
            depth (int): depth of the network, typically 16, 28, etc.
            widen_factor (int): width multiplier
            dropout (float): dropout rate
        Returns:
            nn.Module: WideResNet model
        """
        # Implementation of WideResNet based on typical structure
        # For simplicity, using a custom implementation
        # Note: in practice, replace with a well-tested implementation
        from models.wideresnet import WideResNet
        return WideResNet(depth=depth, widen_factor=widen_factor, dropout=dropout)

    def initialize_weights(self) -> None:
        """
        Initialize the network weights with He initialization for conv and linear layers.
        """
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output logits
        """
        return self.model(x)

    def get_model(self) -> nn.Module:
        """
        Return underlying model for compatibility with optimizer and evaluation.
        Returns:
            nn.Module: the network model
        """
        return self.model
