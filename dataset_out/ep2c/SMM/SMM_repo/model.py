## model.py
import torch
import torchvision.models as models
from typing import Optional

class Model:
    """
    Encapsulates a pre-trained backbone model (e.g., ResNet or ViT) with frozen parameters.
    Supports loading the specified architecture with pretrained weights and provides
    an inference interface.
    """

    def __init__(self, model_name: str = "ResNet50", pretrained: bool = True):
        """
        Loads the specified pre-trained model architecture, freezes its parameters,
        and prepares it for inference.

        Args:
            model_name (str): Name of the model architecture. Supported values:
                              "ResNet50", "ResNet18", "ViT-B32".
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        self.model_name = model_name
        self.pretrained = pretrained

        # Load the model based on architecture name
        if self.model_name == "ResNet50":
            self.model = models.resnet50(pretrained=self.pretrained)
        elif self.model_name == "ResNet18":
            self.model = models.resnet18(pretrained=self.pretrained)
        elif self.model_name == "ViT-B32":
            # Assuming torchvision supports ViT-B32 (if not, replace with appropriate loading)
            # As of torchvision 0.11.1, ViT is not supported, so add custom loading or handle accordingly.
            # For this code, we attempt to load from torchvision if available
            try:
                self.model = models.vit_b_32(pretrained=self.pretrained)
            except AttributeError:
                raise NotImplementedError(
                    "ViT-B32 not available in torchvision models for this version. "
                    "Please implement a custom loader or update torchvision."
                )
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        # Freeze parameters to prevent updates during training
        for param in self.model.parameters():
            param.requires_grad = False

        # Set model to evaluation mode
        self.model.eval()

        # Store device info
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the frozen pre-trained model.

        Args:
            x (torch.Tensor): Batch of input images, shape [B, C, H, W].

        Returns:
            torch.Tensor: Model output logits, shape [B, num_classes].
        """
        with torch.no_grad():
            logits = self.model(x)
        return logits
