## conditioning.py
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import os

class ConditioningEncoder:
    def __init__(self, control_type: str = "edges", device: torch.device = None):
        """
        Initialize the ConditioningEncoder based on control_type.
        Loads appropriate pre-trained encoders or models for each condition type.
        
        Args:
            control_type (str): Type of control map ("edges", "depth", "pose").
            device (torch.device): Device to load models onto.
        """
        self.control_type = control_type.lower()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define target input size based on typical model expectations
        self.target_size = 512  # Can be adjusted as needed
        
        # Initialize the model based on control_type
        if self.control_type == "edges":
            self.model = self._load_edge_encoder()
        elif self.control_type == "depth":
            self.model = self._load_depth_encoder()
        elif self.control_type == "pose":
            self.model = self._load_pose_encoder()
        else:
            raise ValueError(f"Unsupported control_type: {self.control_type}")
        self.model.to(self.device).eval()

        # Define common preprocessing transforms
        self.transform = T.Compose([
            T.Resize((self.target_size, self.target_size)),
            T.ToTensor(),
            # Normalization defaults for ImageNet-compatible encoders; customize if needed
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_edge_encoder(self) -> nn.Module:
        """
        Loads or defines the encoder for edges.
        For simplicity, here we define a lightweight CNN or identity as placeholder.
        Replace with actual trained encoder if available.
        """
        # Example: simple CNN for edge features
        class EdgeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            def forward(self, x):
                return self.features(x)
        return EdgeEncoder()
    
    def _load_depth_encoder(self) -> nn.Module:
        """
        Loads or initializes a depth encoder.
        For simplicity, use a pretrained MiDaS small model.
        """
        import torchvision.models as models
        from torchvision.models.resnet import resnet18
        class DepthEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                # Use pre-trained ResNet18 as placeholder
                self.backbone = resnet18(pretrained=True)
                self.backbone.fc = nn.Identity()  # remove final classification layer
            def forward(self, x):
                return self.backbone(x)
        return DepthEncoder()

    def _load_pose_encoder(self) -> nn.Module:
        """
        Loads or initializes a pose encoder.
        For simplicity, define a placeholder that returns zeros.
        Replace with actual pose encoder such as HRNet or OpenPose.
        """
        class PoseEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                # Placeholder: identity function
            def forward(self, x):
                return torch.zeros_like(x).mean(dim=1, keepdim=True)
        return PoseEncoder()

    def encode(self, condition_map: np.ndarray) -> torch.Tensor:
        """
        Encode the input condition map into a tensor suitable for ControlNet.
        
        Args:
            condition_map (np.ndarray): Input image array (H x W x C) in RGB.
            
        Returns:
            torch.Tensor: Encoded feature tensor [1, C, H, W].
        """
        # Convert to PIL image
        image = Image.fromarray(condition_map)
        # Apply preprocessing transforms
        tensor = self.transform(image).unsqueeze(0).to(self.device)  # shape: [1,3,H,W]
        
        # Forward through the model
        with torch.no_grad():
            feature = self.model(tensor)
        return feature
