## model.py
import torch
import torch.nn as nn
import torchvision.models as models
try:
    import timm
except ImportError:
    timm = None

class SSLModel(nn.Module):
    """
    Encapsulates the backbone encoder and projection head for SSL.
    Supports Common Backbones (ResNet) and Vision Transformers via configuration.
    Provides encode() and project() methods.
    """
    def __init__(self,
                 backbone_name: str = 'ResNet50',
                 projection_dim: int = 8192,
                 projection_layers: int = 2,
                 hidden_dim: int = 4096):
        """
        Args:
            backbone_name (str): 'ResNet50', 'ViT-tiny', 'ViT-small', etc.
            projection_dim (int): Output dimension of the projection head.
            projection_layers (int): Number of layers in projection head.
            hidden_dim (int): Hidden dimension size in projection head.
        """
        super().__init__()
        self.backbone_name = backbone_name

        # Initialize backbone
        if backbone_name.lower() == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            # Remove the final FC layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 2048  # ResNet50 final feature size
        elif backbone_name.lower() == 'vit-tiny':
            # Use timm if available
            if timm is None:
                raise ImportError("timm library is required for ViT models.")
            self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=False)
            self.feature_dim = self.backbone.embed_dim  # Typically 192
        elif backbone_name.lower() == 'vit-small':
            if timm is None:
                raise ImportError("timm library is required for ViT models.")
            self.backbone = timm.create_model('vit_small_patch16_224', pretrained=False)
            self.feature_dim = self.backbone.embed_dim  # Typically 384
        else:
            # Default: use ResNet50 if unspecified
            self.backbone = models.resnet50(pretrained=False)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 2048

        # Initialize projection head
        layers = []
        in_dim = self.feature_dim
        for _ in range(projection_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        # Final layer
        layers.append(nn.Linear(in_dim, projection_dim))
        self.projection_head = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone encoder.
        Args:
            x (torch.Tensor): Input images tensor (B, C, H, W)
        Returns:
            torch.Tensor: Backbone feature vectors (B, feature_dim)
        """
        if 'resnet' in self.backbone_name.lower():
            features = self.backbone(x)  # [B, 2048, 1, 1]
            features = features.view(x.size(0), -1)  # flatten to [B, 2048]
        elif 'vit' in self.backbone_name.lower():
            # Use ViT's forward features
            features = self.backbone.forward_features(x)  # [B, embed_dim]
        else:
            # fallback
            features = self.backbone(x)
        return features

    def project(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Pass backbone features through projection head.
        Args:
            emb (torch.Tensor): Features from encoder (B, feature_dim)
        Returns:
            torch.Tensor: Final projections for SSL (B, projection_dim)
        """
        return self.projection_head(emb)
