## model.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50
import torchvision.transforms as T
import torchvision.models as torchvision_models
from config import MODEL

class Model(nn.Module):
    def __init__(self, backbone_name: str = 'resnet50', feature_dim: int = 512, normalize_features: bool = True):
        """
        Initializes the backbone network and the embedding head.

        Args:
            backbone_name (str): Type of backbone. Options: 'resnet50', 'vit_b/16', 'mobilenet_v2'
            feature_dim (int): Dimensionality of the output embedding.
            normalize_features (bool): If True, apply L2 normalization to features.
        """
        super(Model, self).__init__()
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        self.normalize_features = normalize_features

        # Instantiate backbone based on configuration
        if backbone_name == 'resnet50':
            backbone = resnet50(pretrained=True)
            # Remove final fully connected layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # output: batch x 2048 x 1 x 1
            self.backbone_output_dim = 2048
        elif backbone_name == 'vit_b/16':
            # Using torchvision's ViT model
            from torchvision.models.vision_transformer import vit_b_16
            self.backbone = vit_b_16(pretrained=True)
            # The pooled embedding is usually available as self.backbone.heads.head
            # But for feature extraction, use the 'embeddings' or the pooled output
            # Extract from the 'encoder' or 'head' accordingly
            # Alternatively, define custom forward
            self.backbone_output_dim = 768
        elif backbone_name == 'mobilenet_v2':
            backbone = torchvision_models.mobilenet_v2(pretrained=True)
            # Remove classifier
            self.backbone = backbone.features  # feature extractor layers
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.backbone_output_dim = 1280
        else:
            raise ValueError(f'Unsupported backbone: {backbone_name}')

        # Embedding head: linear projection to feature_dim
        self.embedding = nn.Linear(self.backbone_output_dim, feature_dim)
        # Optional: initialize weights
        nn.init.kaiming_normal_(self.embedding.weight, mode='fan_out', nonlinearity='relu')
        if self.embedding.bias is not None:
            nn.init.constant_(self.embedding.bias, 0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract features and project onto embedding space.

        Args:
            x (Tensor): input images, shape [batch_size, 3, H, W]

        Returns:
            features (Tensor): normalized feature vectors, shape [batch_size, feature_dim]
        """
        if self.backbone_name == 'resnet50':
            # Feature extraction
            feat = self.backbone(x)  # shape: batch x 2048 x 1 x 1
            feat = feat.view(feat.size(0), -1)  # flatten
        elif self.backbone_name == 'vit_b/16':
            # ViT's forward: use pooled output
            # For ViT, the output is in the 'heads' attribute or last hidden
            # Assuming using ViT from torchvision: extract the pooled embedding
            # It's usually available as output of the classifier head
            # But for feature embeddings, detach the pooled embedding
            feat = self.backbone.forward_features(x)  # shape: batch x 768
        elif self.backbone_name == 'mobilenet_v2':
            feat_map = self.backbone(x)  # shape: batch x features x H' x W'
            feat = self.avgpool(feat_map)  # shape: batch x 1280 x 1 x 1
            feat = feat.view(feat.size(0), -1)
        else:
            raise ValueError(f'Unsupported backbone: {self.backbone_name}')

        # Project to embedding vector
        feat = self.embedding(feat)  # shape: batch x feature_dim

        # Optional normalization
        if self.normalize_features:
            feat = nn.functional.normalize(feat, p=2, dim=1)  # L2 normalize on feature_dim axis

        return feat

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Utility function for extracting features (used during evaluation).

        Args:
            x (Tensor): input images

        Returns:
            features (Tensor): feature embeddings
        """
        return self.forward(x)
