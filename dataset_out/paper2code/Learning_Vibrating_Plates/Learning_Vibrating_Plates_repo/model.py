## model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Extract model parameters from config
ARCHITECTURE_TYPE = CONFIG['model'].get('architecture', 'UNet')  # 'UNet', 'FNO', etc.
ENCODER_TYPE = CONFIG['model'].get('encoder', {}).get('type', 'implicit_shape_encoder')
SHAPE_REPRESENTATION = CONFIG['model'].get('encoder', {}).get('shape_representation', 'signed_distance_function') # placeholder
RESPONDER_TYPE = CONFIG['model'].get('response_decoder', {}).get('type', 'velocity_field')  # or 'scalar_response'
CHANNELS = CONFIG['model'].get('channels', 64)
DEPTH = CONFIG['model'].get('depth', 4)
FREQ_EMBED_SIZE = 32  # Size of frequency embedding, can be adjusted or made configurable

# Helper functions
def get_fourier_features(f, num_features=FREQ_EMBED_SIZE):
    """
    Creates Fourier features for scalar frequency input.
    Args:
        f: scalar tensor, shape (batch_size,)
        num_features: int, number of frequency features
    Returns:
        feature tensor of shape (batch_size, 2 * num_features)
    """
    omega = torch.linspace(0., 1., steps=num_features, device=f.device)
    f = f.unsqueeze(-1)  # (batch_size, 1)
    f_scaled = f * omega * 2 * math.pi  # scale
    sin_feat = torch.sin(f_scaled)
    cos_feat = torch.cos(f_scaled)
    return torch.cat([sin_feat, cos_feat], dim=-1)  # (batch_size, 2 * num_features)

# Shape Encoder Modules
class ImplicitShapeEncoder(nn.Module):
    """
    Encodes shape expressed as a Signed Distance Function (SDF) grid
    via a few convolutional layers to produce a fixed-length embedding.
    """
    def __init__(self, input_channels=1, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2)
        self.norm1 = nn.LayerNorm([32, None, None])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.norm2 = nn.LayerNorm([64, None, None])
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.norm3 = nn.LayerNorm([128, None, None])
        self.fc = nn.Linear(128, embedding_dim)
        
    def forward(self, sdf: torch.Tensor):
        """
        Args:
            sdf: tensor shape (batch_size, height, width)
        Returns:
            embedding: tensor shape (batch_size, embedding_dim)
        """
        x = sdf.unsqueeze(1)  # (B, 1, H, W)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        # Global average pooling
        x = x.mean(dim=[2,3])  # (B, 128)
        embedding = self.fc(x)
        return embedding

# Example placeholder for ResNet18 encoder
# For brevity, use torchvision's ResNet if available:
import torchvision.models as models
class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained=False, embedding_dim=128):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        # Remove classification head
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embedding_dim)
    def forward(self, x):
        """
        Args:
            x: tensor (B, C, H, W)
        Returns:
            embedding: tensor (B, embedding_dim)
        """
        x = self.resnet(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)
        return self.fc(x)

# Vision Transformer encoder placeholder
# Implementing a minimal ViT encoder as a subclass
class ViTEncoder(nn.Module):
    def __init__(self, image_size=64, patch_size=16, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        # For simplicity, use nn.TransformerEncoder with patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.embedding = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: tensor (B, 1, H, W)
        Returns:
            embedding: tensor (B, embed_dim)
        """
        x = self.embedding(x)  # (B, embed_dim, Hh, Wh)
        B, C, Hh, Wh = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # (Hh*Wh, B, embed_dim)
        x = self.transformer(x)  # same shape
        # Pool over spatial patches
        x = x.mean(dim=0)  # (B, embed_dim)
        return x

# FiLM conditioning layer
class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, scalar_dim):
        """
        Args:
            feature_dim: dimensionality of the features to condition
            scalar_dim: number of scalar conditioning parameters
        """
        super().__init__()
        self.film_fc = nn.Linear(scalar_dim, 2 * feature_dim)  # gamma and beta

    def forward(self, features: torch.Tensor, scalar_params: torch.Tensor):
        """
        Args:
            features: (B, feature_dim)
            scalar_params: (B, scalar_dim)
        Returns:
            conditioned features: (B, feature_dim)
        """
        gamma_beta = self.film_fc(scalar_params)  # (B, 2*feature_dim)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return features * gamma + beta

# Response decoders
class VelocityFieldUNet(nn.Module):
    """
    UNet architecture for predicting velocity fields conditioned on shape + scalar params + frequency.
    Uses FiLM layers for conditioning.
    """
    def __init__(self, in_channels=1, base_channels=CHANNELS, depth=DEPTH):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth

        # Encoding path
        self.encoders = nn.ModuleList()
        for i in range(depth):
            in_ch = in_channels if i == 0 else base_channels * 2**(i-1)
            out_ch = base_channels * 2**i
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LayerNorm([out_ch, None, None]),
                    nn.ReLU()
                )
            )

        # Decoding path
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth-1)):
            in_ch = base_channels * 2**(i+1)
            out_ch = base_channels * 2**i
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LayerNorm([out_ch, None, None]),
                    nn.ReLU()
                )
            )

        # Final convolution to 2 velocity components
        self.final_conv = nn.Conv2d(base_channels, 2, kernel_size=1)

        # Self-attention layers in encoder and decoder
        self.encoder_self_attn = nn.MultiheadAttention(embed_dim=base_channels, num_heads=4)
        self.decoder_self_attn = nn.MultiheadAttention(embed_dim=base_channels, num_heads=4)

        # FiLM layers will be applied after each encoder block
        # For simplicity, create a list
        self.film_layers = nn.ModuleList([
            FiLMLayer(base_channels * 2 ** i, scalar_dim=7) for i in range(depth)
        ])

    def forward(self, shape_feat: torch.Tensor, scalar_params: torch.Tensor, f: torch.Tensor):
        """
        Args:
            shape_feat: (B, H, W, C) feature map or (B, C) shape embedding
            scalar_params: (B, 7)
            f: (B,) scalar, frequency value
        Returns:
            velocity_field: (B, 2, H, W)
        """
        # For simplicity, assume shape_feat is spatial for unet input or pooled for vector
        # First, expand shape_feat to spatial grid if needed
        # Let's assume input shape_feat is (B, C) and broadcast to spatial
        # Alternatively, shape_feat can be a feature map if coming from a CNN encoder
        # Here, treat shape_feat as a vector; expand spatially
        B = shape_feat.shape[0]
        H, W = 64, 64  # or set according to input; be consistent with dataset
        feat_map = shape_feat.unsqueeze(-1).unsqueeze(-1).expand(B, shape_feat.shape[-1], H, W)

        # Embed frequency
        freq_emb = get_fourier_features(f, num_features=FREQ_EMBED_SIZE)  # (B, 2*FREQ_EMBED_SIZE)
        # Expand freq_emb spatially for FiLM conditioning
        freq_emb_exp = freq_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # Initial input features
        x = feat_map  # (B, C, H, W)

        # Encoder path
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            # Apply FiLM for conditioning
            conditioned = self.film_layers[i](x.permute(0,2,3,1).mean(dim=[1,2]), scalar_params)  # pool spatial
            # Broadcast back
            gamma = conditioned.unsqueeze(1).unsqueeze(2)
            beta = conditioned.unsqueeze(1).unsqueeze(2)
            x = x * gamma + beta
            skips.append(x)
            # Downsample for next layer if needed? (In current implementation, strides are inside convs)

        # Bottleneck act (could add attentions here)
        # For simplicity, skip
        # Decoder path
        for i, decoder in enumerate(self.decoders):
            skip_feat = skips[-(i+2)]
            x = F.interpolate(x, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_feat], dim=1)
            x = decoder(x)

        velocity = self.final_conv(x)  # (B, 2, H, W)
        return velocity

class ResponseMLP(nn.Module):
    """
    Fully-connected network to predict scalar response F(f) given combined features.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, input_dim)
        Returns:
            scalar response: (B, 1)
        """
        return self.layers(x).squeeze(-1)

# Main model class supporting variants
class LearningVibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Instantiate shape encoder
        encoder_type = ARCHITECTURE_TYPE.lower()
        if encoder_type == 'implicit_shape_encoder':
            self.shape_encoder = ImplicitShapeEncoder(embedding_dim=128)
            shape_feat_dim = 128
        elif encoder_type == 'resnet18':
            self.shape_encoder = ResNet18Encoder(pretrained=False, embedding_dim=128)
            shape_feat_dim = 128
        elif encoder_type == 'vit':
            self.shape_encoder = ViTEncoder(embed_dim=128)
            shape_feat_dim = 128
        else:
            raise ValueError(f"Unsupported encoder type: {ARCHITECTURE_TYPE}")

        # Scalar properties embedding (can be normalized, so just linear layer)
        self.prop_fc = nn.Linear(7, 64)

        # FiLM layers to condition shape features
        self.film_layers = nn.ModuleList([
            FiLMLayer(shape_feat_dim, scalar_dim=7) for _ in range(DEPTH)
        ])

        # Response decoder
        decoder_type = RESPONDER_TYPE.lower()
        if decoder_type == 'velocity_field':
            self.response_decoder = VelocityFieldUNet(
                in_channels=1, base_channels=CHANNELS, depth=DEPTH
            )
            self.response_type = 'velocity_field'
        elif decoder_type == 'scalar_response':
            # Input_dim: shape feature + propagated scalar + freq embedding
            self.response_decoder = ResponseMLP(input_dim=shape_feat_dim + 7 + 2 * FREQ_EMBED_SIZE)
            self.response_type = 'scalar_response'
        else:
            raise ValueError(f"Unsupported response decoder type: {RESPONDER_TYPE}")

    def forward(self, shape_input: torch.Tensor, scalar_props: torch.Tensor, frequency: torch.Tensor):
        """
        Args:
            shape_input: shape data, e.g., sdf grid or images, shape depends on encoder
            scalar_props: (B, 7)
            frequency: (B,) frequency scalar values
        Returns:
            response: response prediction (scalar or velocity map)
        """
        # Encode shape
        shape_feat = self.shape_encoder(shape_input)  # (B, feature_dim)
        # Encode properties
        prop_feat = self.prop_fc(scalar_props)  # (B, 64) optionally
        # Conditioning shape features with scalar properties via FiLM
        conditioned_feat = shape_feat
        for i, film in enumerate(self.film_layers):
            conditioned_feat = film(conditioned_feat, scalar_props)  # (B, feature_dim)

        # Embed frequency
        freq_emb = get_fourier_features(frequency, num_features=FREQ_EMBED_SIZE)  # (B, 2*FREQ_EMBED_SIZE)

        if self.response_type == 'velocity_field':
            # Decode velocity field conditioned on shape + frequency + properties
            velocity = self.response_decoder(conditioned_feat, scalar_props, frequency)
            return velocity  # shape (B, 2, H, W)
        else:
            # Concatenate features for scalar response
            combined_input = torch.cat([conditioned_feat, scalar_props, freq_emb], dim=-1)
            response_scalar = self.response_decoder(combined_input)  # (B,)
            return response_scalar

