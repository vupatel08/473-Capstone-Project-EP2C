## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for timesteps.
    Converts scalar timestep t into a high-dimensional embedding.
    """
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): Tensor of timesteps shape (batch, ), assumed to be scalar or 1D.
        Returns:
            embeddings (torch.Tensor): shape (batch, embedding_dim)
        """
        device = t.device
        half_dim = self.embedding_dim // 2
        # Log scale for sinusoid
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -(math.log(10000.0) / (half_dim - 1)))
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # shape: (batch, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 != 0:
            # Pad if odd
            emb = F.pad(emb, (0,1))
        return emb

class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv2d -> LeakyReLU
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # No normalization layers
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))

class DownSampleBlock(nn.Module):
    """
    Downsampling block: ConvBlock followed by downsampling
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.pool(x)

class UpSampleBlock(nn.Module):
    """
    Upsampling block: Upsample + ConvBlock
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Concatenate skip connection along channel dimension
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)

class UNetEncoder(nn.Module):
    """
    Encoder: Stack of downsampling ConvBlocks with skip connections
    """
    def __init__(self, in_channels: int, base_channels: int, depth: int):
        super().__init__()
        self.depth = depth
        self.down_blocks = nn.ModuleList()
        channels = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.down_blocks.append(DownSampleBlock(channels, out_ch))
            channels = out_ch
        self.bottleneck_channels = channels

    def forward(self, x: torch.Tensor):
        features = []
        for down in self.down_blocks:
            features.append(x)
            x = down(x)
        return x, features

class UNetDecoder(nn.Module):
    """
    Decoder: Upsampling with skip connections
    """
    def __init__(self, base_channels: int, depth: int):
        super().__init__()
        self.depth = depth
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            in_ch = base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2 ** i)
            self.up_blocks.append(UpSampleBlock(in_ch, out_ch))
        self.final_conv = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor, features: list):
        # features from encoder are in order: [input, first down, ..., last down]
        for up, feat in zip(self.up_blocks, reversed(features)):
            x = up(x, feat)
        return self.final_conv(x)

class ScoreUNet(nn.Module):
    """
    U-Net architecture with no group norm/self-attention, conditioned on x_T and timestep embedding
    """
    def __init__(self, in_channels=3, base_channels=64, depth=4, embed_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth
        self.embed_dim = embed_dim

        # Timestep embedding module
        self.ts_embedding = TimestepEmbedding(embedding_dim=embed_dim)

        # Input layer: process x_t and x_T concatenated
        self.input_conv = nn.Conv2d(in_channels*2 + embed_dim, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.encoder = UNetEncoder(in_channels=base_channels, base_channels=base_channels, depth=depth)

        # Decoder
        self.decoder = UNetDecoder(base_channels=base_channels, depth=depth)

        # Final output layer: single 1x1 conv to produce \(\hat{\epsilon}_\theta\)
        self.output_conv = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Noisy input image, shape (B, C, H, W)
            x_T (torch.Tensor): Conditioning low-quality or target image, shape (B, C, H, W)
            t (torch.Tensor): Timestep scalar tensor, shape (B,)
        Returns:
            epsilon_pred (torch.Tensor): Predicted scaled noise residual, shape (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Compute timestep embedding
        t_emb = self.ts_embedding(t)  # shape: (B, embed_dim)

        # Expand embedding to spatial shape for injection
        t_emb_expanded = t_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # Concatenate x, conditioning x_T, and timestep embedding
        x_input = torch.cat([x, x_T, t_emb_expanded], dim=1)  # shape: (B, 2C + embed_dim, H, W)
        x_feat = self.input_conv(x_input)

        # Encoding
        bottleneck, features = self.encoder(x_feat)

        # Decoding
        x_dec = self.decoder(bottleneck, features)

        # Output layer
        epsilon_pred = self.output_conv(x_dec)
        return epsilon_pred
