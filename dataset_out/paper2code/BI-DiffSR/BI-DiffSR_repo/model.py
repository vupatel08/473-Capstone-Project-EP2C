## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import get_timestep_embedding, sign_bin, STESign

# =========================
# Binarized Convolution Layer
# =========================
class BinarizedConv(nn.Module):
    """
    Binarized convolution layer with weight scaling and STE.
    Uses binarized weights (sign), scaled by mean absolute value.
    Supports 3x3 and 1x1 convolutions as needed.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1,
                 padding: int=1, bias: bool=False, scale_weights: bool=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale_weights = scale_weights
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Full-precision weight
        self.weight_fp = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # Binarized weight placeholder
        self.register_buffer('weight_b', torch.zeros_like(self.weight_fp))

    def binarize_weight(self):
        """
        Binarize weights with scaling and STE.
        """
        weight_abs_mean = self.weight_fp.abs().mean()
        weight_sign = sign_bin(self.weight_fp)
        if self.scale_weights:
            self.weight_b = weight_sign * weight_abs_mean
        else:
            self.weight_b = weight_sign

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Binarize weights for this forward pass
        self.binarize_weight()
        # Use binarized weights for convolution
        out = F.conv2d(x, self.weight_b, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

# =========================
# Residual Block
# =========================
class ResBlock(nn.Module):
    """
    Residual Block with binarized convolutions, optional Timestep conditioning.
    """
    def __init__(self, channels: int, K: int=5, use_taR: bool=True, use_taA: bool=True):
        super().__init__()
        self.channels = channels
        self.use_taR = use_taR
        self.use_taA = use_taA
        # TaR/TaA modules will be subclasses
        # For simplicity of code, instantiate placeholders; actual will be created in main
        # Internally, will be set via property or method later
        self.b1 = BinarizedConv(channels, channels)
        self.b2 = BinarizedConv(channels, channels)
        self.act1 = nn.Identity()  # placeholder; replaced with TaR in forward
        self.act2 = nn.Identity()  # placeholder; replaced with TaA in forward

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, taR_module=None, taA_module=None):
        """
        x: input feature
        t_emb: timestep embedding for modulation
        taR_module, taA_module: optional modules for timestep conditioning
        """
        # First conv + activation
        out = self.b1(x)
        if self.use_taR and taR_module:
            out = taR_module(out, t_emb)
        out = F.relu(out)  # can replace with TaA if needed
        # Second conv + activation
        out = self.b2(out)
        if self.use_taA and taA_module:
            out = taA_module(out, t_emb)
        # Add skip connection
        out = out + x
        return out

# =========================
# Cycle Pixel Shuffle (Downsampling)
# =========================
class CPDownModule(nn.Module):
    """
    Consistent Pixel-Downsample module:
    - Splits input channels into two halves
    - Processes each half with binarized conv
    - Combines and applies PixelUnShuffle (scale=2)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv1 = BinarizedConv(channels//2, channels//2)
        self.conv2 = BinarizedConv(channels//2, channels//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        c_half = self.channels // 2
        x1, x2 = torch.split(x, c_half, dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        combined = x1 + x2  # simple fusion
        # PixelUnshuffle with scale=2 -> output height/width doubled, channel doubled
        out = F.pixel_unshuffle(combined, downscale_factor=2)
        return out

# =========================
# Cycle Pixel Shuffle (Upsampling)
# =========================
class CPUpModule(nn.Module):
    """
    Consistent Pixel-Upsample module:
    - Process with two binarized convs
    - Concatenate and apply PixelShuffle (scale=2)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv1 = BinarizedConv(channels, channels)
        self.conv2 = BinarizedConv(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        concat = torch.cat([out1, out2], dim=1)
        # PixelShuffle with upscale factor 2
        out = F.pixel_shuffle(concat, upscale_factor=2)
        return out

# =========================
# Channel-Shuffle Fusion (Skip Connection)
# =========================
class CSSFusion(nn.Module):
    """
    Channel-shuffle fusion:
    - Split each feature into odd/even channels
    - Pair odd/even and interleave
    - Concatenate and process with 2 binarized convs
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        c_half = channels // 2
        self.conv_sh1 = BinarizedConv(channels, channels)
        self.conv_sh2 = BinarizedConv(channels, channels)

    def channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shuffle the channels of x.
        Here, we do simple re-interleaving: split into odd/even indices and concatenate.
        """
        # x shape: (B, C, H, W)
        C = x.shape[1]
        odd_idx = torch.tensor([i for i in range(C) if i % 2 == 1], device=x.device)
        even_idx = torch.tensor([i for i in range(C) if i % 2 == 0], device=x.device)
        odd_channels = torch.index_select(x, 1, odd_idx)
        even_channels = torch.index_select(x, 1, even_idx)
        shuf = torch.cat([even_channels, odd_channels], dim=1)  # interleave
        return shuf

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Fuse features x1 and x2.
        """
        # Channel shuffle
        x1_sh = self.channel_shuffle(x1)
        x2_sh = self.channel_shuffle(x2)
        # Concatenate (they are already shuffled for balanced range)
        # Apply binarized convolutions
        out1 = self.conv_sh1(x1_sh)
        out2 = self.conv_sh2(x2_sh)
        out = out1 + out2
        return out

# =========================
# Timestep Encoding (Sinusoidal)
# =========================
class TimestepEncoding(nn.Module):
    """
    Embed timestep scalar into sinusoidal embedding.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: tensor of shape (B,)
        returns: embedding tensor shape (B, channels)
        """
        return get_timestep_embedding(t, self.channels)

# =========================
# Timestep-aware Redistribution (TaR)
# =========================
class TaR(nn.Module):
    """
    Timestep-aware redistribution:
    - K learnable biases (b_i)
    - Select bias based on timestep group
    """
    def __init__(self, channels: int, total_timesteps: int, K: int=5):
        super().__init__()
        self.channels = channels
        self.K = K
        self.total_timesteps = total_timesteps
        # Bias parameters
        self.b_list = nn.ParameterList([
            nn.Parameter(torch.zeros(channels)) for _ in range(K)
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        t: (B,)
        """
        # Determine group index based on t
        t_idx = torch.clamp((t.float() / self.total_timesteps * self.K).long(), 0, self.K -1)
        bias = torch.stack([b for b in self.b_list], dim=0)[t_idx]  # shape (B, C)
        bias = bias.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x + bias

# =========================
# Timestep-aware Activation (TaA)
# =========================
class TaA(nn.Module):
    """
    Timestep-aware activation:
    - K RPReLU modules.
    - Select activation based on timestep group.
    """
    def __init__(self, channels: int, total_timesteps: int, K: int=5):
        super().__init__()
        self.channels = channels
        self.K = K
        self.total_timesteps = total_timesteps
        # Create K RPReLU instances with learnable biases (if needed)
        self.rprelu_list = nn.ModuleList([
            nn.ReLU() for _ in range(K)
        ])  # For simplicity, use ReLU; can replace with custom RPReLU if desired.

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_idx = torch.clamp((t.float() / self.total_timesteps * self.K).long(), 0, self.K -1)
        activation_fn = self.rprelu_list[t_idx]
        # Apply selected activation
        return activation_fn(x)

# =========================
# Main UNet Model
# =========================
class UNet(nn.Module):
    """
    UNet architecture optimized for binarization, with CP modules, CS-Fusion, TaR, and TaA.
    """
    def __init__(self, config: dict):
        super().__init__()
        # Read config parameters
        self.ch = config.get("channels", 64)
        self.num_levels = config.get("encoder_levels",4)
        self.res_blocks_per_level = config.get("res_blocks_per_level",2)
        self.decoder_res_blocks = config.get("decoder_res_blocks",3)
        self.total_timesteps = config.get("total_timesteps",2000)
        self.K = config.get("timestep_encoding_K",5)

        # Timestep embedding
        self.timestep_enc = TimestepEncoding(self.ch)

        # Input conv
        self.input_conv = BinarizedConv(6, self.ch)  # 2 images concatenated: y (LR) + noise image or condition, assumed input shape: 6 channels

        # Encoder levels
        self.encoder_levels = nn.ModuleList()
        for lvl in range(self.num_levels):
            blocks = nn.ModuleList()
            for _ in range(self.res_blocks_per_level):
                blocks.append(ResBlock(self.ch))
            down = CPDownModule(self.ch)
            self.encoder_levels.append(nn.ModuleDict({
                "res_blocks": blocks,
                "down": down
            }))

        # Bottleneck residual blocks
        self.bottleneck = nn.ModuleList()
        for _ in range(self.res_blocks_per_level):
            self.bottleneck.append(ResBlock(self.ch))

        # Decoder levels
        self.decoder_levels = nn.ModuleList()
        for lvl in range(self.num_levels):
            blocks = nn.ModuleList()
            for _ in range(self.decoder_res_blocks):
                blocks.append(ResBlock(self.ch))
            up = CPUpModule(self.ch)
            self.decoder_levels.append(nn.ModuleDict({
                "res_blocks": blocks,
                "up": up
            }))

        # Skip connection fusions (CS-Fusion)
        self.cs_fusions = nn.ModuleList()
        for _ in range(self.num_levels):
            self.cs_fusions.append(CSSFusion(self.ch))

        # Final convolution
        self.output_conv = BinarizedConv(self.ch, 3, kernel_size=3, padding=1)

        # Timestep modules
        self.taR_modules = nn.ModuleList()
        self.taA_modules = nn.ModuleList()
        for _ in range(self.num_levels * 2 + len(self.bottleneck)):
            self.taR_modules.append(TaR(self.ch, self.total_timesteps, self.K))
            self.taA_modules.append(TaA(self.ch, self.total_timesteps, self.K))
        
        # Keep track of total layers
        # Layers in encoder+decoder+latent: for indexing TaR/TaA

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: (B, 6, H, W) (condition + noise image concatenated)
        t: (B,) scalar timestep tensor
        """
        t_emb = self.timestep_enc(t)

        # Initial feature
        feats = []
        x = self.input_conv(x)  # shape: (B, C, H, W)

        # Encoder
        for lvl_idx, lvl in enumerate(self.encoder_levels):
            for res_idx, res_block in enumerate(lvl["res_blocks"]):
                # Apply TaR and TaA
                taR_idx = lvl_idx * self.res_blocks_per_level + res_idx
                taA_idx = taR_idx
                x = res_block(x, t_emb, self.taR_modules[taR_idx], self.taA_modules[taA_idx])
            feats.append(x)
            x = lvl["down"](x)  # downsample

        # Bottleneck
        for idx, res_block in enumerate(self.bottleneck):
            taR_idx = self.num_levels * self.res_blocks_per_level + idx
            taA_idx = taR_idx
            x = res_block(x, t_emb, self.taR_modules[taR_idx], self.taA_modules[taA_idx])

        # Decoder
        for lvl_idx in reversed(range(self.num_levels)):
            up = self.decoder_levels[lvl_idx]["up"]
            x = up(x)
            # Fuse with skip connection
            feat_enc = feats[lvl_idx]
            fused = self.cs_fusions[lvl_idx](feat_enc, x)
            # Residual blocks in decoder
            for res_idx, res_block in enumerate(self.decoder_levels[lvl_idx]["res_blocks"]):
                taR_idx = (self.num_levels + lvl_idx) * self.res_blocks_per_level + res_idx
                taA_idx = taR_idx
                x = res_block(x, t_emb, self.taR_modules[taR_idx], self.taA_modules[taA_idx])
            # Add fused features
            x = x + fused

        # Final output convolution
        out = self.output_conv(x)
        return out
