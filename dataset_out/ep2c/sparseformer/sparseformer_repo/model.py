## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

class SharedBackbone(nn.Module):
    """
    Lightweight convolutional backbone as described:
    7x7 stride-2 conv -> ReLU -> 3x3 stride-2 max pool.
    Output feature map has shape [batch, C, H/4, W/4]
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x  # shape: [batch, C, H/4, W/4]

class MLP(nn.Module):
    """
    Simple MLP with one hidden layer, activation GELU.
    Used for adaptive feature decoding.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TokenEmbeddings(nn.Module):
    """
    Manage learnable token embeddings and associated RoIs.
    """
    def __init__(self, num_tokens: int, token_dim: int, init_rois: torch.Tensor):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(num_tokens, token_dim))
        self.rois = nn.Parameter(init_rois.clone())  # shape: [N, 4], normalized [0,1]
    def get_embeddings(self):
        return self.embeddings
    def get_rois(self):
        return self.rois
    def refine_rois(self, delta: torch.Tensor):
        """
        delta shape: [N, 4], corresponding to (x,y,w,h) adjustments
        RoI update equations:
        x' = x + Δt_x * w
        y' = y + Δt_y * h
        w' = w * exp(Δt_w)
        h' = h * exp(Δt_h)
        """
        x, y, w, h = self.rois[:,0], self.rois[:,1], self.rois[:,2], self.rois[:,3]
        Δx, Δy, Δw, Δh = delta[:,0], delta[:,1], delta[:,2], delta[:,3]
        x_new = x + Δx * w
        y_new = y + Δy * h
        w_new = w * torch.exp(Δw)
        h_new = h * torch.exp(Δh)
        self.rois.data = torch.stack([x_new, y_new, w_new, h_new], dim=1).clamp(0,1)

class BilinearSampler:
    """
    Utility class for bilinear sampling from feature maps
    using sampling locations in normalized coordinates.
    """
    @staticmethod
    def sample(feature_map: torch.Tensor, sampling_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W]
            sampling_points: [N, P, 2], normalized to [0, 1]
        Returns:
            sampled_features: [N, P, C]
        """
        B, C, H, W = feature_map.shape
        N, P, _ = sampling_points.shape
        # Convert normalized coords to absolute xy in feature map
        x = sampling_points[:,:,0] * (W - 1)
        y = sampling_points[:,:,1] * (H - 1)
        grid = torch.stack([x, y], dim=3)  # [N, P, 2]
        # For batch processing, replicate feature_map
        # Note: In this context, sampling is per token, so batch size=1
        grid = grid.unsqueeze(0)  # [1, N, P, 2]
        # Reshape for grid_sample
        # But since we process per token, map batch independently
        # We assume batch size of 1 for simplicity
        grid = grid.squeeze(0).permute(2,0,1).unsqueeze(0)  # [1, 2, N, P]
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode='bilinear',
            align_corners=True
        )  # shape: [B, C, 1, N]
        sampled = sampled.squeeze(2).permute(2,0,1)  # [N, P, C]
        return sampled

class FocusLinearGenerator(nn.Module):
    """
    Generate P sampling offsets conditioned on token embedding t.
    Outputs offsets: [N, P, 2]
    """
    def __init__(self, token_dim: int, P: int):
        super().__init__()
        self.linear = nn.Linear(token_dim, 2 * P)
        self.P = P
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [N, d_c]
        Returns:
            offsets: [N, P, 2]
        """
        offset = self.linear(t)  # [N, 2*P]
        offset = offset.view(-1, self.P, 2)  # [N, P, 2]
        return offset

class RoIAdjuster(nn.Module):
    """
    Generate RoI deltas for refinement from token embedding t.
    Outputs: delta_x, delta_y, delta_w, delta_h each [N, 1]
    """
    def __init__(self, token_dim: int):
        super().__init__()
        self.linear = nn.Linear(token_dim, 4)
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [N, d_c]
        Returns:
            delta: [N, 4]
        """
        delta = self.linear(t)
        return delta

class SparseFocusingTransformer(nn.Module):
    """
    One stage of the focusing transformer:
    - Generate sampling points
    - Sample features
    - Decode features to update token embeddings
    - Refine RoIs
    """
    def __init__(self, token_dim: int, num_points: int, image_size: int, feature_map_size: Tuple[int,int]):
        super().__init__()
        self.P = num_points
        self.image_size = image_size  # e.g., 224
        self.feature_map_size = feature_map_size  # (H_feat, W_feat)
        self.offset_generator = FocusLinearGenerator(token_dim, self.P)
        self.roi_delta_generator = RoIAdjuster(token_dim)
        # Adaptive decoder
        self.decoder = MLP(in_dim=self.P * feature_map_size[0]*feature_map_size[1], hidden_dim=token_dim//4, out_dim=token_dim)
        # Map for converting offsets
        self.norm_std = 3.0  # standard deviations for normalization
    def forward(self, t: torch.Tensor, rois: torch.Tensor, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t: [N, d_c]
            rois: [N, 4], normalized [0,1]
            feature_map: [B, C, Hf, Wf], shared feature map
        Returns:
            new_t: [N, d_c]
            new_rois: [N, 4]
        """
        N, d_c = t.shape
        # Generate offsets conditioned on t
        offsets = self.offset_generator(t)  # [N, P, 2]
        # Normalize offsets with std
        offsets = offsets / self.norm_std
        # Convert relative offsets to absolute sampling locations
        x0, y0, w, h = rois[:,0], rois[:,1], rois[:,2], rois[:,3]
        # [N,1] for broadcasting
        xc = x0.unsqueeze(1)  # [N,1]
        yc = y0.unsqueeze(1)
        W = w.unsqueeze(1)
        H = h.unsqueeze(1)
        # absolute sampling locations
        x_samples = xc + 0.5 * offsets[:,:,0] * W  # [N, P]
        y_samples = yc + 0.5 * offsets[:,:,1] * H
        sampling_points = torch.stack([x_samples, y_samples], dim=2)  # [N, P, 2]
        # Clamp to [0, 1]
        sampling_points = sampling_points.clamp(0,1)
        # Sample features
        sampled_feats = BilinearSampler.sample(feature_map, sampling_points)  # [N, P, C]
        # Decode features with adaptive decoding
        feat_flat = sampled_feats.view(N, -1)  # [N, P*C]
        decoded = self.decoder(feat_flat)  # [N, d_c]
        # Residual update of token
        new_t = t + decoded
        # Generate RoI deltas
        delta = self.roi_delta_generator(t)  # [N,4]
        # Update RoIs
        new_x = rois[:,0] + delta[:,0] * rois[:,2]
        new_y = rois[:,1] + delta[:,1] * rois[:,3]
        new_w = rois[:,2] * torch.exp(delta[:,2])
        new_h = rois[:,3] * torch.exp(delta[:,3])
        new_rois = torch.stack([new_x, new_y, new_w, new_h], dim=1).clamp(0,1)
        return new_t, new_rois

class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer Encoder Layer
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float=4.0, dropout: float=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, embed_dim]
        """
        # MultiheadAttention expects [N, embed_dim], batch_first
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm.unsqueeze(0), x_norm.unsqueeze(0), x_norm.unsqueeze(0))
        attn_output = attn_output.squeeze(0)
        x = residual + self.dropout(attn_output)
        residual = x
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = residual + self.dropout(x_mlp)
        return x

class CortexTransformerEncoder(nn.Module):
    """
    Multiple layers of transformer encoder over token set.
    """
    def __init__(self, embed_dim: int, num_layers: int, num_heads: int=8, mlp_ratio: float=4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class SparseFormer(nn.Module):
    """
    Main class implementing the SparseFormer architecture.
    Combines backbone, token set, focusing transformer, cortex transformer, and classifier.
    """
    def __init__(self, config: dict):
        super().__init__()
        # Extract configuration
        self.num_tokens = config['model'].get('num_tokens', 81)
        self.token_dim = config['model'].get('token_dim', 768)
        self.focusing_layers = config['model'].get('focusing_layers', 1)
        self.cortex_layers = config['model'].get('cortex_layers', 12)
        self.sampling_points = config['model'].get('sampling_points', 36)
        self.stage_repeats = config['model'].get('stage_repeats', 1)
        self.image_size = 224  # as per training config
        # Backbone
        self.backbone = SharedBackbone(in_channels=3, out_channels=96)
        # Initialize token embeddings and RoIs
        init_rois = self._initialize_rois()
        self.token_set = TokenEmbeddings(self.num_tokens, self.token_dim, init_rois)
        # Focusing transformer stage
        self.focusing_transformer = nn.ModuleList([
            SparseFocusingTransformer(
                token_dim=self.token_dim,
                num_points=self.sampling_points,
                image_size=self.image_size,
                feature_map_size=(self.image_size//4, self.image_size//4)
            ) for _ in range(self.focusing_layers)
        ])
        # Cortex transformer
        self.cortex_transformer = CortexTransformerEncoder(
            embed_dim=self.token_dim,
            num_layers=self.cortex_layers
        )
        # Classification head
        self.head = nn.Linear(self.token_dim, 1000)
    def _initialize_rois(self):
        """
        Initialize RoIs to cover the image on a grid.
        """
        # For simplicity, we initialize grid centered at uniform points
        n_grid = int(math.sqrt(self.num_tokens))
        coords = torch.linspace(0.1, 0.9, n_grid)
        centers_x, centers_y = torch.meshgrid(coords, coords)
        centers_x = centers_x.contiguous().view(-1)
        centers_y = centers_y.contiguous().view(-1)
        widths = torch.full_like(centers_x, 0.5)
        heights = torch.full_like(centers_y, 0.5)
        rois = torch.stack([centers_x, centers_y, widths, heights], dim=1)  # [N,4]
        return rois
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W]
        Returns:
            logits: [B, 1000]
        """
        B = images.shape[0]
        # Extract shared feature map
        feature_map = self.backbone(images)  # [B, C, H/4, W/4]
        # Prepare tokens and RoIs
        t = self.token_set.get_embeddings()  # [N, d_c]
        rois = self.token_set.get_rois()     # [N, 4]
        # Repeat for batch
        token_embeddings = t.unsqueeze(0).expand(B, -1, -1)  # [B, N, d_c]
        rois_batch = rois.unsqueeze(0).expand(B, -1, -1)  # [B, N, 4]

        # Initialize tokens for this batch
        tokens = token_embeddings  # [B, N, d_c]
        rois = rois_batch  # [B, N, 4]

        # Focus stages
        for stage_idx in range(self.focusing_layers):
            stage_fn = self.focusing_transformer[stage_idx]
            new_tokens_list = []
            new_rois_list = []
            for b in range(B):
                # For each batch element
                tokens_b = tokens[b]  # [N, d_c]
                rois_b = rois[b]      # [N, 4]
                # Apply focusing transformer stage
                new_tokens, new_rois = stage_fn(tokens_b, rois_b, feature_map[b:b+1])
                new_tokens_list.append(new_tokens)
                new_rois_list.append(new_rois)
            tokens = torch.stack(new_tokens_list, dim=0)  # [B, N, d_c]
            rois = torch.stack(new_rois_list, dim=0)      # [B, N, 4]

        # Prepare tokens for cortex transformer
        tokens = tokens  # [B, N, d_c]
        # Reshape for transformer (batch, seq, embed)
        tokens = tokens
        # Process with cortex transformer
        tokens = self.cortex_transformer(tokens)  # [B, N, d_c]
        # Readout: average over tokens
        pooled = tokens.mean(dim=1)  # [B, d_c]
        logits = self.head(pooled)   # [B, 1000]
        return logits
