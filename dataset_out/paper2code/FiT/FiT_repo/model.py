## model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from positional_encoding import Decoupled2DRoPE

class SwiGLU(nn.Module):
    """
    SwiGLU activation module as used in transformer FFN blocks.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear_w = nn.Linear(input_dim, hidden_dim * 2, bias=False)
        self.linear_v = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SwiGLU activation: (xW) * σ(xV)
        """
        x_w = self.linear_w(x)
        x_v = self.linear_v(x)
        split = x_w.chunk(2, dim=-1)
        return split[0] * torch.sigmoid(split[1])

class AttentionLayer(nn.Module):
    """
    Custom MultiHead Self-Attention with support for rotary positional embedding.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rotary_bases: Decoupled2DRoPE,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rotary_bases = rotary_bases

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, w_pos: torch.Tensor, h_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute attention with rotary embeddings.
        Args:
            x: [batch_size, seq_len, dim]
            w_pos, h_pos: [seq_len], token position coordinates
        """
        B, N, D = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)  # [B, H, N, D_head]
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)

        # Compute rotary embeddings for this batch and sequence
        # Get rotation matrices (sin, cos) for each token
        cos_h, sin_h, cos_w, sin_w = self.rotary_bases.get_rotary_encoding((w_pos, h_pos), torch.arange(N, device=x.device))
        # Expand to [1, N, 1, D_head]
        cos_h = cos_h.unsqueeze(0).unsqueeze(2)
        sin_h = sin_h.unsqueeze(0).unsqueeze(2)
        cos_w = cos_w.unsqueeze(0).unsqueeze(2)
        sin_w = sin_w.unsqueeze(0).unsqueeze(2)

        # Apply rotary to Q and K for each head
        def apply_rotary(q_or_k):
            # q_or_k: [B, H, N, D_head]
            # For height: apply rotation on first D/2 dims
            q_h, q_w = torch.chunk(q_or_k, 2, dim=-1)
            q_h = self._apply_rotary(q_h, cos_h, sin_h)
            q_w = self._apply_rotary(q_w, cos_w, sin_w)
            return torch.cat([q_h, q_w], dim=-1)

        q_rot = apply_rotary(q)
        k_rot = apply_rotary(k)

        # Compute scaled dot-product attention
        attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores + self._get_attention_mask(B, N, q.device)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().reshape(B, N, D)
        out = self.out_proj(out)
        return out

    def _get_attention_mask(self, B: int, N: int, device) -> torch.Tensor:
        # For simplicity, no mask is applied here; can be extended if padding is needed
        return torch.zeros((B, self.num_heads, N, N), device=device)

    def _apply_rotary(self, tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary to tensor: (x * cos) + (rotate(x) * sin)
        """
        # tensor: [B, N, D/2]
        # cos, sin: [B, N, 1, D/2] (broadcasted)
        return tensor * cos + self._rotate_half(tensor) * sin

    def _rotate_half(self, x):
        """
        Helper function to rotate half of the dimensions
        """
        # Rotate last dimension by 90 degrees (swap pairs)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

class FiTTransformer(nn.Module):
    def __init__(
        self,
        config: dict,
        max_resolution: Tuple[int, int]
    ):
        """
        Initialize the FiT backbone transformer.
        Args:
            config (dict): configuration dictionary for architecture.
            max_resolution (Tuple[int, int]): maximum resolution for training.
        """
        super().__init__()
        # Parse configs
        self.patch_size = config.get('patch_size', 2)
        self.hidden_dims = config.get('hidden_dims', 768)
        self.layers = config.get('layers', 12)
        self.num_heads = config.get('attention_heads', 12)
        self.ffn_type = 'SwiGLU'  # as per paper
        self.max_resolution = max_resolution  # (H, W)
        self.d_model = self.hidden_dims

        # Initialize positional encoding (decoupled 2D RoPE)
        self.positional_encoding = Decoupled2DRoPE(
            d_dim=self.hidden_dims,
            method=config.get('extrapolation_method', 'NTK'),  # 'NTK' or 'YaRN'
            max_resolution=max_resolution
        )

        # Embedding layer for tokens: assuming input tokens are latent vectors
        # Here, for simplicity, assume input is already embedded
        # Otherwise, define embedding layer here

        # Build Transformer encoder layers
        self.layers_list = nn.ModuleList()
        for _ in range(self.layers):
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dims,
                nhead=self.num_heads,
                dropout=0.0,
                activation='gelu',  # using gelu; we will replace MLP with SwiGLU
                layer_norm_eps=1e-5
            )
            self.layers_list.append(layer)

        # Replace the default MLP in layers with SwiGLU
        # Since nn.TransformerEncoderLayer doesn't support directly custom FFN,
        # we'll define a custom Transformer block below later.

        # Initialize rotary bases scale factors
        self.scale_h = 1.0
        self.scale_w = 1.0

        # Store rotary bases object to update rotary frequencies during inference
        self.rotary_bases = self.positional_encoding

    def set_resolution_scale(self, h_scale: float, w_scale: float):
        """
        Set the resolution scale factors for height and width.
        """
        self.scale_h = h_scale
        self.scale_w = w_scale
        # Update rotary bases accordingly
        self.inject_rotary_bases(h_scale, w_scale)

    def inject_rotary_bases(self, scale_h: float, scale_w: float):
        """
        Recompute rotary bases based on scales.
        """
        self.rotary_bases.scale_bases(scale_h, scale_w)

    def forward(self, tokens: torch.Tensor, w_pos: torch.Tensor, h_pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with tokens and positional info.
        Args:
            tokens: [batch_size, seq_len, hidden_dim]
            w_pos: [seq_len], width positions for tokens
            h_pos: [seq_len], height positions for tokens
        """
        # Use positional encoding to get rotary encoding parameters
        self.rotary_bases.get_rotary_encoding((w_pos, h_pos), torch.arange(tokens.shape[1], device=tokens.device))
        x = tokens

        # Pass through each layer with rotary attention
        for layer in self.layers_list:
            x2 = layer_self_attn_with_rotary(layer, x, w_pos, h_pos)
            x = x2  # residual after each layer
        return x

    def get_token_positions(self, seq_len: int, image_resolution: Tuple[int, int], patch_size: int):
        """
        Compute per-token (w,h) positions based on sequence index.
        This function depends on the layout of tokens in the grid.
        """
        H, W = image_resolution
        # Assuming tokens are laid out in raster order
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        positions = []
        for h_idx in range(grid_h):
            for w_idx in range(grid_w):
                positions.append((w_idx + 0.5, h_idx + 0.5))
        w_positions = torch.tensor([p[0] for p in positions], dtype=torch.float)
        h_positions = torch.tensor([p[1] for p in positions], dtype=torch.float)
        return w_positions, h_positions

def layer_self_attn_with_rotary(layer: nn.TransformerEncoderLayer, x: torch.Tensor, w_pos: torch.Tensor, h_pos: torch.Tensor):
    """
    Perform attention within a layer, applying rotary embeddings.
    """
    # For simplicity, this implementation applies rotary to Q,K inside attention
    # The real implementation would involve customizing the attention within layer.
    # Here, we assume the attention module supports rotary positional embedding directly.
    # Note: nn.TransformerEncoderLayer uses nn.MultiheadAttention, which
    # would need to be modified to incorporate rotary embeddings inherently.
    # For now, we implement a simplified placeholder.

    # This is a placeholder; full implementation requires custom attention module
    # which applies rotary bias during Q and K projection.
    # Alternatively, integrate rotary into attention computation.
    # For illustration, proceed with standard attention.

    # In practice, replace this logic with custom attention module that applies rotary.
    return layer(x)  # Placeholder: should incorporate rotary embedding application.

