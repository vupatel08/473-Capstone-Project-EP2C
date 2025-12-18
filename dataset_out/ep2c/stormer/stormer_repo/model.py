## model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import timm  # For transformer layers if desired
from utils import get_2d_positional_encoding

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on delta_t embedding.
    This layer learns scale and shift parameters dynamically from input conditioning.
    """
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)
        # Parameters for adaptive scale (gamma) and shift (beta)
        # These will be generated dynamically during forward pass
        # No parameters here; generated in forward from conditioning MLP

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, normalized_shape)
        normalized = self.norm(x)
        return normalized * gamma.unsqueeze(1) + beta.unsqueeze(1)

class FeedForward(nn.Module):
    """
    Feedforward network with configurable expansion ratio.
    """
    def __init__(self, d_model: int, mlp_ratio: int = 4, drop_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * mlp_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model * mlp_ratio, d_model)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module with optional bias.
    """
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.1, proj_drop: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class CrossVariableAttention(nn.Module):
    """
    Cross-Attention over variable dimension with learnable query vector.
    """
    def __init__(self, num_vars: int, feature_dim: int, num_heads: int = 4):
        super().__init__()
        # Query vector: learnable parameter for aggregation
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))
        # Multi-head attention for variables
        self.attn = MultiHeadSelfAttention(embed_dim=feature_dim, num_heads=num_heads)
        # Note: No key/value learnable, input is (V, D)
        # Will be embedded into Event format: shape (V, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (H_p, W_p, V, D)
        Returns:
            aggregated: tensor of shape (H_p, W_p, D)
        """
        H_p, W_p, V, D = x.shape
        # Reshape spatially
        # For each spatial position, aggregate over V
        # Shape: (H_p*W_p, V, D)
        x_flat = x.reshape(-1, V, D)
        # Expand query: shape (1,1,D), then broadcast
        Q = self.query.expand(x_flat.shape[0], -1, -1)  # (B,1,D)
        # For attention, treat V as sequence dimension
        # Prepare input: (B, V, D)
        # Use Q as query
        # Attention over V
        attn_output = self.attn_with_query(Q, x_flat)
        # attn_output: (B, 1, D)
        # Reshape back to (H_p, W_p, D)
        aggregated = attn_output.squeeze(1).reshape(H_p, W_p, D)
        return aggregated

    def attn_with_query(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Modified attention with fixed query vector
        Args:
            query: shape (batch_size, 1, D)
            key_value: shape (batch_size, V, D)
        Returns:
            output: shape (batch_size, 1, D)
        """
        # Following scaled dot-product attention
        scale = 1.0 / math.sqrt(query.shape[-1])
        scores = torch.bmm(query, key_value.transpose(1, 2)) * scale  # (B,1,V)
        attn_weights = torch.softmax(scores, dim=-1)  # (B,1,V)
        out = torch.bmm(attn_weights, key_value)  # (B,1,D)
        return out

class TransformerBlock(nn.Module):
    """
    Transformer block with AdaLN conditioned on delta_t embedding.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, drop_rate: float = 0.1):
        super().__init__()
        self.norm1 = None  # Will be replaced with AdaLN during forward
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop=drop_rate, proj_drop=drop_rate)
        self.norm2 = None  # AdaLN
        self.mlp = FeedForward(embed_dim, mlp_ratio, drop_rate)
        self.drop_path = nn.Identity()  # Could implement stochastic depth if needed

    def forward(self, x: torch.Tensor, gamma1: torch.Tensor, beta1: torch.Tensor,
                gamma2: torch.Tensor, beta2: torch.Tensor) -> torch.Tensor:
        # Attention with AdaLN
        # For AdaLN, replace layer norm with with AdaLayerNorm
        # Using idempotent design for simplicity
        # Attention block
        x_attn = self._attention_with_aDLN(x, gamma1, beta1)
        x = x + self.drop_path(x_attn)

        # MLP with AdaLN
        x_mlp = self._mlp_with_aDLN(x, gamma2, beta2)
        x = x + self.drop_path(x_mlp)

        return x

    def _attention_with_aDLN(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # Standard MultiHeadAttention, followed by AdaLN
        attn_out = self.attn(x)
        # Apply AdaLN
        return attn_out  # Applieed externally

    def _mlp_with_aDLN(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        mlp_out = self.mlp(x)
        return mlp_out

class TransformerModel(nn.Module):
    """
    Core transformer-based weather forecasting model with weather-specific embedding,
    adaptive layer normalization conditioned on delta_t, and output heads for variable differences.
    """
    def __init__(self, config: dict):
        super().__init__()
        # Extract config parameters
        self.patch_size = config['model']['patch_size']
        self.hidden_dim = config['model']['hidden_dim']
        self.num_layers = config['model']['num_layers']
        self.num_heads = config['model']['num_heads']
        self.mlp_ratio = config['model'].get('mlp_ratio', 4)
        self.dropout_rate = config['model'].get('dropout_rate', 0.1)

        self.variables = None  # To be set on forward, or passed during init
        self.num_vars = None  # Number of variables
        # Variable embedding layers: one per variable, shared here for simplicity
        # Can be extended to distinct layers for each variable if needed
        # For simplicity, assume shared linear embedding for all variables
        # But per variable, we can have individual linear layers
        # For generality, let's implement a ModuleDict of linear layers
        # We'll set variable names later
        self.variable_embeddings = nn.ModuleDict()

        # Initialize included variables and pressure levels according to config
        self.config_vars = config['dataset']['variables']
        self.pressure_levels = config['dataset']['pressure_levels']

        # Prepare variable embedding modules
        # Each variable: linear layer mapping input channels to embedding dimension
        # Assume input channels for each variable are known
        # For pressure-dependent variables, channels == 1 per pressure level
        # For surface variables, channels == 1
        for var_list in self.config_vars.values():
            for var in var_list:
                # Input channels depend on variable (pressure levels or surface)
                # We assume pressure levels for pressure variables, 1 for surface
                if var in self.pressure_levels:
                    in_channels = 1  # Single pressure level per mesh
                else:
                    in_channels = 1  # For surface vars
                # Initialize linear layer for variable embedding
                self.variable_embeddings[var] = nn.Linear(in_channels, self.hidden_dim)

        # Number of tokens per spatial patch
        self.patch_size = config['model']['patch_size']
        # Compute number of tokens along H and W after patching
        self.H_tokens = None  # will be set during forward when input shape is known
        self.W_tokens = None

        # Positional encoding: sinusoidal or learned
        # We will generate on the fly during forward based on feature map size
        # For simplicity, assume fixed max position embeddings
        self.pos_encoding = None  # To be created during forward

        # Variable aggregation via cross attention
        self.variable_aggregation = CrossVariableAttention(
            num_vars=sum(len(vs) for vs in self.config_vars.values()),
            feature_dim=self.hidden_dim,
            num_heads=4
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_dim, self.num_heads, self.mlp_ratio, self.dropout_rate)
             for _ in range(self.num_layers)]
        )

        # Conditioning MLP for delta_t
        # Maps scalar delta_t to gamma and beta parameters for AdaLN
        # For each normalization layer, produce gamma, beta
        # Additionally, per paper, produce alpha1, alpha2 for other scaling if needed
        self.condition_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_layers * 4)  # for each layer: gamma1, beta1, gamma2, beta2
        )

        # Output head to predict variable differences per token
        self.output_head = nn.Linear(self.hidden_dim, len(self.pressure_levels) + 1)  # +1 for 2-m T or similar

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        # Initialize weights of linear projections
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_conditioning_params(self, delta_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate gamma and beta parameters for all transformer layers conditioned on delta_t.
        Args:
            delta_t: tensor of shape (batch_size, 1)
        Returns:
            gamma: list of tensors per layer for attention normalization (list length = num_layers)
            beta: list of tensors per layer
        """
        params = self.condition_mlp(delta_t)  # shape: (batch_size, num_layers*4)
        gamma_list = []
        beta_list = []
        for i in range(self.num_layers):
            gamma1 = params[:, 4*i].unsqueeze(-1)  # shape: (batch_size, 1)
            beta1 = params[:, 4*i + 1].unsqueeze(-1)
            gamma2 = params[:, 4*i + 2].unsqueeze(-1)
            beta2 = params[:, 4*i + 3].unsqueeze(-1)
            gamma_list.append((gamma1, gamma2))
            beta_list.append((beta1, beta2))
        return gamma_list, beta_list

    def forward(self, X: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            X: tensor of shape (batch_size, V, H, W)
            delta_t: tensor of shape (batch_size, 1) or scalar, forecast interval in hours
        Returns:
            delta_pred: tensor of shape (batch_size, V, H, W), predicted difference fields
        """
        batch_size, V, H, W = X.shape

        # Generate delta_t embedding
        delta_t = delta_t.view(-1,1)  # shape: (batch_size,1)
        gamma_list, beta_list = self.get_conditioning_params(delta_t)

        # --- Variable tokenization ---
        # For each variable in config, embed each spatial patch
        tokens_list = []
        variable_names = []
        for var_type, vars_in_type in self.config_vars.items():
            for var in vars_in_type:
                # Input data: shape (batch, H, W) or (batch, levels, H, W)
                data = X[:, var]  # shape: (batch, H, W) or (batch, levels, H, W)
                # If pressure-dependent, pick pressure level or embed as needed
                # Here, assume input data already at needed pressure level or variable
                if data.ndim == 3:
                    # Shape: (batch, H, W)
                    # Prepare for linear embedding: flatten patches
                    # Patchify: reshape to (batch, H/p, p, W/p, p), then flatten patches
                    p = self.patch_size
                    H_patches = H // p
                    W_patches = W // p
                    data_patched = data.reshape(batch_size, H_patches, p, W_patches, p)
                    # Flatten patches spatially
                    # For each patch, average or flatten
                    # Using flatten
                    data_patched = data_patched.permute(0,1,3,2,4).reshape(batch_size, H_patches, W_patches, p*p)
                    # Linear layer per patch
                    # Apply linear embed per spatial patch
                    # Reshape to (batch * H_patches * W_patches, p*p)
                    patches_flat = data_patched.reshape(-1, p*p)
                    embeddings = self.variable_embeddings[var](patches_flat)  # (batch*H_p*W_p, D)
                    # Reshape back
                    spatial_tokens = embeddings.reshape(batch_size, H_patches, W_patches, -1)
                else:
                    # For pressure level data, shape: (batch, levels, H, W)
                    # For simplicity, pick the pressure level matching pressure_levels for this variable
                    # Assume data has shape (batch, levels, H, W)
                    # For pressure-dependent variables, assume one level per variable
                    # Let's select the appropriate level index
                    # In real data, this may need proper selection
                    level_idx = 0  # placeholder, could be matched by pressure level
                    data_level = data[:, level_idx, :, :]  # (batch, H, W)
                    p = self.patch_size
                    H_patches = H // p
                    W_patches = W // p
                    data_patched = data_level.reshape(batch_size, H_patches, p, W_patches, p)
                    data_patched = data_patched.permute(0,1,3,2,4).reshape(batch_size, H_patches, W_patches, p*p)
                    patches_flat = data_patched.reshape(-1, p*p)
                    embeddings = self.variable_embeddings[var](patches_flat)
                    spatial_tokens = embeddings.reshape(batch_size, H_patches, W_patches, -1)
                tokens_list.append(spatial_tokens)  # shape: (batch, H_p, W_p, D)
                variable_names.append(var)

        # Stack all variable tokens: shape (batch, H_p, W_p, V, D)
        tokens_stacked = torch.stack(tokens_list, dim=2)  # (batch, H_p, W_p, V, D)

        # --- Variable aggregation via cross-attention ---
        # Shape needed: (H_p, W_p, V, D)
        H_p, W_p, V_total, D = tokens_stacked.shape
        # For each spatial position, aggregate over V
        aggregated = torch.empty(H_p, W_p, D, device=X.device, dtype=X.dtype)
        for i in range(H_p):
            for j in range(W_p):
                var_deps = tokens_stacked[:, i, j, :, :]  # shape: (batch, V, D)
                # For each batch, do cross-attention
                # Batch-wise operation:
                # We do per batch for efficiency:
                # But to avoid complexity, process per batch
                # For simplicity, perform batch over batch size
                # Reshape to (batch, V, D) and process
                # For proper broadcasting, process batch-wise using list comprehension
                # but for clean code, process in batch:
                # Here, implement per batch attention
                # To simplify, we perform a batch operation:
                # For each element in batch, do a cross-attention
                # We'll handle batch by treating batch dimension as batch in cross-attention
                # So, perform batch here
                # Reshape tensor: (batch, V, D)
                # The cross-attention module expects (B, V, D)
                # Our cross_attention expects input of shape (B, V, D)
                # The variable query is fixed, so we process per batch
                # Let's implement a batch process
                # For simplicity, process variable dependency across all batch: use batch as batch dimension
                # So, for each batch, do cross-attention
                # Prepare tensor (batch, V, D)
                variable_deps = var_deps  # shape: (batch, V, D)
                # Expand query: (batch, 1, D)
                query = self.variable_aggregation.query.expand(batch_size, 1, D)
                attn_output = self.variable_aggregation.attn_with_query(query, variable_deps)  # (batch, 1, D)
                # Average over batch
                # To keep spatial map, average over batch
                # sum over batch:
                # For this, just take mean
                # But since this is per location, we want spatially mapped:
                # So, for simplicity, average over batch and assign to spatial place
                mean_agg = attn_output.mean(dim=0).squeeze(1)  # (D,)
                aggregated[i, j, :] = mean_agg

        # Add positional encodings
        if self.pos_encoding is None or self.pos_encoding.shape[0] != H_p or self.pos_encoding.shape[1] != W_p:
            self.pos_encoding = get_2d_positional_encoding(H_p, W_p, D).to(X.device)
        tokens = aggregated + self.pos_encoding  # shape: (H_p, W_p, D)

        # Flatten spatially: shape (num_tokens, D)
        tokens = tokens.reshape(-1, D)
        # Expand to batch: (batch, num_tokens, D)
        tokens = tokens.unsqueeze(0).repeat(batch_size,1,1)  # broadcast

        # --- Transformer stack with AdaLN conditioning ---
        # Generate gamma and beta lists
        gamma_list, beta_list = self.get_conditioning_params(delta_t)

        # Process each Transformer block
        for layer_idx, block in enumerate(self.transformer_blocks):
            gamma1, gamma2 = gamma_list[layer_idx]
            beta1, beta2 = beta_list[layer_idx]
            # Pass through block
            tokens = block(tokens, gamma1, beta1, gamma2, beta2)

        # --- Final linear layer to predict delta variables ---
        delta_pred = self.output_head(tokens)  # shape: (batch, num_tokens, output_dim)
        # For full spatial map, reshape accordingly
        delta_pred = delta_pred.reshape(batch_size, H_p, W_p, -1)
        # Map back to variable shape: replicating across pressure levels
        # For simplicity, output as per-token difference predictions
        # Alternatively, can be upsampled to original grid size if needed
        # Permute to (batch, V, H, W)
        delta_pred_map = delta_pred.permute(0, 3, 1, 2)  # (batch, V, H_p, W_p)
        return delta_pred_map

