## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ReversibleIN(nn.Module):
    """
    Implements Reversible Instance Normalization (RevIN) as described in Kim et al. (2021b).
    It normalizes input sequences per batch during training and allows denormalization.
    """
    def __init__(self, feature_dim: int, epsilon: float = 1e-5):
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        # Learnable scale and shift parameters
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.beta = nn.Parameter(torch.zeros(feature_dim))
        # Running mean and std are computed per batch during normalization, no buffers needed

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes mean/std per batch and normalizes input sequence.
        Args:
            x (Tensor): shape (B, L, D) or (B, D, L)
        Returns:
            normalized (Tensor): same shape as input
        """
        # Permute to (B, D, L) if necessary for consistency
        if x.ndim == 3:
            if x.shape[1] != self.feature_dim:
                x = x.transpose(1, 2)  # ensure shape (B, D, L)
            mu = torch.mean(x, dim=(0, 2), keepdim=True)
            sigma = torch.std(x, dim=(0, 2), keepdim=True) + self.epsilon
            self.mu = mu.squeeze(0).squeeze(1)  # (D,)
            self.sigma = sigma.squeeze(0).squeeze(1)  # (D,)
            x_norm = (x - mu) / sigma
            return x_norm.transpose(1, 2)  # back to (B, L, D)
        else:
            raise ValueError("Input tensor must be 3D.")

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalizes the input sequence using stored mu and sigma.
        Args:
            x (Tensor): shape (B, L, D)
        Returns:
            denormalized (Tensor): same shape
        """
        mu = self.mu.unsqueeze(0).unsqueeze(2)  # (1,1,D)
        sigma = self.sigma.unsqueeze(0).unsqueeze(2)  # (1,1,D)
        return x * sigma + mu

    def compute_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and std over batch for normalization.
        """
        mu = torch.mean(x, dim=(0, 1))
        sigma = torch.std(x, dim=(0, 1)) + self.epsilon
        return mu, sigma


class ChannelWiseAttention(nn.Module):
    """
    Implements the channel-wise attention as per Eq. (4), with softmax row-wise.
    Stores the attention matrix for analysis.
    """
    def __init__(self, input_dim: int, proj_dim: int):
        """
        Args:
            input_dim (int): D, number of features/channels
            proj_dim (int): d_qk, dimension of query/key projections
        """
        super().__init__()
        # Spectrally normalized projection matrices for Q and K
        self.W_q = spectral_norm(nn.Parameter(torch.randn(input_dim, proj_dim)))
        self.W_k = spectral_norm(nn.Parameter(torch.randn(input_dim, proj_dim)))
        # V and O are not used explicitly here but can be added for extended versions
        self.attention_matrix = None  # store for analysis

        # Initialize to Xavier uniform to match standard practice
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights and return attention output.
        Args:
            X (Tensor): shape (B, D, L) or (Batch, D, L)
        Returns:
            A_X (Tensor): shape (D, D), the attention matrix (feature to feature)
        """
        # X shape: (B, D, L)
        # Compute Q and K: (B, D, d_qk)
        Q = torch.einsum('bdk,df->bfk', X, self.W_q)  # shape: (B, D, d_qk)
        K = torch.einsum('bdk,df->bfk', X, self.W_k)  # shape: (B, D, d_qk)

        # Compute attention scores: batch-wise
        # For feature-level attention, aggregate over time: (B, D, D)
        # Use the feature dimension D
        # Compute scaled dot-product for each pair of features
        attn_scores = torch.einsum('bfd,bgd->f g', Q, K)  # shape: (D, D), summed over batch
        attn_scores = attn_scores / (Q.shape[-1] ** 0.5)

        # Apply softmax row-wise (over feature dimension g)
        attn_probs = F.softmax(attn_scores, dim=1)  # shape: (D, D)

        # Store for analysis
        self.attention_matrix = attn_probs.detach()

        # The attention output is applied to inputs
        # For residual, we will compute: (X + A(X) X W_V W_O)
        return attn_probs

    def get_attention_matrix(self) -> torch.Tensor:
        """
        Return stored attention matrix for analysis.
        """
        return self.attention_matrix


class SAMTransformer(nn.Module):
    """
    Implements a shallow transformer with channel-wise attention, residual,
    spectral normalization, and RevIN normalization.
    """
    def __init__(self, config: Dict, feat_stats: Dict):
        """
        Args:
            config (Dict): configuration containing model hyperparameters
            feat_stats (Dict): Dictionary containing feature-wise mean/std for normalization
        """
        super().__init__()
        # Extract relevant hyperparameters
        self.d_m = config.get('d_m', 16)
        self.d_qk = config.get('d_qk', 16)
        self.output_dim = config.get('output_dim', 16)  # W and final linear out dimension
        self.input_dim = feat_stats['mean'].shape[0]  # number of features D
        self.epsilon = 1e-5

        # RevIN normalization layer
        self.revin = ReversibleIN(self.input_dim)

        # Attention module
        self.attention = ChannelWiseAttention(self.input_dim, self.d_qk)

        # Final linear layer W: (D, H) where H is prediction horizon or output size
        self.W = spectral_norm(nn.Linear(self.input_dim, self.output_dim))
        nn.init.xavier_uniform_(self.W.weight)

        # Store the last attention matrix for analysis
        self.attn_mtx = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x (Tensor): shape (B, L, D)
        Returns:
            pred (Tensor): shape (B, L, D) or (B, D, H) for prediction
            attn_mat (Tensor): attention matrix D x D
        """
        # Apply RevIN normalization
        x_norm = self.revin.fit_transform(x)  # shape: (B, L, D)
        x_norm = x_norm.transpose(1, 2)  # to shape (B, D, L)

        # Compute attention matrix
        attn_probs = self.attention(x_norm)  # shape: (D, D)
        self.attn_mtx = attn_probs

        # Apply attention to input features
        # attention matrix shape: (D, D)
        # Input shape: (B, D, L)
        # Compute attention-weighted features
        attn_output = torch.einsum('fg,bgd->bfd', attn_probs, x_norm)  # shape: (B, D, L)

        # Residual connection: add original normalized input
        residual = x_norm + attn_output  # shape: (B, D, L)

        # Linear projection to output dimension
        # flatten residual to shape (B, L, D)
        residual = residual.transpose(1, 2)  # (B, L, D)
        # Final linear layer
        pred = self.W(residual)  # shape: (B, L, output_dim)

        # For residual connection consistency, it might be preferable to return the same shape
        # as input or process accordingly. Here, assuming output is (B, L, D)
        # Optionally, adapt the output size as needed.
        pred = pred.transpose(1, 2)  # (B, D, output_dim) if desired
        return pred, attn_probs

    def get_attention(self) -> torch.Tensor:
        """
        Return stored attention matrix for analysis.
        """
        if self.attn_mtx is None:
            raise RuntimeError("Attention matrix is not computed yet. Run forward pass first.")
        return self.attn_mtx
