## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from math import sqrt
from typing import Optional

# Import configuration from the global config module
from config import config

class AttentionMIL(nn.Module):
    """
    Attention-based MIL model processing precomputed features.
    Consists of an attention network to assign relevance weights to instances,
    aggregates weighted features, and predicts a bag-level output.
    """
    def __init__(self,
                 feature_dim: int = 512,
                 hidden_dim: int = 512,
                 dropout: float = 0.0):
        """
        Initializes the AttentionMIL model.
        Args:
            feature_dim (int): Dimensionality of input instance features.
            hidden_dim (int): Hidden dimension size for attention network.
            dropout (float): Dropout probability.
        """
        super(AttentionMIL, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Attention network components: a small MLP
        self.attention_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )

        # Optional: a small bias term for attention logits
        self.bias = nn.Parameter(torch.zeros(1))

        # Final classifier: linear layer to produce scalar prediction
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, K, D)
        Returns:
            torch.Tensor: Bag prediction logits of shape (batch_size, 1)
        """
        # Compute attention scores for each instance
        # shape: (batch_size, K, 1)
        attn_logits = self.attention_layer(x).squeeze(-1) + self.bias
        # Attention weights: softmax over instances
        attn_weights = F.softmax(attn_logits, dim=1)  # shape: (batch_size, K)

        # Weight instances and sum to get bag representation
        bag_rep = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # shape: (batch_size, D)

        # Final prediction
        out = self.classifier(bag_rep)  # shape: (batch_size, 1)
        return out

class TransMIL(nn.Module):
    """
    Transformer-based MIL model processing precomputed features.
    Uses a Transformer encoder with a class token for global context.
    """
    def __init__(self,
                 feature_dim: int = 512,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 hidden_dim: int = 512,
                 dropout: float = 0.1):
        """
        Initializes the TransMIL model.
        Args:
            feature_dim (int): Dimension of input instance features.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Dimension of transformer feedforward layer.
            dropout (float): Dropout probability.
        """
        super(TransMIL, self).__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Class token: learned embedding prepended to sequence
        self.class_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))
        # Positional encoding can be added if necessary (here omitted for simplicity)

        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(d_model=self.feature_dim,
                                                nhead=self.num_heads,
                                                dim_feedforward=self.hidden_dim,
                                                dropout=self.dropout,
                                                activation='relu')
        self.transformer = TransformerEncoder(encoder_layers, num_layers=self.num_layers)

        # Final classification head applied to class token output
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input features of shape (batch_size, K, D)
        Returns:
            torch.Tensor: Bag prediction logits of shape (batch_size, 1)
        """
        batch_size, K, D = x.shape
        # Expand class token for batch
        class_token = self.class_token.expand(batch_size, -1, -1)  # shape: (batch_size, 1, D)

        # Concatenate class token at sequence start
        seq_input = torch.cat([class_token, x], dim=1)  # shape: (batch_size, K+1, D)

        # Transformer expects input shape: (K+1, batch_size, D) or (batch_size, K+1, D)
        # Using batch-first mode if available: default is batch first, so no change needed
        # transformer expects shape: (seq_len, batch, embed_dim)
        seq_input = seq_input.transpose(0, 1)  # shape: (K+1, batch_size, D)

        # Pass through transformer encoder
        encoder_output = self.transformer(seq_input)  # shape: (K+1, batch_size, D)

        # Extract class token output (first token)
        class_token_output = encoder_output[0]  # shape: (batch_size, D)

        # Compute scalar prediction
        out = self.classifier(class_token_output)  # shape: (batch_size, 1)
        return out

class AdditiveMIL(nn.Module):
    """
    Additive MIL model: predicts bag as sum over per-instance predictions.
    The model is inherently interpretable; each instance's score explains contribution.
    """
    def __init__(self,
                 feature_dim: int = 512,
                 hidden_dim: int = 512,
                 dropout: float = 0.0):
        """
        Initialize the AdditiveMIL model.
        Args:
            feature_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer size for instance scoring.
            dropout (float): Dropout probability.
        """
        super(AdditiveMIL, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Instance scoring network: MLP per instance
        self.instance_scorer = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, K, D)
        Returns:
            torch.Tensor: Bag prediction logits of shape (batch_size, 1)
        """
        # Compute per-instance scores
        # shape: (batch_size, K, 1)
        instance_logits = self.instance_scorer(x)  # shape: (batch_size, K, 1)

        # Sum over instances to get bag score
        bag_logits = torch.sum(instance_logits, dim=1)  # shape: (batch_size, 1)

        return bag_logits
