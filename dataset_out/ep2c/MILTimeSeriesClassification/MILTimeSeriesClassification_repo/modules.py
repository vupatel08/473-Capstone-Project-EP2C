## modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Implements fixed sinusoidal positional encodings as described in Vaswani et al. (2017).
    Generates a tensor of shape (max_length, d_model) that encodes positions with sine and cosine functions.
    """
    def __init__(self, max_length: int = 1008, d_model: int = 128):
        """
        Initialize PositionalEncoding with maximum sequence length and embedding dimension.
        Args:
            max_length (int): Maximum sequence length (default: 1008).
            d_model (int): Embedding dimension (default: 128).
        """
        super().__init__()
        position = torch.arange(0, max_length, dtype=torch.float32).unsqueeze(1)  # shape: (max_length, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * 
            -(np.log(10000.0) / d_model)
        )  # shape: (d_model/2,)
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        self.register_buffer('pe', pe)  # Not a parameter, but persistent buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to input embeddings.
        Args:
            x (torch.Tensor): Input tensor of shape (t, d_model) or (batch, t, d_model).
        Returns:
            torch.Tensor: Tensor with positional encoding added, same shape as input.
        """
        seq_len = x.size(0) if x.dim() == 2 else x.size(1)
        pe_slice = self.pe[:seq_len, :].to(x.device)  # shape: (seq_len, d_model)
        if x.dim() == 2:
            return x + pe_slice
        elif x.dim() == 3:
            return x + pe_slice.unsqueeze(0)
        else:
            raise ValueError("Input tensor must be 2D or 3D for positional encoding.")

class GAPPooling(nn.Module):
    """
    Implements Global Average Pooling (GAP) for time series embeddings.
    Takes mean over the sequence dimension, outputs a single vector per sample.
    """
    def __init__(self):
        super().__init__()
        # No parameters needed

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings (torch.Tensor): shape (batch, 1, t, d)
        Returns:
            torch.Tensor: pooled (batch, 1, 1, d)
        """
        # Average over dimension=2 (time)
        pooled = torch.mean(embeddings, dim=2, keepdim=True)  # shape: (batch, 1, 1, d)
        return pooled

class AttentionPooling(nn.Module):
    """
    Implements attention-based MIL pooling.
    Computes attention weights for each time point, scales embeddings, and pools via weighted sum.
    """
    def __init__(self, d: int=128, attention_heads: int=1, attention_size: int=8):
        """
        Args:
            d (int): Embedding dimension size.
            attention_heads (int): Number of attention heads (default: 1).
            attention_size (int): Hidden size of attention head (default: 8).
        """
        super().__init__()
        self.attention_heads = attention_heads
        self.attention_size = attention_size
        # Attention network: two linear layers with tanh + sigmoid activations
        self.attn_linear1 = nn.Linear(d, attention_size)
        self.attn_linear2 = nn.Linear(attention_size, attention_heads)
        # Initialize weights
        # Note: initialization can be added if needed

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings (torch.Tensor): shape (batch, 1, t, d)
        Returns:
            torch.Tensor: pooled embedding (batch, 1, 1, d)
            attention weights: (batch, 1, t, 1)
        """
        batch_size, _, t, d = embeddings.shape
        # Compute attention scores
        # Step 1: (batch, t, d)
        z = embeddings.squeeze(1)  # (batch, t, d)
        # Attention network: (batch, t, attention_heads)
        a = torch.tanh(self.attn_linear1(z))  # (batch, t, attention_size)
        a = torch.sigmoid(self.attn_linear2(a))  # (batch, t, attention_heads)
        if self.attention_heads > 1:
            # For simplicity, average over attention heads: shape (batch, t, 1)
            a = a.mean(dim=2, keepdim=True)
        else:
            # Already shape (batch, t, 1)
            pass
        # Attention weights: shape (batch, t, 1)
        a_weights = a  # in range (0,1)

        # Element-wise scaling of embeddings by attention weights
        # Expand attention weights to match embeddings
        a_weights_exp = a_weights.permute(0,2,1)  # shape: (batch, 1, t)
        a_weights_exp = a_weights_exp.unsqueeze(-1)  # shape: (batch, 1, t, 1)
        # scale embeddings
        scaled_embeddings = embeddings * a_weights_exp  # broadcasting
        # Pool over t: sum scaled embeddings
        pooled = torch.sum(scaled_embeddings, dim=2, keepdim=True)  # shape: (batch, 1, 1, d)
        return pooled, a_weights  # provide attention weights for interpretability

class InstancePooling(nn.Module):
    """
    Implements instance-level class predictions per time point, then average to get series prediction.
    """
    def __init__(self, d: int=128, c: int=10):
        """
        Args:
            d (int): Embedding dimension size.
            c (int): Number of classes.
        """
        super().__init__()
        self.classifier = nn.Conv1d(d, c, kernel_size=1)  # per-time-point class scores

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings (torch.Tensor): shape (batch, 1, t, d)
        Returns:
            torch.Tensor: class prediction per series (batch, 1, 1, c)
        """
        # Permute to (batch, d, t) for conv1d
        x = embeddings.squeeze(1).permute(0,2,1)  # shape: (batch, t, d)
        # Apply linear classifier at each time point
        preds = self.classifier(x.permute(0,2,1))  # (batch, c, t)
        preds = preds.permute(0,2,1).unsqueeze(1)  # (batch,1,t,c)
        # Average over time dimension
        preds_mean = preds.mean(dim=2, keepdim=True)  # (batch,1,1,c)
        return preds_mean

class AdditivePooling(nn.Module):
    """
    Combines attention and class predictions: computes attention weights, predicts class per time point,
    then scales class predictions by attention and pools.
    """
    def __init__(self, d: int=128, c: int=10, attention_heads: int=1, attention_size: int=8):
        """
        Args:
            d (int): Embedding size.
            c (int): Number of classes.
            attention_heads (int): Number of attention heads.
            attention_size (int): Hidden size for attention network.
        """
        super().__init__()
        self.attention_heads = attention_heads
        self.attention_size = attention_size
        # Attention network for attention scores: same as AttentionPooling
        self.attn_linear1 = nn.Linear(d, attention_size)
        self.attn_linear2 = nn.Linear(attention_size, attention_heads)
        # Classifier to produce class scores per time point
        self.classifier = nn.Conv1d(d, c, kernel_size=1)

    def forward(self, embeddings: torch.Tensor):
        """
        Args:
            embeddings (torch.Tensor): shape (batch, 1, t, d)
        Returns:
            pooled series prediction: (batch, 1, 1, c)
            Also returns scaled class predictions per time point for interpretability
        """
        batch_size, _, t, d = embeddings.shape
        # Compute attention scores
        z = embeddings.squeeze(1)  # (batch, t, d)
        a = torch.tanh(self.attn_linear1(z))  # (batch, t, attention_size)
        a = torch.sigmoid(self.attn_linear2(a))  # (batch, t, attention_heads)
        if self.attention_heads > 1:
            a = a.mean(dim=2, keepdim=True)  # (batch, t, 1)
        else:
            # Already shape (batch, t, 1)
            pass
        a_weights = a  # attention scores in [0,1]
        # Compute class predictions at each time point
        class_scores = self.classifier(z.permute(0,2,1))  # (batch, c, t)
        class_scores = class_scores.permute(0,2,1).unsqueeze(1)  # (batch,1,t,c)
        # Scale class scores by attention weights
        a_exp = a_weights.permute(0,2,1).unsqueeze(-1)  # (batch,1,t,1)
        scaled_preds = class_scores * a_exp  # scale per class
        # Pool over time for final prediction
        pooled_preds = scaled_preds.mean(dim=2, keepdim=True)  # (batch,1,1,c)
        return pooled_preds, scaled_preds, a_weights

class ConjunctivePooling(nn.Module):
    """
    Implements the Conjunctive MIL pooling:
    -- Attention head to produce attention scores per time point.
    -- Class prediction head producing class scores per time point.
    -- Multiply class scores by attention scores element-wise, then average.
    """
    def __init__(self, d: int=128, c: int=10, attention_heads: int=1, attention_size: int=8):
        """
        Args:
            d (int): Embedding size.
            c (int): Number of classes.
            attention_heads (int): Number of attention heads.
            attention_size (int): Hidden layer size in attention network.
        """
        super().__init__()
        self.attention_heads = attention_heads
        self.attention_size = attention_size
        # Attention network for attention scores
        self.attn_linear1 = nn.Linear(d, attention_size)
        self.attn_linear2 = nn.Linear(attention_size, attention_heads)
        # Classifier head for class scores per time point
        self.classifier = nn.Conv1d(d, c, kernel_size=1)

    def forward(self, embeddings: torch.Tensor):
        """
        Args:
            embeddings (torch.Tensor): shape (batch, 1, t, d)
        Returns:
            pooled prediction (batch, 1, 1, c)
        """
        batch_size, _, t, d = embeddings.shape
        z = embeddings.squeeze(1)  # (batch, t, d)
        # Attention head
        attn_a = torch.tanh(self.attn_linear1(z))  # (batch, t, attention_size)
        attn_a = torch.sigmoid(self.attn_linear2(attn_a))  # (batch, t, attention_heads)
        if self.attention_heads > 1:
            attn_a = attn_a.mean(dim=2, keepdim=True)  # (batch, t, 1)
        else:
            # shape: (batch, t, 1)
            pass
        attn_scores = attn_a  # (batch, t, 1)

        # Class predictions at each time point
        class_preds = self.classifier(z.permute(0,2,1))  # (batch, c, t)
        class_preds = class_preds.permute(0,2,1).unsqueeze(1)  # (batch,1,t,c)

        # Element-wise multiply class predictions by attention scores
        attn_exp = attn_scores.permute(0,2,1).unsqueeze(-1)  # (batch,1,t,1)
        class_preds_scaled = class_preds * attn_exp  # shape: (batch,1,t,c)

        # Pool over time: mean across t
        pooled_preds = class_preds_scaled.mean(dim=2, keepdim=True)  # (batch,1,1,c)
        return pooled_preds

