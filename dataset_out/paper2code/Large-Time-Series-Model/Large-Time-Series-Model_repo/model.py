## model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    """Single decoder (self-attention + feedforward) layer with causal masking."""
    def __init__(self, size: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(size, size * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(size * 4, size)
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention with causal mask
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_output)
        # Feedforward network
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + ff_output)
        return x

class TimerTransformer(nn.Module):
    """
    GPT-style decoder-only transformer for large time series modeling.
    
    Supports:
    - Hierarchical scaling via model size and layers
    - Autoregressive next token prediction
    - Optional position and timestamp embeddings
    """
    def __init__(self, 
                 size: int = 512,                     # Hidden dimension D
                 num_layers: int = 6,                 # Number of decoder layers L
                 num_heads: int = 8,                  # Attention heads
                 ff_dim_multiplier: int = 2,          # FFN dimension multiplier
                 max_position_embeddings: int = 1024, # Max sequence length
                 dropout: float = 0.1,                # Dropout probability
                 input_token_length: int = 96,        # Token length S
                 use_positional_embedding: bool = True,
                 use_timestamp_embedding: bool = True):
        super().__init__()
        self.size = size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_position_embeddings
        self.input_token_length = input_token_length
        self.use_pos_emb = use_positional_embedding
        self.use_time_emb = use_timestamp_embedding
        self.ff_dim = size * ff_dim_multiplier
        self.dropout = dropout

        # Token embedding layer (mapping token IDs or continuous vectors)
        # Here, for implementation, we assume input tokens are continuous vectors,
        # so embedding layer is not required. For discrete tokens, implement nn.Embedding.
        # Let's assume we process embeddings outside; so in forward, input is already embedding.
        # For simplicity, we'll provide linear projection later if needed.

        # Positional Embedding
        if self.use_pos_emb:
            self.pos_embedding = nn.Embedding(self.max_seq_len, size)
        else:
            self.pos_embedding = None

        # Timestamp embedding (optional)
        if self.use_time_emb:
            self.timing_embedding = nn.Embedding(self.max_seq_len, size)
        else:
            self.timing_embedding = None

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(size, num_heads, dropout) for _ in range(num_layers)
        ])

        # Final linear decoder: project hidden states to token vectors (regression)
        # Since tokens are continuous values, output is of dimension equal to token dimension S
        # e.g., predicting S-dimensional token vectors directly
        self.output_dim = self.input_token_length
        self.decoder_head = nn.Linear(size, self.output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for positional embeddings and decoder head."""
        if self.pos_embedding is not None:
            nn.init.uniform_(self.pos_embedding.weight, -0.02, 0.02)
        if self.timing_embedding is not None:
            nn.init.uniform_(self.timing_embedding.weight, -0.02, 0.02)
        nn.init.xavier_uniform_(self.decoder_head.weight)
        if self.decoder_head.bias is not None:
            nn.init.zeros_(self.decoder_head.bias)

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for self-attention (size: seq_len x seq_len)."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(next(self.parameters()).device)

    def forward(self, x: torch.Tensor, positional_ids: Optional[torch.Tensor] = None, timestamp_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input token representations, shape (batch_size, seq_len, embed_dim)
            positional_ids (torch.Tensor): Optional positional indices, shape (batch_size, seq_len)
            timestamp_ids (torch.Tensor): Optional timestamp indices, shape (batch_size, seq_len)
        Returns:
            logits (torch.Tensor): Predicted token vectors, shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Add positional embeddings
        if self.use_pos_emb and self.pos_embedding is not None:
            if positional_ids is None:
                positional_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(positional_ids)  # (batch_size, seq_len, size)
        else:
            pos_emb = 0

        # Add timestamp embeddings
        if self.use_time_emb and self.timing_embedding is not None:
            if timestamp_ids is None:
                timestamp_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            time_emb = self.timing_embedding(timestamp_ids)
        else:
            time_emb = 0

        # Add embeddings
        x = x + pos_emb + time_emb
        # Initial input: assume x is already embedded if continuous, else embed here
        # You may add an input embedding layer if input tokens are discrete indices.

        # Create causal mask for attention
        attn_mask = self._generate_causal_mask(seq_len)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, attn_mask)

        # Project final hidden states to produce output token vectors
        output = self.decoder_head(x)  # (batch_size, seq_len, output_dim)
        return output

    def generate(self, 
                 past_tokens: torch.Tensor, 
                 max_new_tokens: int) -> torch.Tensor:
        """
        Generate sequence autoregressively.
        Args:
            past_tokens (torch.Tensor): Initial token embedding sequence, shape (batch_size, seq_len, embed_dim)
            max_new_tokens (int): Max tokens (timesteps) to generate.
        Returns:
            generated (torch.Tensor): Generated token vectors, shape (batch_size, seq_len + max_new_tokens, output_dim)
        """
        batch_size, seq_len, _ = past_tokens.shape
        device = past_tokens.device
        current_seq = past_tokens

        for _ in range(max_new_tokens):
            # Build positional and timestamp IDs for current sequence: assume incremental
            if self.use_pos_emb:
                positional_ids = torch.arange(current_seq.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
            else:
                positional_ids = None
            if self.use_time_emb:
                # For generation, timestamps increase sequentially; placeholder set to zeros or use actual timestamps
                timestamp_ids = torch.zeros_like(positional_ids, dtype=torch.long)
            else:
                timestamp_ids = None

            # Forward pass: get predictions
            pred = self.forward(current_seq, positional_ids, timestamp_ids)
            # Take last token prediction
            next_token_vec = pred[:, -1, :]  # (batch_size, output_dim)
            # Append predicted token vector as embedding for next step
            # Since we're predicting continuous vectors, no embedding lookup is needed.
            # Expand to shape (batch_size, 1, output_dim)
            next_token = next_token_vec.unsqueeze(1)
            # For consistency with input shape, create an embedding-like tensor
            # Here, we just treat the output vector as the embedding for next token
            current_seq = torch.cat([current_seq, next_token], dim=1)

        return current_seq

