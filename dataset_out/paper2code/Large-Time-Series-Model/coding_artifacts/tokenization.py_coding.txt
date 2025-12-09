# tokenization.py
"""
The Tokenizer class is responsible for transforming raw multivariate or univariate time series data 
into a sequence of tokens suitable for large-scale autoregressive training of the Timer model. 
It supports hierarchical segmentation levels, continuous token embedding facilitation, optional timestamp 
embedding incorporation, and handles heterogeneity across datasets and variables.
"""

import numpy as np
from typing import List, Optional, Tuple
import math

class Token:
    """
    Represents a single token extracted from a time series segment,
    optionally with timestamp metadata.
    """
    def __init__(self, values: np.ndarray, token_id: int, timestamp: Optional[float] = None):
        """
        Args:
            values (np.ndarray): The feature vector of the token (shape: (S, V) or (S,))
            token_id (int): The unique ID of this token.
            timestamp (Optional[float]): Starting timestamp associated with this segment.
        """
        self.values = values
        self.token_id = token_id
        self.timestamp = timestamp

class Sequence:
    """
    Represents a sequence of tokens ready for embedding and model input.
    """
    def __init__(self, tokens: List[Token]):
        """
        Args:
            tokens (List[Token]): Ordered list of Token objects.
        """
        self.tokens = tokens

    def to_id_tensor(self) -> Tuple[torch.Tensor, List[Optional[float]]]:
        """
        Converts tokens to a tensor of token IDs for model input,
        and returns associated timestamps.
        Returns:
            ids (torch.Tensor): Shape (N,), sequence of token IDs.
            timestamps (List[Optional[float]]): Corresponding list of timestamps.
        """
        import torch
        ids = torch.tensor([t.token_id for t in self.tokens], dtype=torch.long)
        timestamps = [t.timestamp for t in self.tokens]
        return ids, timestamps

class Tokenizer:
    """
    Handles hierarchical segmentation, encoding, and embedding of time series data.
    """
    def __init__(self, 
                 hierarchy_levels: List[str] = ['small', 'medium', 'large'],
                 segment_lengths: List[int] = [96, 672, 1440],
                 use_timestamps: bool = True,
                 max_sequence_length: int = 1440,
                 embedding_dim: int = 512):
        """
        Args:
            hierarchy_levels (List[str]): Names or identifiers for levels of hierarchy.
            segment_lengths (List[int]): Corresponding segment lengths for each hierarchy level.
            use_timestamps (bool): Whether to include timestamp info as embeddings.
            max_sequence_length (int): Max tokens per sequence to handle large datasets.
            embedding_dim (int): Embedding size consistent with model input.
        """
        self.hierarchy_levels = hierarchy_levels
        self.segment_lengths = segment_lengths
        self.use_timestamps = use_timestamps
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim

        # Initialize learned embedding weights for tokens
        # For continuous tokens, we'll have a learnable embedding layer
        # Here, for simplicity, we implement as a dictionary or via numpy (in practice, use nn.Embedding)
        self.token_embedding_table = None  # Placeholder for embedding lookup

    def _initialize_embeddings(self, vocab_size: int):
        """
        Initialize embedding weights for tokens. For continuous tokens, this could be a learned lookup.
        """
        import torch
        self.token_embedding_table = torch.nn.Embedding(vocab_size, self.embedding_dim)

    def _segment_series(self, series: np.ndarray, length: int, step: int = None) -> List[np.ndarray]:
        """
        Segment a series into non-overlapping or overlapping windows.
        Args:
            series (np.ndarray): Series data, shape (T, V) or (T,).
            length (int): Segment length.
            step (int): Step size between segments; default is length for non-overlapping.
        Returns:
            List of segmented array slices.
        """
        T = series.shape[0]
        if step is None:
            step = length
        segments = []
        for start_idx in range(0, T - length + 1, step):
            segments.append(series[start_idx:start_idx + length])
        return segments

    def _normalize_series(self, series: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Apply normalization to a series.
        """
        return (series - mean) / (std + 1e-8)

    def _interpolate_missing(self, series: np.ndarray, timestamps: Optional[np.ndarray]) -> np.ndarray:
        """
        Fill missing values in series via linear interpolation.
        """
        # If series has NaNs, interpolate
        if np.isnan(series).any():
            if timestamps is None:
                # Regular series: interpolate directly
                for col in range(series.shape[1]) if series.ndim > 1 else range(series.shape[0]):
                    arr = series[:, col] if series.ndim > 1 else series
                    indices = np.arange(series.shape[0])
                    valid_mask = ~np.isnan(arr)
                    if valid_mask.sum() > 1:
                        arr[np.isnan(arr)] = np.interp(indices[np.isnan(arr)], indices[valid_mask], arr[valid_mask])
                    if series.ndim > 1:
                        series[:, col] = arr
                    else:
                        series = arr
            else:
                # Irregular sampling: interpolate along timestamps
                for col in range(series.shape[1]) if series.ndim > 1 else range(series.shape[0]):
                    arr = series[:, col] if series.ndim > 1 else series
                    valid_mask = ~np.isnan(arr)
                    if valid_mask.sum() > 1:
                        arr[np.isnan(arr)] = np.interp(timestamps[np.isnan(arr)], timestamps[valid_mask], arr[valid_mask])
                    if series.ndim > 1:
                        series[:, col] = arr
                    else:
                        series = arr
        return series

    def tokenize_series(self, 
                        series: np.ndarray, 
                        timestamp: Optional[float] = None,
                        hierarchy_level: str = 'small') -> List[Token]:
        """
        Segment a series into tokens, with optional timestamp embedding.
        Args:
            series (np.ndarray): Series data shape (T, V) or (T,)
            timestamp (Optional[float]): Starting timestamp for series, if present.
            hierarchy_level (str): The hierarchy level (small, medium, large).
        Returns:
            List[Token]: Sequence of tokens from the series.
        """
        # Select segmentation length based on hierarchy
        try:
            level_idx = self.hierarchy_levels.index(hierarchy_level)
        except ValueError:
            level_idx = 0  # default to first if unknown
        seg_length = self.segment_lengths[min(level_idx, len(self.segment_lengths)-1)]

        # Segment the series
        tokens = []
        series_T = series.shape[0]
        start_idx = 0
        while start_idx + seg_length <= series_T:
            segment = series[start_idx:start_idx + seg_length]
            # For univariate: shape (S,)
            # for multivariate: shape (S, V)
            # For token_id: treat as continuous vector (append as float array)
            token_id = self._compute_token_id(segment)
            token_timestamp = timestamp + start_idx if timestamp is not None else None
            tokens.append(Token(values=segment, token_id=token_id, timestamp=token_timestamp))
            start_idx += seg_length
        return tokens

    def _compute_token_id(self, segment: np.ndarray) -> int:
        """
        Map a segment to a token ID.
        For simplicity, we can hash the segment's bytes or compute a hash of the flattened array.
        Alternatively, maintain a learned embedding layer for continuous tokens.
        """
        # Hash the flattened segment bytes
        array_bytes = segment.tobytes()
        # Use Python's hash function or a deterministic hash
        token_hash = hash(array_bytes)
        return token_hash & 0xffffffff  # Convert to 32-bit int

    def convert_series_to_sequence(self, 
                                   series: np.ndarray, 
                                   timestamp: Optional[float] = None,
                                   hierarchy_level: str = 'small') -> Sequence:
        """
        Convert a full series into a sequence of tokens (a 'sentence').
        Args:
            series (np.ndarray): Series data, shape (T, V)
            timestamp (Optional[float]): Starting timestamp for the series
            hierarchy_level (str): The segmentation level
        Returns:
            Sequence: The sequence object containing tokens
        """
        tokens = self.tokenize_series(series=series, timestamp=timestamp, hierarchy_level=hierarchy_level)
        return Sequence(tokens)

    def build_embedding_table(self, vocab_size: int):
        """
        Initialize embedding table for token IDs, if using learned embedding
        """
        import torch
        self._initialize_embeddings(vocab_size)

    def get_token_id(self, token: np.ndarray) -> int:
        """
        Generate or lookup token ID for a segment
        """
        return self._compute_token_id(token)

    def embed_token(self, token: np.ndarray) -> torch.Tensor:
        """
        Convert a token into an embedding vector, using learned embedding.
        """
        import torch
        if self.token_embedding_table is None:
            raise RuntimeError("Embedding table not initialized. Call build_embedding_table().")
        token_id = self._compute_token_id(token)
        token_id_mod = token_id % self.token_embedding_table.num_embeddings
        return self.token_embedding_table.weight[token_id_mod]

    def process_series_for_model(self, 
                                 series: np.ndarray, 
                                 timestamp: Optional[float] = None, 
                                 hierarchy_level: str = 'small') -> Tuple[torch.Tensor, List[Optional[float]]]:
        """
        Convert a raw series into sequence IDs and timestamp list for model input
        """
        sequence = self.convert_series_to_sequence(series, timestamp, hierarchy_level)
        ids, timestamps = sequence.to_id_tensor()
        return ids, timestamps

# Note:
# - The actual implementation for embedding management (e.g., torch.nn.Embedding) should be part of
#   the larger model code, but here we provide a placeholder interface.
# - For continuous tokens, in practice, integrate with a learned embedding layer or vector quantization.
# - Handling of irregular timestamps and variable sampling rates may require resampling beforehand.
# - This class provides core logic; integration with the overall pipeline (training, inference) will use
#   process_series_for_model() and sequence construction methods.
```