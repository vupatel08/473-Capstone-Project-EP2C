## attention_wrapper.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from collections import OrderedDict

# Import the configuration parameters from the YAML config if needed
# But per instruction, they are passed or set externally; will set defaults here.

class AttentionWrapper:
    """
    Implements a customized attention mechanism for long sequence processing,
    combining sliding window local attention with relevance-based external memory retrieval.
    Ensures compatibility with HuggingFace transformer models.
    """

    def __init__(
        self,
        embed_dim: int,
        local_window_size: int = 4096,
        memory_block_size: int = 512,
        top_k_representative_tokens: int = 4,
        relevance_decay: float = 0.1,
        cache_memory_size: int = 64,
    ):
        """
        Initialize the attention wrapper with hyperparameters.
        """
        self.embed_dim = embed_dim
        self.local_window_size = local_window_size
        self.memory_block_size = memory_block_size
        self.top_k = top_k_representative_tokens
        self.relevance_decay = relevance_decay
        self.cache_size = cache_memory_size

        # Projection layers for queries, keys, values are assumed to be from the base model
        # Alternatively, they can be passed or integrated; here, we expect to get projected tensors

        # Cache for key-value tensors to speed up attention
        self.cache = None  # Will manage memory relevance separately
        # For simplicity, no internal linear layers here; expect that the queries, keys, values
        # are provided externally from the model's forward pass, or they are integrated outside.

        # Optional: Keep track of cache hit/miss statistics for logging
        self.cache_hits = 0
        self.cache_misses = 0

    def call(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        current_tokens: torch.Tensor,
        memory: 'ExternalMemory',
        queries_proj: Optional[torch.nn.Module] = None,
        keys_proj: Optional[torch.nn.Module] = None,
        values_proj: Optional[torch.nn.Module] = None,
        layer_idx: int = 0
    ) -> torch.Tensor:
        """
        Perform attention with sliding window + memory relevance.

        Args:
            query (torch.Tensor): [batch, seq_len, hidden_dim], the query vectors.
            key (torch.Tensor): [batch, total_len, hidden_dim], all keys (local + memory).
            value (torch.Tensor): [batch, total_len, hidden_dim], all values.
            attention_mask (torch.Tensor): [batch, seq_len, total_len], mask indicating allowed attention.
            current_tokens (torch.Tensor): [batch, seq_len], current token IDs for relevance calculations.
            memory (ExternalMemory): ExternalMemory object providing relevant memory units.
            * Additional projections if needed.

        Returns:
            torch.Tensor: Attention output of shape [batch, seq_len, hidden_dim].
        """

        batch_size, seq_len, hidden_dim = query.size()

        # 1. Compute relevance scores between current tokens and memory units
        relevant_memory_units = memory.retrieve_relevant(current_tokens, self.top_k)
        mem_k: torch.Tensor = relevant_memory_units['keys']  # [total_relevant, block_size, hidden_dim]
        mem_v: torch.Tensor = relevant_memory_units['values']  # [total_relevant, block_size, hidden_dim]
        num_relevant_units = mem_k.size(0)

        # 2. Compute importance scores for tokens in the current block
        #    Here, for simplicity, assume queries and keys are batch aligned
        #    and that the local queries/keys are already provided.
        #    We will perform relevance scoring between current query and memory keys.
        #    Alternatively, compute importance scores per token in batch.

        # For relevance between tokens and memory units, flatten relevant memory for this step
        # Reshape memorized keys to [total_relevant * block_size, hidden_dim]
        flat_mem_k = mem_k.reshape(-1, hidden_dim)  # [total_relevant * block_size, hidden_dim]
        # Similarly, compute relevance per token
        # For efficiency, perform batched matrix multiplications

        # Compute importance scores for each token with each relevant unit
        # Q: [batch, seq_len, hidden_dim], K: [total_relevant * block_size, hidden_dim]
        # We can compute: scores = Q @ K^T
        queries_flat = query.reshape(-1, hidden_dim)  # [batch*seq_len, hidden_dim]
        relevance_scores = torch.matmul(queries_flat, flat_mem_k.T)  # [batch*seq_len, total_relevant*block_size]

        # For each token, compute mean relevance with its local window (simulate importance score)
        # Since the formula in the paper: r_m = mean_{j=1}^{l_L} (q_{m+j} @ k_m)
        # Here, we approximate with relevance between each token's query and corresponding key.
        # For simplicity, we skip the local window averaging, or approximate using available projected layer.

        # 3. Select top-k relevant memory units per token based on relevance scores
        # For computational efficiency, aggregate relevance per unit
        # Sum relevance scores over tokens to get unit relevance
        unit_relevance = relevance_scores.reshape(batch_size, seq_len, -1).mean(dim=1)  # [batch, total_relevant*block_size]
        # Sum over the block tokens in each unit: first, sum relevance over block tokens
        unit_scores_per_unit = []
        for i in range(num_relevant_units):
            start_idx = i * self.memory_block_size
            end_idx = start_idx + self.memory_block_size
            # sum relevance across tokens in this block
            unit_score = relevance_scores[:, start_idx:end_idx].mean(dim=1)  # [batch]
            unit_scores_per_unit.append(unit_score)
        # Stack to shape [batch, num_relevant_units]
        relevant_scores = torch.stack(unit_scores_per_unit, dim=1)  # [batch, num_relevant_units]

        # 4. Use MemoryManager to select top units based on relevance
        # For each batch, select top-k units
        topk_scores, topk_indices = torch.topk(relevant_scores, k=min(self.cache_memory_size, num_relevant_units), dim=1)

        # 5. Retrieve key-value pairs for selected relevant units
        selected_keys_list = []
        selected_values_list = []

        for b in range(batch_size):
            for idx in topk_indices[b]:
                mem_idx = idx.item()
                selected_keys_list.append(mem_k[mem_idx])    # [block_size, hidden_dim]
                selected_values_list.append(mem_v[mem_idx])  # [block_size, hidden_dim]

        selected_keys = torch.stack(selected_keys_list, dim=0)  # [k, block_size, hidden_dim]
        selected_values = torch.stack(selected_values_list, dim=0)  # [k, block_size, hidden_dim]

        # 6. Concatenate local attention context with relevant memory context
        # For local context: use existing local_key: local_value tensors
        # -- Assuming local_key and local_value are computed externally and supplied.
        # For demonstration, assume key and value are already in combined form.
        # Alternatively, key/value should be precomputed; we proceed with provided tensors.

        # 7. Construct combined key, value tensors
        # Concatenate along sequence dimension
        combined_key = torch.cat([key, selected_keys], dim=1)  # [batch, total_len + k*block_size, hidden_dim]
        combined_value = torch.cat([value, selected_values], dim=1)

        # 8. Construct attention mask
        # The mask should block attention outside local window and irrelevant units
        # For simplicity, we keep the existing attention_mask, but augment it

        # 9. Compute scaled dot-product attention
        # Q: [batch, seq_len, hidden_dim]
        # K: [batch, total_len + relevant_units, hidden_dim]
        # V: [batch, total_len + relevant_units, hidden_dim]
        # attention_mask masks disallowed positions (e.g., outside local window, or irrelevant units if needed)
        scores = torch.matmul(query, combined_key.transpose(-2, -1)) / (self.embed_dim ** 0.5)  # [batch, seq_len, total_len + k*block_size]

        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # [batch, seq_len, total_len + k*block_size]
        attn_output = torch.matmul(attn_weights, combined_value)  # [batch, seq_len, hidden_dim]

        # 10. Possibly update cache statistics
        # For logging
        # self.cache_hits += number of cache hits (number of relevant units reused)
        # self.cache_misses += number of cache misses (units not in cache)

        return attn_output

    def compute_importance_scores(
        self,
        block_tokens: torch.Tensor,
        queries: torch.Tensor,
        keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance scores r_m for tokens within a block.
        """
        # block_tokens: [block_size]
        # queries: [chunk_size, hidden_dim]
        # keys: [block_size, hidden_dim]
        # For each token m in block, compute r_m as average over local window
        # For simplicity, we compute r_m = mean over j (query_{m+j} @ k_m)

        # Initialize importance scores
        block_size = block_tokens.size(0)
        r_m = torch.zeros(block_size, device=queries.device)

        for m in range(block_size):
            # local window indices (m+j), clamp to within queries
            start_idx = m
            end_idx = min(m + self.local_window_size, queries.size(0))
            local_queries = queries[start_idx:end_idx]  # [local_window, hidden_dim]
            k_m = keys[m]  # [hidden_dim]

            # Compute dot product, then mean
            dot_products = torch.matmul(local_queries, k_m)  # [local_window]
            r_m[m] = dot_products.mean()

        return r_m

    def construct_attention_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Construct a mask allowing attention within local window.
        """
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start_idx = max(0, i - self.local_window_size)
            end_idx = min(seq_len, i + self.local_window_size + 1)
            mask[i, start_idx:end_idx] = 1
        return mask.unsqueeze(0)  # [1, seq_len, seq_len], batch dimension optional
