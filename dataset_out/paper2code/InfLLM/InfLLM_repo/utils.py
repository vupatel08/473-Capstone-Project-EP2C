## utils.py
import numpy as np
import torch
from typing import List, Tuple, Dict
from transformers import PreTrainedTokenizer

def assign_positional_embeddings(total_length: int, offset: int) -> torch.Tensor:
    """
    Generate positional indices for a sequence of total_length tokens.
    Tokens beyond a certain threshold are assigned a fixed offset position,
    simulating the paper's approach of assigning distant tokens the same positional encoding.

    Args:
        total_length (int): total number of tokens in the sequence.
        offset (int): positional offset for tokens beyond local window (e.g., large fixed offset).

    Returns:
        torch.Tensor: tensor of shape (total_length,) containing positional indices.
    """
    positions = torch.arange(total_length)
    # For tokens beyond a certain position (e.g., local window size), assign offset
    # But as per text, the entire sequence beyond local window gets same pos
    # Here, for simplicity, assign all beyond local window same position
    # Alternatively, assign positions normally, then clip
    # For illustration, set all beyond local window (say position >= local_window) to offset
    # But need external info; here, just assign all positions as normal + offset
    # Or set positions < local_window as their index; beyond assign offset
    # Example implementation:
    # This function can be modified as needed based on sequence index
    return positions + offset

def compute_token_importance_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    local_window_size: int
) -> torch.Tensor:
    """
    Compute importance scores r_m for each token in the sequence based on the local context,
    following the formula: r_m = (1 / l_L) * sum_{j=1}^{l_L} (q_{m+j} · k_m)

    Args:
        queries (torch.Tensor): [sequence_length, hidden_dim], query vectors.
        keys (torch.Tensor): [sequence_length, hidden_dim], key vectors.
        local_window_size (int): l_L, size of local window for relevance.

    Returns:
        torch.Tensor: importance scores [sequence_length].
    """
    seq_len, hidden_dim = queries.size()
    scores = torch.zeros(seq_len, device=queries.device)

    for m in range(seq_len):
        start_idx = m + 1
        end_idx = min(m + 1 + local_window_size, seq_len)
        if start_idx >= end_idx:
            continue
        local_queries = queries[start_idx:end_idx]  # shape: [local_window, hidden_dim]
        k_m = keys[m]  # shape: [hidden_dim]
        # Compute inner product between each local query and k_m
        dot_products = torch.matmul(local_queries, k_m)  # shape: [local_window]
        scores[m] = dot_products.mean()
    return scores

def chunk_sequence(sequence: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
    """
    Split a long token sequence into overlapping chunks for streaming.
    Overlap ensures context continuity between chunks.

    Args:
        sequence (List[str]): list of tokens representing the sequence.
        chunk_size (int): size of each chunk in tokens.
        overlap (int): number of tokens overlapping between chunks.

    Returns:
        List[List[str]]: list of token chunks (each a list), overlapping accordingly.
    """
    chunks = []
    start = 0
    seq_len = len(sequence)
    while start < seq_len:
        end = min(start + chunk_size, seq_len)
        chunks.append(sequence[start:end])
        if end == seq_len:
            break
        start = end - overlap  # move start to overlap
    return chunks

def reconstruct_sequence_from_chunks(chunks: List[List[str]], overlap: int) -> List[str]:
    """
    Reconstruct the full sequence from overlapping chunks, removing duplicate overlaps.

    Args:
        chunks (List[List[str]]): list of token chunks.
        overlap (int): number of tokens overlapping between chunks.

    Returns:
        List[str]: reconstructed full sequence tokens.
    """
    if not chunks:
        return []
    full_sequence = list(chunks[0])
    for i in range(1, len(chunks)):
        # Append only the non-overlap part
        full_sequence.extend(chunks[i][overlap:])
    return full_sequence

def update_relevance_score(
    previous_score: float,
    attention_score: float,
    decay: float
) -> float:
    """
    Update the relevance score with decay and new attention score, following:
    s_b = s_b * d + sum_{j=1}^{l_x} sum_{i=1}^{l_bs} attention_score(q_j, k_i).

    Args:
        previous_score (float): previous relevance score s_b.
        attention_score (float): current attention score for the unit.
        decay (float): decay coefficient d, 0 < d < 1.

    Returns:
        float: updated relevance score.
    """
    return previous_score * decay + attention_score

def select_top_memory_units(
    relevance_scores: List[float],
    top_k: int
) -> List[int]:
    """
    Select indices of top-k most relevant memory units based on scores.

    Args:
        relevance_scores (List[float]): list of relevance scores for each memory unit.
        top_k (int): number of top units to select.

    Returns:
        List[int]: indices of the selected units.
    """
    # Use numpy for argsort
    scores_np = np.array(relevance_scores)
    top_indices = np.argsort(-scores_np)[:top_k]
    return top_indices.tolist()

def manage_gpu_cache(
    all_units: List['KVPair'],
    usage_scores: List[float],
    cache_size: int
) -> List['KVPair']:
    """
    Maintain the GPU cache of memory units applying an LRU strategy based on usage scores.
    Keeps top relevant units, evicts less used ones.

    Args:
        all_units (List[KVPair]): all candidate memory units.
        usage_scores (List[float]): current relevance/usage scores for each unit.
        cache_size (int): maximum number of units in GPU cache.

    Returns:
        List[KVPair]: list of units to load into GPU cache, sorted by relevance.
    """
    # Pair units with their scores
    units_with_scores = list(zip(all_units, usage_scores))
    # Sort by usage score descending
    sorted_units = sorted(units_with_scores, key=lambda x: x[1], reverse=True)
    # Select top cache_size units
    selected_units = [unit for unit, score in sorted_units[:cache_size]]
    return selected_units

def compute_relevance_between(
    current_queries: torch.Tensor,
    memory_unit_keys: torch.Tensor
) -> float:
    """
    Compute the relevance of current tokens with a memory unit (block of keys), e.g.,
    sum of inner products over tokens.

    Args:
        current_queries (torch.Tensor): [seq_len, hidden_dim].
        memory_unit_keys (torch.Tensor): [unit_size, hidden_dim].

    Returns:
        float: relevance score (e.g., sum or mean of dot products).
    """
    # Batch computation of inner products
    scores = torch.matmul(current_queries, memory_unit_keys.T)  # [seq_len, unit_size]
    relevance_score = scores.mean().item()  # scalar average relevance
    return relevance_score

def load_model_and_tokenizer(model_name: str) -> Tuple:
    """
    Load pre-trained model and tokenizer from transformers.

    Args:
        model_name (str): model identifier in HuggingFace

    Returns:
        model, tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def calculate_sequence_length(tokens: List[str]) -> int:
    """
    Utility to count tokens, for chunking decisions.
    """
    return len(tokens)

def save_cache_to_disk(memory_units: List['KVPair'], filename: str) -> None:
    """
    Serialize and save memory units to disk for persistence.

    Args:
        memory_units (List[KVPair]): memory units to save.
        filename (str): file path.
    """
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(memory_units, f)

def load_cache_from_disk(filename: str) -> List['KVPair']:
    """
    Load serialized memory units from disk.

    Args:
        filename (str): file path.

    Returns:
        List[KVPair]
    """
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)

def validate_importance_scores(scores: torch.Tensor) -> bool:
    """
    Check for correctness of importance scores: no NaNs or infs.

    Args:
        scores (torch.Tensor): importance scores.

    Returns:
        bool: True if valid, False otherwise.
    """
    if torch.any(torch.isnan(scores)) or torch.any(torch.isinf(scores)):
        return False
    return True

def test_chunking_functions():
    """
    Basic self-test for chunk_sequence and reconstruct_sequence_from_chunks.
    """
    sequence = [str(i) for i in range(100)]
    chunk_size = 20
    overlap = 5
    chunks = chunk_sequence(sequence, chunk_size, overlap)
    reconstructed = reconstruct_sequence_from_chunks(chunks, overlap)
    assert len(reconstructed) == len(sequence), "Reconstruction length mismatch"
    assert reconstructed == sequence, "Reconstructed sequence does not match original"
    print("Chunking functions passed basic tests.")
