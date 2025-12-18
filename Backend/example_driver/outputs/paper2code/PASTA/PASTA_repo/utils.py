## utils.py

import re
from typing import List, Tuple, Optional
import torch

def tokenize(text: str, tokenizer) -> List[int]:
    """
    Tokenize the input text into token IDs using the provided tokenizer.
    """
    return tokenizer.encode(text, add_special_tokens=False)

def detokenize(token_ids: List[int], tokenizer) -> str:
    """
    Convert token IDs back into a string using the tokenizer.
    """
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=True)

def extract_emphasis_token_indices(prompt: str, marker: str='*') -> List[int]:
    """
    Extract emphasized token indices from a styled prompt.
    Assumes emphasis markers (e.g., '*') enclose emphasized spans.
    Returns a list of token indices (integers) that are emphasized.
    """
    # Find all emphasized segments (pairwise matches of marker)
    pattern = re.escape(marker) + '(.*?)' + re.escape(marker)
    matches = list(re.finditer(pattern, prompt))
    emphasized_char_spans: List[Tuple[int, int]] = []

    # For each emphasized span, record start and end character positions
    for match in matches:
        start_char = match.start()
        end_char = match.end()
        # The span text without markers
        span_text = match.group(1)
        emphasized_char_spans.append((start_char, end_char))

    # Remove emphasis markers to get clean prompt
    clean_prompt = re.sub(pattern, lambda m: m.group(1), prompt)

    # Tokenize the clean prompt with offset mapping
    # Requires tokenizer to support return_offsets_mapping=True
    # We will assume the caller passes tokenizer and do tokenization here
    # For robustness, check if tokenizer supports it
    # But since this is a utility, we expect to pass tokenizer as an argument
    # Let's implement the core here (to be called outside)
    return list_of_emphasized_token_indices_from_char_spans(clean_prompt, emphasized_char_spans)

def list_of_emphasized_token_indices_from_char_spans(
    prompt_text: str,
    emphasized_char_spans: List[Tuple[int, int]],
    tokenizer=None
) -> List[int]:
    """
    Map character spans to token indices.
    If tokenizer is provided, use it to compute token offsets. Otherwise, return empty.
    """
    token_indices: List[int] = []
    if tokenizer is None:
        return token_indices

    # Obtain token offsets
    encoding = tokenizer(
        prompt_text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    offsets = encoding['offset_mapping']  # list of (start_char, end_char) per token

    # For each emphasized span, find tokens overlapping with span
    for (char_start, char_end) in emphasized_char_spans:
        for idx, (tok_start, tok_end) in enumerate(offsets):
            # Check overlap
            if tok_end <= char_start:
                continue
            if tok_start >= char_end:
                continue
            # Overlap exists
            token_indices.append(idx)

    # Optionally, remove duplicates
    token_indices = sorted(set(token_indices))
    return token_indices

def extract_emphasis_token_indices(prompt: str, tokenizer, marker: str='*') -> List[int]:
    """
    Parses the prompt, extracts emphasized spans marked with `marker`,
    and returns token indices corresponding to emphasized tokens.
    """
    pattern = re.escape(marker) + '(.*?)' + re.escape(marker)
    matches = list(re.finditer(pattern, prompt))
    emphasized_char_spans: List[Tuple[int, int]] = []

    for match in matches:
        start_char = match.start()
        end_char = match.end()
        span_text = match.group(1)
        emphasized_char_spans.append((start_char, end_char))
    # Remove emphasis markers in prompt
    clean_prompt = re.sub(pattern, lambda m: m.group(1), prompt)
    # Get token offsets
    encoding = tokenizer(
        clean_prompt,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    offsets = encoding['offset_mapping']
    token_indices: List[int] = []

    # Map each emphasized char span to token indices
    for (char_start, char_end) in emphasized_char_spans:
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= char_start:
                continue
            if tok_start >= char_end:
                continue
            # Overlap
            if idx not in token_indices:
                token_indices.append(idx)
    token_indices = sorted(set(token_indices))
    return token_indices

def normalize_attention_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Normalize attention scores over the last dimension so rows sum to 1.
    Adds epsilon for numerical stability.
    """
    epsilon = 1e-9
    sums = scores.sum(dim=-1, keepdim=True) + epsilon
    return scores / sums

def scale_attention_scores(
    scores: torch.Tensor,
    emphasis_indices: List[int],
    alpha: float = 0.01
) -> torch.Tensor:
    """
    Reweight attention scores by scaling down non-emphasized token scores.
    Scores: tensor of shape (batch_size, num_heads, seq_len, seq_len)
    emphasis_indices: list of token indices to emphasize
    """
    # Create mask tensor: shape (seq_len,)
    seq_len = scores.shape[-1]
    device = scores.device
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[emphasis_indices] = True
    # Expand mask for each batch and head
    # shape: (1, 1, seq_len)
    # We will broadcast in the next step
    mask_broadcast = mask.unsqueeze(0).unsqueeze(0)  # shape: (1,1,seq_len)
    # For each position in the query sequence (dim=-2)
    # scale scores where the key token is not emphasized
    # Use torch.where for element-wise multiplication
    # Expand for batch and head dims
    # scores shape: (batch_size, num_heads, seq_len_q, seq_len_k)
    # We need to scale scores for non-emphasized key tokens
    scores_scaled = scores.clone()
    # For each position in query, scale scores of non-emphasized tokens
    # Broadcast mask over query positions
    # Create a mask for the key tokens: shape (seq_len,)
    # Expand to (1, 1, 1, seq_len)
    mask_expanded = ~mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # shape: (1,1,1,seq_len)
    # Convert to float for multiplication
    scale_mask = mask_expanded.type(scores.dtype)
    # Multiply corresponding scores by alpha
    scores_scaled = torch.where(
        mask_expanded,
        scores * alpha,
        scores
    )
    # Renormalize
    scores_norm = normalize_attention_scores(scores_scaled)
    return scores_norm
