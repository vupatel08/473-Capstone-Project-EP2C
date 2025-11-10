"""
utils.py

This module provides utility functions for the TruthX method implementation.
It includes helper functions for:
    1. Token matching: extracting common tokens or their indices from paired answers.
    2. Attention fusion: a custom attention mechanism to fuse semantic (h_sem) and
       truthful (h_truth) latent representations.
    3. Loss functions: reconstruction (MSE), contrastive (InfoNCE-style) and editing losses.
    4. A cosine similarity function with numerical stability.

All functions use explicit type hints and default values where needed. Configuration
parameters such as the temperature for contrastive loss are set with default values but
can be overridden via function arguments.
    
Author: [Your Name]
Date: [Today's Date]
"""

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def get_common_indices(seq1: List[str], seq2: List[str]) -> Tuple[List[int], List[int]]:
    """
    For two sequences of tokens (as strings), compute the indices of tokens
    that are common to both sequences.

    Args:
        seq1 (List[str]): First sequence of tokens.
        seq2 (List[str]): Second sequence of tokens.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing two lists:
            - indices1: The indices in seq1 where the token is also present in seq2.
            - indices2: The indices in seq2 where the token is also present in seq1.
    Example:
        >>> seq1 = ["the", "quick", "brown", "fox"]
        >>> seq2 = ["quick", "fox", "jumps"]
        >>> get_common_indices(seq1, seq2)
        ([1, 3], [0, 1])
    """
    common_tokens = set(seq1).intersection(set(seq2))
    indices1 = [i for i, token in enumerate(seq1) if token in common_tokens]
    indices2 = [i for i, token in enumerate(seq2) if token in common_tokens]
    return indices1, indices2


def attention_fusion(query: Tensor, key: Tensor, scaling: bool = True) -> Tensor:
    """
    Computes an attention-based fusion between query (typically h_sem) and key (typically h_truth).
    The fusion is computed as:
         fusion = query + Attn(query, key)
    where the attention output is computed with scaled dot-product attention.

    Args:
        query (Tensor): Tensor of shape (..., d_latent), representing the semantic latent codes.
        key (Tensor): Tensor of shape (..., d_latent), representing the truthful latent codes.
        scaling (bool): If True, scales the dot product by sqrt(d_latent). Default: True.

    Returns:
        Tensor: The attention output tensor of the same shape as query.
    
    Note:
        This function supports inputs of shape (B, d) or (B, T, d).
    """
    d_latent = query.size(-1)
    # Compute dot-product attention scores
    # scores shape: (..., query_length, key_length)
    scores = torch.matmul(query, key.transpose(-2, -1))
    if scaling:
        scores = scores / math.sqrt(d_latent)
    # Normalize scores to probabilities along the key dimension
    attn_weights = F.softmax(scores, dim=-1)
    # Compute weighted sum of key vectors; shape will match query
    attn_output = torch.matmul(attn_weights, key)
    return attn_output


def mse_reconstruction_loss(x: Tensor, x_prime: Tensor) -> Tensor:
    """
    Computes the Mean Squared Error (MSE) loss between the original hidden representation x
    and its reconstruction x_prime.

    Args:
        x (Tensor): Original hidden representation, shape (..., d_model).
        x_prime (Tensor): Reconstructed representation, same shape as x.

    Returns:
        Tensor: Scalar tensor representing the MSE loss.
    """
    return F.mse_loss(x_prime, x)


def cosine_similarity(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Computes the cosine similarity between two tensors along the last dimension.
    
    Args:
        a (Tensor): Tensor of shape (..., d).
        b (Tensor): Tensor of shape (..., d).
        eps (float): Small constant to prevent division by zero. Default: 1e-8.
    
    Returns:
        Tensor: Tensor of cosine similarities with shape equal to the broadcasted shape of a and b,
                excluding the last dimension.
    """
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=-1)


def contrastive_loss(anchor: Tensor,
                     positives: Tensor,
                     negatives: Tensor,
                     temperature: float = 0.1,
                     eps: float = 1e-8) -> Tensor:
    """
    Computes a contrastive loss (InfoNCE-style) for a batch of anchor representations.
    
    For each anchor vector, this function computes:
         loss = -log ( sum(exp(sim(anchor, pos)/temperature)) /
                      (sum(exp(sim(anchor, pos)/temperature)) + sum(exp(sim(anchor, neg)/temperature)) )
    
    Args:
        anchor (Tensor): Anchor representations of shape (B, d).
        positives (Tensor): Positive sample representations of shape (B, N_pos, d).
        negatives (Tensor): Negative sample representations of shape (B, N_neg, d).
        temperature (float): Temperature scaling factor. Default: 0.1.
        eps (float): Small constant to prevent numerical issues. Default: 1e-8.
    
    Returns:
        Tensor: Scalar tensor representing the averaged contrastive loss over the batch.
    """
    # Expand anchor to (B, 1, d) to broadcast with positives and negatives
    anchor_expanded = anchor.unsqueeze(1)  # Shape: (B, 1, d)
    # Compute cosine similarities for positives and negatives
    pos_sim = cosine_similarity(anchor_expanded.expand_as(positives), positives, eps=eps)  # (B, N_pos)
    neg_sim = cosine_similarity(anchor_expanded.expand_as(negatives), negatives, eps=eps)  # (B, N_neg)
    
    # Scale similarities by temperature
    pos_sim_scaled = pos_sim / temperature  # (B, N_pos)
    neg_sim_scaled = neg_sim / temperature  # (B, N_neg)
    
    # Compute numerator and denominator of the InfoNCE loss
    numerator = torch.exp(pos_sim_scaled).sum(dim=1)  # (B,)
    denominator = numerator + torch.exp(neg_sim_scaled).sum(dim=1)  # (B,)
    
    # Compute log probability and loss
    loss_per_sample = -torch.log(numerator / (denominator + eps) + eps)  # (B,)
    loss = loss_per_sample.mean()
    return loss


def editing_loss(x_target: Tensor, x_swapped: Tensor) -> Tensor:
    """
    Computes the editing loss as the Mean Squared Error (MSE) between the target hidden representation
    and the reconstruction obtained after swapping the truthful latent representations.
    
    Args:
        x_target (Tensor): The ground truth hidden representation (from either a truthful or untruthful sample).
                           Shape: (..., d_model)
        x_swapped (Tensor): The reconstructed hidden representation obtained after swapping latent codes.
                            Shape: same as x_target.
    
    Returns:
        Tensor: Scalar tensor representing the MSE editing loss.
    """
    return F.mse_loss(x_swapped, x_target)
