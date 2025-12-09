# loss.py
"""
This module implements multiple loss functions central to CARE—namely, the contrastive (InfoNCE) loss,
the equivariance (angle preservation) loss, the uniformity loss, and their combined form.
All losses should be differentiable and compatible with batch processing to enable efficient training.
"""

import torch
import torch.nn.functional as F
import yaml

# Load configuration for hyperparameters
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract relevant hyperparameters with defaults if not specified
TEMPERATURE_NCE = config.get('training', {}).get('temperature_infonce', 0.5)
LAMBDA_EQUIV = config.get('training', {}).get('lambda_equiv', 0.001)
USE_INVARIANCE = config.get('loss', {}).get('invariance', True)
USE_UNIFORMITY = config.get('loss', {}).get('uniformity', True)

def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMPERATURE_NCE) -> torch.Tensor:
    """
    Computes the InfoNCE contrastive loss for a batch of positive pairs.
    Args:
        z1 (torch.Tensor): Embeddings of first views (batch_size, embedding_dim), assumed normalized.
        z2 (torch.Tensor): Embeddings of second views (batch_size, embedding_dim), assumed normalized.
        temperature (float): Temperature scaling hyperparameter.
    Returns:
        torch.Tensor: Scalar loss value.
    """
    batch_size = z1.shape[0]
    # Compute pairwise cosine similarity scaled by temperature
    similarity_matrix = torch.matmul(z1, z2.t()) / temperature  # shape: (batch_size, batch_size)

    # For stability, subtract max
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    # Labels: diagonal elements are the positive pairs
    labels = torch.arange(batch_size, device=z1.device)

    loss = F.cross_entropy(logits, labels)
    return loss

def equivariance_loss(z_x: torch.Tensor, z_a_x: torch.Tensor) -> torch.Tensor:
    """
    Computes the equivariance loss that enforces the angle-preserving condition:
    inner products of augmented embeddings should match those of original embeddings.
    Args:
        z_x (torch.Tensor): Embeddings of original inputs (batch_size, embedding_dim), assumed normalized.
        z_a_x (torch.Tensor): Embeddings of augmented inputs (batch_size, embedding_dim), assumed normalized.
    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Since embeddings are normalized, inner product is cosine similarity
    # Compute inner products
    # Shape: (batch_size, batch_size)
    inner_ori = torch.mm(z_x, z_x.t())  # similarity between original embeddings
    inner_aug = torch.mm(z_a_x, z_a_x.t())  # similarity between augmented embeddings

    # Our loss: mean squared difference between inner products
    loss = F.mse_loss(inner_aug, inner_ori)
    return loss

def uniformity_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Computes the uniformity loss to prevent collapse.
    Encourages embeddings to be uniformly distributed on the sphere.
    Args:
        z (torch.Tensor): Embeddings (batch_size, embedding_dim), assumed normalized.
    Returns:
        torch.Tensor: Scalar loss value.
    """
    similarity_matrix = torch.matmul(z, z.t())  # shape: (batch_size, batch_size)
    # Exclude diagonal to avoid trivial zero differences
    mask = ~torch.eye(z.size(0), dtype=bool, device=z.device)
    sims = similarity_matrix[mask]
    # Compute the mean of exponentiated similarities
    exp_sims = torch.exp(sims)
    mean_exp = torch.mean(exp_sims)
    # Use negative log to encourage spread
    loss = -torch.log(mean_exp + 1e-8)
    return loss

def compute_total_loss(z1: torch.Tensor, z2: torch.Tensor,
                       z_x: torch.Tensor, z_a_x: torch.Tensor,
                       mode: str = 'train') -> torch.Tensor:
    """
    Computes the total CARE loss combining contrastive, invariance, uniformity, and equivariance.
    Parameters:
        z1, z2 (torch.Tensor): Representations for contrastive loss.
        z_x, z_a_x (torch.Tensor): Representations for equivariance loss.
        mode (str): Mode of training; can be 'train' or 'eval', determines loss components.
    Returns:
        torch.Tensor: Scalar total loss.
    """
    # Contrastive (InfoNCE) loss
    loss_infonc = contrastive_loss(z1, z2, temperature=TEMPERATURE_NCE)

    # Equivariance loss (angle preservation)
    loss_equiv = equivariance_loss(z_x, z_a_x)

    # Invariance loss: optional, encourage similarity of original and augmented within batch
    # Implemented as mean squared difference or cosine similarity
    # Here, we use cosine similarity for invariance measure
    loss_invar = None
    if USE_INVARIANCE:
        # The typical invariance loss is encouraging f(a(x)) ≈ f(x)
        # For simplicity, we can define as mean squared difference or negative cosine similarity
        # But in the paper, it is more like an invariance term encouraging f(a(x)) ≈ f(x)
        # For numerical stability, use cosine similarity
        loss_invar = 1 - torch.mean(torch.sum(z_x * z_a_x, dim=1))
        # Alternatively, implement as MSE between normalized vectors if desired
    else:
        loss_invar = torch.tensor(0.0, device=z1.device)

    # Uniformity loss
    loss_uniform = torch.tensor(0.0, device=z1.device)
    if USE_UNIFORMITY:
        # On the set of embeddings z (either z1, z2, or combined)
        # Here, compound all embeddings for simplicity
        z_all = torch.cat([z1, z2], dim=0)
        loss_uniform = uniformity_loss(z_all)
    else:
        loss_uniform = torch.tensor(0.0, device=z1.device)

    # Final total loss
    total = loss_invar + loss_uniform + LAMBDA_EQUIV * loss_equiv

    return total
