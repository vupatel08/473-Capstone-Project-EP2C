"""
model.py

This module implements the TruthXModel class for enhancing the truthfulness of 
large language models (LLMs) by editing their internal representations in a 
dedicated "truthful space" using an autoencoder. The autoencoder decouples the 
LLM's hidden state into a truthful component and a semantic component, fuses 
them via an attention mechanism, and reconstructs the original representation 
through a decoder. Additionally, the model provides methods to compute a global 
truth-editing direction and to edit internal representations using this direction.

The architecture details are:
  - Two encoder modules (TruthEnc and SemEnc): two-layer MLPs mapping x ∈ ℝ^(d_model) 
    to latent representations h_truth and h_sem (each of dimension latent_dim).
  - Fusion mechanism: fuses h_sem and h_truth via dot-product attention 
    (implemented by the attention_fusion helper from utils.py) as:
        fused = h_sem + Attn(h_sem, h_truth)
  - Decoder: a two-layer MLP mapping the fused latent representation back to ℝ^(d_model).

Core methods include:
  - compute_latents(x): returns (h_truth, h_sem)
  - fuse_latents(h_sem, h_truth): performs the attention-based fusion.
  - decode(fused): reconstructs x from fused latent code.
  - forward(x): autoencoder forward pass; returns (x_recon, h_truth, h_sem).
  - compute_edit_direction(truthful_latents, untruthful_latents): computes δ 
    as the difference between the means of truthful and untruthful latent codes.
  - edit_representation(x, delta, alpha): edits the representation x by converting 
    δ from latent space to an additive perturbation Δ in the original representation 
    space and returns x̂ = x + α * Δ.

Configuration parameters (e.g., d_model, encoder_dims, decoder_dims, latent_dim) 
are read from the provided configuration dictionary, typically loaded from config.yaml.

Author: [Your Name]
Date: [Today's Date]
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the attention fusion helper function from utils.py
from utils import attention_fusion


class TruthXModel(nn.Module):
    """
    TruthXModel implements an autoencoder-based method to decouple an LLM's hidden state 
    into semantic and truth-related latent spaces, fuse them using an attention mechanism, 
    and reconstruct the original representation. It also provides methods to compute the 
    truth-editing direction and to edit internal representations.
    """
    def __init__(self, config: dict) -> None:
        """
        Initializes the TruthXModel with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing model parameters.
                Expected keys under "model":
                  - d_model: Dimension of the LLM hidden state (default: 4096)
                  - encoder_dims: List of dimensions for the encoder MLP (default: [4096, 2048, 1024])
                  - decoder_dims: List of dimensions for the decoder MLP (default: [1024, 2048, 4096])
                  - latent_dim: Dimension of the latent representation (default: 1024)
                  - k_edit_layers: Number of top layers selected for editing (default: 10)
        """
        super(TruthXModel, self).__init__()

        model_config = config.get("model", {})
        self.d_model: int = model_config.get("d_model", 4096)
        # encoder_dims: expected as [d_model, intermediate_dim, latent_dim]
        self.encoder_dims: list = model_config.get("encoder_dims", [4096, 2048, 1024])
        # decoder_dims: expected as [latent_dim, intermediate_dim, d_model]
        self.decoder_dims: list = model_config.get("decoder_dims", [1024, 2048, 4096])
        self.latent_dim: int = model_config.get("latent_dim", 1024)
        self.k_edit_layers: int = model_config.get("k_edit_layers", 10)

        # Define the Truthful Encoder (TruthEnc): 2-layer MLP 
        # from d_model -> encoder_dims[1] -> encoder_dims[2] (latent_dim)
        self.TruthEnc = nn.Sequential(
            nn.Linear(self.d_model, self.encoder_dims[1]),
            nn.ReLU(),
            nn.Linear(self.encoder_dims[1], self.encoder_dims[2])
        )

        # Define the Semantic Encoder (SemEnc): Same architecture as TruthEnc
        self.SemEnc = nn.Sequential(
            nn.Linear(self.d_model, self.encoder_dims[1]),
            nn.ReLU(),
            nn.Linear(self.encoder_dims[1], self.encoder_dims[2])
        )

        # Define the Decoder: 2-layer MLP mapping fused latent representation to original space
        # from latent_dim -> decoder_dims[1] -> decoder_dims[2] (d_model)
        self.Dec = nn.Sequential(
            nn.Linear(self.latent_dim, self.decoder_dims[1]),
            nn.ReLU(),
            nn.Linear(self.decoder_dims[1], self.decoder_dims[2])
        )

    def compute_latents(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the latent representations of input x using TruthEnc and SemEnc.

        Args:
            x (torch.Tensor): Input hidden representation with shape (B, d_model).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: h_truth and h_sem,
                each of shape (B, latent_dim).
        """
        h_truth: torch.Tensor = self.TruthEnc(x)
        h_sem: torch.Tensor = self.SemEnc(x)
        return h_truth, h_sem

    def fuse_latents(self, h_sem: torch.Tensor, h_truth: torch.Tensor) -> torch.Tensor:
        """
        Fuses the semantic (h_sem) and truthful (h_truth) latent codes using an attention mechanism.

        The fusion is computed as:
            fused = h_sem + Attn(h_sem, h_truth)
        where Attn(h_sem, h_truth) is computed using a scaled dot-product attention.

        Args:
            h_sem (torch.Tensor): Semantic latent code of shape (B, latent_dim).
            h_truth (torch.Tensor): Truthful latent code of shape (B, latent_dim).

        Returns:
            torch.Tensor: Fused latent representation with shape (B, latent_dim).
        """
        attn_out: torch.Tensor = attention_fusion(query=h_sem, key=h_truth)
        fused: torch.Tensor = h_sem + attn_out
        return fused

    def decode(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Decodes the fused latent representation back to the original hidden state space.

        Args:
            fused (torch.Tensor): Fused latent code of shape (B, latent_dim).

        Returns:
            torch.Tensor: Reconstructed representation x_recon with shape (B, d_model).
        """
        x_recon: torch.Tensor = self.Dec(fused)
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the autoencoder.
        
        Steps:
          1. Compute latent codes h_truth and h_sem.
          2. Fuse latent codes using attention: fused = h_sem + Attn(h_sem, h_truth).
          3. Decode the fused code to reconstruct the representation.

        Args:
            x (torch.Tensor): Input hidden representation of shape (B, d_model).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_recon (torch.Tensor): Reconstructed representation, shape (B, d_model).
                - h_truth (torch.Tensor): Truthful latent code, shape (B, latent_dim).
                - h_sem (torch.Tensor): Semantic latent code, shape (B, latent_dim).
        """
        h_truth, h_sem = self.compute_latents(x)
        fused = self.fuse_latents(h_sem, h_truth)
        x_recon = self.decode(fused)
        return x_recon, h_truth, h_sem

    def compute_edit_direction(self, truthful_latents: torch.Tensor, untruthful_latents: torch.Tensor) -> torch.Tensor:
        """
        Computes the truth-editing direction δ as the difference between the 
        means of the truthful latent representations and the untruthful latent representations.

        Args:
            truthful_latents (torch.Tensor): Tensor of shape (N_truth, latent_dim) for truthful samples.
            untruthful_latents (torch.Tensor): Tensor of shape (N_untruth, latent_dim) for untruthful samples.

        Returns:
            torch.Tensor: The editing direction δ of shape (latent_dim,).
        """
        delta: torch.Tensor = torch.mean(truthful_latents, dim=0) - torch.mean(untruthful_latents, dim=0)
        return delta

    def edit_representation(self, x: torch.Tensor, delta: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Edits an internal hidden representation x to steer the LLM toward truthfulness.

        The editing uses the conversion mechanism:
          1. Compute latent codes: (h_truth, h_sem) = compute_latents(x).
          2. Compute modified truthful latents: h_truth_plus = h_truth + delta, h_truth_minus = h_truth - delta.
          3. Fuse with semantic latent codes:
                fused_plus = h_sem + Attn(h_sem, h_truth_plus)
                fused_minus = h_sem + Attn(h_sem, h_truth_minus)
          4. Decode both to get x_plus and x_minus.
          5. Compute Δ = x_plus - x_minus.
          6. Return the edited representation: x_edited = x + alpha * Δ.

        Args:
            x (torch.Tensor): Original hidden representation with shape (B, d_model).
            delta (torch.Tensor): Editing direction in latent space (shape: latent_dim,).
            alpha (float): Editing strength hyperparameter.

        Returns:
            torch.Tensor: The edited representation x_edited with shape (B, d_model).
        """
        # Compute latent representations for x.
        h_truth, h_sem = self.compute_latents(x)  # Both have shape (B, latent_dim)

        # Expand delta to match batch dimension: shape (B, latent_dim).
        delta_expanded = delta.unsqueeze(0).expand_as(h_truth)

        # Compute modified truthful latent codes.
        h_truth_plus = h_truth + delta_expanded
        h_truth_minus = h_truth - delta_expanded

        # Fuse with semantic latent codes using attention.
        fused_plus = h_sem + attention_fusion(query=h_sem, key=h_truth_plus)
        fused_minus = h_sem + attention_fusion(query=h_sem, key=h_truth_minus)

        # Decode the fused latent representations.
        x_plus = self.decode(fused_plus)
        x_minus = self.decode(fused_minus)

        # Compute the conversion vector Δ in the representation space.
        delta_representation = x_plus - x_minus

        # Edit the original representation.
        x_edited = x + alpha * delta_representation
        return x_edited
