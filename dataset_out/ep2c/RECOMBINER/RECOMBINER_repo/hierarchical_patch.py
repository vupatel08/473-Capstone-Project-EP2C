## hierarchical_patch.py
"""
Implements the HierarchicalPatchModel class for managing hierarchical Bayesian priors
over high-resolution data subdivided into patches, as described in Appendix B.2 and Figure 2.
It models, infers, and updates global, group, and patch-level weight representations,
supports permutation strategies, and maintains the dependencies and sharing necessary
for the hierarchical prior in the RECOMBINER framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import List, Optional, Tuple

class GaussianVariational:
    """
    Helper class to represent Gaussian variational distributions: q(μ,ρ)
    for hierarchical variables, with methods for sampling and KL.
    """
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor):
        """
        Initialize Gaussian variational distribution.
        Args:
            mu: mean tensor
            rho: log-variance tensor
        """
        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample from q(μ, ρ): h = μ + ε * σ
        Args:
            num_samples: number of samples
        Returns:
            samples: tensor of shape [num_samples, *mu.shape]
        """
        std = torch.exp(0.5 * self.rho)
        eps = torch.randn((num_samples,) + self.mu.shape, device=self.mu.device)
        return self.mu.unsqueeze(0) + eps * std.unsqueeze(0)

    def kl_divergence(self, prior_mu: torch.Tensor, prior_rho: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence KL(q || p) per Gaussian with diagonal covariance.
        Args:
            prior_mu: prior mean
            prior_rho: prior log-variance
        Returns:
            scalar tensor of KL
        """
        sigma_q = torch.exp(0.5 * self.rho)
        sigma_p = torch.exp(0.5 * prior_rho)
        term1 = (sigma_q ** 2) / (sigma_p ** 2)
        term2 = ((prior_mu - self.mu) ** 2) / (sigma_p ** 2)
        kl = 0.5 * torch.sum(term1 + term2 - 1 + torch.log((sigma_p ** 2) / (sigma_q ** 2) + 1e-8))
        return kl

class HierarchicalPatchModel:
    """
    Implements the hierarchical Bayesian prior for patch-based high-res signals,
    with global, group, and patch-level latent variables, permutation strategies,
    and variational inference.
    """
    def __init__(self, config, total_patches: int, group_size: int = 16, hierarchy_levels: int = 3,
                 seed: int = 42):
        """
        Initialize hierarchical prior and variational parameters.
        Args:
            config (dict): Configuration including prior variances, init params.
            total_patches (int): number of lowest-level patches.
            group_size (int): number of patches per group at second level.
            hierarchy_levels (int): total hierarchy levels (3 here).
            seed (int): random seed for permutations.
        """
        torch.manual_seed(seed)
        random.seed(seed)
        self.total_patches = total_patches
        self.group_size = group_size
        self.hierarchy_levels = hierarchy_levels

        # --- Priors over global h_w (top level) ---
        mu_global = torch.zeros_like(torch.tensor([]))  # will be set per dataset
        rho_global = torch.full_like(torch.tensor([]), fill_value=0.0)  # Variances initialized to 1

        # These will be set during training using dataset statistics; placeholders here
        self.prior_mu_global = torch.zeros(1)  # placeholder, updated during training
        self.prior_rho_global = torch.zeros(1)

        # --- Priors over group deviations at level 2 ---
        mu_group = torch.zeros(self.hierarchy_levels - 2, dtype=torch.float32)
        rho_group = torch.zeros(self.hierarchy_levels - 2, dtype=torch.float32)  # log-variance

        # --- Priors over patch deviations at level 1 ---
        mu_patch = torch.zeros(self.total_patches, dtype=torch.float32)
        rho_patch = torch.zeros(self.total_patches, dtype=torch.float32)

        # Variational parameters for global
        # Initialize with small variance (e.g., log-variance = -12)
        self.q_mu_global = nn.Parameter(torch.zeros_like(self.prior_mu_global))
        self.q_rho_global = nn.Parameter(torch.full_like(self.prior_rho_global, -12.0))
        self.q_global = GaussianVariational(self.q_mu_global, self.q_rho_global)

        # Variational for groups: shape (number of groups, mu, rho)
        self.num_groups = int(math.ceil(self.total_patches / self.group_size))
        self.q_mu_groups = nn.Parameter(torch.zeros(self.num_groups))
        self.q_rho_groups = nn.Parameter(torch.full((self.num_groups,), -12.0))
        self.q_groups = [GaussianVariational(self.q_mu_groups[i:i+1], self.q_rho_groups[i:i+1])
                         for i in range(self.num_groups)]

        # Variational for patches: shape (total_patches, mu, rho)
        self.q_mu_patches = nn.Parameter(torch.zeros(self.total_patches))
        self.q_rho_patches = nn.Parameter(torch.full((self.total_patches,), -12.0))
        self.q_patches = [GaussianVariational(self.q_mu_patches[i:i+1], self.q_rho_patches[i:i+1])
                          for i in range(self.total_patches)]

        # --- Permutation matrices/vectors ---
        self.perm_patch_current = torch.arange(self.total_patches)
        # Permutation for patches (shuffling across patches)
        self.permutation_patch = torch.randperm(self.total_patches)
        # Permutation within groups (across patches in each group)
        self.permutation_groups = [torch.randperm(self.group_size) for _ in range(self.num_groups)]

        # Save group assignments for each patch
        self.patch_to_group = [i // self.group_size for i in range(self.total_patches)]

        # --- Additional parameters for dependency modeling, if needed ---
        # For simplicity, assume no hyper-priors over covariances here.
        # Users can extend with hyper-priors if desired.

    def sample_global(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample global h_w from variational posterior.
        Returns:
            Tensor: shape (num_samples, global_dim)
        """
        return self.q_global.sample(num_samples)

    def sample_group(self, group_idx: int, num_samples: int=1) -> torch.Tensor:
        """
        Sample group deviation h_w^g.
        Args:
            group_idx: index of the group
        Returns:
            Tensor: shape (num_samples, group_dim)
        """
        return self.q_groups[group_idx].sample(num_samples)

    def sample_patch(self, patch_idx: int, num_samples: int=1) -> torch.Tensor:
        """
        Sample patch deviation h_w^π.
        Args:
            patch_idx: index of the patch
        Returns:
            Tensor: shape (num_samples, patch_dim)
        """
        return self.q_patches[patch_idx].sample(num_samples)

    def get_patch_weights(self, global_h: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate the per-patch weights from global and deviations.
        Args:
            global_h: sampled global h_w, shape (global_dim,)
        Returns:
            List of tensors: each shape (patch_dim,)
        """
        device = global_h.device
        patch_weights = []
        for pi in range(self.total_patches):
            group_idx = self.patch_to_group[pi]
            # Sample deviations
            h_g_samples = self.sample_group(group_idx)  # shape (samples, group_dim)
            h_p_samples = self.sample_patch(pi)        # shape (samples, patch_dim)

            # For deterministic extraction, take the mean of the variational posterior
            # Alternatively, sample once:
            # h_g_mean = h_g_samples.mean(0)
            # h_p_mean = h_p_samples.mean(0)
            # But typically, during inference, we take expectation (mean):
            h_g_mean = self.q_groups[group_idx].mu
            h_p_mean = self.q_patches[pi].mu
            # Combine: h_w^(\pi) = global + deviation (group and patch)
            h_w_pi = global_h + h_g_mean + h_p_mean
            patch_weights.append(h_w_pi)
        return patch_weights

    def sample_global_posterior(self) -> torch.Tensor:
        """
        Sample global h_w from the variational posterior
        """
        return self.q_global.sample()

    def sample_all_patch_weights(self, global_h: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate patch weights for all patches given global h_w.
        """
        return self.get_patch_weights(global_h)

    def update_variational_params(self, new_mu_global, new_rho_global,
                                    new_mu_groups, new_rho_groups,
                                    new_mu_patches, new_rho_patches):
        """
        Update all variational parameters with new values.
        """
        self.q_mu_global.data.copy_(new_mu_global)
        self.q_rho_global.data.copy_(new_rho_global)
        for i in range(self.num_groups):
            self.q_mu_groups.data[i:i+1] = new_mu_groups[i]
            self.q_rho_groups.data[i:i+1] = new_rho_groups[i]
        for pi in range(self.total_patches):
            self.q_mu_patches.data[pi:pi+1] = new_mu_patches[pi]
            self.q_rho_patches.data[pi:pi+1] = new_rho_patches[pi]

    def compute_kl(self,
                   prior_mu_global: torch.Tensor,
                   prior_rho_global: torch.Tensor,
                   prior_mu_group: torch.Tensor,
                   prior_rho_group: torch.Tensor,
                   prior_mu_patch: torch.Tensor,
                   prior_rho_patch: torch.Tensor) -> torch.Tensor:
        """
        Compute the total KL divergence upper bound (Equation 4)
        for the hierarchical model, summing over global, groups, and patches.
        """
        kl_global = self.q_global.kl_divergence(prior_mu_global, prior_rho_global)
        kl_groups = 0.0
        for i in range(self.num_groups):
            kl_groups += self.q_groups[i].kl_divergence(prior_mu_group, prior_rho_group)
        kl_patches = 0.0
        for pi in range(self.total_patches):
            kl_patches += self.q_patches[pi].kl_divergence(prior_mu_patch, prior_rho_patch)
        return kl_global + kl_groups + kl_patches

    def apply_permutation(self):
        """
        Permute the individual matrices/hierarchies as per permutation vectors.
        This applies to the concatenated representation matrix H(ℓ) at each level.
        For simplicity, user should perform permutations outside this class.
        """
        # Example: permute patch order
        self.perm_patch_current = self.permutation_patch
        # For groups, per-group permutation
        self.permutation_groups = [permutation for permutation in self.permutation_groups]
        # These permutation vectors can be used during representation stacking
  
    def get_permuted_indices(self, level: str) -> torch.Tensor:
        """
        Access current permutation indices for a given level.
        """
        if level == 'patch':
            return self.perm_patch_current
        elif level == 'group':
            # For group level, return list of permutations
            return self.permutation_groups
        else:
            return torch.arange(self.total_patches)

    # Additional utility methods for handling matrix stacking, slicing, etc.,
    # can be added as needed for the full encoding/decoding pipeline.

