## spectral_transform.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class SpectralTransformer:
    """
    Implements spectral modulation of embeddings based on covariance spectra.
    Supports whitening, power law modulation, and iterative Newton (IterNorm).
    Designed with compatibility to configuration parameters from 'config.yaml'.
    """

    def __init__(
        self,
        T: int = 4,
        method: str = 'Newton',
        p: float = 0.5,
        epsilon: float = 1e-5
    ):
        """
        Args:
            T (int): Number of iterations T (for iterative Newton).
            method (str): 'whitening', 'power', 'Newton' (for IterNorm).
            p (float): Power parameter for 'power' method, should be near 0.5.
            epsilon (float): Small value for numerical stability, e.g., 1e-5.
        """
        self.T = T
        self.method = method
        self.p = p
        self.epsilon = epsilon

    def compute_covariance(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute empirical covariance matrix of embeddings.
        Args:
            Z (torch.Tensor): shape (d, m), embedding batch
        Returns:
            Sigma (torch.Tensor): shape (d, d), covariance matrix
        """
        m = Z.shape[1]
        Z_mean = Z - Z.mean(dim=1, keepdim=True)
        Sigma = (Z_mean @ Z_mean.t()) / (m - 1)  # unbiased estimator
        return Sigma

    def eig_decompose(self, Sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Eigen decomposition of symmetric matrix Sigma.
        Args:
            Sigma (torch.Tensor): shape (d, d)
        Returns:
            U (torch.Tensor): eigenvectors (d, d)
            Lambda (torch.Tensor): eigenvalues (d,)
        """
        # Use torch.linalg.eigh for symmetric matrices
        # Ensure numerical stability
        Sigma_stable = Sigma + self.epsilon * torch.eye(Sigma.shape[0], device=Sigma.device)
        Lambda, U = torch.linalg.eigh(Sigma_stable)
        # Eigenvalues sorted in ascending order
        return U, Lambda

    def spectral_modulate(self, Lambda: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral modulation function g(λ) = λ^{-p} or whitening.
        Args:
            Lambda (torch.Tensor): eigenvalues (d,)
        Returns:
            gLambda (torch.Tensor): modulated eigenvalues
        """
        if self.method.lower() == 'whitening':
            # g(λ) = λ^{-0.5}
            gLambda = torch.clamp(Lambda, min=self.epsilon) ** -0.5
        elif self.method.lower() == 'power':
            # g(λ) = λ^{-p}
            gLambda = torch.clamp(Lambda, min=self.epsilon) ** -self.p
        elif self.method.lower() == 'newton':
            # Use iterative Newton's method approximation
            # For Newton, apply the f_T function to eigenvalues
            # We'll implement this separately as a method
            # For simplicity, here we do a placeholder: use power law close to 0.5
            gLambda = torch.clamp(Lambda, min=self.epsilon) ** -self.p
        else:
            raise ValueError(f"Unknown spectral modulation method: {self.method}")
        return gLambda

    def apply_iter_norm(self, Sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute the approximate whitening matrix via Newton iteration.
        Returns:
            Phi_T (torch.Tensor): shape (d, d), approximate whitening matrix
        """
        # Initialize P_0
        P = torch.eye(Sigma.shape[0], device=Sigma.device)
        # Compute tr(Sigma)
        tr_sigma = torch.trace(Sigma)
        # Normalize Sigma
        Sigma_N = Sigma / (tr_sigma + self.epsilon)
        for _ in range(self.T):
            # Newton's update: P_k+1 = (3/2)*P_k - (1/2)*P_k^3 * Sigma_N
            P_cubed = P @ P @ P
            P = 0.5 * (3 * P - P_cubed @ Sigma_N)
        Phi_T = P / torch.sqrt(tr_sigma + self.epsilon)
        return Phi_T

    def transform(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Perform spectral transformation on embedding batch Z.
        Args:
            Z (torch.Tensor): shape (d, m)
        Returns:
            Z_hat (torch.Tensor): shape (d, m), spectrally modulated embeddings
        """
        Sigma = self.compute_covariance(Z)
        U, Lambda = self.eig_decompose(Sigma)

        if self.method.lower() == 'Newton':
            # Use iterative Newton's method to approximate Sigma^{-0.5}
            Phi = self.apply_iter_norm(Sigma)
            Z_hat = Phi @ Z
        else:
            # For whitening or power methods, spectral modulation on eigenvalues
            gLambda = self.spectral_modulate(Lambda)
            Z_hat = self.reconstruct_embeddings(U, gLambda, Z)

        return Z_hat

    def reconstruct_embeddings(self, U: torch.Tensor, gLambda: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct transformed embeddings: Z_hat = U (diag(gLambda)) U^T Z
        Args:
            U (torch.Tensor): eigenvectors (d, d)
            gLambda (torch.Tensor): eigenvalues (d,)
            Z (torch.Tensor): original embeddings (d, m)
        Returns:
            Z_hat (torch.Tensor): transformed embeddings (d, m)
        """
        # Construct diagonal matrix of gLambda
        G = torch.diag(gLambda)
        # Compute U G U^T Z
        Z_hat = U @ G @ U.t() @ Z
        return Z_hat

    def log_eigenvalues(self, Lambda: torch.Tensor):
        """
        Optional diagnostic: log spectrum.
        """
        return torch.log(Lambda + self.epsilon)

    # Additional helper functions if needed could be added here
    # For example, a method to compute g(λ) based on configuration

# Note:
# - This module relies on the supplied configuration settings.
# - For 'Newton' mode, the Newton iteration is used to approximate Sigma^{-0.5}.
# - For 'power' mode, apply elementwise power.
# - Proper handling of numerical instability (epsilon adjustment) is embedded.
# - Eigen-decomposition is performed via torch.linalg.eigh with regularization.
# - The class is designed for integration into the training pipeline, transforming batch embeddings.
