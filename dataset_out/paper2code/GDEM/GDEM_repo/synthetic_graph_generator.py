## synthetic_graph_generator.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from scipy.linalg import svd
from scipy.sparse import csr_matrix
from utils import project_onto_stiefel  # Ensure this is implemented as per interfaces, for orthogonal projection

class SyntheticGraphGenerator:
    """
    Generates a synthetic graph that mimics the spectral properties of a real graph
    by matching eigenbasis and spectrum, and optimizes node features accordingly.
    """

    def __init__(self, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor, features: torch.Tensor,
                 lambda_e: float = 1.0, lambda_d: float = 1.0, lambda_o: float = 1.0,
                 match_lr: float = 1e-3, device: torch.device = torch.device('cpu')):
        """
        Initialize the generator with real spectral info and initial node features.
        Args:
            eigenvalues (torch.Tensor): [K], real graph's eigenvalues.
            eigenvectors (torch.Tensor): [N, K], real graph's eigenbasis.
            features (torch.Tensor): [N', d], initial node features for synthetic graph.
            lambda_e (float): weight for eigenbasis matching loss.
            lambda_d (float): weight for discrimination (category) loss.
            lambda_o (float): weight for orthogonality regularization.
            match_lr (float): learning rate for basis matching.
            device (torch.device): computation device.
        """
        self.device = device
        self.eigenvalues = eigenvalues.to(self.device)  # [K]
        self.eigenvectors = eigenvectors.to(self.device)  # [N, K]
        self.features = features.to(self.device)  # [N', d]

        self.N_prime = self.features.shape[0]
        self.K = self.eigenvalues.shape[0]
        self.d = self.features.shape[1]

        # Initialize the synthetic eigenbasis U' [N', K]
        # Start with a random orthonormal basis
        U_rand = torch.randn(self.N_prime, self.K, device=self.device)
        self.U_prime = self._orthogonalize(U_rand)  # [N', K]

        # Node features optimization tensor
        self.X_prime = self.features.clone().detach().requires_grad_(True)

        # Optimization parameters
        self.lambda_e = lambda_e
        self.lambda_d = lambda_d
        self.lambda_o = lambda_o
        self.match_lr = match_lr

    def _orthogonalize(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Projects matrix onto the Stiefel manifold (orthogonal basis) via QR.
        Args:
            matrix (torch.Tensor): [N', K]
        Returns:
            torch.Tensor: orthonormal basis [N', K]
        """
        Q, R = torch.linalg.qr(matrix, mode='reduced')
        return Q

    def construct_spectrum_matrices(self):
        """
        Build synthetic adjacency and Laplacian matrices from eigenbasis and eigenvalues.
        Returns:
            A_prime (torch.Tensor): [N', N']
            L_prime (torch.Tensor): [N', N']
        """
        # Compute U' U'^T [N', N']
        U = self.U_prime  # [N', K]
        # Spectrum-based constructions
        # Broadcast eigenvalues to shape [K]
        lambda_k = self.eigenvalues  # [K]
        # Outer product: [N', K] x [K, N'] -> [N', N']
        # Construct A' and L' as matrices
        # Using tensor operations for efficiency
        outer_U = torch.einsum('ik,jk->ij', U, U)  # [N', N']
        A_prime = torch.einsum('k,ik,jk->ij', 1 - lambda_k, U, U)  # sum over k
        L_prime = torch.einsum('k,ik,jk->ij', lambda_k, U, U)
        return A_prime, L_prime

    def build_adjacency(self):
        """
        Computes adjacency matrix A' from spectrum and eigenbasis.
        Returns:
            torch.Tensor: [N', N']
        """
        A_prime, _ = self.construct_spectrum_matrices()
        return A_prime

    def build_laplacian(self):
        """
        Computes Laplacian matrix L' from spectrum and eigenbasis.
        Returns:
            torch.Tensor: [N', N']
        """
        _, L_prime = self.construct_spectrum_matrices()
        return L_prime

    def optimize_eigenbasis(self, real_basis: torch.Tensor, steps: int = 3000):
        """
        Optimize the synthetic eigenbasis U' to match the real eigenbasis U.
        Args:
            real_basis (torch.Tensor): [N, K], real graph's principal eigenvectors.
            steps (int): number of gradient steps.
        """
        real_basis = real_basis.to(self.device)
        U_prime = self.U_prime.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([U_prime], lr=self.match_lr)

        for step in range(steps):
            optimizer.zero_grad()
            # Compute \mathcal{L}_e = Frobenius norm of difference of outer products
            # U U^T vs U' U'^T
            UUT = self.eigenvectors @ self.eigenvectors.t()  # [N, N]
            U_prime_proj = U_prime
            U_prime_UT = U_prime_proj @ U_prime_proj.t()
            basis_loss = torch.norm(UUT - U_prime_UT, p='fro') ** 2

            # Orthogonality regularization
            UtU = U_prime.t() @ U_prime
            orth_loss = torch.norm(UtU - torch.eye(self.K, device=self.device), p='fro') ** 2

            total_loss = self.lambda_e * basis_loss + self.lambda_o * orth_loss
            total_loss.backward()
            optimizer.step()

            # Enforce orthogonality after each step
            with torch.no_grad():
                U_prime = self._orthogonalize(U_prime)

        # Save the optimized basis
        self.U_prime = U_prime.detach()

    def optimize_features(self, schedule_params):
        """
        Optimize node features X' given current U' to minimize spectral discrepancy.
        Args:
            schedule_params (dict): contains 'steps', 'lr', etc.
        """
        steps = schedule_params.get('steps', 3000)
        lr = schedule_params.get('lr', 1e-3)
        optimizer = optim.Adam([self.X_prime], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            # Compute spectral loss functions
            # Build spectrum matrices
            # A
            A_pred, _ = self.construct_spectrum_matrices()  # [N', N']
            # Spectrum regularization (spectral discrepancy)
            # For simplicity, combine spectral discrepancies in a single loss
            # For real graph, have real eigenvalues
            lambda_k = self.eigenvalues  # [K]
            U = self.U_prime  # [N', K]
            diag_lambda = torch.diag(lambda_k)  # [K, K], optional for quadratic forms
            # Calculate quadratic form: trace(X'^\top L' X') or similar
            # For now, use spectral discrepancy based on the spectrum
            # Can also implement more elaborate spectral loss if required.

            # Here, implement a simple spectral discrepancy:
            # Reconstruction loss of spectral relation: minimize || X'^T L' X' - X^T L X || 
            # or minimize the difference between spectrum
            # For now, approximate as:
            # (Optional) Implement other spectral discrepancy as in the paper if needed
            # For simplicity, use the Frobenius norm between the constructed A' (adj) and spectrum influenced adjacency
            # But since only spectra are used, a practical way:
            # Use the eigenvalues: minimize || eigenvalues - estimated eigenvalues via X' and U' (not directly available)
            # Alternatively, use the spectral basis approximation:
            # Approximate the spectrum via the current eigenbasis

            # For this code, we'll use the discrepancy between the current spectrum and the known real eigenvalues
            # to guide X' updates (this is a simplified placeholder).
            # For a more rigorous implementation, define spectral loss based on quadratic forms, etc.
            # But in this simplified demonstration, we skip explicit spectral loss to focus on the core logic.

            # As a placeholder, define a dummy loss (e.g., norm of features), or incorporate a spectral loss as needed.
            # For now, we build a spectral loss based on current U' and eigenvalues:
            # Using the Rayleigh quotient approximation
            spectral_estimate = torch.sum((self.X_prime @ self.U_prime) ** 2, dim=1)
            spectrum_loss = torch.nn.functional.mse_loss(spectral_estimate, self.eigenvalues.repeat(self.N_prime // self.K + 1)[:self.N_prime])

            # Alternatively, can define a more sophisticated spectral loss here.

            # Compute other losses if needed, e.g., structure constraints, distribution matching.
            # For illustration, we minimize the spectrum loss.
            total_loss = self.lambda_e * spectrum_loss

            total_loss.backward()
            optimizer.step()

        # End, store optimized features
        self.X_prime = self.X_prime.detach()

    def get_synthetic_graph(self):
        """
        Build the final synthetic adjacency and features after optimization.
        Returns:
            A_final (torch.Tensor): [N', N']
            L_final (torch.Tensor): [N', N']
            U_final (torch.Tensor): [N', K]
            X_final (torch.Tensor): [N', d]
        """
        # Final U'
        U_final = self.U_prime
        # Build spectrum-based adjacency and Laplacian
        A_final, L_final = self.construct_spectrum_matrices()
        # Use the spectrum to construct adjacency
        return A_final, L_final, U_final, self.X_prime

    def run(self, real_basis: torch.Tensor, schedule_params: dict):
        """
        Run the full alternating optimization schedule.
        Args:
            real_basis (torch.Tensor): [N, K], real eigenvectors for basis matching.
            schedule_params (dict): scheduling info with 'tau_1', 'tau_2', 'eigenbasis_steps', 'feature_steps', etc.
        """
        tau_1 = schedule_params.get('tau_1', 3000)  # eigenbasis matching steps
        tau_2 = schedule_params.get('tau_2', 3000)  # feature optimization steps
        total_epochs = schedule_params.get('epochs', 6000)
        for epoch in range(0, total_epochs, tau_1 + tau_2):
            # Eigenbasis matching
            self.optimize_eigenbasis(real_basis, steps=tau_1)
            # Feature optimization
            self.optimize_features(schedule_params)
        # After full schedule
        return self.get_synthetic_graph()
