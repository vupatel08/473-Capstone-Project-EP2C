## sm_operator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmOperator(nn.Module):
    """
    Implements the smooth operator as an iterative graph Laplacian smoothing based on the paper.
    It approximates the inverse (I + gamma * L)^(-1) via T iteration steps, promoting
    local consistency of instance embeddings.
    """
    def __init__(self, num_steps: int = 10, alpha_init: float = 0.5, use_spectral_norm: bool = True):
        """
        Args:
            num_steps (int): Number of T steps in the iterative approximation.
            alpha_init (float): Initial value for the alpha parameter, in (0,1). It will be learned.
            use_spectral_norm (bool): Whether to apply spectral normalization to alpha (if learnable).
        """
        super().__init__()
        self.num_steps = num_steps
        # Initialize alpha as a learnable parameter, constrained between 0 and 1
        # Use sigmoid parametrization for stability
        self._logit_alpha = nn.Parameter(torch.logit(torch.tensor(alpha_init)))
        self.use_spectral_norm = use_spectral_norm

        if self.use_spectral_norm:
            # Optional: apply spectral normalization to the alpha parameter
            self.alpha = nn.utils.spectral_norm(self._logit_alpha.unsqueeze(0)).squeeze()
        else:
            self.alpha = self._logit_alpha

    def forward(self, embeddings: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings (Tensor): shape (N, D), instance embeddings before smoothing.
            adjacency (Tensor): shape (N, N), adjacency matrix (symmetric, non-negative).
        Returns:
            smoothed_embeddings (Tensor): shape (N, D), smoothed instance embeddings.
        """
        # Validate inputs
        assert embeddings.dim() == 2, "embeddings should be (N, D)"
        assert adjacency.dim() == 2 and adjacency.size(0) == adjacency.size(1) == embeddings.size(0), \
            "adjacency should be (N, N)"

        N = embeddings.shape[0]
        device = embeddings.device

        # Step 1: Compute degree matrix D
        degrees = torch.clamp(adjacency.sum(dim=1), min=1e-12)  # avoid division by zero
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees))
        # Step 2: Compute normalized adjacency: A_norm = D^{-1/2} A D^{-1/2}
        A_norm = D_inv_sqrt @ adjacency @ D_inv_sqrt  # shape (N, N)

        # Step 3: Compute the normalized Laplacian: L = I - A_norm
        L = torch.eye(N, device=device) - A_norm  # shape (N, N)
        I = torch.eye(N, device=device)

        # Convert alpha to a sigmoid for stability, ensure it's in [0,1]
        alpha = torch.sigmoid(self._logit_alpha)

        # Initialize G(0) = embeddings
        G = embeddings

        # Precompute (I - L) matrix
        # Note: (I - L) = A_norm
        A_tilde = A_norm  # (N, N)

        # Iterative process
        for _ in range(self.num_steps):
            # Update based on eq: G(t) = alpha * (I - L) * G(t-1) + (1 - alpha) * embeddings
            G = alpha * (A_tilde @ G) + (1 - alpha) * embeddings

        return G

    def get_alpha(self) -> float:
        """
        Returns current alpha value (in [0,1])
        """
        return torch.sigmoid(self._logit_alpha).item()

    def set_alpha(self, new_alpha: float):
        """
        Set alpha as a float in (0, 1), updates the parameter accordingly.
        """
        # Convert to logit
        new_logit = torch.logit(torch.tensor(new_alpha))
        self._logit_alpha.data = new_logit
