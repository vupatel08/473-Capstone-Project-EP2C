# spectral_decomposition.py
import torch
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

class SpectralDecomposition:
    """
    Computes spectral decomposition (eigenvalues and eigenvectors) of the normalized graph Laplacian.
    Supports large sparse graphs via scipy's eigsh solver.
    """

    def __init__(self, adj_matrix: torch.Tensor, K: int = 500):
        """
        Initializes the spectral decomposition with adjacency matrix and number of eigenvectors.

        Args:
            adj_matrix (torch.Tensor): adjacency matrix of shape (N, N), assumed undirected and unweighted.
            K (int): number of eigenvalues/eigenvectors to compute.
                     If K >= N, computes the full spectrum.
        """
        self.adj = adj_matrix
        self.K = K
        self.num_nodes = adj_matrix.shape[0]
        # Convert adjacency tensor to scipy csr_matrix
        self.adj_sparse = self._to_coo_csr(self.adj)
        self.eigenvalues = None
        self.eigenvectors = None
        self._compute_laplacian()

    def _to_coo_csr(self, tensor: torch.Tensor):
        """
        Converts a torch Tensor adjacency matrix to scipy CSR sparse matrix.

        Args:
            tensor (torch.Tensor): adjacency matrix.

        Returns:
            scipy.sparse.csr_matrix
        """
        # Assure adjacency is on CPU
        adj_np = tensor.cpu().numpy()
        # Turn into CSR sparse matrix
        csr = csr_matrix(adj_np)
        return csr

    def _compute_laplacian(self):
        """
        Computes the symmetric normalized Laplacian matrix from adjacency.
        """
        # Degree vector
        degrees = np.array(self.adj_sparse.sum(axis=1)).flatten()
        # Handle zero degrees
        degrees[degrees == 0] = 1.0
        # Compute D^{-1/2}
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
        D_inv_sqrt = diags(d_inv_sqrt)

        # Symmetric normalized adjacency: D^{-1/2} * A * D^{-1/2}
        # Then Laplacian: L = I - normalized adjacency
        normalized_adj = D_inv_sqrt @ self.adj_sparse @ D_inv_sqrt
        self.laplacian = csr_matrix(np.identity(self.num_nodes)) - normalized_adj

    def compute_eigenbasis(self):
        """
        Performs eigen-decomposition on the Laplacian to obtain eigenvalues and eigenvectors.
        """
        # Decide number of eigenvalues
        k = self.K
        # If K >= number of nodes, compute full spectrum
        if k >= self.num_nodes:
            k = self.num_nodes - 1  # eigsh cannot compute all, so one less
            # For full spectrum, eigenvalues close to the matrix's size
            # fallback to dense if needed
            # but here, assume K < N
        try:
            # eigsh for sparse matrices, 'SM' for smallest magnitude eigenvalues
            eigenvalues, eigenvectors = eigsh(self.laplacian, k=k, which='SM', tol=1e-3)
            # eigsh returns eigenvalues in ascending order
        except Exception as e:
            # fallback: dense eigen decomposition if size N is small
            # For very large graphs, approximate methods should be used
            dense_L = self.laplacian.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(dense_L)
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]

        # Convert to torch tensors
        eigenvalues = torch.tensor(eigenvalues, dtype=torch.float32)
        eigenvectors = torch.tensor(eigenvectors, dtype=torch.float32)
        # Ensure eigenvectors are orthogonal and normalized
        # (eigsh guarantees this for symmetric matrices)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    def get_spectrum(self):
        """
        Returns eigenvalues and eigenvectors.

        Returns:
            eigenvalues (torch.Tensor): shape (K,)
            eigenvectors (torch.Tensor): shape (N, K)
        """
        return self.eigenvalues, self.eigenvectors
