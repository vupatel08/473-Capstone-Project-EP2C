## pca_transform.py
import os
import torch
import math
from typing import List, Dict, Optional

class PCAProcessor:
    """
    Handles PCA analysis of activation signals for each transformer layer.
    Computes covariance matrices, eigen-decompositions, and stores eigenvectors (Q)
    for use in model transformations and weight slicing.
    """

    def __init__(
        self,
        signals: torch.Tensor,
        model_wrapper,
        spectrum_threshold: str = "auto",
        save_path: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize PCAProcessor with signals and parameters.
        Args:
            signals (torch.Tensor): Concatenated signals of shape (N, D), already collected.
            model_wrapper: Reference to ModelWrapper to infer model details.
            spectrum_threshold (str or float): Threshold for eigen spectrum retention.
                - If float between 0 and 1, it specifies variance coverage.
                - If "auto", use heuristic of spectral decay.
            save_path (str, optional): Directory path to save/load eigenvectors.
            debug (bool): If True, print spectrum info for debugging.
        """
        self.signals = signals  # shape: (N, D)
        self.model_wrapper = model_wrapper
        self.spectrum_threshold = spectrum_threshold
        self.save_path = save_path
        self.debug = debug

        # Will contain a list of Q matrices, one per layer
        self.Qs: List[torch.Tensor] = []

        # Eigenvalues per layer (for spectrum analysis)
        self.eigenvalues: List[torch.Tensor] = []

        # Placeholder for number of layers (inferred or set externally)
        # Assuming model_wrapper can provide number of layers
        self.layer_count = len(self.model_wrapper._get_layers())

    def collect_covariance_matrices(self) -> List[torch.Tensor]:
        """
        For each layer, compute the covariance matrix of signals: C = sum_i X_i^T X_i
        Returns:
            List of covariance matrices (D x D), one per layer.
        """
        cov_matrices: List[torch.Tensor] = []

        start_idx = 0
        # Collect signals for each layer by splitting signals accordingly
        # For simplicity, assume signals per layer are pre-extracted or that
        # signals are layer-specific. Therefore, signals is a dictionary.
        # But to match the description, here, 'signals' are concatenated over layers.
        # In practice, signals would be collected per layer in a list during extraction.

        # So, for this code, let's assume signals is a dict: layer_idx -> signals tensor
        # To maintain the design, assume 'collect_signals' is called per layer and stored
        # separately. Here, we'll proceed with an abstract flexible approach:
        # (In real implementation, signals per layer should be stored beforehand)
        # For the placeholder, assume 'signals' is a list of signals per layer.
        pass

    def compute_eigenvectors(self, spectrum_threshold: Optional[float] = None):
        """
        For each covariance matrix, compute eigen-decomposition, sort eigenvectors,
        and determine how many components to retain based on threshold.
        Args:
            spectrum_threshold (float): Percentage of variance to retain (0-1). If None, use internal logic.
        """
        self.Qs = []
        self.eigenvalues = []

        # Compute covariance matrix
        # signals shape assumed (N, D) for one layer at a time, or list of signals
        # For this code, we assume 'signals_per_layer' is a list of tensors
        # For simplicity, assume signals is a list: length = number of layers.
        # For real implementation, signals should be stored accordingly.
        # Here, implement with dummy placeholders for demonstration.
        # In practice, replace 'self.signals' with per-layer signals.
        signals_per_layer = self._separate_signals_by_layer()
        for layer_idx in range(self.layer_count):
            X = signals_per_layer[layer_idx]  # shape: (N_i, D)
            # Centering signals (optional): not required for covariance eigen-decomposition
            # Compute covariance matrix: sum over outer products
            cov = torch.zeros((X.shape[1], X.shape[1]), dtype=torch.float64)
            for i in range(X.shape[0]):
                xi = X[i]
                cov += torch.ger(xi, xi)
            # Normalize covariance matrix
            cov /= X.shape[0]

            # Eigen-decomposition with FP64
            eigvals, eigvecs = torch.linalg.eigh(cov)
            # Sort in descending order
            eigvals, indices = torch.sort(eigvals, descending=True)
            eigvecs = eigvecs[:, indices]

            if self.debug:
                print(f"Layer {layer_idx}: Eigen spectrum top eigenvalues: {eigvals[:10]}")

            # Spectrum-based cutoff
            if spectrum_threshold == "auto" or spectrum_threshold is None:
                # Use heuristic: retain eigenvectors covering 95% variance
                spectrum_ratio = eigvals / eigvals.sum()
                cumulative = torch.cumsum(spectrum_ratio, dim=0)
                keep_k = torch.searchsorted(cumulative, 0.95) + 1
            else:
                # Use fixed variance coverage
                spectrum_ratio = eigvals / eigvals.sum()
                cumulative = torch.cumsum(spectrum_ratio, dim=0)
                keep_k = torch.searchsorted(cumulative, spectrum_threshold) + 1

            # Save eigenvectors (Q) for retained components
            Q_current = eigvecs[:, :keep_k]  # shape: D x keep_k
            self.Qs.append(Q_current)
            self.eigenvalues.append(eigvals.cpu())

            # Optionally save to disk
            if self.save_path is not None:
                filename = os.path.join(self.save_path, f"Q_layer_{layer_idx}.pt")
                torch.save(Q_current, filename)

    def save_eigenvectors(self, path: str):
        """
        Save all eigenvector matrices to disk.
        Args:
            path (str): Directory to save eigenvectors.
        """
        os.makedirs(path, exist_ok=True)
        for idx, Q in enumerate(self.Qs):
            filename = os.path.join(path, f"Q_layer_{idx}.pt")
            torch.save(Q, filename)

    def load_eigenvectors(self, path: str):
        """
        Load eigenvector matrices from disk.
        Args:
            path (str): Directory containing eigenvector `.pt` files.
        """
        self.Qs = []
        for layer_idx in range(self.layer_count):
            filename = os.path.join(path, f"Q_layer_{layer_idx}.pt")
            if os.path.isfile(filename):
                Q = torch.load(filename)
                self.Qs.append(Q)
            else:
                raise FileNotFoundError(f"Eigenvector file not found: {filename}")

    def _separate_signals_by_layer(self) -> List[torch.Tensor]:
        """
        Placeholder method to separate or provide signals per layer.
        In practice, this should be implemented to match how signals are collected.
        """
        # This is a critical point in actual code, where signals would be stored per layer.
        # Here, we'll assume signals are provided externally or implemented separately.
        # For the scope of this code, raise NotImplementedError.
        raise NotImplementedError("Provide implementation for signals separation per layer.")

    def analyze_spectrum(self):
        """
        Optional: plot or log spectrum decay for debugging/inspection.
        """
        import matplotlib.pyplot as plt
        for idx, eigs in enumerate(self.eigenvalues):
            plt.plot(eigs.cpu().numpy(), label=f'Layer {idx}')
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Eigenvalue (log scale)")
        plt.yscale('log')
        plt.legend()
        plt.show()

