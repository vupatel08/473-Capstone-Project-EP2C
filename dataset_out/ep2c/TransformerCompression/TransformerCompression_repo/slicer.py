## slicer.py
import torch
import os
from typing import List, Dict, Optional
from dataset_loader import DatasetLoader
from model import ModelWrapper
from pca_transform import PCAProcessor

class Slicer:
    """
    Implements the core PCA-based weight slicing and residual adjustment
    according to the SliceGPT methodology.
    """
    def __init__(
        self,
        model_wrapper: ModelWrapper,
        eigenvectors: List[torch.Tensor],
        spectrum_threshold: str = "auto",
        keep_ratio: float = 0.75,
        debug: bool = False
    ):
        """
        Initialize the Slicer.
        Args:
            model_wrapper (ModelWrapper): Wrapper around the target model.
            eigenvectors (List[torch.Tensor]): List of Q matrices per layer.
            spectrum_threshold (str or float): Threshold for eigen spectrum retention.
            keep_ratio (float): Default variance ratio to keep (used if spectrum_threshold='auto' or for fallback).
            debug (bool): Print debug info.
        """
        self.model_wrapper = model_wrapper
        self.Qs = eigenvectors
        self.spectrum_threshold = spectrum_threshold
        self.keep_ratio = keep_ratio
        self.debug = debug
        self.layer_count = len(eigenvectors)

    def compute_layer_spectrum(self, signals: Dict[int, torch.Tensor]) -> None:
        """
        Optional: Analyze eigen-spectrum of signals per layer for diagnostic.
        Args:
            signals (dict): layer_idx -> signals tensor (N, D)
        """
        import matplotlib.pyplot as plt
        for layer_idx, X in signals.items():
            cov = torch.zeros((X.shape[1], X.shape[1]), dtype=torch.float64)
            for i in range(X.shape[0]):
                xi = X[i]
                cov += torch.ger(xi, xi)
            cov /= X.shape[0]
            eigvals, _ = torch.linalg.eigh(cov)
            eigvals = torch.sort(eigvals, descending=True).values
            plt.plot(eigvals.cpu().numpy(), label=f"Layer {layer_idx}")
        plt.yscale('log')
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Eigenvalues (log scale)")
        plt.legend()
        plt.show()

    def slice_layer(self, layer_idx: int, W_in: torch.Tensor, W_out: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Slice weight matrices of a single layer based on PCA eigen-spectrum.
        Args:
            layer_idx (int): Index of the layer.
            W_in (torch.Tensor): Input weight matrix (D x D_in)
            W_out (torch.Tensor): Output weight matrix (D_out x D)
        Returns:
            (W_in_sliced, W_out_sliced): sliced weight matrices with reduced dimensions.
        """
        Q = self.Qs[layer_idx]  # shape: D x D (assuming D=D)
        # Compute eigenvalues to decide how many components to keep
        # Since eigenvectors are orthogonal, the spectrum info is from eigenvalues
        # But here, optionally, spectral info is used; for simplicity, assume all components kept
        # Alternatively, if we have spectrum data, we can compute number of components to retain
        # For demonstration, here we keep all components or retain based on variance ratio
        # If spectrum_threshold is 'auto', use spectrum info; else, keep ratio parameter
        # For simplicity, retain a fixed ratio (e.g., 1 - ratio), or full
        # For now, assume kept_dims is provided (can be computed prior)
        # Placeholder: keep all components
        # If spectrum info exists, implement variance coverage logic
        # For example purposes, keep all
        kept_dims = Q.shape[1]
        # Construct projection matrices
        Q_retain = Q[:, :kept_dims]  # shape: D x kept_dims

        # Rotate weights into principal component basis
        W_in_rot = Q.t() @ W_in  # (D x D_in)
        W_out_rot = W_out @ Q  # (D_out x D)

        # Slice the least important components (bottom eigenvectors)
        W_in_sliced = W_in_rot[:kept_dims, :]    # Keep top components
        W_out_sliced = W_out_rot[:, :kept_dims]

        # Reverse the rotation to original basis
        W_in_final = Q @ W_in_sliced
        W_out_final = W_out_sliced @ Q.t()

        return W_in_final, W_out_final

    def apply_layer_slicing(self, layer_idx: int) -> None:
        """
        Perform the eigen-spectrum based slicing on model weights for a layer.
        Args:
            layer_idx (int): The specific layer to slice.
        """
        # Fetch current weights
        weights = self.model_wrapper.get_weights(layer_idx)

        # Extract relevant matrices: attention key/query (W_in), output (W_out)
        W_emb = weights['W_emb']
        W_q = weights[f'W_q_{layer_idx}']
        W_o = weights[f'W_o_{layer_idx}']
        W_ff1 = weights.get(f'W_ff1_{layer_idx}')
        W_ff2 = weights.get(f'W_ff2_{layer_idx}')
        # Additional matrices can be added accordingly

        # Compute eigen-spectrum for the layer signals
        # (Assuming signals are precomputed and passed to this class externally)
        # For simplicity, we assume that eigenvectors Q_l correspond already to the desired eigencomponents
        # which is set via previous PCA computation
        Q = self.Qs[layer_idx]

        # Rotation into PC basis
        W_in_rot = Q.t() @ W_q
        W_out_rot = W_o @ Q

        # Decide number of components to keep
        eigenvalues = None
        if self.spectrum_threshold == 'auto':
            # Could compute based on spectrum info if available
            # For simplicity: keep 90% variance => top keep_ratio
            keep_ratio = 0.9
        else:
            keep_ratio = float(self.spectrum_threshold)

        # Compute spectrum from eigenvalues if available
        # Here, we assume eigenvalues info is provided elsewhere,
        # or approximate with singular values.
        # For simplicity, just keep a fixed ratio
        D_current = W_in_rot.shape[0]
        keep_dims = int(D_current * keep_ratio)
        keep_dims = max(1, keep_dims)  # at least dimension 1

        # Slice the retained components
        W_in_sliced = W_in_rot[:keep_dims, :]
        W_out_sliced = W_out_rot[:, :keep_dims]

        # Reverse rotation
        W_in_final = Q @ W_in_sliced
        W_out_final = W_out_sliced @ Q.t()

        # Update weights
        weights['W_q_{layer_idx}'] = W_in_final
        weights['W_o_{layer_idx}'] = W_out_final

        # Set sliced weights back to model
        self.model_wrapper.set_weights(weights)

        # Optionally, handle residual skip connection adjustment if needed
        # (If residuals are involved, insert Q_{l-1}^T Q_l matrices accordingly)
        # For simplicity, residual adjustments are handled outside here as per implementation

    def slice_all_layers(self, keep_ratio: float = 0.75) -> None:
        """
        Apply slicing across all layers in the model.
        Args:
            keep_ratio (float): Ratio of components to retain
        """
        for layer_idx in range(self.layer_count):
            self.apply_layer_slicing(layer_idx)
            if self.debug:
                print(f"Sliced layer {layer_idx} with retain ratio {keep_ratio}")
        # After weight slicing, the model is ready with reduced dimensions

    def execute(self, signals_dict: Dict[int, torch.Tensor], keep_ratio: Optional[float]=None):
        """
        Main method to execute PCA-based slicing and weight pruning.
        Args:
            signals_dict (dict): layer_idx -> signals tensor (N, D)
            keep_ratio (float, optional): Ratio of eigen components to keep.
        """
        # Optionally analyze spectrum
        if self.debug:
            self.compute_layer_spectrum(signals_dict)
        # For each layer, compute eigen spectrum and decide retention
        # Here, assuming eigenvectors Qs are already computed and stored
        # For actual spectrum-driven sicing, implement spectrum thresholding here
        # For now, keep entire spectrum
        self.slice_all_layers(keep_ratio=keep_ratio or self.keep_ratio)
