## explanation.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, List
from torch import Tensor

from config import config

class Explanation:
    """
    Implements instance-wise relevance attribution for MIL models using xMIL-LRP.
    Supports models: attention-based, transformer-based, additive MIL.
    Computes relevance scores per instance (and per feature), revealing positive/negative evidence.
    """
    def __init__(self, model: nn.Module, explanation_method: str = "xMIL-LRP"):
        """
        Initialize Explanation object.
        Args:
            model (nn.Module): Trained MIL model.
            explanation_method (str): Method to use, default "xMIL-LRP".
        """
        self.model = model
        self.method = explanation_method
        # Select relevance propagation rules according to configuration
        self.rules = config.explanation_method['relevance_rules']
        # Epsilon for epsilon-rule (numerical stability in linear relevance propagation)
        self.epsilon = 1e-6

    def compute_relevance(self, features: Tensor, prediction: Optional[Tensor] = None, target_class: Optional[int] = None) -> List[float]:
        """
        Computes relevance scores for input features of shape (K, D) for one bag.
        Args:
            features (torch.Tensor): The instance features tensor (K, D), requires_grad=False.
            prediction (torch.Tensor): Model's output score for the bag, if precomputed.
            target_class (int): Index of class of interest; if None, use model's predicted class.
        Returns:
            List[float]: Relevance scores aggregated per feature for each instance, then flattened.
        """
        # Setup input features
        features = features.requires_grad_(True)
        # Forward pass
        output_score = self._get_model_output(features) if prediction is None else prediction
        if target_class is None:
            target_score = output_score
        else:
            # Assuming binary classification as in paper (output scalar), so treat as class 0 or 1
            target_score = output_score

        # Initialize relevance at output as the score itself
        R = target_score
        # Store relevance at each layer
        relevance_scores = {}
        # Start backward propagation
        relevance_scores['output'] = R

        # Backpropagate relevance through the network layers
        # (Wrap in a recursive or iterative method)
        relevance_scores = self._relprop(self.model, features, relevance_scores['output'])
        # relevance_scores now contain relevance at input features (per feature)
        # Sum over features to get per-instance relevance
        instance_relevance = []
        for k in range(features.shape[0]):
            rel_feat = relevance_scores['input'][k]  # shape: (D,)
            epsilon_score = rel_feat.sum().item()
            instance_relevance.append(epsilon_score)
        return instance_relevance

    def _get_model_output(self, features: Tensor, class_idx: Optional[int] = None) -> Tensor:
        """
        Forward pass through the model given features input.
        Args:
            features (Tensor): (K, D)
            class_idx (int): Optional, index for class-specific explanation.
        Returns:
            Tensor: Scalar output score for given features.
        """
        # Expand features to batch size 1
        feats = features.unsqueeze(0)  # shape: (1, K, D)
        # Forward pass depending on the model type
        if hasattr(self.model, 'forward'):
            output = self.model(feats)
            if isinstance(output, (list, tuple)):
                output = output[0]
            # output shape: (1, 1)
        else:
            raise RuntimeError("Model does not have forward method.")
        return output.squeeze()

    def _relprop(self, model: nn.Module, features: Tensor, R_out: Tensor):
        """
        Recursive relevance propagation for each layer.
        Args:
            model (nn.Module): The model or sub-module (attention, linear, norm).
            features (Tensor): Input features to current layer.
            R_out (Tensor): Relevance output from the layer (scalar or vector).
        Returns:
            Dict: relevance at previous layer's neurons/features, with keys:
                'input': tensor of relevance scores per feature for each input instance.
        """
        # Placeholder: Determine the layer type and apply appropriate relevance rule
        # for simplicity, assume linear or attention layers are functions we can call
        # in practice, this function needs to traverse model structure (e.g., via hooks or a custom wrapper)
        # But for demonstration, implement core logic for basic linear and attention rules

        # --- Linear layer relevance propagation ---
        if isinstance(model, nn.Linear):
            return self._linear_relprop(model, features, R_out)
        elif hasattr(model, 'attention') or ('attention' in str(model).lower()):
            return self._attention_relprop(model, features, R_out)
        elif isinstance(model, nn.LayerNorm):
            return self._layernorm_relprop(model, features, R_out)
        elif isinstance(model, nn.Sequential) or isinstance(model, nn.Module):
            # Recursively apply to contained modules
            # For simplicity, assume the model wrapper handles propagation
            # Here, we need a custom wrapper for the full model......
            # For code completeness, we assume direct linear or attention layer
            return {'input': features, 'relevance': R_out}
        elif isinstance(model, nn.ReLU):
            # ReLU does not change relevance magnitude; relevance flows unchanged where activation >0
            return {'input': features, 'relevance': R_out}
        else:
            # Default: return relevance unchanged
            return {'input': features, 'relevance': R_out}

    def _linear_relprop(self, layer: nn.Linear, inputs: Tensor, R_out: Tensor):
        """
        Propagate relevance for linear layer using epsilon rule.
        """
        with torch.no_grad():
            W = layer.weight  # shape: (out_dim, in_dim)
            b = layer.bias  # shape: (out_dim)
            inputs = inputs.requires_grad_(True)
            # Forward pass contribution
            Z = W @ inputs.T + b.unsqueeze(1)  # shape: (out_dim, batch_size)
            # Stabilize denominator
            Z += self.epsilon * torch.sign(Z)
            # Distribute relevance proportionally
            # R_out shape: (batch_size,)
            # assume R_out spread evenly across out_dim
            # For scalar output, R_out is scalar
            denom = Z.sum(dim=0, keepdim=True)  # sum over out_dim
            # Compute relevance for each input feature
            # Layer contribution: R_opt * (Z / denom)
            # For simplicity, assume scalar R_out
            relevance_input = torch.zeros_like(inputs)
            # No batch assumption: shape: (in_dim,)
            # For vectorized code, need to handle batch; here, single instance
            # For batch, expand accordingly
            # For simplicity, assume R_out is scalar and features shape: (K, D)
            # Recompute accordingly
            # As per paper, for batch, distribute relevance proportionally
            for i in range(inputs.shape[0]):  # over in_dim
                relevance_input[i] = (inputs[i] * torch.sum(W[:, i] * R_out))
            return {'input': relevance_input}

    def _attention_relprop(self, layer, inputs, R_out):
        """
        Relevance redistribution for attention modules based on AH-rule.
        """
        # Assume attention layer has attributes: attention scores, value input
        # For simplicity, assume layer has stored attention scores (if not, provide externally)
        attention_scores = self._get_attention_scores(layer)  # shape: (K,)
        # Distribute relevance proportionally to attention scores and value features
        value_input = inputs  # shape: (K, D)
        relevance_input = torch.zeros_like(value_input)
        sum_scores = attention_scores.sum()
        for k in range(len(attention_scores)):
            weight = attention_scores[k] / (sum_scores + self.epsilon)
            relevance_input[k] = weight * R_out
        return {'input': relevance_input}

    def _layernorm_relprop(self, layer: nn.LayerNorm, inputs, R_out):
        """
        Relevance propagation through LayerNorm, per LN-rule.
        """
        # As per Appendix A.2, propagate relevance proportionally
        # LayerNorm normalizes over features; the relevance is distributed per feature
        mean = inputs.mean(dim=1, keepdim=True)
        std = inputs.std(dim=1, keepdim=True) + self.epsilon
        # Relevance is proportionally distributed
        relevance_input = ((inputs - mean) / std) * R_out
        return {'input': relevance_input}

    def _get_attention_scores(self, layer):
        """
        Placeholder for obtaining attention scores from a layer, if stored.
        """
        # For actual implementation, store attention weights during forward pass
        # For now, return uniform or dummy
        # This must be replaced by actual attention scores during forward.
        # For demonstration, assign equal attention
        K = 10  # placeholder, should be number of instances
        return torch.ones(K)

    def generate_heatmap(self, bag_patches: List, relevance_scores: List[float]):
        """
        Generate visualization heatmap overlayed on patch images.
        Args:
            bag_patches (List): List of patch image objects (PIL images).
            relevance_scores (List): Corresponding relevance scores per instance.
        Returns:
            heatmap_img (PIL.Image): Combined heatmap visualization.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        num_patches = len(bag_patches)
        # Normalize relevance scores for color mapping
        q1 = np.percentile(relevance_scores, 25)
        q3 = np.percentile(relevance_scores, 75)
        whisker = 1.5 * (q3 - q1)
        min_val = np.min(relevance_scores)
        max_val = np.max(relevance_scores)
        clipped_scores = np.clip(relevance_scores, min_val - whisker, max_val + whisker)
        norm_scores = (np.array(clipped_scores) - min_val) / (max_val - min_val + 1e-8)

        # Create heatmaps per patch
        heatmaps = []
        for idx, score in enumerate(norm_scores):
            color = (1, 0, 0) if score > 0.5 else (0, 0, 1)
            alpha = abs(score - 0.5) * 2  # range 0-1
            overlay = bag_patches[idx].copy()
            overlay = overlay.convert("RGBA")
            mask = Image.new("RGBA", overlay.size, color + (int(255 * alpha),))
            combined = Image.alpha_composite(overlay, mask)
            heatmaps.append(combined)

        # Compose final heatmap image
        cols = int(np.sqrt(num_patches))
        rows = (num_patches + cols - 1) // cols
        width, height = bag_patches[0].size
        new_img = Image.new('RGBA', (cols * width, rows * height))
        for i, img in enumerate(heatmaps):
            row = i // cols
            col = i % cols
            new_img.paste(img, (col * width, row * height))
        return new_img
