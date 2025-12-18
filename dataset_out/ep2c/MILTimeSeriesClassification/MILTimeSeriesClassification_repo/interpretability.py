## interpretability.py

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import roc_auc_score

class Interpretability:
    """
    Provides methods for computing interpretability scores, perturbation analysis,
    and ND CG evaluation for MIL-based TSC models with different pooling methods.
    """
    def __init__(self, model: torch.nn.Module, pooling_method: str = 'GAP', config: dict = None):
        """
        Initialize with trained model, pooling method, and configuration.
        Args:
            model (torch.nn.Module): Trained MIL model with pooling.
            pooling_method (str): One of 'GAP', 'Attention', 'Instance', 'Additive', 'Conjunctive'.
            config (dict): Additional configuration, optional.
        """
        self.model = model
        self.pooling_method = pooling_method
        self.config = config if config is not None else {}
        # Set default interpretability parameters
        self.n_repeats = self.config.get('interpretability', {}).get('evaluation_repeat', 3)
        # Determine model type and extract necessary parts
        self.device = next(model.parameters()).device
        # Placeholders for extracting importance scores
        self._prepare_model_for_extraction()

    def _prepare_model_for_extraction(self):
        """
        Prepares model for extracting importance scores based on pooling method.
        """
        # Depending on the pooling method, different model parts provide importance info
        # For Attention: use attention weights directly
        # For class-specific: use per-time-point class predictions / CAM
        # For CAM, ensure model supports hook registration (not implemented here)
        pass

    def compute_scores(self, series: np.ndarray, class_idx: int = None) -> np.ndarray:
        """
        Compute importance scores for a single series, optionally for a specific class.
        Args:
            series (np.ndarray): 1D array, shape (t,)
            class_idx (int): Index of class for class-specific methods; if None, use predicted class.
        Returns:
            np.ndarray: importance scores, shape (t,)
        """
        self.model.eval()
        series_tensor = torch.tensor(series, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape (1,1,t)
        series_tensor = series_tensor.to(self.device)

        # Forward pass through the model
        with torch.no_grad():
            # The model should output relevant intermediates based on pooling
            # For example, for attention, retrieve attention weights, etc.
            output, attentions, per_time_preds = self._forward_with_intermediate(series_tensor)

        # Determine which importance scores to produce based on method
        if self.pooling_method == 'Attention':
            # Use attention weights directly for importance
            # attentions: shape (1, t, 1)
            attn_weights = attentions.squeeze(0).squeeze(-1).cpu().numpy()
            scores = attn_weights
            return scores

        elif self.pooling_method in ['GAP', 'Additive', 'Conjunctive']:
            # Use per-time-point class predictions
            # per_time_preds: shape (1, t, c)
            preds = per_time_preds.squeeze(0)  # (t, c)
            # Determine class index: predicted or provided
            if class_idx is None:
                class_idx = preds.mean(dim=0).argmax().item()
            class_scores = preds[:, class_idx].cpu().numpy()
            return class_scores

        elif self.pooling_method == 'Instance':
            # Use class predictions per time point
            preds = per_time_preds.squeeze(0)  # (t, c)
            if class_idx is None:
                class_idx = preds.mean(dim=0).argmax().item()
            class_scores = preds[:, class_idx].cpu().numpy()
            return class_scores

        else:
            # Default: if no specific, fallback to class scores
            # For safety, produce equal importance
            return np.ones(series.shape[0])

    def _forward_with_intermediate(self, series_tensor: torch.Tensor):
        """
        Forward pass that returns model outputs and importance-related intermediates.
        Must be implemented based on model specifics and pooling.
        Returns:
            output (tensor): class logits
            attentions (tensor): attention weights if available
            per_time_preds (tensor): per-time-step predictions if available
        """
        # For illustration, assuming model returns these:
        # e.g., model(series_tensor) -> (logits, attentions, per_time_preds)
        # Users must adapt or modify according to their model.
        # Here, we assume model returns a dict or have attributes.
        # If not, implement custom hooks or modify models accordingly.
        # For this implementation, we assume:
        # - If model has attribute 'attention_scores': use it.
        # - Else, use per-time predictions from model output.

        # Pseudo code: replace or adapt accordingly
        output = self.model(series_tensor)  # e.g., class logits
        attentions = None
        per_time_preds = None

        if hasattr(self.model, 'attention_weights'):
            attentions = self.model.attention_weights  # shape (1, t, 1)
        if hasattr(self.model, 'per_time_predictions'):
            per_time_preds = self.model.per_time_predictions  # shape (1, t, c)
        else:
            # For models without explicit per_time_predictions, try to get gradients or saliency maps if implemented
            pass

        return output, attentions, per_time_preds

    def compute_perturbation(self, series: np.ndarray, class_idx: int = None):
        """
        Sequentially remove important points, record predicted class confidence.
        Args:
            series (np.ndarray): 1D array (t,)
            class_idx (int): class index, if None, use predicted class from full series.
        Returns:
            decay_curve (list): model confidence at each removal step
            aopcr_score (float): area over perturbation curve
        """
        t = len(series)
        importance_scores = self.compute_scores(series, class_idx=class_idx)
        # Rank points from most important to least
        order = importance_scores.argsort()[::-1]
        # Initialize perturbed series
        perturbed_series = series.copy()

        decay_curve = []
        # Decide how many points to remove, e.g., 50%
        n_remove = max(1, int(0.5 * t))
        for i in range(n_remove):
            # Remove the top importance point(s)
            idx_to_remove = order[i]
            # Replace with mean value or zero (here, zero)
            perturbed_series[idx_to_remove] = 0.0  # or np.mean(series)

            # Convert to tensor and predict
            series_tensor = torch.tensor(perturbed_series, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            series_tensor = series_tensor.to(self.device)
            with torch.no_grad():
                output, _, _ = self._forward_with_intermediate(series_tensor)
                probs = torch.softmax(output, dim=1)
                pred_conf = probs.max(dim=1)[0].item()  # Confidence of predicted class
                decay_curve.append(pred_conf)

        # Compute AOPCR: area over the decay curve vs random
        aopcr_score = self._compute_AOPCR(decay_curve)

        return decay_curve, aopcr_score

    def _compute_AOPCR(self, decay_curve: list):
        """
        Compute Area over Perturbation Curve relative to random.
        Args:
            decay_curve (list): model confidence at each step.
        Returns:
            float: AOPCR score.
        """
        x = np.arange(1, len(decay_curve)+1)
        # Curves:
        curve = np.array(decay_curve)
        # Random curve average
        rand_curves = []
        for _ in range(self.n_repeats):
            np.random.shuffle(x)
            rand_curve = np.array([curve[xx-1] for xx in x])
            rand_curves.append(np.mean(rand_curve))
        rand_avg = np.mean(rand_curves)
        # Area over the curve: using trapezoidal rule
        area_curve = np.trapz(curve, x)
        area_rand = np.trapz(np.array(rand_curves), x)
        # Normalize by max possible area (max confidence * length)
        max_area = max(np.max(curve), np.max(np.array(rand_curves))) * len(curve)
        # Compute normalized difference
        aopcr = (area_rand - area_curve) / max_area
        return aopcr

    def compute_ndcgc(self, series: np.ndarray, true_signature_indices: list):
        """
        Computes normalized discounted cumulative gain at n (assuming true_signature_indices).
        Args:
            series (np.ndarray): 1D array (t,)
            true_signature_indices (list): list of true signature point indices.
        Returns:
            float: ND CG score between 0 and 1.
        """
        importance_scores = self.compute_scores(series)
        # Rank importance scores descending
        ranked_indices = importance_scores.argsort()[::-1]
        n_signatures = len(true_signature_indices)
        rel = np.zeros(n_signatures)
        for i, sig_idx in enumerate(true_signature_indices):
            # Position of true signature ranked
            rank_pos = np.where(ranked_indices == sig_idx)[0]
            if len(rank_pos) > 0:
                rel[i] = rank_pos[0] + 1  # 1-based rank
            else:
                # Not found, assign worst rank
                rel[i] = len(importance_scores) + 1
        # Compute weighted sum with discount (log base 2)
        denom = np.sum(1/np.log2(np.arange(2, n_signatures+2)))
        score = 0.0
        for i, r in enumerate(rel):
            score += (1.0 / np.log2(r+1))
        ndcg = score / denom if denom > 0 else 0.0
        # Normalize to [0,1], higher means better rank
        return ndcg

