## evaluation.py
import numpy as np
import scipy.stats
from typing import List, Tuple
from .likelihood import compute_divergence

class Evaluation:
    """
    Handles evaluation of distributional meaning representations:
    - Semantic similarity via correlation metrics
    - Entailment (directionality) classification
    - Hyponym/hypernym relation prediction
    """

    def __init__(self, prediction_scores: List[float], labels: List[float]):
        """
        Initialize the Evaluation object.
        Args:
            prediction_scores (List[float]): Divergence or similarity scores; higher similarity or lower divergence indicates stronger relation.
            labels (List[float]): Ground-truth labels (e.g., human similarity scores, binary labels).
        """
        self.scores = np.array(prediction_scores)
        self.labels = np.array(labels)

    def calculate_spearman(self) -> float:
        """
        Compute Spearman rank correlation between scores and labels.
        Returns:
            float: Spearman correlation coefficient scaled by 100.
        """
        rho, _ = scipy.stats.spearmanr(self.scores, self.labels)
        return rho * 100.0 if rho is not None else float('nan')

    def calculate_accuracy(self, predictions: List[int], labels: List[int]) -> float:
        """
        Compute accuracy given binary predictions and true labels.
        Args:
            predictions (List[int]): Predicted labels (0 or 1).
            labels (List[int]): Ground truth labels (0 or 1).
        Returns:
            float: Accuracy score in [0, 1].
        """
        correct = sum(p == l for p, l in zip(predictions, labels))
        return correct / len(labels) if len(labels) > 0 else float('nan')

    def evaluate_similarity(self) -> float:
        """
        Evaluate semantic similarity via Spearman correlation.
        Returns:
            float: Spearman correlation scaled by 100.
        """
        return self.calculate_spearman()

    def evaluate_entailment(self, divergence_uv: List[float], divergence_vu: List[float], labels: List[int]) -> float:
        """
        Infer entailment direction from divergences and evaluate accuracy.
        Args:
            divergence_uv (List[float]): Divergence scores d(M_u, M_v).
            divergence_vu (List[float]): Divergence scores d(M_v, M_u).
            labels (List[int]): Ground truth labels (1: u entails v, 0: v entails u).
        Returns:
            float: Accuracy of entailment prediction.
        """
        predictions = [1 if du < dv else 0 for du, dv in zip(divergence_uv, divergence_vu)]
        return self.calculate_accuracy(predictions, labels)

    def evaluate_hypernymy(self, divergence_u: List[float], divergence_v: List[float], labels: List[int]) -> float:
        """
        Predict hyponym/hypernym relation based on divergences.
        Args:
            divergence_u (List[float]): Divergence between word u's distribution and v's.
            divergence_v (List[float]): Divergence between word v's distribution and u's.
            labels (List[int]): Ground truth relation labels (e.g., 1 for v hyponym of u, 0 otherwise).
        Returns:
            float: Accuracy of hyponym/hypernym prediction.
        """
        predictions = [1 if dv < du else 0 for du, dv in zip(divergence_u, divergence_v)]
        return self.calculate_accuracy(predictions, labels)

    def evaluate_multimodal_similarity(self, divergence_scores: List[float], labels: List[float]) -> float:
        """
        Evaluate similarities in multimodal setting (e.g., image-image, image-caption).
        Args:
            divergence_scores (List[float]): Divergence scores between multimodal pairs.
            labels (List[float]): Human-annotated similarity scores.
        Returns:
            float: Spearman correlation scaled by 100.
        """
        return self.calculate_spearman()

    def report(self):
        """
        Optional: Provide a summary report of evaluation metrics.
        """
        print(f"Spearman correlation: {self.scores}")
        # Further detailed reports can be added as needed.
