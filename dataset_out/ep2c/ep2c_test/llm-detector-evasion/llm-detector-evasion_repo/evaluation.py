## evaluation.py
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import expit
from typing import List, Dict, Optional
from detectors import DetectorAPI
from model import ModelWrapper

class Evaluation:
    """
    Evaluation class to compute metrics such as AUROC and perplexity for a given
    set of texts, using specified detectors and a language model.
    """

    def __init__(
        self,
        model: ModelWrapper,
        detectors: List[DetectorAPI],
        device: str = "cuda",
        detector_names: Optional[List[str]] = None,
        human_texts: Optional[List[str]] = None,
        human_labels: Optional[List[int]] = None,
        texts: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
    ):
        """
        Initializes the evaluator.
        Args:
            model (ModelWrapper): The language model for perplexity eval.
            detectors (List[DetectorAPI]): List of detector instances.
            device (str): 'cuda' or 'cpu'.
            detector_names (Optional[List[str]]): Names corresponding to detectors.
            human_texts (Optional[List[str]]): Texts presumed human-written (for AUROC).
            human_labels (Optional[List[int]]): Labels for human texts (1).
            texts (Optional[List[str]]): Texts to evaluate for metrics.
            labels (Optional[List[int]]): True labels for texts, 1=human, 0=AI.
        """
        self.model = model
        self.detectors = detectors
        self.device = device
        self.detector_names = detector_names or [f"Detector_{i}" for i in range(len(detectors))]
        self.human_texts = human_texts
        self.human_labels = human_labels
        self.texts = texts
        self.labels = labels

        # Validate inputs
        if self.human_texts is not None and self.human_labels is None:
            raise ValueError("If human_texts is provided, human_labels must be provided.")
        if self.texts is not None and self.labels is None:
            raise ValueError("If texts are provided, labels must be provided.")

    def compute_detector_scores(
        self,
        texts: List[str],
    ) -> Dict[str, List[float]]:
        """
        Obtain scores from all detectors for each text.
        Args:
            texts (List[str]): Texts to score.
        Returns:
            Dict[str, List[float]]: A dict mapping detector name to list of scores.
        """
        scores_dict = {}
        for det, name in zip(self.detectors, self.detector_names):
            scores = []
            for text in texts:
                score = det.score(text)
                scores.append(score)
            scores_dict[name] = scores
        return scores_dict

    def evaluate_detector_auroc(
        self,
        detector_scores: Dict[str, List[float]],
        true_labels: List[int],
    ) -> Dict[str, float]:
        """
        Compute AUROC for each detector given scores and ground truth labels.
        Args:
            detector_scores (Dict[str, List[float]]): Scores per detector.
            true_labels (List[int]): True labels (1=human, 0=AI).
        Returns:
            Dict[str, float]: AUROC value per detector.
        """
        auroc_dict = {}
        for name in detector_scores:
            scores = detector_scores[name]
            try:
                auroc = roc_auc_score(true_labels, scores)
            except Exception:
                auroc = float('nan')
            auroc_dict[name] = auroc
        return auroc_dict

    def compute_perplexity(self, texts: List[str]) -> float:
        """
        Compute the average perplexity over a list of texts.
        Args:
            texts (List[str]): List of texts.
        Returns:
            float: Average perplexity.
        """
        total_neg_log_likelihood = 0.0
        total_tokens = 0
        for text in texts:
            # Use model's log_prob method
            ll = self.model.log_prob(text)
            tokens = self.model.tokenizer.tokenize(text)
            token_count = max(1, len(tokens))
            # Approximate negative log-likelihood
            neg_ll = -ll
            total_neg_log_likelihood += neg_ll
            total_tokens += token_count
        if total_tokens == 0:
            return float('nan')
        avg_neg_ll = total_neg_log_likelihood / total_tokens
        perplexity = np.exp(avg_neg_ll)
        return perplexity

    def evaluate_texts(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
    ) -> Dict:
        """
        Evaluate texts: compute detector scores, AUROC (if labels provided),
        and perplexity.
        Args:
            texts (List[str]): List of texts to evaluate.
            labels (Optional[List[int]]): Ground truth labels for AUROC.
        Returns:
            Dict: Dictionary with metrics.
        """
        scores_per_detector = self.compute_detector_scores(texts)
        result = {}
        # Compute detector scores and stats
        for name, scores in scores_per_detector.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            result[f"{name}_mean_score"] = mean_score
            result[f"{name}_std_score"] = std_score

        # Compute AUROC if labels are provided
        if labels is not None:
            for name, scores in scores_per_detector.items():
                try:
                    auroc = roc_auc_score(labels, scores)
                except Exception:
                    auroc = float('nan')
                result[f"AUROC_{name}"] = auroc

        # Compute perplexity
        perplexity = self.compute_perplexity(texts)
        result['perplexity'] = perplexity

        return result

    def human_preference_evaluation(
        self,
        responses_pairs: List[Tuple[str, str]],
        detector: DetectorAPI,
    ) -> Dict[str, float]:
        """
        Human evaluation based on detector scores of response pairs.
        Args:
            responses_pairs (List[Tuple[str, str]]): List of (response1, response2).
            detector (DetectorAPI): To score responses.
        Returns:
            Dict[str, float]: Proportion of responses where response1 is more human.
        """
        count_response1_preferred = 0
        total = len(responses_pairs)
        for r1, r2 in responses_pairs:
            score1 = detector.score(r1)
            score2 = detector.score(r2)
            # Higher detector score indicates more human-like
            if score1 > score2:
                count_response1_preferred += 1
        proportion = count_response1_preferred / total if total > 0 else 0.0
        return {
            "response1_better_proportion": proportion,
            "total_pairs": total
        }

    # Optional: add additional metrics or human annotation analysis as needed
