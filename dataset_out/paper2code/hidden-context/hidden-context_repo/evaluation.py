## evaluation.py
"""
Evaluation module for preference models trained under various settings, focusing on:
- Calculating preference accuracy against ground truth utilities or preferences.
- Computing rank correlation metrics (Spearman's ρ, Kendall's τ) between model predictions and true utilities.
- Deriving Borda counts from model pairwise preference probabilities and comparing their orderings.
- Detecting the influence of hidden context via analysis of distributional outputs (mean, variance, or categorical probs).
Provides flexibility for synthetic and real datasets, distributional and scalar models.
"""

import numpy as np
from scipy.stats import spearmanr, kendalltau

class Evaluation:
    def __init__(self, model, dataset, ground_truth_utils=None, model_output_type='scalar'):
        """
        Initialize the evaluator.
        
        Args:
            model: The preference model instance, with method predict() that returns model outputs.
            dataset: Dataset object containing ComparisonPair instances.
            ground_truth_utils: Optional dict mapping alternative to true utility (for synthetic data).
            model_output_type: str, one of {'scalar', 'mean_var', 'categorical'} indicating model's output form.
        """
        self.model = model
        self.dataset = dataset
        self.ground_truth_utils = ground_truth_utils
        self.model_output_type = model_output_type
        # Parse dataset into structured form for evaluation
        self._prepare_data()

    def _prepare_data(self):
        """
        Prepare data arrays for evaluation:
        - All alternatives involved
        - Ground truth utilities if available
        """
        # Collect all unique alternatives
        self.alternatives = set()
        for pair in self.dataset.pairs:
            self.alternatives.add(pair.a)
            self.alternatives.add(pair.b)
        self.alternatives = sorted(list(self.alternatives))
        self.idx_map = {a: i for i, a in enumerate(self.alternatives)}
        # Store ground truth utilities if provided
        if self.ground_truth_utils:
            self.true_utils = np.array([self.ground_truth_utils.get(a, 0.0) for a in self.alternatives])
        else:
            self.true_utils = None

        # Initialize container for model predictions
        self.predicted_utils = np.zeros(len(self.alternatives))
        # To hold distributional info if needed
        if self.model_output_type != 'scalar':
            self.dist_params = [None] * len(self.alternatives)
        else:
            self.dist_params = None
        # Collect pairwise preferences for preference accuracy
        self.preference_labels = []

    def evaluate(self):
        """
        Compute evaluation metrics:
        - Preference accuracy
        - Rank correlation (Spearman, Kendall)
        - Borda counts and their comparison
        - Hidden context detection metrics
        Returns a dictionary of metrics.
        """
        preference_acc = self._compute_preference_accuracy()
        spearman_corr, _ = spearmanr(self.predicted_utils, self.true_utils) if self.true_utils is not None else (None, None)
        kendall_tau, _ = kendalltau(self.predicted_utils, self.true_utils) if self.true_utils is not None else (None, None)
        borda_scores = self.compute_borda_counts()

        detection_metrics = self.detect_hidden_context()

        metrics = {
            'preference_accuracy': preference_acc,
            'spearman_correlation': spearman_corr,
            'kendall_tau': kendall_tau,
            'borda_counts': borda_scores,
            'hidden_context_detection': detection_metrics
        }
        return metrics

    def _compute_preference_accuracy(self):
        """
        Compute the fraction of preference pairs correctly predicted by the model.
        """
        correct = 0
        total = 0
        for pair in self.dataset.pairs:
            a_idx = self.idx_map[pair.a]
            b_idx = self.idx_map[pair.b]

            pred = self._predict_preference_for_pair(pair.a, pair.b)
            if pred is None:
                continue  # Skip if prediction not computable
            # ground truth preference
            gt_pref = pair.preference
            pred_pref = 1 if pred > 0.5 else 0
            if pred_pref == gt_pref:
                correct += 1
            total += 1
        return correct / total if total > 0 else None

    def _predict_preference_for_pair(self, a, b):
        """
        Estimate preference probability that a > b based on model outputs.
        """
        a_idx = self.idx_map[a]
        b_idx = self.idx_map[b]

        if self.model_output_type == 'scalar':
            ua = self._get_utility_value(a_idx)
            ub = self._get_utility_value(b_idx)
        elif self.model_output_type == 'mean_var':
            ua = self._get_mean_variance(a_idx)[0]
            ub = self._get_mean_variance(b_idx)[0]
        elif self.model_output_type == 'categorical':
            ua = self._get_expected_utility_categorical(a_idx)
            ub = self._get_expected_utility_categorical(b_idx)
        else:
            raise ValueError(f"Unknown model output type: {self.model_output_type}")

        diff = ua - ub
        pred_prob = 1 / (1 + np.exp(-diff))
        return pred_prob

    def _get_utility_value(self, idx):
        """
        Get scalar utility value for alternative at index, from model.
        """
        # Obtain model prediction
        a_str = self.dataset.pairs[0].prompt_response_a  # Placeholder if needed
        # For batch evaluation, generate a batch of prompts as needed
        # For simplicity, assuming model.predict(a) which returns scalar
        try:
            pred_output = self.model.predict(self.dataset.pairs[idx].prompt_response_a, None)
            return pred_output['utility'].item()
        except Exception:
            # fallback or default
            return 0.0

    def _get_mean_variance(self, idx):
        """
        Retrieve mean and variance from distributional model output.
        """
        try:
            pred_output = self.model.predict(self.dataset.pairs[idx].prompt_response_a, None)
            mean = pred_output['mean'].item()
            var = pred_output['variance'].item()
            return mean, var
        except Exception:
            return 0.0, 0.0

    def _get_expected_utility_categorical(self, idx):
        """
        Compute expected utility from categorical distribution.
        """
        try:
            pred_output = self.model.predict(self.dataset.pairs[idx].prompt_response_a, None)
            probs = pred_output['probs'].detach().cpu().numpy()
            u_bins = np.linspace(0, 1, self.model.num_outputs)
            expected_util = np.sum(probs * u_bins)
            return expected_util
        except Exception:
            return 0.0

    def compute_borda_counts(self):
        """
        Calculate Borda counts for each alternative based on predicted pairwise preference probabilities.
        """
        n = len(self.alternatives)
        BC = np.zeros(n)
        for i, a in enumerate(self.alternatives):
            sum_probs = 0.0
            for j, b in enumerate(self.alternatives):
                if i == j:
                    continue
                prob_ab = self._predict_pairwise_probability(a, b)
                sum_probs += prob_ab
            BC[i] = sum_probs / (n - 1)
        # Store or output BC scores
        return {a: BC[i] for i, a in enumerate(self.alternatives)}

    def _predict_pairwise_probability(self, a: float, b: float):
        """
        Predict probability that alternative a is preferred over b.
        """
        a_idx = self.idx_map[a]
        b_idx = self.idx_map[b]
        if self.model_output_type == 'scalar':
            ua = self._get_utility_value(a_idx)
            ub = self._get_utility_value(b_idx)
        elif self.model_output_type == 'mean_var':
            ua = self._get_mean_variance(a_idx)[0]
            ub = self._get_mean_variance(b_idx)[0]
        elif self.model_output_type == 'categorical':
            ua = self._get_expected_utility_categorical(a_idx)
            ub = self._get_expected_utility_categorical(b_idx)
        else:
            raise ValueError(f"Unknown model output type: {self.model_output_type}")

        # Logistic probability
        prob = 1 / (1 + np.exp(-(ua - ub)))
        return prob

    def detect_hidden_context(self):
        """
        Analyze distributional parameters or variance to identify signals of hidden context influence.
        For models outputting variance, high variance may suggest high hidden context.
        For models with explained variance (e.g., r^2), compare residual vs. total variance.
        """
        high_variance_alternatives = []
        explained_variances = []
        for idx in range(len(self.alternatives)):
            if self.model_output_type != 'scalar' and hasattr(self.model, 'dist_params') and self.model.dist_params:
                # For distributional outputs
                params = self.model.dist_params[idx]
                if params is None:
                    continue
                if 'variance' in params:
                    var = params['variance']
                    # Threshold arbitrarily set at some value (e.g., >0.1)
                    if var > 0.1:
                        high_variance_alternatives.append(self.alternatives[idx])
                elif 'probs' in params:
                    # For categorical, variance can be derived
                    probs = params['probs']
                    expected_util = self._get_expected_utility_categorical(idx)
                    variance = np.sum(probs.numpy() * (np.linspace(0,1,self.model.num_outputs) - expected_util)**2)
                    if variance > 0.1:
                        high_variance_alternatives.append(self.alternatives[idx])
        # Optionally, compute aggregate metrics
        return {
            'high_variance_alternatives': high_variance_alternatives
        }
