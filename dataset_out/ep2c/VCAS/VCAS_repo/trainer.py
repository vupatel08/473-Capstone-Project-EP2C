## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, Any, List, Optional

from dataset_loader import DatasetLoader
from importance_sampler import ImportanceSampler
from variance_estimator import VarianceEstimator
from evaluation import Evaluation

class Trainer:
    """
    Implements training routine with Variance-Controlled Adaptive Sampling (VCAS).
    Handles scheduling of importance ratio updates, integration with importance sampling,
    variance estimation, and optimizer steps.
    """
    def __init__(
        self,
        model: nn.Module,
        dataset_config: Dict[str, Any],
        importance_sampler: ImportanceSampler,
        variance_estimator: VarianceEstimator,
        hyperparameters: Dict[str, Any]
    ):
        """
        Initialize trainer with model, dataset, importance sampler, variance estimator and hyperparameters.

        Args:
            model (nn.Module): the model to train
            dataset_config (dict): dataset configuration from yaml
            importance_sampler (ImportanceSampler): importance sampler instance
            variance_estimator (VarianceEstimator): variance estimator instance
            hyperparameters (dict): hyperparameters such as tau thresholds, alpha, beta, F, M
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Load dataset
        self.dataset_loader = DatasetLoader(**dataset_config)
        self.train_loader = self.dataset_loader.get_loader()
        # For evaluation, load validation/test set similarly
        # Here, for simplicity, assume dataset_config contains validation config too
        # else, initialize Evaluation separately later
        self.eval = None  # Will instantiate after training if needed

        self.importance_sampler = importance_sampler
        self.variance_estimator = variance_estimator

        # Hyperparameters from config
        self.tau_act = hyperparameters['variance_control_thresholds'].get('activation', 0.025)
        self.tau_w = hyperparameters['variance_control_thresholds'].get('weight', 0.025)
        self.alpha = hyperparameters.get('update_step_alpha', 0.01)
        self.beta = hyperparameters.get('ratio_scaling_beta', 0.95)
        self.F = hyperparameters.get('variance_update_frequency', 100)
        self.M = hyperparameters.get('monte_carlo_samples', 4)
        self.num_epochs = hyperparameters['training'].get('epochs', 3)
        self.total_steps = hyperparameters['training'].get('total_steps', 10000)

        # Initialize importance ratios (for data points and tokens in each layer)
        # These will be dynamically updated
        self.rho = self.importance_sampler.rho  # list of stratified ratios per layer (activation)
        self.nu = self.importance_sampler.nu    # list for token importance ratios per layer

        # Allocate storage for importance ratios
        self._init_sampling_ratios()
        self.current_step = 0

        # Set optimizer and scheduler (use AdamW as default)
        lr = hyperparameters['training'].get('learning_rate', 2e-5)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        # For simplicity, no explicit scheduler here. Can be added later.
    
    def _init_sampling_ratios(self):
        """Initialize ratios and hyperparameters."""
        # ratios already initialized in importance_sampler
        # Here, we can also initialize any additional variables if needed
        pass

    def train(self):
        """
        Main training loop with importance sampling, variance estimation and ratio adaptation.
        """
        progress_bar = tqdm(total=self.total_steps, desc='Training')
        for epoch in range(self.num_epochs):
            for batch_idx, batch in enumerate(self.train_loader):
                if self.current_step >= self.total_steps:
                    break

                self.current_step += 1
                progress_bar.update(1)

                # ============================
                # 1. Prepare batch
                # ============================
                input_ids = batch.get('input_ids', None)
                attention_mask = batch.get('attention_mask', None)
                labels = batch.get('labels', None)
                # Send to device
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.to(self.device)
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.to(self.device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)

                # ============================
                # 2. Importance scores estimation
                # (Before forward)
                # ============================
                # Here, we should approximate importance scores. For simplicity, assume
                # current importance scores are stored or estimated based on previous gradients.
                # For autoflow, you may implement a fast heuristic or stored importance scores.
                # We'll proceed assuming importance scores are computed externally at each update interval.

                # For the sake of this code, assume importance scores are provided or estimated:
                activation_importance_scores = self._estimate_activation_importance(batch)
                weight_importance_scores = self._estimate_weight_importance()

                # ============================
                # 3. Sample indices based on importance
                # ============================
                # For each layer, sample data points and tokens based on current importance ratios
                # Returns masks indicating importance-wise selection
                self.importance_sampler.update_ratios({
                    'activation': self.rho,
                    'weight': self.nu
                })  # Update ratios using previous variance info if needed

                # For each layer, get importance masks and importance weights for activation
                # and weight sampling
                # These would typically be computed on importance scores
                # The sample_indices function creates masks consistent with importance ratios
                # For activation gradients:
                activation_mask_list, activation_scale_list = [], []
                for l in range(len(self.rho)):
                    importance_scores_layer = activation_importance_scores[l]
                    p_data = self.rho[l]
                    mask = self.importance_sampler.sample_indices(importance_scores_layer, p_data)
                    scale = torch.zeros_like(importance_scores_layer)
                    scale[mask] = 1.0 / max(self.rho[l], 1e-8)  # importance weight scaling
                    activation_mask_list.append(mask)
                    activation_scale_list.append(scale)

                # For weight gradients (tokens):
                token_mask_list, token_scale_list = [], []
                for l in range(len(self.nu)):
                    # importance scores for tokens (simulate, e.g., from importance scores or previous)
                    importance_scores_tokens = self._estimate_token_importance_scores()
                    p_tokens = self.nu[l]
                    mask_tokens = self.importance_sampler.sample_tokens(importance_scores_tokens, p_tokens)
                    scale_tokens = torch.zeros_like(importance_scores_tokens)
                    scale_tokens[mask_tokens] = 1.0 / max(self.nu[l], 1e-8)
                    token_mask_list.append(mask_tokens)
                    token_scale_list.append(scale_tokens)

                # ============================
                # 4. Forward pass
                # ============================
                # Attach importance masks/scales to model (via hooks or directly)
                # For modularity, assume model has method to set importance masks/scales
                self._set_importance_masks_masks_scale(activation_mask_list, activation_scale_list,
                                                       token_mask_list, token_scale_list)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs.logits, labels)
                self.optimizer.zero_grad()
                loss.backward()

                # ============================
                # 5. Variance estimation and ratio adjustment
                # (Every F steps)
                # ============================
                if self.current_step % self.F == 0:
                    # Monte Carlo variance estimation
                    variance_estimates = self._estimate_variance(activation_importance_scores, importance_scores_tokens=None)
                    self._adjust_sampling_ratios(variance_estimates)

                # ============================
                # 6. Optimization update
                # ============================
                self.optimizer.step()

        progress_bar.close()

    def _estimate_activation_importance(self, batch):
        """
        Estimate importance scores for activation gradients.
        For simplicity, here we may use stored or proxy metrics, or previous gradient norms.
        """
        # In practice, you'd run a small forward/backward pass to estimate importance
        # ranging from current gradients, or use heuristics.
        # For demonstration, return scaled norms or random importance scores.
        importance_scores = []
        for l in range(len(self.rho)):
            # Mock importance score: use random or fixed importance
            # E.g., sum of gradient norms from previous iteration
            importance_scores.append(torch.rand(self.device, size=(batch['input_ids'].shape[0],)))
        return importance_scores

    def _estimate_weight_importance(self):
        """
        Estimate importance scores for weights (leverage scores or similar).
        """
        # Placeholder: in real implementation, compute approximate leverage scores
        # Here, just return dummy importance per layer.
        importance_scores = []
        for _ in range(len(self.rho)):
            importance_scores.append(torch.tensor(1.0, device=self.device))
        return importance_scores

    def _estimate_token_importance_scores(self):
        """
        Estimate importance scores for tokens in the sequence.
        """
        # Placeholder: need actual importance scores based on gradient norms or leverage scores
        # For simplicity, return dummy.
        return torch.rand(10, device=self.device)  # dummy size, e.g., 10 tokens

    def _set_importance_masks_masks_scale(self, activation_masks, activation_scales,
                                          token_masks, token_scales):
        """
        Set importance masks and scales inside the model for backward modifications.
        Implemented via hooks or direct attribute mutation depending on model.
        """
        # For modularity: assume model has a method to set importance masks/scales
        # Implement model set_importance_masks() as needed.
        # For code simplicity, this is left as a placeholder.
        pass

    def _estimate_variance(self, activation_scores, importance_scores_tokens):
        """
        Use VarianceEstimator to estimate current variance for important gradients.
        """
        # Collect variance estimates for activation and weight gradients
        variance_results: Dict[str, Any] = {}
        # Activate importance scores could be used directly as importance metrics
        # in the Monte Carlo estimation inside VarianceEstimator
        # For simplicity, the implementation of VarianceEstimator is outside scope.
        # Assume it returns a dict: {'activation': float, 'weight': float}
        var_estimates = self.variance_estimator.estimate_variance(
            batch_size=activation_scores[0].shape[0],
            activation_importance_scores=activation_scores,
            weight_importance_scores=self._get_weight_importance_scores(),
            importance_ratios={'activation': self.rho, 'weight': self.nu}
        )
        return var_estimates

    def _adjust_sampling_ratios(self, variance_estimates):
        """
        Update importance sampling ratios based on variance estimates.
        Implements the rules described in Sec. 5 and equations 4-7.
        """
        # Update s (gradient norm preserver ratio) for activation
        # For each layer, use the variance estimates and thresholds
        for l in range(len(self.rho)):
            V_act = variance_estimates['activation']  # could be per layer; here simplified
            V_w = variance_estimates['weight']
            # As a heuristic, compare variance to thresholds
            # The actual techniques would apply formulas from Sec.5, e.g., Eq. 4 and Eq. 5
            # For demonstration, naive proportional adjustment:
            if V_act > self.tau_act:
                # Increase ratio to reduce variance
                self.rho[l] = min(1.0, self.rho[l] * (1 + self.alpha))
            else:
                # Decrease ratio to promote speed
                self.rho[l] = max(0.1, self.rho[l] * (1 - self.alpha))
            # Similarly, for weight importance ratios
            if V_w > self.tau_w:
                self.nu[l] = min(1.0, self.nu[l] * (1 + self.alpha))
            else:
                self.nu[l] = max(0.1, self.nu[l] * (1 - self.alpha))
        # Clamp ratios in [0.1, 1.0]
        self.rho = [max(0.1, min(r, 1.0)) for r in self.rho]
        self.nu = [max(0.1, min(n, 1.0)) for n in self.nu]

    def _get_weight_importance_scores(self):
        """
        Return current importance scores for weights (simulate or from previous computation).
        """
        importance_scores = []
        for _ in range(len(self.rho)):
            importance_scores.append(torch.tensor(1.0, device=self.device))
        return importance_scores
