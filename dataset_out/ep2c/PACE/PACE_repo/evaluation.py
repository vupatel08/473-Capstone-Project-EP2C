## evaluation.py
import torch
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np
from datasets import Dataset

class Evaluation:
    """
    Class to evaluate a model on a given dataset split, computing specified metrics.
    Supports optional gradient norm computation for analysis of regularization effects.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        config: Dict,
        device: Optional[str] = 'cuda:0',
        verbose: bool = True
    ):
        """
        Initialize Evaluation instance.
        Args:
            model: The trained PyTorch model instance.
            dataset: Dataset object providing data loader method.
            config: Dict with evaluation settings, including metrics.
            device: String or torch.device for inference.
            verbose: If True, print detailed logs.
        """
        self.model = model
        self.dataset = dataset
        self.device = torch.device(device if isinstance(device, str) else device)
        self.verbose = verbose

        # Extract metrics configuration
        self.metrics = config.get('evaluation', {}).get('metrics', ['accuracy'])
        self.metric_target = config.get('evaluation', {}).get('metric_target', 'validation')
        self.save_best = config.get('evaluation', {}).get('save_best_model', True)
        self.eval_interval = config.get('evaluation', {}).get('evaluation_interval', 10)

        # Metrics storage
        self.best_score = None
        self.best_model_state_dict = None
        self.log_metrics_history = []

        # Set model in evaluation mode
        self.model.eval()
        self.model.to(self.device)

        # Handle optional gradient norm tracking
        self.compute_grad_norms = self._check_need_grad_norm()

        if self.compute_grad_norms:
            self.grad_norms = []

    def _check_need_grad_norm(self) -> bool:
        """
        Decide whether to compute gradient norms based on metrics or config.
        For simplicity, assume always false unless explicitly requested.
        """
        # For this implementation, as per guidelines, compute only if specified.
        # For demonstration, set False; can be extended to make conditional.
        return False

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the dataset split, compute metrics.
        Returns:
            Dictionary with metrics values.
        """
        dataloader = self.dataset.get_dataloader(split=self.metric_target, batch_size=128, shuffle=False)
        all_preds = []
        all_labels = []

        # For gradient norms, manage gradient context
        if self.compute_grad_norms:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)

        total_loss = 0.0
        total_samples = 0
        metrics_results = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs, labels = batch
                inputs = self._to_device(inputs)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model.forward(inputs, perturb_params={'apply_perturb': False})
                # outputs shape expected: for classification, logits (batch x classes)
                # for regression, continuous values, etc.

                # Collect predictions
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())

                # Compute per-batch loss if relevant
                # For classification: cross-entropy
                # For regression: MSE
                # For now, just accumulate for metrics
                # Could implement specific losses if needed in config

        # Concatenate all outputs and labels
        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Compute requested metrics
        for metric in self.metrics:
            if metric.lower() == 'accuracy' or metric.lower() == 'acc':
                acc = self.compute_accuracy(preds, labels)
                metrics_results['accuracy'] = acc
                if self.verbose:
                    print(f"Validation Accuracy: {acc:.4f}")
                # Check for best
                self._update_best(metric_score=acc, metric_name='accuracy')
            elif metric.lower() == 'mse':
                mse = self.compute_mse(preds, labels)
                metrics_results['mse'] = mse
                if self.verbose:
                    print(f"Validation MSE: {mse:.4f}")
            elif metric.lower() == 'correlation':
                corr = self.compute_correlation(preds, labels)
                metrics_results['correlation'] = corr
                if self.verbose:
                    print(f"Validation Corr: {corr:.4f}")
            else:
                if self.verbose:
                    print(f"Unknown metric '{metric}', skipping.")

        # Optional: compute gradient norms
        if self.compute_grad_norms:
            grad_norm = self._compute_model_gradient_norm()
            metrics_results['gradient_norm'] = grad_norm
            if self.verbose:
                print(f"Gradient Norm: {grad_norm:.4f}")

        return metrics_results

    def _to_device(self, inputs: Dict) -> Dict:
        """
        Transfer inputs, assuming inputs is a dict of tensors.
        """
        # For datasets with dict inputs (images, input_ids, etc.)
        device_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                device_inputs[k] = v.to(self.device)
            else:
                device_inputs[k] = v
        return device_inputs

    def compute_accuracy(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute classification accuracy.
        Assumes preds are logits.
        """
        if preds.ndim > 1 and preds.shape[1] > 1:
            pred_labels = torch.argmax(preds, dim=1)
        else:
            # For binary classification or regression, possibly threshold
            pred_labels = (preds > 0.5).long() if preds.ndim > 1 else (preds > 0).long()
        correct = (pred_labels == labels).sum().item()
        total = labels.numel()
        return correct / total

    def compute_mse(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute Mean Squared Error.
        """
        mse = F.mse_loss(preds.squeeze(), labels.float()).item()
        return mse

    def compute_correlation(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute Pearson correlation coefficient.
        """
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        if preds_np.ndim > 1 and preds_np.shape[1] == 1:
            preds_np = preds_np.squeeze()
        if labels_np.ndim > 1 and labels_np.shape[1] == 1:
            labels_np = labels_np.squeeze()
        if len(preds_np) == 0:
            return 0.0
        corr = np.corrcoef(preds_np, labels_np)[0,1]
        return corr

    def _compute_model_gradient_norm(self) -> float:
        """
        Compute L2 norm of gradients across all trainable model parameters.
        """
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_sq += torch.sum(p.grad.data ** 2).item()
        total_norm = np.sqrt(total_norm_sq)
        return total_norm

    def _update_best(self, metric_score: float, metric_name: str):
        """
        If current metric improves on best, save model state.
        """
        if self.best_score is None or metric_score > self.best_score:
            self.best_score = metric_score
            self.best_model_state_dict = self.model.state_dict()

    def _save_checkpoint(self, path: str):
        """
        Save model checkpoint to file.
        """
        torch.save(self.model.state_dict(), path)

