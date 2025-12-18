## evaluation.py
import torch
from typing import Dict

class Evaluation:
    """
    Handles model evaluation on a dataset, computes specified metrics.
    
    Attributes:
        model (torch.nn.Module): The trained model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for validation/test data.
        device (torch.device): Device to run inference on.
        metrics (list of str): Metrics to compute, e.g., ['accuracy', 'loss'].
        loss_fn (callable): Loss function used for 'loss' metric.
        correct_sum (int): Accumulated number of correct predictions.
        total_samples (int): Total number of samples evaluated.
        loss_sum (float): Accumulated loss (if 'loss' in metrics).
    """
    def __init__(
        self,
        model,
        data_loader,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        metrics: list = ['accuracy']
    ):
        """
        Initialize Evaluation instance.
        Args:
            model (torch.nn.Module): The trained model.
            data_loader (DataLoader): DataLoader for evaluation dataset.
            device (torch.device): Device for inference.
            metrics (list): List of metrics to compute ('accuracy', 'loss', ...).
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.metrics = metrics
        # Use CrossEntropyLoss if 'loss' is requested
        if 'loss' in self.metrics:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = None
        # Initialize accumulators
        self.reset_metrics()
    
    def reset_metrics(self) -> None:
        """
        Reset all accumulator variables before evaluation.
        """
        self.correct_sum = 0
        self.total_samples = 0
        self.loss_sum = 0.0

    def evaluate(self) -> Dict[str, float]:
        """
        Run model on dataset, compute metrics, and return results.
        Returns:
            dict: {'accuracy': float, 'loss': float} depending on metrics
        """
        self.model.eval()
        with torch.no_grad():
            for batch in self.data_loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)

                # Compute accuracy if needed
                if 'accuracy' in self.metrics:
                    preds = torch.argmax(outputs, dim=1)
                    self.correct_sum += (preds == labels).sum().item()

                # Compute loss if needed
                if 'loss' in self.metrics:
                    loss = self.loss_fn(outputs, labels)
                    self.loss_sum += loss.item() * inputs.size(0)

                self.total_samples += inputs.size(0)

        results = {}
        if 'accuracy' in self.metrics:
            accuracy = self.correct_sum / self.total_samples if self.total_samples > 0 else 0.0
            results['accuracy'] = accuracy
        if 'loss' in self.metrics:
            mean_loss = self.loss_sum / self.total_samples if self.total_samples > 0 else 0.0
            results['loss'] = mean_loss
        return results
