## evaluation.py
import time
from typing import Dict, Any, Optional
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, spearmanr, roc_auc_score
import math
import torch.nn as nn

class Evaluation:
    """
    Handles evaluation of a trained model on a dataset, computing relevant performance metrics
    according to dataset/task type and logging resource usage metrics (time, GPU memory).
    """
    def __init__(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset,
                 dataset_name: str = '', task_type: str = '', # specify dataset/task info if available
                 device: Optional[str] = None,
                 enable_time_tracking: bool = True,
                 enable_memory_tracking: bool = True):
        """
        Initialize Evaluation with trained model and dataset.
        """
        self.model = model
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_time_tracking = enable_time_tracking
        self.enable_memory_tracking = enable_memory_tracking

        # Resource logger setup
        # Using time module for timing, torch.cuda for memory
        # No dedicated logger class needed, handle within evaluate()

    def evaluate(self) -> Dict[str, Any]:
        """
        Perform inference over the dataset, compute metrics, log resources.
        Returns a dict with metrics and resource usage info.
        """
        import torch.cuda

        # Set model to eval mode
        self.model.eval()
        self.model.to(self.device)

        # Prepare data loader
        dataloader = None
        if hasattr(self.dataset, '__iter__') and hasattr(self.dataset, '__len__'):
            # Dataset object could be Dataset or DataLoader; try to build a DataLoader
            if isinstance(self.dataset, torch.utils.data.DataLoader):
                dataloader = self.dataset
            else:
                # Build DataLoader
                # Use default batch size 128 if possible
                dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=128,
                                                          shuffle=False, num_workers=4, pin_memory=True)
        else:
            # Fallback: assume dataset is indexable
            dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=128, shuffle=False,
                                                      num_workers=4, pin_memory=True)

        # Initialize storage for predictions and labels
        all_preds = []
        all_targets = []

        # Track resource usage
        start_time = None
        total_time = None
        max_mem = 0.0

        # Optionally start cuda memory tracking
        if self.enable_memory_tracking and torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()
        
        # Start timing
        if self.enable_time_tracking:
            start_time = time.perf_counter()

        with torch.no_grad():
            for batch in dataloader:
                # Parse batch depending on dataset type
                inputs, labels = self._prepare_batch(batch)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        # End timing
        if self.enable_time_tracking:
            total_time = time.perf_counter() - start_time

        # Record max GPU memory
        if self.enable_memory_tracking and torch.cuda.is_available():
            max_mem_bytes = torch.cuda.max_memory_allocated()
            max_mem = max_mem_bytes / (1024 ** 2)  # MB
        else:
            max_mem = 0.0

        # Convert all predictions and targets to numpy arrays for metric computations
        y_pred = np.array(all_preds)
        y_true = np.array(all_targets)

        # Determine appropriate metrics based on dataset/task
        metrics: Dict[str, Any] = {}

        # Placeholder: if dataset_name and task_type not provided, try inference
        # For the purpose of the implementation, we'll assume:
        # - Image and text classification: compute accuracy and F1 (macro)
        # - Regression tasks: compute Spearman correlation
        # - Graph classification: accuracy, ROC AUC if binary

        # Default task detection: for simplicity here, assume classification if integer labels
        # Use dataset_name or task_type if provided, else default to classification
        # For real implementation, pass in task_type explicitly
        dataset_name = self.dataset_name.lower()
        task_type = self.task_type.lower() if self.task_type else ''

        # Use heuristics for task type
        if 'cifar' in dataset_name or 'imagenet' in dataset_name or 'nlp' in dataset_name or 'glue' in dataset_name:
            # Possibly classification
            # Calculate accuracy
            accuracy = accuracy_score(y_true, y_pred)
            metrics['accuracy'] = accuracy

            # F1 Score (macro)
            try:
                f1 = f1_score(y_true, y_pred, average='macro')
            except Exception:
                f1 = None
            metrics['f1_score'] = f1

            # Matthews correlation coefficient
            try:
                mcc = matthews_corrcoef(y_true, y_pred)
            except Exception:
                mcc = None
            metrics['matthews'] = mcc

            # For binary classes, compute ROC-AUC
            if len(np.unique(y_true)) == 2:
                try:
                    probas = self._get_model_probs(inputs, batch, method='classification')
                    roc_auc = roc_auc_score(y_true, probas[:,1])
                except Exception:
                    roc_auc = None
                metrics['roc_auc'] = roc_auc
            else:
                metrics['roc_auc'] = None

        elif 'graph' in dataset_name:
            # Graph classification, reasonably assume binary or multiclass accuracy
            accuracy = accuracy_score(y_true, y_pred)
            metrics['accuracy'] = accuracy

            # For binary classification, ROC-AUC
            try:
                probas = self._get_model_probs(inputs, batch, method='classification')
                if probas.shape[1] == 2:
                    roc_auc = roc_auc_score(y_true, probas[:,1])
                else:
                    roc_auc = None
            except Exception:
                roc_auc = None
            metrics['roc_auc'] = roc_auc

        elif 'regression' in task_type:
            # For regression tasks, compute Spearman
            try:
                # Obtain regression outputs
                preds = self._get_raw_outputs(inputs, batch)
                spearman_corr, _ = spearmanr(y_true, preds)
            except Exception:
                spearman_corr = None
            metrics['spearman'] = spearman_corr
        else:
            # Default fallback
            accuracy = accuracy_score(y_true, y_pred)
            metrics['accuracy'] = accuracy

        # Add resource usage info
        metrics['evaluation_time_sec'] = total_time if total_time else None
        metrics['max_gpu_memory_MB'] = max_mem

        return metrics

    def _prepare_batch(self, batch) -> (torch.Tensor, torch.Tensor):
        """
        Parse the batch to inputs and labels depending on dataset type.
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
            else:
                raise ValueError("Unexpected batch format, expected at least two elements.")
        elif isinstance(batch, dict):
            inputs = batch.get('inputs', None)
            labels = batch.get('labels', None)
        else:
            raise TypeError("Batch type not supported.")
        return inputs, labels

    def _get_raw_outputs(self, inputs, batch) -> np.ndarray:
        """
        Get raw model outputs for regression tasks, e.g., continuous values.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            preds = outputs.cpu().numpy()
        return preds

    def _get_model_probs(self, inputs: torch.Tensor, batch: Any, method='classification') -> np.ndarray:
        """
        Obtain model's probability outputs for metrics like ROC-AUC.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)
            return probs.cpu().numpy()
