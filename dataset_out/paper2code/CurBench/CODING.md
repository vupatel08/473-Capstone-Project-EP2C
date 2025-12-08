# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## curriculum_strategies.py

```python
## curriculum_strategies.py
import math
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable
import numpy as np

class CurriculumStrategy(ABC):
    """
    Base class for curriculum learning strategies.
    Defines the interface for obtaining sample weights and optional updates.
    """
    def __init__(self,
                 start_ratio: float = 0.0,
                 grow_epochs: int = 20,
                 grow_fn: str = 'linear',
                 warm_epochs: int = 5,
                 **kwargs):
        self.start_ratio = start_ratio
        self.grow_epochs = grow_epochs
        self.grow_fn = grow_fn
        self.warm_epochs = warm_epochs
        # Strategy-specific hyperparameters
        self.params = kwargs
        # For strategies that need to cache difficulty scores
        self.difficulty_scores_cache = None

    def _compute_progress(self, epoch: int) -> float:
        """
        Compute the progress ratio [0, 1] of curriculum based on current epoch.
        """
        if epoch < self.warm_epochs:
            return 0.0
        progress = min(float(epoch - self.warm_epochs) / max(1, self.grow_epochs), 1.0)
        return progress

    def _apply_grow_fn(self, progress: float) -> float:
        """
        Apply the growth function ('linear', 'exponential') to progress.
        """
        if self.grow_fn == 'linear':
            return progress
        elif self.grow_fn == 'exponential':
            # Exponential growth: curve from start_ratio to 1
            base = 2.0
            exp_progress = (math.exp(base * progress) - 1) / (math.exp(base) - 1)
            return exp_progress
        elif self.grow_fn == 'logarithmic':
            # Logarithmic growth (clamped for progress>0)
            if progress == 0:
                return 0.0
            log_progress = math.log(1 + 9 * progress) / math.log(10)
            return log_progress
        else:
            # Default to linear if unknown
            return progress

    def get_progress_weight(self, epoch: int) -> float:
        """
        Compute the current curriculum weight scale factor based on epoch.
        """
        progress = self._compute_progress(epoch)
        scaled_progress = self._apply_grow_fn(progress)
        return scaled_progress

    @abstractmethod
    def get_sample_weights(self, dataset, model, epoch: int):
        """
        Return tensor of size [dataset_size] with sample importance weights (or probabilities).
        To be implemented by subclasses.
        """
        pass

    def update_strategy(self, epoch: int, model, dataset):
        """
        Optional method for adaptive curricula to update internal difficulty scores.
        Default: do nothing.
        """
        pass

class DifficultyBasedCurriculum(CurriculumStrategy):
    """
    Uses explicit difficulty scores per sample (e.g., loss, noise level).
    """
    def __init__(self,
                 difficulty_metric: str = 'loss',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # difficulty_metric: 'loss', 'confidence', 'entropy', 'noise_level' (extendable)
        self.difficulty_metric = difficulty_metric

    def get_sample_weights(self, dataset, model, epoch: int):
        """
        Compute normalized difficulty scores and derive sample weights.
        Easy samples (lower difficulty) get higher weights in curriculum.
        """
        dataset_size = len(dataset)
        if self.difficulty_scores_cache is None:
            # Attempt to compute difficulty scores
            scores = []
            for idx in range(dataset_size):
                sample = dataset[idx]
                # For torchvision datasets: sample is (data, label)
                # Placeholder: assume sample[0] is input tensor
                input_tensor = sample[0].unsqueeze(0).to(next(model.parameters()).device)
                with torch.no_grad():
                    output = model(input_tensor)
                    if self.difficulty_metric == 'loss':
                        # Use negative log probability as difficulty (higher loss)
                        probs = F.softmax(output, dim=1)
                        max_probs, _ = torch.max(probs, dim=1)
                        # Difficulty: low confidence => high difficulty
                        difficulty_score = 1.0 - max_probs.item()
                    elif self.difficulty_metric == 'confidence':
                        probs = F.softmax(output, dim=1)
                        max_probs, _ = torch.max(probs, dim=1)
                        difficulty_score = 1.0 - max_probs.item()
                    elif self.difficulty_metric == 'entropy':
                        probs = F.softmax(output, dim=1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                        difficulty_score = entropy.item() / np.log(probs.size(1))
                    else:
                        # Default to zero if unknown metric
                        difficulty_score = 0.0
                scores.append(difficulty_score)
            # Normalize scores to [0,1]
            scores = np.array(scores)
            norm_scores = (scores - scores.min()) / (scores.ptp() + 1e-8)
            self.difficulty_scores_cache = norm_scores
        else:
            norm_scores = self.difficulty_scores_cache
        # Progress scale
        scale = self.get_progress_weight(epoch)
        # Convert difficulty to weights: lower difficulty -> higher weight
        weights = 1.0 - norm_scores
        weights = torch.tensor(weights, dtype=torch.float32)
        # Optionally, scale weights based on curriculum progress
        weights = weights * scale + 1e-8  # avoid zeros
        return weights

class ConfidenceBasedCurriculum(CurriculumStrategy):
    """
    Uses model confidence or uncertainty as difficulty measure.
    """
    def __init__(self,
                 confidence_metric: str = 'confidence',  # or 'entropy'
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_metric = confidence_metric

    def get_sample_weights(self, dataset, model, epoch: int):
        """
        Compute confidence scores and derive weights.
        Lower confidence samples are emphasized early.
        """
        dataset_size = len(dataset)
        confidences = []
        for idx in range(dataset_size):
            sample = dataset[idx]
            input_tensor = sample[0].unsqueeze(0).to(next(model.parameters()).device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                max_prob, _ = torch.max(probs, dim=1)
                confidence_score = max_prob.item()
                confidences.append(confidence_score)
        confidences = np.array(confidences)
        # Normalize to [0,1]
        norm_conf = (confidences - confidences.min()) / (confidences.ptp() + 1e-8)
        # Inverse: low confidence = high difficulty
        difficulty_scores = 1.0 - norm_conf
        # Get curriculum scale
        scale = self.get_progress_weight(epoch)
        weights = difficulty_scores * scale + 1e-8
        weights = torch.tensor(weights, dtype=torch.float32)
        return weights

class SelfPacedCurriculum(CurriculumStrategy):
    """
    Implements Self-Paced Learning: starts easy, gradually adds hard samples.
    """
    def get_sample_weights(self, dataset, model, epoch: int):
        """
        Use the model loss or confidence to score difficulty.
        Samples with lower loss or higher confidence are regarded as easier.
        """
        dataset_size = len(dataset)
        losses = []
        model.eval()
        for idx in range(dataset_size):
            sample = dataset[idx]
            input_tensor = sample[0].unsqueeze(0).to(next(model.parameters()).device)
            label = sample[1]
            with torch.no_grad():
                output = model(input_tensor)
                loss = F.cross_entropy(output, torch.tensor([label], device=output.device))
                losses.append(loss.item())
        model.train()
        # Normalize losses
        losses = np.array(losses)
        norm_losses = (losses - losses.min()) / (losses.ptp() + 1e-8)
        # Curriculum progress as a cutoff: keep samples below a threshold
        progress = self.get_progress_weight(epoch)
        cutoff = progress
        # Assign weights: 1 for samples below cutoff, 0 otherwise
        weights = (norm_losses <= cutoff).astype(float)
        # Convert to tensor
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        # To avoid zero weights, add small epsilon
        weights_tensor = weights_tensor + 1e-8
        return weights_tensor

    def update_strategy(self, epoch: int, model, dataset):
        """
        Optionally, recompute difficulty scores if needed.
        Here we use static measures, so do nothing.
        """
        pass

class NoiseCurriculum(CurriculumStrategy):
    """
    Focuses on reducing the influence of noisy samples.
    Assumes a noise estimate or surrogate difficulty.
    """
    def __init__(self,
                 noise_estimator_fn: Optional[Callable] = None,
                 *args,
                 **kwargs):
        """
        noise_estimator_fn: Function to estimate sample noise or label correctness.
        """
        super().__init__(*args, **kwargs)
        self.noise_estimator_fn = noise_estimator_fn

    def get_sample_weights(self, dataset, model, epoch: int):
        """
        Use the estimator or model loss to identify noisy samples.
        Assign lower weights to suspected noisy data as curriculum advances.
        """
        dataset_size = len(dataset)
        scores = []
        for idx in range(dataset_size):
            sample = dataset[idx]
            input_tensor = sample[0].unsqueeze(0).to(next(model.parameters()).device)
            label = sample[1]
            with torch.no_grad():
                output = model(input_tensor)
                loss = F.cross_entropy(output, torch.tensor([label], device=output.device))
                scores.append(loss.item())
        scores = np.array(scores)
        # Higher loss -> higher difficulty
        norm_scores = (scores - scores.min()) / (scores.ptp() + 1e-8)
        # Gradually filter out noisy samples
        scale = self.get_progress_weight(epoch)
        weights = 1.0 - norm_scores
        weights = weights * scale + 1e-8
        weights = torch.tensor(weights, dtype=torch.float32)
        return weights

# Note: Additional specific curricula can be implemented similarly,
# adapting the get_sample_weights() method with specific difficulty calculations
# and optional update_strategy() for online/recurrent adaptation.
```

## dataset_loader.py

```python
## dataset_loader.py
import os
import random
import numpy as np
from typing import Dict, Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms

# Optional: For graph datasets
try:
    from torch_geometric.data import Data as GraphData
    from torch_geometric.datasets import TUDataset, MoleculeNet
except ImportError:
    # If torch_geometric is not installed, handle gracefully
    pass

class DatasetLoader:
    """
    Handles dataset loading, preprocessing, applying label noise and class imbalance.
    Supports CV datasets (CIFAR, Tiny-ImageNet, MNIST), NLP datasets (placeholder),
    and graph datasets via PyTorch Geometric.
    """
    def __init__(
        self,
        dataset_name: str,
        split_ratios: Dict[str, float],
        noise_ratio: float = 0.0,
        imbalance_factor: float = 1.0,
        apply_augmentation: bool = True,
        difficulty_scores: Optional[Dict] = None,
        seed: int = 42,
        device: str = 'cuda'
    ):
        """
        Initialize DatasetLoader with dataset parameters.
        """
        self.dataset_name = dataset_name
        self.split_ratios = split_ratios
        self.noise_ratio = noise_ratio
        self.imbalance_factor = imbalance_factor
        self.apply_augmentation = apply_augmentation
        self.difficulty_scores = difficulty_scores  # Not used here, but placeholder
        self.seed = seed
        self.device = device

        # Set seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load the dataset based on name. Performs train/val/test split.
        Returns datasets for train, val, and test.
        """
        dataset_lower = self.dataset_name.lower()

        if dataset_lower.startswith('cifar10'):
            full_train = datasets.CIFAR10(
                root='./data', train=True, download=True,
                transform=self._get_transform('train'))
            test_set = datasets.CIFAR10(
                root='./data', train=False, download=True,
                transform=self._get_transform('test'))
            dataset_name = 'CIFAR10'

        elif dataset_lower.startswith('cifar100'):
            full_train = datasets.CIFAR100(
                root='./data', train=True, download=True,
                transform=self._get_transform('train'))
            test_set = datasets.CIFAR100(
                root='./data', train=False, download=True,
                transform=self._get_transform('test'))
            dataset_name = 'CIFAR100'

        elif 'tinyimagenet' in dataset_lower:
            # Load Tiny-ImageNet: assume data is extracted and structured
            # For simplicity, load using torchvision.datasets.folder.ImageFolder
            # Custom dataset class may be needed in practice
            train_dir = './tinyimagenet/train'
            val_dir = './tinyimagenet/val'
            # Load datasets
            full_train = datasets.ImageFolder(train_dir, transform=self._get_transform('train'))
            val_set = datasets.ImageFolder(val_dir, transform=self._get_transform('test'))
            # Create a dataset list combining train + val to then split as per ratio
            # Or directly split train dataset further
            test_set = val_set  # Use validation set as test, per paper
            dataset_name = 'Tiny-ImageNet'

        elif dataset_lower.startswith('mnist'):
            full_train = datasets.MNIST(
                root='./data', train=True, download=True,
                transform=self._get_transform('train'))
            test_set = datasets.MNIST(
                root='./data', train=False, download=True,
                transform=self._get_transform('test'))
            dataset_name = 'MNIST'

        elif dataset_lower in ['rte', 'mrpc', 'sst-2', 'qnli', 'qqp', 'mnli', 'cola']:
            # Placeholder for NLP datasets from GLUE
            # Actual loading might involve huggingface transformers
            # Here, we leave None, as focus is on CV datasets
            # TODO: Extend for NLP datasets
            raise NotImplementedError(f"{self.dataset_name} loading not implemented in this code.")
        
        elif dataset_lower in ['mutag', 'proteins', 'nci1', 'ogbg-molhiv']:
            # Graph datasets (PyTorch Geometric)
            if 'torch_geometric' in globals():
                if 'mutag' in dataset_lower:
                    # Load MUTAG
                    dataset_obj = TUDataset('./data', name='MUTAG')
                elif 'proteins' in dataset_lower:
                    dataset_obj = TUDataset('./data', name='PROTEINS')
                elif 'nci1' in dataset_lower:
                    dataset_obj = TUDataset('./data', name='NCI1')
                elif 'ogbg-molhiv' in dataset_lower:
                    dataset_obj = TUDataset('./data', name='ogbg-molhiv')  # Requires OGB
                else:
                    raise ValueError(f"Unknown graph dataset: {self.dataset_name}")
                full_train = dataset_obj
                test_set = dataset_obj  # Will split later if needed
            else:
                raise ImportError("torch_geometric not installed.")
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' not recognized or supported.")

        # Split datasets into train, val, test as per ratios
        if 'cifar' in dataset_lower or 'tinyimagenet' in dataset_lower or 'mnist' in dataset_lower:
            # For cv datasets, perform split
            total_train = len(full_train)
            train_size = int(self.split_ratios.get('train', 0.8) * total_train)
            val_size = int(self.split_ratios.get('validation', 0.1) * total_train)
            test_size = len(full_train) - train_size - val_size

            indices = list(range(total_train))
            np.random.seed(self.seed)
            np.random.shuffle(indices)

            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]

            self.train_dataset = Subset(full_train, train_idx)
            self.val_dataset = Subset(full_train, val_idx)
            self.test_dataset = test_set
        elif 'graph' in dataset_lower:
            # For graph datasets, assume full dataset, split accordingly
            total_graphs = len(full_train)
            train_size = int(self.split_ratios.get('train', 0.8) * total_graphs)
            indices = list(range(total_graphs))
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            train_idx = indices[:train_size]
            val_idx = indices[train_size:]
            self.train_dataset = Subset(full_train, train_idx)
            self.val_dataset = Subset(full_train, val_idx)
            self.test_dataset = full_train  # Or better, a specific test split if available
        else:
            # For other datasets, simply return the full dataset
            self.train_dataset = full_train
            self.val_dataset = None
            self.test_dataset = test_set

        return self.train_dataset, self.val_dataset, self.test_dataset

    def apply_noise_or_imbalance(self, dataset: Dataset):
        """
        Apply label noise and class imbalance to the training dataset.
        Returns a new Dataset object with modifications.
        """
        dataset_type = type(dataset)
        # Support for standard datasets
        if 'torchvision' in globals() and isinstance(dataset, Dataset):
            # For torchvision datasets
            # Extract all labels
            # Note: For Subset, access underlying dataset
            # For datasets like CIFAR10
            dataset_obj = dataset.dataset if hasattr(dataset, 'dataset') else dataset

            # Get original targets
            try:
                targets = np.array(dataset_obj.targets)
            except AttributeError:
                # For datasets like MNIST: attribute is 'labels'
                targets = np.array(dataset_obj.labels)
            # Save original targets for returning
            original_targets = targets.copy()

            # Apply noise if specified
            if self.noise_ratio > 0:
                np.random.seed(self.seed)
                n_samples = len(targets)
                n_noisy = int(self.noise_ratio * n_samples)
                noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
                num_classes = len(np.unique(targets))
                for idx in noisy_indices:
                    current_label = targets[idx]
                    # Choose a random label different from the current
                    possible_labels = list(range(num_classes))
                    possible_labels.remove(current_label)
                    new_label = np.random.choice(possible_labels)
                    targets[idx] = new_label

            # Apply imbalance if specified
            if self.imbalance_factor != 1.0:
                # Create class-wise indices
                class_indices = {}
                for idx, label in enumerate(targets):
                    class_indices.setdefault(label, []).append(idx)
                # Calculate number of samples per class based on imbalance factor
                min_size = min(len(idxs) for idxs in class_indices.values())
                # For ratio r, define the largest class size
                max_size = int(min_size * self.imbalance_factor)
                new_indices = []
                for label, idxs in class_indices.items():
                    n_samples_cls = int(max_size / self.imbalance_factor)
                    n_samples_cls = min(n_samples_cls, len(idxs))
                    np.random.seed(self.seed)
                    selected_idx = np.random.choice(idxs, n_samples_cls, replace=False)
                    new_indices.extend(selected_idx)
            else:
                # No imbalance, keep original indices
                new_indices = list(range(len(targets)))
                np.random.seed(self.seed)

            # Create subset with new indices
            # To support DataLoader, create a custom Dataset wrapper
            subset_dataset = _SubsetWithTargets(dataset_obj, new_indices, targets)
            return subset_dataset

        elif 'torch_geometric' in globals():
            # For graph datasets
            # Noise and imbalance at graph level may be less common,
            # assuming label noise on graph labels
            # Placeholder: implement similarly if needed
            # For now, return the dataset as is
            return dataset

        elif 'pandas' in globals():
            # Placeholder for NLP datasets
            return dataset

        else:
            # If dataset type unsupported
            raise NotImplementedError("apply_noise_or_imbalance not implemented for this dataset type.")

    def _get_transform(self, phase: str):
        """
        Return data transformation pipeline based on dataset and phase.
        """
        # For CV datasets, define basic transforms
        if 'cifar' in self.dataset_name.lower() or 'tinyimagenet' in self.dataset_name.lower():
            if phase == 'train':
                return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.247, 0.243, 0.261))
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.247, 0.243, 0.261))
                ])
        elif 'mnist' in self.dataset_name.lower():
            if phase == 'train':
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
        else:
            # Placeholder or extend for other datasets, NLP, graphs, etc.
            return None

class _SubsetWithTargets(Subset):
    """
    Helper class to create a subset with specified targets (labels),
    supporting potential curriculum algorithms that need access to targets.
    """
    def __init__(self, dataset: Dataset, indices: list, targets: np.ndarray):
        super().__init__(dataset, indices)
        self._targets = targets

    @property
    def targets(self):
        return self._targets
```

## evaluation.py

```python
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
```

## main.py

```python
## main.py
import yaml
import os
import torch
from utils import setup_logger, set_seed, save_metrics
from dataset_loader import DatasetLoader
from models import ModelFactory
from curriculum_strategies import CurriculumStrategyFactory
from trainer import Trainer
from evaluation import Evaluation
from resources import ResourceLogger

def main():
    # 1. Load configuration from 'config.yaml'
    config_path = 'config.yaml'
    assert os.path.exists(config_path), f"Config file {config_path} not found."
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 2. Setup logger for console and file output
    setup_logger(log_file='experiment.log')
    print("Configuration loaded successfully.")

    # 3. Set seeds for reproducibility
    seed = cfg['hyperparameters'].get('seed', 42)
    set_seed(seed)
    # Optionally, set deterministic backend (add if needed)
    
    # 4. Initialize resource logger if enabled
    resource_config = cfg.get('resource_logging', {})
    resource_logger = ResourceLogger(
        enable_time=resource_config.get('enable_time_tracking', True),
        enable_memory=resource_config.get('enable_memory_tracking', True)
    )

    # 5. Load & prepare dataset
    dataset_cfg = cfg['dataset']
    noise_ratio = dataset_cfg.get('noise_ratio', 0.0)
    imbalance_factor = dataset_cfg.get('imbalance_factor', 1.0)
    split_ratios = dataset_cfg.get('split_ratios', {'train':0.8,'validation':0.1,'test':0.1})
    dataset_loader = DatasetLoader(
        dataset_name=dataset_cfg['name'],
        split_ratios=split_ratios,
        noise_ratio=noise_ratio,
        imbalance_factor=imbalance_factor,
        seed=seed
    )
    train_dataset, val_dataset, test_dataset = dataset_loader.load_data()

    # For validation, we assume a fixed dataset; no noise/imbalance applied
    # 6. Initialize model
    model_cfg = cfg['model']
    model_type = model_cfg['type']
    model_hparams = model_cfg.get('hyperparameters', {})
    # Add explicit number of classes based on dataset
    if hasattr(train_dataset, 'dataset'):
        # For torchvision datasets
        num_classes = len(train_dataset.dataset.classes) if hasattr(train_dataset.dataset, 'classes') else None
    else:
        num_classes = None  # default or set from dataset properties
    # Set num_classes
    if num_classes is None:
        # fallback: from dataset info or hardcoded for known datasets
        if 'cifar' in dataset_cfg['name'].lower():
            num_classes = 10 if 'cifar10' in dataset_cfg['name'].lower() else 100
        elif 'mnist' in dataset_cfg['name'].lower():
            num_classes = 10
        elif 'tinyimagenet' in dataset_cfg['name'].lower():
            num_classes = 200
        elif 'glue' in dataset_cfg['name'].lower():
            # NLP datasets: set accordingly or default
            num_classes = 2 # Placeholder
        else:
            num_classes = 10
    model_hparams['num_classes'] = num_classes
    model = ModelFactory(model_type, model_hparams)

    # 7. Initialize optimizer & scheduler
    train_cfg = cfg['train']
    lr = train_cfg.get('learning_rate', 0.0001)
    optimizer_type = train_cfg.get('optimizer', 'Adam')
    weight_decay = train_cfg.get('weight_decay', 1e-4)
    
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    # Scheduler if specified
    scheduler_cfg = train_cfg.get('scheduler', {})
    if scheduler_cfg:
        sched_type = scheduler_cfg.get('type', 'StepLR')
        if sched_type == 'StepLR':
            step_size = scheduler_cfg.get('step_size', 30)
            gamma = scheduler_cfg.get('gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            scheduler = None
    else:
        scheduler = None

    # 8. Initialize curriculum strategy
    curriculum_cfg = cfg.get('curriculum', {})
    strategy_name = curriculum_cfg.get('strategy', 'DifficultyBased')
    curriculum_cls = CurriculumStrategyFactory.get_strategy(strategy_name)
    # Parameters for curriculum
    start_ratio = curriculum_cfg.get('start_ratio', 0.0)
    grow_epochs = curriculum_cfg.get('grow_epochs', 20)
    grow_fn = curriculum_cfg.get('grow_fn', 'linear')
    warm_epochs = curriculum_cfg.get('warm_epochs', 5)
    curriculum_params = dict(
        start_ratio=start_ratio,
        grow_epochs=grow_epochs,
        grow_fn=grow_fn,
        warm_epochs=warm_epochs
    )
    curriculum = curriculum_cls(**curriculum_params)

    # 9. Initialize Trainer
    hyperparams = cfg['hyperparameters']
    trainer_kwargs = dict(
        model=model,
        dataset=train_dataset,
        curriculum=curriculum,
        hyperparameters=hyperparams,
        optimizer=optimizer,
        scheduler=scheduler,
        resource_logger=resource_logger
    )
    trainer = Trainer(**trainer_kwargs)

    # 10. Run training loop
    total_epochs = hyperparams.get('epochs', 200)
    log_interval = hyperparams.get('log_interval', 10)
    for epoch in range(1, total_epochs + 1):
        # Resource tracking start
        resource_logger.start_epoch()

        # Optional curriculum update
        if hasattr(curriculum, 'update_strategy'):
            try:
                curriculum.update_strategy(epoch, model, train_dataset)
            except Exception as e:
                print(f"Warning: Curriculum update_strategy error at epoch {epoch}: {e}")

        # Obtain sample weights from curriculum
        with torch.no_grad():
            sample_weights = None
            if hasattr(curriculum, 'get_sample_weights'):
                sample_weights = curriculum.get_sample_weights(train_dataset, model, epoch)
                # normalize weights if needed
                if isinstance(sample_weights, torch.Tensor) and sample_weights.sum() != 0:
                    sample_weights = torch.where(
                        sample_weights == 0,
                        torch.ones_like(sample_weights) * 1e-8,
                        sample_weights
                    )
        # Create data loader considering curriculum schedule
        train_loader = trainer._create_curriculum_dataloader(train_dataset, sample_weights)

        # Train one epoch
        trainer.train_one_epoch(train_loader)

        # Log resource usage
        resource_logger.end_epoch(epoch)

        # Periodic logging
        if epoch % log_interval == 0 or epoch == 1 or epoch == total_epochs:
            print(f"Epoch {epoch}/{total_epochs} completed. "
                  f"Time: {resource_logger.epoch_logs[-1]['time_sec']:.2f}s, "
                  f"Max Mem: {resource_logger.epoch_logs[-1]['max_memory_MB']:.2f}MB.")

        # Step scheduler
        if scheduler:
            scheduler.step()

    # 11. Evaluation on test set
    eval_model = model
    evaluator = Evaluation(eval_model, test_dataset)
    eval_metrics = evaluator.evaluate()

    # 12. Save results
    results = {
        'config': cfg,
        'seed': seed,
        'resource_logs': resource_logger.epoch_logs,
        'final_metrics': eval_metrics
    }
    save_metrics(results, 'final_results.json')
    print("Experiment completed. Results saved to 'final_results.json'.")

if __name__ == "__main__":
    main()
```

## models.py

```python
## models.py
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, GPT2Model
try:
    from torch_geometric.nn import GCNConv, GATConv, GINConv
    from torch_geometric.data import Data as PyGData
except ImportError:
    GCNConv = None
    GATConv = None
    GINConv = None

class LeNet(nn.Module):
    """Implementation of LeNet-5 architecture for small images."""
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        # No pretrained weights, initialize from scratch
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2)
        )
        # Calculate the output shape after conv & pool
        # Assuming input size 32x32; adjust as needed
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, params.get('num_classes', 10))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNet18(nn.Module):
    """ResNet-18 with optional pretrained weights and custom output classes."""
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        pretrained = params.get('pretrained', False)
        num_classes = params.get('num_classes', 10)
        self.model = models.resnet18(pretrained=pretrained)
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ViT(nn.Module):
    """Vision Transformer from Hugging Face transformers."""
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        from transformers import ViTForImageClassification
        num_classes = params.get('num_classes', 10)
        pretrained = params.get('pretrained', False)
        # Load ViT model with or without pretrained weights
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k' if pretrained else None,
            num_labels=num_classes
        )

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        outputs = self.model.pixel_values(x)
        return outputs.logits

class LSTM(nn.Module):
    """Standard LSTM for sequence tasks."""
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        input_size = params.get('input_size', 300)
        hidden_size = params.get('hidden_size', 256)
        num_layers = params.get('num_layers', 1)
        bidirectional = params.get('bidirectional', False)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.output_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(self.output_dim, params.get('num_classes', 2))

    def forward(self, x):
        # x: [seq_len, batch_size, input_size]
        out, _ = self.lstm(x)
        # Use last hidden state
        last_hidden = out[-1]
        logits = self.fc(last_hidden)
        return logits

class BERT(nn.Module):
    """BERT model for NLP tasks."""
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        from transformers import BertModel
        pretrained = params.get('pretrained', False)
        num_labels = params.get('num_classes', 2)
        if pretrained:
            self.model = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = BertModel(config={'num_labels': num_labels})
        # Append classifier head if desired
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

class GPT2(nn.Module):
    """GPT-2 for language modeling / classification."""
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        from transformers import GPT2Model
        pretrained = params.get('pretrained', False)
        num_labels = params.get('num_classes', 2)
        if pretrained:
            self.model = GPT2Model.from_pretrained('gpt2')
        else:
            self.model = GPT2Model(config={'n_positions': 1024})
        # Add classifier head if needed
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        # Pooling: mean pooling over seq_len
        pooled_output = last_hidden.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits

class GCN(nn.Module):
    """Graph Convolutional Network using torch_geometric."""
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        if GCNConv is None:
            raise ImportError("torch_geometric is not installed.")
        hidden_channels = params.get('hidden_channels', 64)
        num_layers = params.get('num_layers', 2)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = params.get('in_channels', 9 if 'ogbg-molhiv' in params else 1)
            out_channels = hidden_channels
            conv = GCNConv(in_channels, out_channels)
            self.convs.append(conv)
            # in_channels for next layer
            params['in_channels'] = out_channels
        self.lin = nn.Linear(hidden_channels, params.get('num_classes', 2))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        # Global pooling or readout: mean
        out = torch_geometric.nn.global_mean_pool(x, data.batch)
        out = self.lin(out)
        return out

class GAT(nn.Module):
    """Graph Attention Network using torch_geometric."""
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        if GATConv is None:
            raise ImportError("torch_geometric is not installed.")
        hidden_channels = params.get('hidden_channels', 64)
        num_heads = params.get('num_heads', 4)
        num_layers = params.get('num_layers', 2)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = params.get('in_channels', 9 if 'ogbg-molhiv' in params else 1)
            out_channels = hidden_channels
            conv = GATConv(in_channels, out_channels, heads=num_heads)
            self.convs.append(conv)
            params['in_channels'] = out_channels * num_heads
        self.lin = nn.Linear(params['in_channels'], params.get('num_classes', 2))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        out = torch_geometric.nn.global_mean_pool(x, data.batch)
        out = self.lin(out)
        return out

class GIN(nn.Module):
    """Graph Isomorphism Network using torch_geometric."""
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        if GINConv is None:
            raise ImportError("torch_geometric is not installed.")
        nn_full = nn.Sequential(
            nn.Linear(params.get('in_channels', 9 if 'ogbg-molhiv' in params else 1)), nn.ReLU(), nn.Linear(params.get('hidden', 64))
        )
        self.convs = nn.ModuleList()
        num_layers = params.get('num_layers', 2)
        for _ in range(num_layers):
            conv = GINConv(nn_full)
            self.convs.append(conv)
        self.lin = nn.Linear(params.get('hidden', 64), params.get('num_classes', 2))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        out = torch_geometric.nn.global_mean_pool(x, data.batch)
        out = self.lin(out)
        return out
```

## resources.py

```python
## resources.py
import torch
import time
from typing import Optional, Dict, Any

class ResourceLogger:
    """
    Handles resource tracking during training epochs, including timing and GPU memory.
    Records per-epoch resource usage metrics for analysis in CurBench.
    """
    def __init__(
        self,
        enable_time_tracking: bool = True,
        enable_memory_tracking: bool = True
    ):
        self.enable_time_tracking = enable_time_tracking
        self.enable_memory_tracking = enable_memory_tracking
        self.epoch_logs = []  # To store logs per epoch: list of dicts
        self.start_time = None
        self.peak_memory_bytes = 0  # Maximum GPU memory during epoch in bytes

    def start_epoch(self):
        """
        Call at the beginning of an epoch.
        - Records start time if enabled.
        - Resets GPU peak memory stats if enabled.
        """
        if self.enable_time_tracking:
            self.start_time = time.perf_counter()
        if self.enable_memory_tracking:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                # For environments without CUDA or GPU unavailable
                pass
        self.peak_memory_bytes = 0

    def end_epoch(self, epoch_idx: int):
        """
        Call after epoch ends.
        - Computes elapsed time if enabled.
        - Records maximum GPU memory allocated during epoch if enabled.
        - Appends resource usage info for this epoch.
        """
        epoch_time = None
        max_mem_mb = None
        # Record elapsed time
        if self.enable_time_tracking and self.start_time is not None:
            epoch_time = time.perf_counter() - self.start_time
        # Record max GPU memory
        if self.enable_memory_tracking:
            try:
                max_mem_bytes = torch.cuda.max_memory_allocated()
                if max_mem_bytes > self.peak_memory_bytes:
                    self.peak_memory_bytes = max_mem_bytes
                max_mem_mb = max_mem_bytes / 1e6  # Convert bytes to MB
            except Exception:
                max_mem_mb = None  # Could happen if CUDA not available
        # Save per-epoch resource info
        self.epoch_logs.append({
            'epoch': epoch_idx,
            'time_sec': epoch_time,
            'max_memory_MB': max_mem_mb
        })

    def get_total_time(self) -> float:
        """
        Return total cumulative training time across epochs if needed.
        """
        total_time = sum(log['time_sec'] for log in self.epoch_logs if log['time_sec'] is not None)
        return total_time

    def get_max_memory_MB(self) -> float:
        """
        Return maximum GPU memory used across epochs in MB.
        """
        return self.peak_memory_bytes / 1e6

    def export_logs_as_json(self, filename: str):
        """
        Save recorded resource logs to a JSON file.
        """
        import json
        with open(filename, 'w') as f:
            json.dump(self.epoch_logs, f, indent=4)

    def export_logs_as_csv(self, filename: str):
        """
        Save resource logs as CSV for easy visualization.
        """
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'time_sec', 'max_memory_MB'])
            writer.writeheader()
            for entry in self.epoch_logs:
                writer.writerow(entry)

    def reset(self):
        """
        Reset logs and internal states if needed.
        """
        self.epoch_logs = []
        self.start_time = None
        self.peak_memory_bytes = 0
```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, WeightedRandomSampler
from resources import ResourceLogger
from utils import set_seed, save_metrics
from typing import Dict, Any, Optional

class Trainer:
    """
    Orchestrates the training process for a model with optional curriculum learning.
    Implements resource logging and adheres to the experimental setup in CurBench.
    """
    def __init__(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        curriculum: Optional[object],  # Instance of CurriculumStrategy
        hyperparameters: Dict[str, Any]
    ):
        """
        Initialize the trainer with model, dataset, curriculum, and hyperparameters.
        """
        # Save core components
        self.model = model
        self.dataset = dataset
        self.curriculum = curriculum
        # Set hyperparameters with defaults
        self.epochs = hyperparameters.get('epochs', 200)
        self.batch_size = hyperparameters.get('batch_size', 128)
        self.learning_rate = hyperparameters.get('learning_rate', 0.0001)
        self.optimizer_type = hyperparameters.get('optimizer', 'Adam')
        self.weight_decay = hyperparameters.get('weight_decay', 0.0)
        self.device = hyperparameters.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.log_interval = hyperparameters.get('log_interval', 10)

        # Prepare device
        self.device = torch.device(self.device)

        # Set seed for reproducibility
        seed = hyperparameters.get('seed', 42)
        set_seed(seed)

        # Initialize model
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = self._init_optimizer()

        # Optional: learning rate scheduler
        sched_cfg = hyperparameters.get('scheduler', None)
        if sched_cfg:
            sched_type = sched_cfg.get('type', 'StepLR')
            if sched_type == 'StepLR':
                step_size = sched_cfg.get('step_size', 30)
                gamma = sched_cfg.get('gamma', 0.1)
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
            else:
                self.scheduler = None
        else:
            self.scheduler = None

        # Initialize resource logger
        enable_time = hyperparameters.get('enable_time_tracking', True)
        enable_memory = hyperparameters.get('enable_memory_tracking', True)
        self.resource_logger = ResourceLogger(enable_time, enable_memory)

        # Prepare DataLoader
        self.train_loader = self._init_dataloader(self.dataset, shuffle=True)
        self.val_loader = None  # Will set after validation dataset is available

        # Loss criterion
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # For weighted loss

        # Collect logs
        self.metrics_log = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_time': [],
            'epoch_memory': []
        }

        # Number of training samples for curriculum weighting
        self.dataset_indices = list(range(len(self.dataset)))

    def _init_optimizer(self):
        """Initialize optimizer based on configuration."""
        if self.optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def _init_dataloader(self, dataset: torch.utils.data.Dataset, shuffle: bool = True):
        """Create DataLoader for the dataset."""
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

    def train(self):
        """Main training loop with curriculum learning integration, resource tracking, and logging."""
        for epoch in range(1, self.epochs + 1):
            # Reset or start resource logging
            start_time = self.resource_logger.log_time_start()
            self.resource_logger.reset_memory()

            # If curriculum is adaptive, update the internal curriculum parameters
            if self.curriculum and hasattr(self.curriculum, 'update_strategy'):
                try:
                    self.curriculum.update_strategy(epoch, self.model, self.dataset)
                except Exception as e:
                    # Fail silently or log
                    print(f"Warning: Curriculum update_strategy error at epoch {epoch}: {e}")

            # Obtain sample weights or indices from curriculum strategy
            sample_weights = None
            if self.curriculum and hasattr(self.curriculum, 'get_sample_weights'):
                with torch.no_grad():
                    sample_weights = self.curriculum.get_sample_weights(self.dataset, self.model, epoch)

            # Prepare DataLoader with sample weights if provided
            dataloader = self._create_curriculum_dataloader(self.dataset, sample_weights)

            # Training phase
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch_idx, batch in enumerate(dataloader):
                # Batch inputs: adapt based on dataset type
                inputs, labels = self._prepare_batch(batch)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # Compute loss with optional sample weights
                loss_values = self.criterion(outputs, labels)
                if sample_weights is not None:
                    # Get corresponding weights for current batch samples
                    batch_indices = self._get_batch_indices(batch)
                    weights = sample_weights[batch_indices].to(self.device)
                    loss = (loss_values * weights).mean()
                else:
                    loss = loss_values.mean()

                loss.backward()
                self.optimizer.step()

                # Accumulate statistics
                total_loss += loss.item() * labels.size(0)
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            # Step scheduler if used
            if self.scheduler:
                self.scheduler.step()

            # Record epoch metrics
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples

            self.metrics_log['train_loss'].append(avg_loss)
            self.metrics_log['train_acc'].append(accuracy)

            # Resource usage logging
            epoch_time = self.resource_logger.log_time_end(start_time)
            epoch_memory = self.resource_logger.get_max_memory_MB()
            self.metrics_log['epoch_time'].append(epoch_time)
            self.metrics_log['epoch_memory'].append(epoch_memory)

            # Logging
            if epoch % self.log_interval == 0 or epoch == 1 or epoch == self.epochs:
                print(f"Epoch {epoch}/{self.epochs} - "
                      f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | "
                      f"Time: {epoch_time:.2f}s | Max Mem: {epoch_memory:.2f}MB")

        # After training completes, save the logs or perform evaluation
        save_metrics(self.metrics_log, 'training_metrics.json')

    def _create_curriculum_dataloader(self, dataset, sample_weights: Optional[torch.Tensor]):
        """
        Create DataLoader that considers sample weights or selection from curriculum.
        """
        if sample_weights is not None:
            # Use WeightedRandomSampler with normalized weights
            # For stability, normalize weights for sampling probabilities
            probs = sample_weights / sample_weights.sum()
            sampler = WeightedRandomSampler(probs, num_samples=len(probs), replacement=True)
            return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        else:
            # Default shuffling
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def _prepare_batch(self, batch):
        """
        Extract inputs and labels from batch depending on dataset type.
        """
        # Batch is a tuple or dict; we assume tuple for torchvision/datasets
        # Extend as needed for NLP or Graph datasets
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                inputs, labels = batch
            elif len(batch) == 3:
                inputs, labels, extra = batch
            else:
                raise ValueError("Unexpected batch format.")
        elif isinstance(batch, dict):
            inputs = batch['inputs']
            labels = batch['labels']
        else:
            # Fallback: assume batch is tuple (inputs, labels)
            inputs, labels = batch

        return inputs, labels

    def _get_batch_indices(self, batch):
        """
        For weighted sampling, retrieve indices for samples in batch.
        This may involve storing indices during dataset loading. For simplicity, assume batch has attribute or mapping.
        """
        # Since DataLoader shuffles or samples, in this simplified example, 
        # we arbitrarily assign indices by position.
        # For more accurate association, dataset or sampler should track indices.
        # Here, we'll assume sequential order (not exact but illustrative).
        # In practice, you'd pass indices explicitly.
        batch_size = len(batch[1]) if isinstance(batch, (list, tuple)) else len(batch['labels'])
        # Fake indices: just range for now (not accurate if shuffling)
        # Should be replaced with real indices tracking if necessary.
        start_idx = 0  # Placeholder
        return torch.arange(start_idx, start_idx + batch_size)

```

## utils.py

```python
## utils.py
import os
import time
import logging
import json
import yaml
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from typing import Any, Dict, Optional

class ConfigManager:
    """Handles loading and accessing nested configuration parameters from YAML file."""
    def __init__(self, filepath: str = 'config.yaml'):
        self.config = self.load_config(filepath)

    @staticmethod
    def load_config(filepath: str) -> Dict[str, Any]:
        """Load YAML configuration file into a dictionary."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        with open(filepath, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg

    def get(self, key_path: str, default: Optional[Any] = None) -> Any:
        """Retrieve nested configuration value using dot notation, e.g., 'dataset.noise_ratio'."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enforce deterministic behavior (can impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ResourceLogger:
    """Handles timing and GPU memory tracking."""
    def __init__(self, enable_time: bool = True, enable_memory: bool = True):
        self.enable_time = enable_time
        self.enable_memory = enable_memory
        self.start_time = None
        self.max_memory_bytes = 0

    def log_time_start(self) -> float:
        """Record start time."""
        if self.enable_time:
            self.start_time = time.perf_counter()
            return self.start_time
        return 0.0

    def log_time_end(self, start_time: float) -> float:
        """Calculate elapsed time."""
        if self.enable_time and self.start_time is not None:
            elapsed = time.perf_counter() - start_time
            return elapsed
        return 0.0

    def reset_memory(self):
        """Reset max memory counter."""
        if self.enable_memory and torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()

    def log_gpu_memory(self) -> float:
        """Return current maximum GPU memory allocated in MB."""
        if self.enable_memory and torch.cuda.is_available():
            max_mem_bytes = torch.cuda.max_memory_allocated()
            # Update overall max if needed
            if max_mem_bytes > self.max_memory_bytes:
                self.max_memory_bytes = max_mem_bytes
            return max_mem_bytes / (1024 ** 2)  # Convert to MB
        return 0.0

    def get_max_memory_MB(self) -> float:
        """Get maximum GPU memory used during monitoring in MB."""
        if self.enable_memory and torch.cuda.is_available():
            return self.max_memory_bytes / (1024 ** 2)
        return 0.0

def setup_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure global logger."""
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=log_format, handlers=handlers)

def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics dictionary into a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_metrics(metrics: Dict[str, list], metric_name: str = 'Score', save_path: str = 'metrics_plot.png') -> None:
    """Plot training/validation metrics over epochs."""
    plt.figure(figsize=(8, 6))
    for label, values in metrics.items():
        plt.plot(values, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Metrics over epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_data_transform(dataset_name: str, phase: str = 'train'):
    """Provide data transformation pipeline based on dataset and phase."""
    from torchvision import transforms
    # Example for CV datasets; extend for NLP/graphs accordingly
    if dataset_name.lower().startswith('cifar') or dataset_name.lower().startswith('tinyimagenet'):
        if phase == 'train':
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.247, 0.243, 0.261))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.247, 0.243, 0.261))
            ])
    else:
        # Placeholder: extend for NLP or graph datasets
        return None

def set_deterministic(seed: int = 42):
    """Apply deterministic settings for reproducibility, if needed."""
    set_seed(seed)
    # Additional deterministic backend configs can be added here
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\CurBench\CurBench_repo`
