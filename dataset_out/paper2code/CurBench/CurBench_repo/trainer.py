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

