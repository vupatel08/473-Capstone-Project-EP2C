## trainer.py

import torch
from torch.optim import Optimizer
from typing import Callable, Dict, Any

class Trainer:
    """
    Class: Trainer
    Purpose:
        Manage the training loop for the InvertedTransformer model.
        Handles per-epoch training, logging, validation, and checkpointing.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: Optimizer,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 data_loader: Any,
                 device: str = 'cpu',
                 config: Dict[str, Any] = None):
        """
        Initialize the trainer.

        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer instance.
            loss_fn (callable): Loss function, e.g., nn.MSELoss().
            data_loader (Any): Dataset loader object providing get_*__batches methods.
            device (str): Device to run computation on ('cpu' or 'cuda').
            config (dict): Configuration dictionary, optional, including training hyperparams.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.device = device
        self.config = config if config is not None else {}

        # Extract hyperparameters with defaults
        self.epochs = self.config.get('epochs', 50)
        self.batch_size = self.config.get('batch_size', 64)
        self.learning_rate = self.config.get('learning_rate', 1e-3)
        self.clip_norm = self.config.get('clip_norm', None)  # e.g., 1.0
        self.save_dir = self.config.get('save_dir', 'checkpoints/')
        self.save_freq = self.config.get('save_frequency', 10)
        self.validation_interval = self.config.get('validation_interval', 1)  # in epochs

        # Create checkpoints directory if not exists
        import os
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self):
        """
        Run one epoch of training over the training dataset.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        batch_count = 0

        # Get iterable batch generator
        train_batches = self.data_loader.get_train_batches(self.batch_size)

        for X_batch, Y_batch in train_batches:
            # Move to device
            X_batch = X_batch.to(self.device)  # shape: [batch, T, N]
            Y_batch = Y_batch.to(self.device)  # shape: [batch, S, N]

            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model.forward(X_batch)  # shape: [batch, N, S]
            predictions = predictions.permute(0, 2, 1)  # to shape: [batch, S, N]

            # Compute loss
            loss = self.loss_fn(predictions, Y_batch)
            loss.backward()

            # Gradient clipping if specified
            if self.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

            # Optimizer update
            self.optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        return avg_loss

    def train(self):
        """
        Run training over all epochs, with validation and checkpointing.
        """
        best_val_metric = float('inf')
        for epoch in range(1, self.epochs + 1):
            # Train epoch
            train_loss = self.train_epoch()

            # Log training loss
            print(f"Epoch [{epoch}/{self.epochs}] - Training Loss: {train_loss:.6f}")

            # Validation step
            if epoch % self.validation_interval == 0:
                val_metrics = self.evaluate(self.data_loader, split='val')
                val_mse = val_metrics.get('MSE', None)
                print(f"Epoch [{epoch}] - Validation MSE: {val_mse:.6f}")

                # Save best model based on validation MSE
                if val_mse is not None and val_mse < best_val_metric:
                    best_val_metric = val_mse
                    save_path = os.path.join(self.save_dir, 'best_model.pth')
                    self.save_checkpoint(save_path)
                    print(f"Saved new best model at epoch {epoch}")

            # Save checkpoint periodically
            if epoch % self.save_freq == 0:
                save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
                self.save_checkpoint(save_path)
                print(f"Saved checkpoint at epoch {epoch}")

    def save_checkpoint(self, save_path: str):
        """
        Save model and optimizer state_dicts.

        Args:
            save_path (str): Path where to save checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, save_path)

    def evaluate(self, data_loader: Any, split: str = 'val') -> Dict[str, float]:
        """
        Evaluate model on validation or test set.

        Args:
            data_loader (Any): Dataset loader with get_*_batches methods.
            split (str): 'val' or 'test'.

        Returns:
            dict: Metrics, e.g., {'MSE': value, 'MAE': value}
        """
        self.model.eval()
        total_preds = []
        total_targets = []

        # Get validation/test batches
        if split == 'val':
            batches = data_loader.get_val_batches(self.batch_size)
        elif split == 'test':
            batches = data_loader.get_test_batches(self.batch_size)
        else:
            raise ValueError(f"Unknown split: {split}")

        with torch.no_grad():
            for X_batch, Y_batch in batches:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                preds = self.model.forward(X_batch)  # shape: [batch, N, S]
                preds = preds.permute(0, 2, 1)  # shape: [batch, S, N]

                total_preds.append(preds.cpu().numpy())
                total_targets.append(Y_batch.cpu().numpy())

        # Concatenate all batches
        preds_concat = np.concatenate(total_preds, axis=0)  # shape: [samples, S, N]
        targets_concat = np.concatenate(total_targets, axis=0)  # same shape

        # Compute metrics
        metrics = {}
        for metric_name in self.config.get('metrics', ['MSE', 'MAE']):
            if metric_name == 'MSE':
                mse = np.mean((preds_concat - targets_concat) ** 2)
                metrics['MSE'] = mse
            elif metric_name == 'MAE':
                mae = np.mean(np.abs(preds_concat - targets_concat))
                metrics['MAE'] = mae
            # Add other metrics if needed

        return metrics
