# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
from utils import set_seed


class Trainer:
    """
    This class manages the training process for a neural network model, including batching,
    optimization, early stopping, and logging. It interacts with the provided model for forward
    passes and parameter updates.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray],
        train_labels: np.ndarray,
        val_labels: Optional[np.ndarray],
        config: dict,
        device: str = 'cpu'
    ):
        """
        Initialize the trainer with model, datasets, and training configuration.
        Args:
            model (nn.Module): The neural network model to train.
            train_data (np.ndarray): Training feature array of shape (N_train, input_dim).
            val_data (np.ndarray): Validation feature array of shape (N_val, input_dim), optional.
            train_labels (np.ndarray): Training labels, shape (N_train,).
            val_labels (np.ndarray): Validation labels, shape (N_val,), optional.
            config (dict): Configuration parameters from 'config.yaml'.
            device (str): 'cpu' or 'cuda'. Defaults to 'cpu'.
        """
        self.model = model.to(device)
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.device = device
        self.epochs = config.get('training', {}).get('epochs', 50)
        self.batch_size = config.get('training', {}).get('batch_size', 512)
        self.learning_rate = config.get('training', {}).get('learning_rate', 0.001)
        self.early_stopping_patience = config.get('training', {}).get('early_stopping_patience', 10)
        self.verbose = True
        # Set seed for reproducibility
        set_seed()

        # Setup optimizer and loss
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Initialize early stopping variables
        self.best_validation_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_state_dict = None

        # Prepare data batches for train and validation
        self.train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(self.train_data, dtype=torch.float32),
            torch.tensor(self.train_labels, dtype=torch.float32)
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        if self.val_data is not None:
            self.val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(self.val_data, dtype=torch.float32),
                torch.tensor(self.val_labels, dtype=torch.float32)
            )
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )

    def train(self):
        """
        Executes the main training loop over epochs, with early stopping based on validation loss.
        Saves the best model weights.
        """
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_train_loss = 0.0
            for batch_x, _ in self.train_loader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_x)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * batch_x.size(0)
            avg_train_loss = total_train_loss / len(self.train_loader.dataset)

            if self.verbose:
                print(f"[Epoch {epoch}/{self.epochs}] - Training loss: {avg_train_loss:.6f}")

            # Validate if validation data is provided
            if self.val_data is not None:
                val_loss = self._validate()
                if self.verbose:
                    print(f"[Epoch {epoch}/{self.epochs}] - Validation loss: {val_loss:.6f}")
                # Check for improvement
                if val_loss < self.best_validation_loss - 1e-8:
                    self.best_validation_loss = val_loss
                    self.epochs_no_improve = 0
                    # Save best model weights
                    self.best_state_dict = self.model.state_dict()
                else:
                    self.epochs_no_improve += 1
                # Early stopping
                if self.epochs_no_improve >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    # Restore best weights
                    if self.best_state_dict is not None:
                        self.model.load_state_dict(self.best_state_dict)
                    break
        # Load best model after training
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

    def _validate(self) -> float:
        """
        Runs validation on the validation dataset.
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, _ in self.val_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_x)
                total_loss += loss.item() * batch_x.size(0)
        return total_loss / len(self.val_loader.dataset)

    def train_model(self):
        """
        High-level method to run training.
        """
        self.train()

    def save_checkpoint(self, filename: str):
        """
        Save model state dict to file.
        Args:
            filename (str): Path to save checkpoint.
        """
        torch.save(self.model.state_dict(), filename)

    def load_checkpoint(self, filename: str):
        """
        Load model state dict from checkpoint file.
        Args:
            filename (str): Path to checkpoint.
        """
        self.model.load_state_dict(torch.load(filename))
