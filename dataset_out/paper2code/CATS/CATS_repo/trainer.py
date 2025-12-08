## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from utils import generate_mask, apply_mask
from typing import Dict, Optional

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset: Dict,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer with model, dataset, configs, and device.
        Args:
            model (nn.Module): The CATS model instance.
            dataset (Dict): Dictionary with keys 'train', 'val', 'test' containing Dataset objects or tensors.
            config (Dict): Hyperparameters and settings.
            device (Optional[torch.device]): Computing device.
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Training hyperparameters
        self.lr = self.config['training'].get('learning_rate', 1e-3)
        self.batch_size = self.config['training'].get('batch_size', 32)
        self.epochs = self.config['training'].get('epochs', 30)
        self.dropout_rate = self.config['training'].get('dropout_rate', 0.1)
        self.p_mask = self.config['training'].get('mask_probability', 0.2)
        self.patience = self.config['training'].get('patience', 10)
        self.optimizer_type = self.config['training'].get('optimizer', 'Adam')
        self.weight_decay = self.config['training'].get('weight_decay', 1e-4)

        # Dataset splits
        self.train_data = self.dataset['train']
        self.val_data = self.dataset['val']
        self.test_data = self.dataset['test']

        # Initialize optimizer
        if self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # For early stopping
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        # For reproducibility
        torch.manual_seed(self.config.get('training', {}).get('seed', 42))
        np.random.seed(self.config.get('training', {}).get('seed', 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.get('training', {}).get('seed', 42))
    
    def _get_loader(self, data_tensor: torch.Tensor) -> torch.utils.data.DataLoader:
        """
        Wrap tensor dataset into DataLoader.
        """
        dataset = torch.utils.data.TensorDataset(data_tensor)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
    
    def train(self):
        """
        Run the full training process with early stopping.
        """
        train_loader = self._get_loader(self.train_data)
        val_loader = self._get_loader(self.val_data)

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._validate(val_loader)

            print(f"Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")
            self.scheduler.step(val_loss)

            # Check for early stopping
            if val_loss < self.best_val_loss:
                print("Validation loss improved. Saving model...")
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self._save_checkpoint('best.pth')
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        # Load best model weights after training
        self._load_checkpoint('best.pth')

    def _train_one_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Train over one epoch.
        """
        self.model.train()
        total_loss = 0.0
        progress = tqdm(dataloader, desc="Training", leave=False)
        for batch in progress:
            input_seq = batch[0].to(self.device)  # shape (B, L, D)
            # Generate horizon queries inside the model
            # Forward pass with masking
            self.optimizer.zero_grad()

            # Generate random masks for queries if needed
            # During training, apply stochastic query masking
            with torch.no_grad():
                # This could be handled in model; here kept simple
                pass

            # Forward pass
            forecast = self.model(input_seq, training=True)  # shape (B, T, 1)

            # Ground truth extraction: assumes batch contains corresponding target seqs
            # For simplicity, suppose dataset yields input sequences and target sequences.
            # Here, adjust according to data pipeline.
            target_seq = batch[1].to(self.device)  # shape (B, T, D)
            loss_fn = nn.MSELoss()
            loss = loss_fn(forecast, target_seq)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        return total_loss / len(dataloader)

    def _validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Evaluate on validation set.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                input_seq = batch[0].to(self.device)
                target_seq = batch[1].to(self.device)

                forecast = self.model(input_seq, training=False)
                loss_fn = nn.MSELoss()
                loss = loss_fn(forecast, target_seq)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def _save_checkpoint(self, filename: str):
        """
        Save model state dict.
        """
        torch.save(self.model.state_dict(), filename)

    def _load_checkpoint(self, filename: str):
        """
        Load saved model weights.
        """
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
    
    def evaluate(self):
        """
        Run evaluation on test dataset, compute metrics.
        """
        test_loader = self._get_loader(self.test_data)
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_seq = batch[0].to(self.device)
                target_seq = batch[1].to(self.device)
                forecast = self.model(input_seq, training=False)
                all_preds.append(forecast.cpu())
                all_targets.append(target_seq.cpu())
                loss_fn = nn.MSELoss()
                loss = loss_fn(forecast, target_seq)
                total_loss += loss.item()

        # Concatenate all predictions and targets
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        mse = nn.functional.mse_loss(preds, targets).item()
        mae = nn.functional.l1_loss(preds, targets).item()
        print(f"Test MSE: {mse:.6f} | Test MAE: {mae:.6f}")
        return {'MSE': mse, 'MAE': mae}
