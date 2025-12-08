# trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import math
from typing import Optional, List, Dict

# Import the Model class
from model import TimerTransformer

class Trainer:
    """
    Trainer manages the training loop, learning rate scheduling, checkpointing,
    and evaluation for the Timer large time series model.
    """
    def __init__(self, 
                 model: TimerTransformer,
                 train_dataset: List[torch.Tensor],
                 val_dataset: Optional[List[torch.Tensor]],
                 config: Dict):
        """
        Args:
            model (TimerTransformer): The pre-initialized or randomly initialized model.
            train_dataset (List[torch.Tensor]): List of tokenized training sequences.
            val_dataset (Optional[List[torch.Tensor]]): List of tokenized validation sequences.
            config (dict): Configuration from YAML containing hyperparameters.
        """
        # Store parameters from config with defaults
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        # Hyperparameters
        self.batch_size: int = self.config.get('training', {}).get('batch_size', 2048)
        self.epochs: int = self.config.get('training', {}).get('epochs', 10)
        self.learning_rate: float = self.config.get('training', {}).get('learning_rate', 3e-5)
        self.warmup_steps: int = self.config.get('training', {}).get('warmup_steps', 1000)
        self.decay_strategy: str = self.config.get('training', {}).get('decay_strategy', 'exponential')
        self.decay_rate: float = self.config.get('training', {}).get('decay_rate', 0.5)
        self.save_dir: str = self.config.get('logging', {}).get('save_dir', 'checkpoints/')
        self.log_interval: int = self.config.get('logging', {}).get('log_interval', 100)
        self.save_interval: int = self.config.get('logging', {}).get('save_interval', 1)

        os.makedirs(self.save_dir, exist_ok=True)

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)

        # Learning rate scheduler with warmup + decay
        self.global_step = 0
        self._init_scheduler()

        # Prepare DataLoader
        self.train_loader = self._create_dataloader(self.train_dataset, self.batch_size)
        if self.val_dataset is not None:
            self.val_loader = self._create_dataloader(self.val_dataset, self.batch_size, shuffle=False)
        else:
            self.val_loader = None

        # Logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        self.logger = logging.getLogger('Trainer')

        # Tracking best validation performance (e.g., lowest val loss)
        self.best_val_loss = float('inf')
        self.best_checkpoint_path = None

    def _create_dataloader(self, dataset: List[torch.Tensor], batch_size: int, shuffle: bool = True) -> DataLoader:
        """
        Create DataLoader for dataset with collate fn to handle batching.
        """
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)

    def _collate_fn(self, batch: List[torch.Tensor]) -> Dict:
        """
        Collate function to batch sequences, convert to tensor, create causal mask.
        """
        # Stack sequences: batch_size x seq_len
        input_ids = torch.stack(batch, dim=0)  # (B, L)
        input_ids = input_ids.to(self.device)

        # Create causal attention mask (lower triangular)
        seq_len = input_ids.shape[1]
        attn_mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device)).bool()

        return {
            'input_ids': input_ids,
            'attn_mask': attn_mask
        }

    def _init_scheduler(self):
        """
        Initialize learning rate scheduler as per decay strategy.
        """
        total_steps = len(self.train_loader) * self.epochs
        # Using a custom scheduler: exponential decay post warmup
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(self.warmup_steps)
            else:
                # Steps after warmup
                decay_steps = total_steps - self.warmup_steps
                progress = float(current_step - self.warmup_steps) / decay_steps
                return self.decay_rate ** progress
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def train(self):
        """
        Run the full training loop over epochs.
        """
        for epoch in range(1, self.epochs + 1):
            self.logger.info(f"Starting epoch {epoch}")
            epoch_loss = 0.0
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
            for step, batch in enumerate(pbar, start=1):
                self.global_step += 1
                self.optimizer.zero_grad()

                input_ids = batch['input_ids']  # (B, L)
                attn_mask = batch['attn_mask']   # (L, L)

                # For autoregressive generation, model input and labels: input[:-1], target[:-1]
                # Direct prediction of next tokens: input sequence shifted
                # But in GPT, typically input is sequence, label is next token shifted by one
                # For simplicity, using full sequence; loss computed on each token shift
                logits = self.model(input_ids)

                # Shift inputs and targets for causal prediction
                target = input_ids[:, 1:].contiguous()
                logits = logits[:, :-1, :]  # (B, L-1, vocab_size or output_dim)

                # Loss: CrossEntropy over token IDs
                loss_fn = nn.CrossEntropyLoss()
                # Reshape logits and targets
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # optional

                self.optimizer.step()
                # Step scheduler
                if self.decay_strategy == 'exponential':
                    self.lr_scheduler.step()

                epoch_loss += loss.item()

                if step % self.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(f"Epoch {epoch} Step {step}/{len(self.train_loader)} "
                                     f"Loss: {loss.item():.4f} LR: {current_lr:.6f} Total Loss: {epoch_loss/step:.4f}")

            # End of epoch: evaluate & save checkpoint if needed
            avg_loss = epoch_loss / len(self.train_loader)
            self.logger.info(f"Epoch {epoch} finished with average loss: {avg_loss:.4f}")

            # Optional: validation
            if self.val_loader is not None:
                val_loss = self.evaluate()
                self.logger.info(f"Validation Loss after epoch {epoch}: {val_loss:.4f}")
                # Save checkpoint if best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    checkpoint_path = os.path.join(self.save_dir, f"best_epoch_{epoch}.pt")
                    self.save_checkpoint(checkpoint_path)
                    self.best_checkpoint_path = checkpoint_path
            # Save checkpoint periodically
            if epoch % self.save_interval == 0:
                checkpoint_path = os.path.join(self.save_dir, f"epoch_{epoch}.pt")
                self.save_checkpoint(checkpoint_path)

    def evaluate(self):
        """
        Run evaluation on validation dataset, compute average loss.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids']
                attn_mask = batch['attn_mask']
                logits = self.model(input_ids)
                target = input_ids[:, 1:].contiguous()
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, path: str):
        """
        Save model and optimizer states.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'epoch': getattr(self, 'current_epoch', 0),
            'global_step': self.global_step
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load saved checkpoint for resuming training.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)

