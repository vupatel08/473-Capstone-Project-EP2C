# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import math
from utils import normalize_features, compute_statistics
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model: nn.Module, dataloader, loss_fn, optimizer, config: dict):
        """
        Args:
            model (nn.Module): The self-supervised model with encoder (and predictor if needed).
            dataloader (DataLoader): DataLoader providing batches of augmented data.
            loss_fn (UniformityLoss): Loss function integrating SSL and uniformity.
            optimizer (torch.optim.Optimizer): Optimizer instance.
            config (dict): Configuration dictionary with training parameters.
        """
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # Extract configuration parameters
        self.epochs = config.get('training', {}).get('epochs', 500)
        self.warmup_epochs = config.get('training', {}).get('warmup_epochs', 10)
        self.lambda_max = config.get('training', {}).get('lambda_uniformity', 0.1)
        self.lambda_min = 0.0   # For decay
        self.total_epochs = self.epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # For logging
        self.global_step = 0
        os.makedirs(config.get('logging', {}).get('log_dir', './logs'), exist_ok=True)

        # Save the initial state for checkpointing
        self.save_dir = config.get('logging', {}).get('log_dir', './logs')
        self.save_every = config.get('logging', {}).get('save_model_every', 50)

    def get_lambda_t(self, epoch: int) -> float:
        """
        Linearly decay lambda from max to min over total epochs.
        """
        if epoch >= self.total_epochs:
            return self.lambda_min
        return self.lambda_max - (self.lambda_max - self.lambda_min) * (epoch / self.total_epochs)

    def train(self):
        """
        Main training loop.
        """
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            epoch_uniformity = 0.0
            epoch_base_loss = 0.0
            epoch_count = 0

            lambda_t = self.get_lambda_t(epoch)

            tqdm_loader = tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.epochs}")
            for batch in tqdm_loader:
                # Batch may contain multiple views: e.g., {'view1': ..., 'view2': ...}
                # or a single tensor if only one view
                # For simplicity, assuming dataset yields a dictionary with 'view1', 'view2', (or just 'view')
                # Implement as needed
                # Cast all to device
                # Example assuming batch['view1'], batch['view2']
                x1 = batch['view1'].to(self.device)
                x2 = batch.get('view2', None)
                if x2 is not None:
                    x2 = x2.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass: extract features
                # Assume model.extract_features returns raw features before projection/predictor
                z_a = self.model.extract_features(x1)
                if x2 is not None:
                    z_b = self.model.extract_features(x2)
                else:
                    # For methods without second view
                    z_b = None

                # Normalize features to the sphere
                z_a_norm = normalize_features(z_a)
                if z_b is not None:
                    z_b_norm = normalize_features(z_b)

                # Compute base SSL loss
                if hasattr(self.model, 'predictor') and self.model.predictor is not None:
                    # For BYOL or similar
                    base_loss = self.loss_fn.compute_base_loss(z_a_norm, z_b_norm)
                else:
                    # For contrastive methods or others
                    base_loss = self.loss_fn.compute_base_loss(z_a_norm, z_b_norm)

                # Compute uniformity loss based on features
                # For the batch, collect features (all features from views)
                features_for_uniformity = torch.cat([z_a_norm, z_b_norm], dim=0) if z_b is not None else z_a_norm
                # Compute statistics
                mean, cov = compute_statistics(features_for_uniformity)
                # Compute negative Wasserstein distance
                neg_W2 = self.loss_fn.compute_wasserstein_distance(mean, cov)

                # Dynamic lambda - decay schedule
                lambda_now = lambda_t

                # Total loss
                total_loss = base_loss + lambda_now * neg_W2

                # Backpropagation
                total_loss.backward()
                self.optimizer.step()

                # Logging
                batch_size = x1.shape[0]
                epoch_loss += total_loss.item() * batch_size
                epoch_base_loss += base_loss.item() * batch_size
                epoch_uniformity += neg_W2 * batch_size
                epoch_count += batch_size

                self.global_step += 1

            # Average metrics for epoch
            avg_loss = epoch_loss / epoch_count
            avg_base_loss = epoch_base_loss / epoch_count
            avg_uniformity = epoch_uniformity / epoch_count

            # Save checkpoint
            if epoch % self.save_every == 0:
                checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)

            # Log epoch metrics
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, BaseLoss={avg_base_loss:.4f}, Uniformity={avg_uniformity:.4f}, Lambda={lambda_now:.4f}")

        # End of training
        print("Training completed.")

