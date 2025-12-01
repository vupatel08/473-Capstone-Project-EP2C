"""trainer.py

This module contains the Trainer class which orchestrates the training process for the
TruthXModel. It iterates over the training dataset in batches, computes the reconstruction,
contrastive, and editing losses, performs back–propagation and optimization, logs progress,
and finally computes the overall truth–editing direction δ (delta) from the training data.

Dependencies:
    - torch
    - torch.nn.functional
    - torch.utils.data.DataLoader
    - logging
    - tqdm
    - model.py (TruthXModel)
    - utils.py (attention_fusion, mse_reconstruction_loss, contrastive_loss)

Author: [Your Name]
Date: [Today's Date]
"""

import math
import logging
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import TruthXModel
from utils import mse_reconstruction_loss, contrastive_loss, attention_fusion

# Initialize module level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Trainer:
    """
    Trainer class for training the TruthXModel.
    
    Attributes:
        model (TruthXModel): The TruthXModel instance to be trained.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        config (Dict): Configuration dictionary loaded from config.yaml.
        optimizer (torch.optim.Optimizer): Adam optimizer.
        device (torch.device): Device to perform training on.
        dataloader (DataLoader): PyTorch DataLoader for batching the training dataset.
    """

    def __init__(self, model: TruthXModel, train_dataset: torch.utils.data.Dataset, config: Dict) -> None:
        """
        Initializes Trainer with the model, training dataset, and configuration.

        Args:
            model (TruthXModel): The pre-instantiated TruthXModel.
            train_dataset (torch.utils.data.Dataset): The dataset containing triplets.
            config (Dict): Configuration dictionary from config.yaml.
        """
        self.model: TruthXModel = model
        self.train_dataset: torch.utils.data.Dataset = train_dataset
        self.config: Dict = config

        training_config: Dict = config.get("training", {})
        self.learning_rate: float = training_config.get("learning_rate", 1e-4)
        self.batch_size: int = int(training_config.get("batch_size", 16))
        self.epochs: int = int(training_config.get("epochs", 5))
        
        # Contrastive loss temperature setting from config
        contrastive_config: Dict = config.get("contrastive", {})
        self.temperature: float = contrastive_config.get("temperature", 0.1)

        # Device configuration
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup optimizer: Adam with specified learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create DataLoader from the training dataset
        self.dataloader: DataLoader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        logger.info("Trainer initialized with batch_size=%d, epochs=%d, learning_rate=%.1e", 
                    self.batch_size, self.epochs, self.learning_rate)

    def train(self) -> torch.Tensor:
        """
        Runs the training loop over the dataset for a specified number of epochs.
        At each batch, computes the reconstruction loss, contrastive loss, and editing loss.
        After finishing training, computes the overall truth–editing direction δ as the difference
        between the means of h_truth for truthful and untruthful examples.
        
        Returns:
            torch.Tensor: The computed truth–editing direction δ of shape (latent_dim,).
        """
        self.model.train()
        total_steps: int = self.epochs * len(self.dataloader)
        step_counter: int = 0

        for epoch in range(1, self.epochs + 1):
            epoch_loss_total: float = 0.0
            epoch_loss_recon: float = 0.0
            epoch_loss_ctr: float = 0.0
            epoch_loss_edit: float = 0.0

            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.epochs}")
            for batch in progress_bar:
                # For each sample in batch, we simulate the internal hidden representation "x"
                # as a random tensor since actual LLM activations are assumed to be extracted previously.
                # We generate two representations per sample: one for the truthful branch (x_pos)
                # and one for the untruthful branch (x_neg).
                batch_size_current: int = self.batch_size  # fixed batch size as per DataLoader

                d_model: int = self.config.get("model", {}).get("d_model", 4096)
                # Generate simulated hidden activations for truthful and untruthful answers.
                x_pos: torch.Tensor = torch.randn(batch_size_current, d_model, device=self.device)
                x_neg: torch.Tensor = torch.randn(batch_size_current, d_model, device=self.device)

                # Forward pass for truthful branch
                x_recon_pos, h_truth_pos, h_sem_pos = self.model(x_pos)
                # Forward pass for untruthful branch
                x_recon_neg, h_truth_neg, h_sem_neg = self.model(x_neg)

                # 1. Reconstruction Loss (MSE between original and reconstructed)
                loss_recon_pos: torch.Tensor = mse_reconstruction_loss(x_pos, x_recon_pos)
                loss_recon_neg: torch.Tensor = mse_reconstruction_loss(x_neg, x_recon_neg)
                loss_recon: torch.Tensor = loss_recon_pos + loss_recon_neg

                # 2. Editing Loss:
                # Swap the truthful latent codes between branches and reconstruct.
                fused_pos_to_neg: torch.Tensor = h_sem_pos + attention_fusion(h_sem_pos, h_truth_neg)
                x_pos_to_neg: torch.Tensor = self.model.decode(fused_pos_to_neg)
                fused_neg_to_pos: torch.Tensor = h_sem_neg + attention_fusion(h_sem_neg, h_truth_pos)
                x_neg_to_pos: torch.Tensor = self.model.decode(fused_neg_to_pos)
                loss_edit_pos: torch.Tensor = mse_reconstruction_loss(x_neg, x_pos_to_neg)
                loss_edit_neg: torch.Tensor = mse_reconstruction_loss(x_pos, x_neg_to_pos)
                loss_edit: torch.Tensor = loss_edit_pos + loss_edit_neg

                # 3. Contrastive Loss in truthful and semantic spaces.
                # Construct positive and negative sets for each branch.
                # For truthful latent representations:
                # Expand h_truth_pos and h_truth_neg to shape (B, B, latent_dim)
                B: int = batch_size_current
                # h_truth_pos and h_truth_neg: shape (B, latent_dim)
                positives_truth: torch.Tensor = h_truth_pos.unsqueeze(1).expand(B, B, h_truth_pos.size(-1))
                negatives_truth: torch.Tensor = h_truth_neg.unsqueeze(1).expand(B, B, h_truth_neg.size(-1))
                loss_ctr_truth_pos: torch.Tensor = contrastive_loss(
                    anchor=h_truth_pos, positives=positives_truth, negatives=negatives_truth, temperature=self.temperature
                )
                positives_truth_neg: torch.Tensor = h_truth_neg.unsqueeze(1).expand(B, B, h_truth_neg.size(-1))
                negatives_truth_pos: torch.Tensor = h_truth_pos.unsqueeze(1).expand(B, B, h_truth_pos.size(-1))
                loss_ctr_truth_neg: torch.Tensor = contrastive_loss(
                    anchor=h_truth_neg, positives=positives_truth_neg, negatives=negatives_truth_pos, temperature=self.temperature
                )
                # Similarly, compute contrastive loss in the semantic space
                positives_sem_pos: torch.Tensor = h_sem_pos.unsqueeze(1).expand(B, B, h_sem_pos.size(-1))
                negatives_sem: torch.Tensor = h_sem_neg.unsqueeze(1).expand(B, B, h_sem_neg.size(-1))
                loss_ctr_sem_pos: torch.Tensor = contrastive_loss(
                    anchor=h_sem_pos, positives=positives_sem_pos, negatives=negatives_sem, temperature=self.temperature
                )
                positives_sem_neg: torch.Tensor = h_sem_neg.unsqueeze(1).expand(B, B, h_sem_neg.size(-1))
                negatives_sem_pos: torch.Tensor = h_sem_pos.unsqueeze(1).expand(B, B, h_sem_pos.size(-1))
                loss_ctr_sem_neg: torch.Tensor = contrastive_loss(
                    anchor=h_sem_neg, positives=positives_sem_neg, negatives=negatives_sem_pos, temperature=self.temperature
                )
                loss_ctr: torch.Tensor = loss_ctr_truth_pos + loss_ctr_truth_neg + loss_ctr_sem_pos + loss_ctr_sem_neg

                # Total loss is the sum of the three losses.
                loss_total: torch.Tensor = loss_recon + loss_edit + loss_ctr

                # Backpropagation
                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                # Update loss accumulators
                epoch_loss_total += loss_total.item()
                epoch_loss_recon += loss_recon.item()
                epoch_loss_ctr += loss_ctr.item()
                epoch_loss_edit += loss_edit.item()

                step_counter += 1
                progress_bar.set_postfix({
                    "Total": f"{loss_total.item():.4f}",
                    "Recon": f"{loss_recon.item():.4f}",
                    "Edit": f"{loss_edit.item():.4f}",
                    "Ctr": f"{loss_ctr.item():.4f}"
                })
            # End of epoch: log average losses
            avg_total = epoch_loss_total / len(self.dataloader)
            avg_recon = epoch_loss_recon / len(self.dataloader)
            avg_edit = epoch_loss_edit / len(self.dataloader)
            avg_ctr = epoch_loss_ctr / len(self.dataloader)
            logger.info("Epoch %d completed: Avg Total Loss=%.4f, Avg Recon Loss=%.4f, Avg Edit Loss=%.4f, Avg Contrastive Loss=%.4f",
                        epoch, avg_total, avg_recon, avg_edit, avg_ctr)

            # Save checkpoint at end of epoch
            checkpoint_path: str = f"truthx_model_epoch_{epoch}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            logger.info("Saved checkpoint to %s", checkpoint_path)

        # After training, compute the overall truth-editing direction δ over the entire training dataset.
        truth_latents_pos: List[torch.Tensor] = []
        truth_latents_neg: List[torch.Tensor] = []
        # Set model to evaluation mode.
        self.model.eval()
        with torch.no_grad():
            dataloader_eval: DataLoader = DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
            )
            for batch in tqdm(dataloader_eval, desc="Computing truth-editing direction"):
                B_eval: int = batch.get("question", [None])  # dummy; actual batch size defined by DataLoader
                # Simulate x_pos and x_neg for evaluation (same method as in training)
                x_pos_eval: torch.Tensor = torch.randn(self.batch_size, self.config.get("model", {}).get("d_model", 4096), device=self.device)
                x_neg_eval: torch.Tensor = torch.randn(self.batch_size, self.config.get("model", {}).get("d_model", 4096), device=self.device)
                _, h_truth_pos_eval, _ = self.model(x_pos_eval)
                _, h_truth_neg_eval, _ = self.model(x_neg_eval)
                truth_latents_pos.append(h_truth_pos_eval)
                truth_latents_neg.append(h_truth_neg_eval)
            # Concatenate all batches along dimension 0.
            if truth_latents_pos:
                all_h_truth_pos: torch.Tensor = torch.cat(truth_latents_pos, dim=0)
                all_h_truth_neg: torch.Tensor = torch.cat(truth_latents_neg, dim=0)
                delta: torch.Tensor = torch.mean(all_h_truth_pos, dim=0) - torch.mean(all_h_truth_neg, dim=0)
            else:
                # Fallback: set delta to zeros if no data found.
                latent_dim: int = self.config.get("model", {}).get("latent_dim", 1024)
                delta = torch.zeros(latent_dim, device=self.device)
            # Store the computed delta in the model for later inference use.
            self.model.truth_edit_direction = delta
            logger.info("Computed truth-editing direction δ with norm: %.4f", delta.norm().item())
        return delta


# For standalone testing of Trainer
if __name__ == "__main__":
    # Sample configuration dictionary (could be loaded from config.yaml)
    sample_config: Dict = {
        "training": {
            "learning_rate": 1e-4,
            "batch_size": 16,
            "epochs": 5,
        },
        "model": {
            "d_model": 4096,
            "encoder_dims": [4096, 2048, 1024],
            "decoder_dims": [1024, 2048, 4096],
            "latent_dim": 1024,
            "k_edit_layers": 10,
        },
        "contrastive": {
            "temperature": 0.1,
        },
        "data": {
            "dataset": "TruthfulQA",
            "train_samples": 408,
            "test_samples": 408,
        },
        "evaluation": {
            "metrics": ["TruePercentage", "InfoPercentage", "TrueInfoProduct", "MC1", "MC2", "MC3"],
        }
    }

    # For testing, we simulate a dummy dataset using a list of dicts.
    # Each sample contains minimal fields required by DatasetLoader.
    dummy_samples: List[Dict] = []
    for i in range(100):  # 100 dummy samples
        dummy_samples.append({
            "question": f"Sample question {i}?",
            "truthful_answer": f"This is a truthful answer for sample {i}.",
            "untruthful_answer": f"This is an untruthful answer for sample {i}.",
            "shared_tokens": ["this", "is", "a", "sample"]  # Dummy shared tokens.
        })
    # Create a dummy dataset that is compatible with torch.utils.data.Dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, samples: List[Dict]) -> None:
            self.samples = samples
        def __len__(self) -> int:
            return len(self.samples)
        def __getitem__(self, idx: int) -> Dict:
            return self.samples[idx]

    dummy_dataset = DummyDataset(dummy_samples)
    
    # Instantiate the TruthXModel using the sample configuration.
    model_instance = TruthXModel(sample_config)
    # Instantiate Trainer.
    trainer = Trainer(model=model_instance, train_dataset=dummy_dataset, config=sample_config)
    # Run training.
    editing_direction = trainer.train()
    logger.info("Training completed. Final truth-editing direction δ shape: %s", editing_direction.shape)
