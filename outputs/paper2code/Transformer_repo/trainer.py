"""trainer.py
This module implements the Trainer class responsible for the training loop of the Transformer model.
It orchestrates optimizer initialization, learning rate scheduling according to the paper,
loss computation with label smoothing, checkpoint saving and averaging, and logging & validation.
All configuration parameters are read from config.yaml.
"""

import os
import time
import math
import logging
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import get_learning_rate, average_checkpoints, setup_logger


class Trainer:
    """Trainer class to manage training of the Transformer model.

    Attributes:
        model (nn.Module): The Transformer model.
        data (Dict[str, Any]): A dictionary containing training and validation data loaders.
        config (Dict[str, Any]): Configuration parameters loaded from config.yaml.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        warmup_steps (int): Number of warmup steps for learning rate scheduling.
        d_model (int): Model embedding dimension.
        global_step (int): Global step counter.
        checkpoint_paths (List[str]): List of checkpoint file paths saved during training.
        label_smoothing (float): Label smoothing factor.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        checkpoint_dir (str): Directory to save checkpoints.
        validate_interval (int): Interval (in steps) to perform validation.
        checkpoint_interval (int): Interval (in steps) to save checkpoints.
        model_variant (str): Model variant, either "base_model" or "big_model".
        total_steps (int): Total training steps as specified in the configuration.
        device (torch.device): The training device (CPU or CUDA).
    """

    def __init__(self, model: nn.Module, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Initializes the Trainer with the provided model, data, and configuration.

        Args:
            model (nn.Module): The Transformer model.
            data (Dict[str, Any]): Dictionary containing data loaders.
            config (Dict[str, Any]): Configuration dictionary from config.yaml.
        """
        self.model = model
        self.data = data
        self.config = config

        training_config: Dict[str, Any] = config.get("training", {})
        beta1: float = training_config.get("beta1", 0.9)
        beta2: float = training_config.get("beta2", 0.98)
        epsilon: float = training_config.get("epsilon", 1e-9)
        self.warmup_steps: int = training_config.get("warmup_steps", 4000)

        hyperparams: Dict[str, Any] = config.get("hyperparameters", {})
        base_model_params: Dict[str, Any] = hyperparams.get("base_model", {"d_model": 512, "label_smoothing": 0.1})
        big_model_params: Dict[str, Any] = hyperparams.get("big_model", {"d_model": 1024, "label_smoothing": 0.1})
        # Determine model variant based on the embedding dimension.
        model_embedding_dim: int = getattr(self.model.embedding, "embedding_dim", base_model_params.get("d_model", 512))
        if model_embedding_dim == big_model_params.get("d_model", 1024):
            self.model_variant = "big_model"
            self.total_steps: int = training_config.get("big_train_steps", 300000)
            self.checkpoint_count: int = 20
        else:
            self.model_variant = "base_model"
            self.total_steps: int = training_config.get("base_train_steps", 100000)
            self.checkpoint_count: int = 5

        self.d_model: int = model_embedding_dim

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0, betas=(beta1, beta2), eps=epsilon
        )
        self.global_step: int = 0
        self.checkpoint_paths: List[str] = []

        # Get label smoothing factor from corresponding hyperparameters.
        self.label_smoothing: float = hyperparams.get(self.model_variant, {}).get("label_smoothing", 0.1)

        # Setup device and move model to device.
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Retrieve training and validation data loaders.
        if "translation" in self.data and "en_de" in self.data["translation"]:
            self.train_dataloader = self.data["translation"]["en_de"].get("train")
            self.val_dataloader = self.data["translation"]["en_de"].get("val")
        elif "translation" in self.data:
            first_key: str = list(self.data["translation"].keys())[0]
            self.train_dataloader = self.data["translation"][first_key].get("train")
            self.val_dataloader = self.data["translation"][first_key].get("val")
        elif "parsing" in self.data and "wsj" in self.data["parsing"]:
            self.train_dataloader = self.data["parsing"]["wsj"].get("train")
            self.val_dataloader = self.data["parsing"]["wsj"].get("val")
        else:
            raise ValueError("No valid training data found in the provided data dictionary.")

        self.checkpoint_dir: str = "checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Set default intervals for validation and checkpointing.
        self.validate_interval: int = 1000
        self.checkpoint_interval: int = 1000

        self.logger = setup_logger()

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the label-smoothed cross-entropy loss for the given logits and target tokens.

        Args:
            logits (torch.Tensor): Logits from the model of shape (batch_size, seq_len, vocab_size).
            target (torch.Tensor): Ground truth token indices of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The computed loss value.
        """
        vocab_size: int = logits.size(-1)
        log_probs: torch.Tensor = F.log_softmax(logits, dim=-1)
        # Flatten logits and target tensors.
        log_probs_flat: torch.Tensor = log_probs.contiguous().view(-1, vocab_size)
        target_flat: torch.Tensor = target.contiguous().view(-1)
        smoothing: float = self.label_smoothing

        with torch.no_grad():
            true_dist: torch.Tensor = torch.zeros_like(log_probs_flat)
            true_dist.fill_(smoothing / (vocab_size - 1))
            true_dist.scatter_(1, target_flat.unsqueeze(1), 1.0 - smoothing)
        # Create mask to ignore pad tokens (assumed pad index is 0).
        non_pad_mask: torch.Tensor = target_flat.ne(0).float()
        loss: torch.Tensor = -(true_dist * log_probs_flat).sum(dim=1)
        loss = loss * non_pad_mask
        average_loss: torch.Tensor = loss.sum() / non_pad_mask.sum().clamp(min=1)
        return average_loss

    def save_checkpoint(self) -> None:
        """
        Saves a checkpoint of the model and optimizer state at the current global step.
        The checkpoint is stored in the designated checkpoint directory.
        """
        checkpoint: Dict[str, Any] = {
            "step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        checkpoint_file: str = os.path.join(self.checkpoint_dir, f"checkpoint_{self.global_step}.pt")
        torch.save(checkpoint, checkpoint_file)
        self.checkpoint_paths.append(checkpoint_file)
        self.logger.info(f"Saved checkpoint: {checkpoint_file}")

    def average_checkpoints_final(self) -> None:
        """
        Averages the last few saved checkpoints and saves the averaged model state as the final checkpoint.
        The number of checkpoints averaged is determined by the model variant.
        """
        if len(self.checkpoint_paths) < self.checkpoint_count:
            self.logger.warning("Not enough checkpoints to average. Skipping checkpoint averaging.")
            return
        selected_checkpoints: List[str] = self.checkpoint_paths[-self.checkpoint_count:]
        self.logger.info(f"Averaging the last {self.checkpoint_count} checkpoints...")
        averaged_state: Dict[str, torch.Tensor] = average_checkpoints(selected_checkpoints)
        final_checkpoint_file: str = os.path.join(self.checkpoint_dir, "final_checkpoint.pt")
        torch.save(averaged_state, final_checkpoint_file)
        self.logger.info(f"Saved averaged final checkpoint: {final_checkpoint_file}")

    def validate(self) -> Dict[str, float]:
        """
        Runs the model in evaluation mode on the validation dataset and computes the average validation loss.

        Returns:
            Dict[str, float]: A dictionary containing validation metrics (e.g., average loss).
        """
        self.model.eval()
        total_loss: float = 0.0
        total_tokens: int = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                src: torch.Tensor = batch["src"].to(self.device)
                tgt: torch.Tensor = batch["tgt"].to(self.device)
                src_mask: torch.Tensor = batch["src_mask"].to(self.device)
                tgt_mask: torch.Tensor = batch["tgt_mask"].to(self.device)
                # Teacher forcing: use decoder input as tgt[:, :-1] and target for loss as tgt[:, 1:].
                tgt_input: torch.Tensor = tgt[:, :-1]
                tgt_target: torch.Tensor = tgt[:, 1:]
                tgt_mask_input: torch.Tensor = tgt_mask[:, :-1]
                logits: torch.Tensor = self.model(src, tgt_input, src_mask, tgt_mask_input)
                loss: torch.Tensor = self.compute_loss(logits, tgt_target)
                non_pad: int = tgt_target.ne(0).sum().item()
                total_loss += loss.item() * non_pad
                total_tokens += non_pad
        avg_loss: float = total_loss / total_tokens if total_tokens > 0 else float('inf')
        self.logger.info(f"Validation Loss at step {self.global_step}: {avg_loss:.4f}")
        self.model.train()
        return {"validation_loss": avg_loss}

    def train(self) -> None:
        """
        Executes the training loop for the Transformer model.
        The loop carries out forward passes, loss computation with label smoothing, backpropagation,
        optimizer updates with a custom learning rate scheduler, periodic validation, and checkpoint saving.
        """
        self.model.train()
        train_iterator = iter(self.train_dataloader)
        progress_bar = tqdm(total=self.total_steps, desc="Training", unit="step")
        start_time_overall = time.perf_counter()

        while self.global_step < self.total_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)

            step_start_time = time.perf_counter()

            # Move batch tensors to the designated device.
            src: torch.Tensor = batch["src"].to(self.device)
            tgt: torch.Tensor = batch["tgt"].to(self.device)
            src_mask: torch.Tensor = batch["src_mask"].to(self.device)
            tgt_mask: torch.Tensor = batch["tgt_mask"].to(self.device)

            # Prepare decoder inputs and targets (shift by one position).
            tgt_input: torch.Tensor = tgt[:, :-1]
            tgt_target: torch.Tensor = tgt[:, 1:]
            tgt_mask_input: torch.Tensor = tgt_mask[:, :-1]

            # Forward pass through the model.
            logits: torch.Tensor = self.model(src, tgt_input, src_mask, tgt_mask_input)

            # Compute label-smoothed loss.
            loss: torch.Tensor = self.compute_loss(logits, tgt_target)

            # Backpropagation.
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Increment global step.
            self.global_step += 1

            # Update learning rate as per the schedule.
            current_lr: float = get_learning_rate(self.d_model, self.global_step, self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

            step_end_time = time.perf_counter()
            step_time: float = step_end_time - step_start_time

            progress_bar.update(1)
            progress_bar.set_postfix({
                "Step": self.global_step,
                "Loss": f"{loss.item():.4f}",
                "LR": f"{current_lr:.6f}",
                "StepTime": f"{step_time:.4f}s"
            })

            # Perform periodic validation.
            if self.global_step % self.validate_interval == 0:
                self.validate()

            # Save checkpoint at defined intervals.
            if self.global_step % self.checkpoint_interval == 0:
                self.save_checkpoint()

        progress_bar.close()
        total_time: float = time.perf_counter() - start_time_overall
        self.logger.info(f"Training completed in {total_time:.2f} seconds after {self.global_step} steps.")
        self.validate()
        self.average_checkpoints_final()
