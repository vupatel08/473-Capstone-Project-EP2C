## trainer.py
import os
from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils import get_pressure_weights
from dataset import WeatherDataset
from model import TransformerModel

class StormerTrainer(pl.LightningModule):
    def __init__(self, 
                 config: Dict,
                 train_dataset: WeatherDataset,
                 valid_dataset: WeatherDataset,
                 test_dataset: WeatherDataset,
                 ):
        """
        Initialize the Lightning Module for training Stormer.
        """
        super().__init__()
        self.config = config
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        # Extract architecture parameters
        model_cfg = self.config['model']
        self.model = TransformerModel(model_cfg)

        # Prepare pressure weights for loss
        variables = (self.config['dataset'].get('variables') or {})
        pressure_levels = self.config['dataset']['pressure_levels']
        # Pressure weights: assign higher weight to near-surface
        # Using the scheme in paper: 1.0 for T2m, 0.1 for others
        var_weights = {
            'T2m': 1.0,
            'MSLP': 0.1,
            'U10': 0.1,
            'V10': 0.1
        }
        # For atmospheric variables at pressure levels, assign same weight
        self.pressure_weights = get_pressure_weights(
            variables=sum(variables.values(), []),
            pressure_levels=pressure_levels,
            variable_pressure_mapping=None,
            surface_vars=['T2m', 'MSLP', 'U10', 'V10']
        )
        # Convert to tensor for device usage
        self.pressure_weight_tensor = torch.tensor(
            [self.pressure_weights.get(var, 0.1) for var in sum(variables.values(), [])],
            dtype=torch.float32
        )

        # Loss weights for variables
        self.variable_weights = self.pressure_weights
        # Set to self variables
        self.loss_weight_map = self.variable_weights

        # Training phase flags
        self.current_phase = 1
        self.K = 1  # rollout steps
        self.epochs_phase1 = self.config['training'].get('epochs_phase1', 100)
        self.epochs_phase2 = self.config['training'].get('epochs_finetune_2', 20)
        self.epochs_phase3 = self.config['training'].get('epochs_finetune_3', 20)
        self.warmup_epochs = self.config['training'].get('warmup_epochs', 10)
        self.n_epochs = 0

        # Optimizer and scheduler will be configured later
        self.optimizer = None
        self.scheduler = None

        # Checkpoint callbacks (handled externally in fit() but define here for clarity)
        # Will instantiate in main script

        # For logging training loss
        self.train_loss = []

    def configure_optimizers(self):
        # Set different LR for phases if needed, or manage via scheduler
        # For simplicity, manage LR schedule externally with scheduler
        self.optimizer = AdamW(self.parameters(),
                               lr=self.get_current_lr(),
                               weight_decay=self.config['training'].get('weight_decay',1e-5),
                               betas=(0.9, 0.95))
        # LR schedule: cosine
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        return [self.optimizer], [self.scheduler]

    def get_current_lr(self):
        # Return current learning rate based on phase or epoch
        if self.current_phase == 1:
            return self.config['training']['learning_rate_phase1']
        elif self.current_phase == 2:
            return self.config['training']['learning_rate_finetune_2']
        elif self.current_phase == 3:
            return self.config['training']['learning_rate_finetune_3']
        else:
            return self.config['training']['learning_rate_phase1']

    def training_step(self, batch, batch_idx):
        """
        Process one batch: compute loss with randomized δt, rollout, multi-step loss, etc.
        """
        # Extract initial state and true data
        X0 = batch['X0']  # dict of variable tensors: (batch, H, W)
        delta_T = batch['delta_T']  # dict of variable tensors: (batch, H, W)
        lead_hours = batch['lead_hours'].item()

        batch_size = X0[next(iter(X0))].shape[0]

        # Sample δt for this batch from uniform over [6,12,24]
        delta_hours_choices = [6, 12, 24]
        delta_t_hours = np.random.choice(delta_hours_choices, size=batch_size)
        delta_t_hours = torch.tensor(delta_t_hours, dtype=torch.float32, device=self.device).unsqueeze(1)  # (batch,1)

        # Map δt hours to model input: shape (batch, 1)
        # The model expects delta_t in hours as float
        delta_t_input = delta_t_hours

        # Prepare input tensor X: shape (batch, V, H, W), concatenate variables
        # For simplicity, assume batch of dicts: construct input tensor
        V = len(X0)
        H = next(iter(X0.values())).shape[1]
        W = next(iter(X0.values())).shape[2]
        variable_list = list(X0.keys())

        # Stack variables along new dimension
        X_tensor = torch.stack([X0[var] for var in variable_list], dim=1)  # (batch, V, H, W)

        # Forward pass through model
        delta_pred = self.model(X_tensor, delta_t_input)
        # delta_pred shape: (batch, V, H, W)

        # Compute pressure weights for the batch (broadcasted)
        pressure_weights = self.pressure_weight_tensor.to(self.device)  # (V,)
        # Need to expand to match variable dimensions if necessary
        # For per-variable error, multiply squared error
        loss = 0.0
        total_weight = 0.0

        # For each variable, compute weighted MSE
        for idx, var in enumerate(variable_list):
            pred_var = delta_pred[:, idx, :, :]  # (batch, H, W)
            true_delta = batch['delta_T'][var]    # (batch, H, W)
            # Compute squared error
            se = (pred_var - true_delta).pow(2)
            # Weight with pressure and variable weight
            weight = self.loss_weight_map.get(var, 1.0)
            # For pressure weight, if variable is pressure-dependent, scale accordingly
            # For simplicity, assume pressure_weight tensor applies
            var_weight = se * pressure_weights[idx] * weight
            loss += var_weight.sum()
            total_weight += pressure_weights[idx] * weight * batch_size * H * W

        # Normalize loss
        loss = loss / total_weight

        # Multi-step rollout loss
        if self.K > 1:
            # approximate multi-step: generate K-step rollouts during training
            # Using the same delta_t for K steps
            X_current = X_tensor
            total_multi_loss = 0.0
            for step_i in range(1, self.K):
                # Predict Δ at each step
                delta_pred_k = self.model(X_current, delta_t_input)
                # Update current state: X_{k} = X_{k-1} + Δ
                X_next = X_current + delta_pred_k
                # Compute true Δ for the (k+1)th step: during training, approximate equal to one step
    
                # For simplicity, do not simulate true data here, just compute error
                # Actual implementation could involve more accurate multi-step data
                # For now, assume model is trained for K steps, sum loss over all steps
                # But here, for efficiency, we skip true data for subsequent steps, as in paper
                # So, just compute the loss with model predictions
                for idx, var in enumerate(variable_list):
                    pred_var = delta_pred_k[:, idx, :, :]
                    true_delta = batch['delta_T'][var]
                    se = (pred_var - true_delta).pow(2)
                    weight = self.loss_weight_map.get(var, 1.0)
                    var_weight = se * pressure_weights[idx] * weight
                    total_multi_loss += var_weight.sum()
                X_current = X_next
            # Average multi-step loss
            total_multi_loss = total_multi_loss / (self.K - 1)
            loss = 0.5 * loss + 0.5 * total_multi_loss  # weighting can be tuned

        # Log loss
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Evaluate model on validation batch: compute metrics.
        """
        X0 = batch['X0']
        delta_T = batch['delta_T']
        lead_hours = batch['lead_hours'].item()

        # Use a fixed delta_t during validation (e.g., mean or smallest)
        delta_t_hours = torch.tensor([lead_hours], dtype=torch.float32, device=self.device)
        delta_t_input = delta_t_hours

        V = len(X0)
        variable_list = list(X0.keys())

        X_tensor = torch.stack([X0[var] for var in variable_list], dim=1)  # (batch, V, H, W)

        delta_pred = self.model(X_tensor, delta_t_input)

        # Compute metrics
        preds = {}
        trues = {}
        for idx, var in enumerate(variable_list):
            preds[var] = delta_pred[:, idx, :, :]
            trues[var] = batch['delta_T'][var]
        # Compute validation loss
        val_loss = 0.0
        total_weight = 0.0
        for idx, var in enumerate(variable_list):
            pred_var = preds[var]
            true_delta = trues[var]
            se = (pred_var - true_delta).pow(2)
            weight = self.loss_weight_map.get(var, 1.0)
            var_weight = se * pressure_weights[idx] * weight
            val_loss += var_weight.sum()
            total_weight += pressure_weights[idx] * weight * batch['X0'][var].shape[0] * true_delta.shape[1] * true_delta.shape[2]
        val_loss = val_loss / total_weight

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': val_loss}

    def configure_callbacks(self):
        # Callbacks for checkpointing and early stopping
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.config['logging']['save_dir'],
            filename='stormer-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min',
            every_n_epochs=self.config['logging'].get('save_checkpoint_interval', 10)
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.config['logging'].get('early_stopping_patience', 15),
            mode='min'
        )
        return [checkpoint_callback, early_stop_callback]

    def on_train_epoch_end(self):
        """
        Manage phase transitions based on epoch count.
        """
        epoch = self.current_epoch
        total_epochs_phase1 = self.epochs_phase1
        total_epochs_phase2 = total_epochs_phase1 + self.epochs_phase2
        total_epochs_phase3 = total_epochs_phase2 + self.epochs_phase3

        if epoch >= total_epochs_phase1 and self.current_phase == 1:
            # Transition to phase 2
            self.current_phase = 2
            self.K = 4
            self._load_checkpoint(self.config['logging'].get('checkpoint_phase2'))
            self._update_lr()
        elif epoch >= total_epochs_phase2 and self.current_phase == 2:
            # Transition to phase 3
            self.current_phase = 3
            self.K = 8
            self._load_checkpoint(self.config['logging'].get('checkpoint_phase3'))
            self._update_lr()

    def _load_checkpoint(self, checkpoint_path: Optional[str]):
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(checkpoint['state_dict'])

    def _update_lr(self):
        # Update optimizer LR based on current phase
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_current_lr()

# Utility functions for training loop outside this class:
# - training loop manages epoch counts, calls self.train(), validation, saves checkpoints, etc.
