# trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import copy
from typing import Optional, Dict, Any
from dataset_loader import DatasetLoader
from model import PretrainedModel
from prompt_module import PromptGenerator
from label_mapping import LabelMapper
from evaluation import Evaluator

class Trainer:
    def __init__(
        self,
        model: PretrainedModel,
        prompts: PromptGenerator,
        dataset: Dict[str, Dict[str, Any]],
        label_mapper: LabelMapper,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None,
        config: Dict[str, Any]=None,
    ):
        """
        Initialize the trainer with model, prompts, datasets, label mapper, optimizer, and configs.
        Args:
            model: PretrainedModel instance (frozen backbone).
            prompts: PromptGenerator instance with trainable prompts.
            dataset: dict with 'train' and 'val' DataLoader objects.
            label_mapper: LabelMapper instance for label conversion.
            optimizer: optimizer for prompt and label mapper parameters.
            lr_scheduler: optional LR scheduler.
            config: configuration dictionary from YAML.
        """
        self.model = model
        self.prompts = prompts
        self.label_mapper = label_mapper
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_train = dataset['train']
        self.dataset_val = dataset['val']
        self.epochs = config['training'].get('epochs', 50)
        self.batch_size = config['training'].get('batch_size', 32)
        self.early_stop_patience = config['training'].get('early_stop_patience', 3)
        self.total_iterations = config['training'].get('total_iterations', None)
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.early_stop_counter = 0
        self.checkpoint_path = os.path.join(config['logging'].get('log_dir', './logs'), 'best_model.pth')
        self.validation_metrics = {}
        # Prepare DataLoaders
        self.train_loader = self.dataset_train
        self.val_loader = self.dataset_val
        # Set prompts and label mapper trainable parameters
        self._prepare_train_params()

    def _prepare_train_params(self):
        """
        Collect parameters for optimization: prompts and label_mapper if trainable.
        Backbone is frozen.
        """
        params = []
        # Prompts prompts are trainable tensors
        params += list(self.prompts.prompt_tensor.parameters()) if hasattr(self.prompts.prompt_tensor, 'parameters') else []
        # For frequency prompts, include real and imaginary parts if trainable
        if hasattr(self.prompts, 'real_coeffs'):
            params += list(self.prompts.real_coeffs.parameters()) if hasattr(self.prompts.real_coeffs, 'parameters') else []
        if hasattr(self.prompts, 'imag_coeffs'):
            params += list(self.prompts.imag_coeffs.parameters()) if hasattr(self.prompts.imag_coeffs, 'parameters') else []
        # Label mapping parameters
        if self.label_mapper.strategy == 'FullyMap':
            params += list(self.label_mapper.linear_mapping.parameters())
        # Initialize optimizer with these only
        # Assuming optimizer is already constructed outside and passed in
        # No need to re-initialize here, but confirm optimizer's params
        pass

    def train(self):
        """
        Main training loop with early stopping.
        """
        best_epoch_state = None
        train_loss_history = []
        val_acc_history = []

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            train_loss, train_acc = self._train_one_epoch()
            val_metrics = self._validate()

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Validation Accuracy: {val_metrics['accuracy']:.2f}%")
            
            # Check for improvement
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.early_stop_counter = 0
                # Save best model state
                best_epoch_state = {
                    'prompts': copy.deepcopy(self.prompts),
                    'label_mapper': copy.deepcopy(self.label_mapper),
                    'model_state_dict': copy.deepcopy(self.model.model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                    'epoch': epoch,
                    'val_acc': val_metrics['accuracy']
                }
                self._save_checkpoint(self.checkpoint_path)
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_patience:
                    print("Early stopping triggered.")
                    break

            # Step LR scheduler if provided
            if self.lr_scheduler:
                self.lr_scheduler.step()

        # Load best model
        if best_epoch_state:
            self._load_checkpoint(self.checkpoint_path)
            self.prompts = best_epoch_state['prompts']
            self.label_mapper = best_epoch_state['label_mapper']
            self.model.model.load_state_dict(best_epoch_state['model_state_dict'])

    def _train_one_epoch(self):
        """
        Run a single epoch of training.
        """
        self.model.model.eval()  # freeze backbone
        self.prompts.prompt_tensor.train()
        if hasattr(self.prompts, 'real_coeffs'):
            self.prompts.real_coeffs.train()
        if hasattr(self.prompts, 'imag_coeffs'):
            self.prompts.imag_coeffs.train()
        self._set_optimizer_params()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        for batch in tqdm(self.train_loader):
            imgs = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            # Resize images according to scale (if learnable, differentiable)
            imgs_resized = self._resize_images(imgs)

            # Get current prompts
            prompt = self.prompts.get_prompt()  # shape (C, p, p)
            # Apply prompts to images
            prompted_imgs = self._apply_prompt(imgs_resized, prompt)

            # Forward pass with backbone
            with torch.no_grad():
                preds = self.model.forward(prompted_imgs)  # shape (N, K_s) or features
            # For classification, assume preds are logits
            # Map logits (or features) to target labels
            mapped_preds = self.label_mapper.map(preds, train_data_preds=None)
            loss = self.loss_fn(mapped_preds, labels)
            total_loss += loss.item() * imgs.shape[0]
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            pred_labels = torch.argmax(mapped_preds, dim=1)
            correct += (pred_labels == labels).sum().item()
            total_samples += labels.shape[0]
        epoch_loss = total_loss / total_samples
        epoch_acc = 100.0 * correct / total_samples
        return epoch_loss, epoch_acc

    def _validate(self):
        """
        Run validation epoch
        """
        self.model.model.eval()
        self.prompts.prompt_tensor.eval()
        if hasattr(self.prompts, 'real_coeffs'):
            self.prompts.real_coeffs.eval()
        if hasattr(self.prompts, 'imag_coeffs'):
            self.prompts.imag_coeffs.eval()

        correct = 0
        total_samples = 0
        for batch in tqdm(self.val_loader):
            imgs = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            imgs_resized = self._resize_images(imgs)
            with torch.no_grad():
                prompt = self.prompts.get_prompt()
                prompted_imgs = self._apply_prompt(imgs_resized, prompt)
                preds = self.model.forward(prompted_imgs)
                mapped_preds = self.label_mapper.map(preds)
                pred_labels = torch.argmax(mapped_preds, dim=1)
                correct += (pred_labels == labels).sum().item()
                total_samples += labels.shape[0]
        accuracy = 100.0 * correct / total_samples
        return {'accuracy': accuracy}

    def _save_checkpoint(self, path: str):
        """
        Save the current best model, prompts, label mapper
        """
        checkpoint = {
            'prompts': self.prompts,
            'label_mapper': self.label_mapper,
            'model_state_dict': self.model.model.state_dict(),
        }
        torch.save(checkpoint, path)

    def _load_checkpoint(self, path: str):
        """
        Load saved checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        # Load model backbone if needed
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        # Prompts and label mapper are deep copies
        # (assuming they are serializable or have state_dict methods)
        self.prompts = checkpoint['prompts']
        self.label_mapper = checkpoint['label_mapper']

    def _resize_images(self, imgs: torch.Tensor):
        """
        Resize images according to current scale factor and differentiable if needed.
        For simplicity, assumes fixed scale; can be extended.
        """
        # Placeholder: if scale is fixed, return imgs directly
        # For learnable scale, integrate kornia.transform
        return imgs

    def _apply_prompt(self, imgs: torch.Tensor, prompt: torch.Tensor):
        """
        Add pixel prompts or frequency prompts as per prompt_module design.
        For pixel prompts: overlay/pad prompts onto images.
        """
        # Assume prompt is (C, p, p), images are (N, C, H, W)
        # For simplicity, padding images with prompts (simulate Eq.1)
        p = prompt.shape[1]
        # Use padding to embed prompts
        # This is a simplified example; in practice, the method depends on prompt strategy
        batch_size, C, H, W = imgs.shape
        pad = p
        # For pixel prompts, just overlay or concatenate as needed
        # Placeholder: simple padding with zero
        padded_imgs = F.pad(imgs, (pad, pad, pad, pad), mode='constant', value=0)
        return padded_imgs

    def _set_optimizer_params(self):
        """
        Ensure optimizer only updates prompts and label mapping params.
        """
        params = []
        if hasattr(self.prompts, 'prompt_tensor'):
            params += list(self.prompts.prompt_tensor.parameters()) if hasattr(self.prompts.prompt_tensor, 'parameters') else []
        if hasattr(self.prompts, 'real_coeffs'):
            params += list(self.prompts.real_coeffs.parameters())
        if hasattr(self.prompts, 'imag_coeffs'):
            params += list(self.prompts.imag_coeffs.parameters())
        if self.label_mapper.strategy == 'FullyMap':
            params += list(self.label_mapper.linear_mapping.parameters())
        # Reinitialize optimizer with only these params
        # Assuming optimizer is passed from outside, but if not, can do:
        # self.optimizer = torch.optim.Adam(params, lr=...)
        # For robustness, do not alter externally created optimizer here.
        pass
