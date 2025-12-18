## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import os
from typing import Optional
import numpy as np

from dataset_loader import DatasetLoader, SlideSample
from model import AttentionMIL, TransMIL, AdditiveMIL
from explanation import Explanation
from utils import plot_heatmaps, compute_metrics
from config import config

class Trainer:
    """
    Handles training, validation, checkpointing, and testing of MIL models.
    Implements early stopping based on validation AUC (or other metrics).
    """
    def __init__(self,
                 model: nn.Module,
                 train_dataset: list,
                 val_dataset: list,
                 test_dataset: list,
                 config_dict: dict):
        """
        Initialize the Trainer.
        Args:
            model (nn.Module): The MIL model to train.
            train_dataset (list): List of SlideSample objects for training.
            val_dataset (list): List of SlideSample objects for validation.
            test_dataset (list): List of SlideSample objects for testing.
            config_dict (dict): Hyperparameters, paths, device info from config.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config_dict
        self.device = torch.device(self.config['hardware'].get('device', 'cuda'))
        self.model.to(self.device)
        
        # Prepare DataLoaders
        self.train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_dataset, shuffle=False)
        self.test_loader = self._create_dataloader(self.test_dataset, shuffle=False)

        # Set optimizer
        self.optimizer = self._create_optimizer(self.model, self.config['training'])
        
        # Loss function: binary cross entropy (for binary labels) or use BCEWithLogitsLoss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Learning rate scheduler (optional)
        # Use ReduceLROnPlateau if desired; here, for simplicity, we skip it.
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.early_stop_patience = 10  # can be set from config
        self.patience_counter = 0
        self.checkpoint_path = self.config['save'].get('model_checkpoint_path', './checkpoints/')
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def _create_dataloader(self, dataset: list, shuffle: bool) -> DataLoader:
        """
        Create DataLoader from dataset of SlideSamples.
        """
        return DataLoader(dataset, batch_size=self.config['training'].get('batch_size', 32), shuffle=shuffle, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        """
        Collate function to handle variable-sized bags if needed.
        For simplicity, assume all bags are processed individually; batch contains lists.
        """
        return batch

    def _create_optimizer(self, model: nn.Module, training_cfg: dict):
        """
        Instantiate optimizer.
        """
        lr = training_cfg.get('learning_rate', 0.001)
        opt_name = training_cfg.get('optimizer', 'Adam').lower()
        if opt_name == 'adam':
            return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        elif opt_name == 'sgd':
            return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def train(self):
        """
        Main training loop with validation, early stopping, and checkpointing.
        """
        max_epochs = self.config['training'].get('epochs', 1000)
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            epoch_losses = []
            train_preds = []
            train_labels = []

            # Loop over training batches
            for batch in self.train_loader:
                features_batch, labels_batch = self._prepare_batch(batch)
                features_batch = features_batch.to(self.device)
                labels_batch = labels_batch.to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(features_batch).squeeze(-1)  # shape: (batch_size,)
                loss = self.criterion(outputs, labels_batch)
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

                # Collect predictions for metrics
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                train_preds.extend(probs)
                train_labels.extend(labels_batch.cpu().numpy())

            train_loss = np.mean(epoch_losses)
            train_auc = roc_auc_score(train_labels, train_preds) if len(train_labels) > 0 else 0.5

            # Validation
            val_metrics = self.validate()
            val_auc = val_metrics.get('AUROC', 0)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_AUC={train_auc:.4f}, val_AUC={val_auc:.4f}")

            # Check for improvement
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stop_patience:
                print("Early stopping triggered.")
                break

        # Load best model checkpoint after training
        self._load_checkpoint()

        # Final evaluation on test set
        test_metrics, test_heatmaps = self.evaluate()
        print(f"Test AUROC: {test_metrics.get('AUROC', 0):.4f}")
        if self.config['evaluation']['metrics'].get('AUPRC2', False):
            print(f"Test AUPRC-2: {test_metrics.get('AUPRC2', 0):.4f}")

        # Save heatmaps if needed
        if self.config['evaluation']['visualization'].get('heatmaps', False):
            self._save_heatmaps(test_heatmaps)

    def _prepare_batch(self, batch):
        """
        Converts batch of SlideSample objects into tensors for features and labels.
        Assumes batch is a list of tuples or objects containing features and labels.
        """
        # For simplicity, assume each sample contains features (tensor) and label
        features_list = []
        labels_list = []
        for sample in batch:
            # sample can be a tuple or object
            if isinstance(sample, tuple) or hasattr(sample, 'features'):
                features = getattr(sample, 'features', None)
                label = getattr(sample, 'label', None)
            else:
                # fallback: assume dict with 'features' and 'label'
                features = sample['features']
                label = sample['label']
            # features shape: (K, D)
            # For batch processing, keep list of features and labels
            features_list.append(torch.tensor(features, dtype=torch.float))
            labels_list.append(label)

        # Stack features to tensor (batch_size, max_patches, feature_dim)
        # For variable-sized bags, might need padding; for simplicity, assume all same size
        features_batch = torch.stack(features_list, dim=0)  # shape: (B, K, D)
        labels_batch = torch.tensor(labels_list, dtype=torch.float)  # shape: (B,)
        return features_batch, labels_batch

    def _save_checkpoint(self, epoch: int):
        """
        Save model state_dict and optimizer.
        """
        save_path = os.path.join(self.checkpoint_path, f"best_model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_auc': self.best_val_auc
        }, save_path)
        print(f"Checkpoint saved at epoch {epoch} to {save_path}.")

    def _load_checkpoint(self):
        """
        Load the best saved checkpoint.
        """
        # Find the checkpoint with best val_auc
        checkpoints = [f for f in os.listdir(self.checkpoint_path) if f.endswith('.pt')]
        if not checkpoints:
            print("No checkpoint found.")
            return
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
        latest_ckpt = checkpoints[0]
        ckpt_path = os.path.join(self.checkpoint_path, latest_ckpt)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with val_auc={checkpoint['val_auc']:.4f}")

    def validate(self):
        """
        Evaluate model on validation set, compute metrics.
        """
        self.model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in self.val_loader:
                features_batch, labels_batch = self._prepare_batch(batch)
                features_batch = features_batch.to(self.device)
                labels_batch = labels_batch.to(self.device).float()
                outputs = self.model(features_batch).squeeze(-1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(labels_batch.cpu().numpy())
        auroc = roc_auc_score(val_labels, val_preds) if len(val_labels) > 0 else 0
        auprc2 = self._compute_auprc2(val_labels, val_preds)
        return {'AUROC': auroc, 'AUPRC2': auprc2}

    def _compute_auprc2(self, true_labels, pred_scores):
        """
        Compute the average of AUPRC for positive and negative evidence detection
        (as per AUPRC-2 measurement).
        """
        # For binary classification, AUPRC for positives and negatives
        auprc_pos = average_precision_score(true_labels, pred_scores)
        # For negatives, invert scores
        auprc_neg = average_precision_score(true_labels, [-s for s in pred_scores])
        return 0.5 * (auprc_pos + auprc_neg)

    def _save_heatmaps(self, heatmaps):
        """
        Save final heatmaps for analysis.
        """
        save_dir = self.config['save'].get('explanation_heatmaps_path', './heatmaps/')
        os.makedirs(save_dir, exist_ok=True)
        for idx, heatmap_img in enumerate(heatmaps):
            heatmap_path = os.path.join(save_dir, f"slide_{idx}_heatmap.png")
            heatmap_img.save(heatmap_path)
        print(f"Heatmaps saved to {save_dir}.")

    def evaluate(self):
        """
        Run inference on test set, compute metrics, generate heatmaps.
        """
        self.model.eval()
        test_preds = []
        test_labels = []
        all_heatmaps = []
        with torch.no_grad():
            for batch in self.test_loader:
                features_batch, labels_batch = self._prepare_batch(batch)
                features_batch = features_batch.to(self.device)
                labels_batch = labels_batch.to(self.device).float()

                outputs = self.model(features_batch).squeeze(-1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                test_preds.extend(probs)
                test_labels.extend(labels_batch.cpu().numpy())

                # Generate explanations/heatmaps for this batch if needed
                # Could be added here: e.g., compute relevance maps and visualize
                # For demonstration, we skip that step in bulk evaluation
        auroc = roc_auc_score(test_labels, test_preds) if len(test_labels) > 0 else 0
        auprc2 = self._compute_auprc2(test_labels, test_preds)
        metrics = {'AUROC': auroc, 'AUPRC2': auprc2}

        # Placeholder: generate heatmaps for samples if desired
        # For simplicity, return empty list
        heatmaps_list = []
        return metrics, all_heatmaps
