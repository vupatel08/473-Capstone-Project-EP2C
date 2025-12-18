## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from copy import deepcopy
from evaluation import evaluate_accuracy, evaluate_AUROC
from utils import set_seed
from model import BackboneNetwork
from modules import PositionalEncoding
from interpretability import Interpretability

class Trainer:
    def __init__(self, dataset, config: dict):
        """
        Initialize the Trainer with dataset and configuration.
        Args:
            dataset (Dataset): Dataset object providing train and validation data loaders.
            config (dict): Parsed configuration dictionary from YAML.
        """
        self.dataset = dataset
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set seed for reproducibility
        seed = self.config.get('training', {}).get('seed', 123)
        set_seed(seed)

        # Initialize model
        self.model = self._initialize_model()
        self.model.to(self.device)

        # Initialize optimizer
        lr = self.config.get('training', {}).get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training parameters
        self.epochs = self.config.get('training', {}).get('epochs', 1500)
        self.batch_size = self.config.get('training', {}).get('batch_size', 16)
        self.early_stopping = self.config.get('training', {}).get('early_stopping', True)

        # Early stopping parameters
        self.patience = 20  # default patience
        self.best_val_loss = np.inf
        self.best_state_dict = None
        self.no_improve_counter = 0

        # Data loaders
        self.train_loader = self._create_dataloader(self.dataset.train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(self.dataset.val_dataset, shuffle=False)

    def _initialize_model(self):
        """
        Instantiate the MILLET model based on config.
        """
        backbone_arch = self.config['model'].get('backbone', 'InceptionTime')
        embedding_dim = self.config['model'].get('embedding_dim', 128)
        dropout_rate = self.config['model'].get('dropout_rate', 0.1)
        architecture_params = self.config['model'].get('architecture_params', {})
        pooling_method = self.config['model'].get('pooling_method', 'Conjunctive')
        pooling_params = self.config['model'].get('pooling_params', {})

        # Instantiate backbone network
        backbone = BackboneNetwork(architecture=backbone_arch,
                                   embedding_dim=embedding_dim,
                                   architecture_params=architecture_params)
        # Instantiate pooling module
        pooling_method_name = self.config['model'].get('pooling_method', 'Conjunctive')
        pooling_params_dict = self.config['model'].get('pooling_params', {})

        # Select and instantiate pooling
        from modules import PositionalEncoding
        pooling_module = None
        if pooling_method_name == 'GAP':
            from modules import GAPPooling
            pooling_module = GAPPooling()
        elif pooling_method_name == 'Attention':
            from modules import AttentionPooling
            attention_heads = pooling_params_dict.get('attention_heads', 1)
            attention_size = pooling_params_dict.get('attention_size', 8)
            pooling_module = AttentionPooling(d=embedding_dim, attention_heads=attention_heads, attention_size=attention_size)
        elif pooling_method_name == 'Instance':
            from modules import InstancePooling
            c = self.dataset.y.max() + 1 if hasattr(self.dataset, 'y') else 10
            pooling_module = InstancePooling(d=embedding_dim, c=c)
        elif pooling_method_name == 'Additive':
            from modules import AdditivePooling
            attention_heads = pooling_params_dict.get('attention_heads', 1)
            attention_size = pooling_params_dict.get('attention_size', 8)
            c = self.dataset.y.max() + 1 if hasattr(self.dataset, 'y') else 10
            pooling_module = AdditivePooling(d=embedding_dim, c=c, attention_heads=attention_heads, attention_size=attention_size)
        elif pooling_method_name == 'Conjunctive':
            from modules import ConjunctivePooling
            attention_heads = pooling_params_dict.get('attention_heads', 1)
            attention_size = pooling_params_dict.get('attention_size', 8)
            c = self.dataset.y.max() + 1 if hasattr(self.dataset, 'y') else 10
            pooling_module = ConjunctivePooling(d=embedding_dim, c=c, attention_heads=attention_heads, attention_size=attention_size)
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method_name}")
        
        # Build full model: backbone + positional encoding + optional dropout + pooling + classifier
        # Since models in model.py are designed to output features, we'll wrap them
        model = nn.Sequential()
        # Backbone
        model_backbone = backbone
        # Add positional encoding if specified
        if self.config.get('model', {}).get('positional_encoding', True):
            pe = PositionalEncoding(max_length=self.dataset.length, d_model=embedding_dim)
            # Wrap backbone to add positional encoding
            class BackboneWithPE(nn.Module):
                def __init__(self, backbone_net, pe_module):
                    super().__init__()
                    self.backbone_net = backbone_net
                    self.pe_module = pe_module
                def forward(self, x):
                    features = self.backbone_net(x)
                    # features shape: (batch, t, d)
                    features_pe = self.pe_module(features)
                    return features_pe
            model_backbone = BackboneWithPE(backbone, pe)
        # Set dropout if specified
        dropout_rate = self.config['model'].get('dropout_rate', 0.1)
        # Compose full model
        class MILModel(nn.Module):
            def __init__(self, backbone, pooling, dropout_rate, num_classes):
                super().__init__()
                self.backbone = backbone
                self.pooling = pooling
                self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
                self.classifier = nn.Linear(self._get_feature_dim(), num_classes)

            def _get_feature_dim(self):
                # Determine feature dimension after pooling
                return embedding_dim

            def forward(self, x):
                # x shape: (batch, 1, t)
                features = self.backbone(x)  # (batch, t, d)
                # add dropout
                features = self.dropout(features)
                # pooling: returns pooled embedding (batch, 1, 1, d) or (batch, 1, d)
                pooled = self.pooling(features)
                pooled = pooled.squeeze(1)  # shape: (batch, d)
                logits = self.classifier(pooled)
                # For interpretability, also output per-time-point predictions based on pooling
                # For simplicity, this example only outputs class logits
                return logits
        num_classes = len(np.unique(self.dataset.y))
        model_instance = MILModel(model_backbone, pooling_module, dropout_rate, num_classes)
        return model_instance

    def _create_dataloader(self, dataset_obj, shuffle=True):
        """
        Create DataLoader from dataset object.
        """
        return DataLoader(dataset_obj, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def run(self):
        """
        Run the training loop with early stopping.
        """
        for epoch in range(1, self.epochs + 1):
            train_loss, train_correct, train_total = 0, 0, 0
            self.model.train()
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)  # shape: (batch, 1, t)
                batch_y = batch_y.to(self.device)  # shape: (batch,)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)  # shape: (batch, num_classes)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                preds = torch.argmax(outputs, dim=1)
                train_correct += (preds == batch_y).sum().item()
                train_total += batch_y.size(0)

            avg_train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validation step
            self.model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for val_X, val_y in self.val_loader:
                    val_X = val_X.to(self.device)
                    val_y = val_y.to(self.device)
                    val_outputs = self.model(val_X)
                    loss_val = self.criterion(val_outputs, val_y)
                    val_loss += loss_val.item() * val_X.size(0)
                    preds_val = torch.argmax(val_outputs, dim=1)
                    val_correct += (preds_val == val_y).sum().item()
                    val_total += val_y.size(0)
            avg_val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            print(f"Epoch {epoch:04d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} "
                  f"| Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Early stopping
            if self.early_stopping:
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.best_state_dict = deepcopy(self.model.state_dict())
                    self.no_improve_counter = 0
                else:
                    self.no_improve_counter += 1
                if self.no_improve_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        # Load best model weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        print("Training completed.")
        return self.model

