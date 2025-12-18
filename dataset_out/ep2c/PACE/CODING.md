# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset.py

```python
## dataset.py
import os
import random
import numpy as np
from typing import Optional, Dict
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

class CustomVisionDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # list of PIL Images or tensors
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        y = self.labels[idx]
        return x, y

class DatasetLoader:
    def __init__(self, config: Dict):
        # Read dataset info from config
        self.dataset_name = config.get('dataset_name', 'VTAB-1K')
        self.train_split_name = config.get('train_split', 'train')
        self.validation_split_name = config.get('validation_split', 'validation')
        self.test_split_name = config.get('test_split', 'test')
        self.seed = config.get('seed', 42)
        self.device = config.get('device', 'cuda:0')
        # Additional dataset params can be added here if needed

        # Fix global seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def load_dataset(self):
        """
        Load dataset based on dataset_name.
        Supports standard datasets via datasets library.
        For custom datasets (VTAB, FGVC, etc.), handle accordingly.
        """
        name = self.dataset_name.lower()
        dataset = None
        if name.startswith('vtab'):
            # For VTAB-1K, use a predefined local or remote implementation
            # Here, we'll assume a mock implementation; in practice, replace with actual API
            dataset = self._load_vtab()
        elif name.startswith('fgvc'):
            dataset = self._load_fgvc()
        elif name.startswith('glue'):
            dataset = self._load_glue()
        elif name.startswith('gsm-8k'):
            dataset = self._load_gsm8k()
        elif 'imagenet' in name:
            dataset = self._load_imagenet()
        elif 'cifar' in name:
            dataset = self._load_cifar()
        else:
            # fallback: attempt to load by datasets library
            dataset = load_dataset(name)
        return dataset

    def _load_vtab(self):
        # Placeholder: define local loading for VTAB-1K datasets
        # Usually, load each dataset separately with fixed splits
        # For demonstration, load VTAB from datasets
        dataset_dict = {}
        try:
            dataset_dict = load_dataset("google/vtab")
            # For each dataset, use fixed splits
            # For simplicity, pick small subset for train/val/test
        except:
            raise NotImplementedError("VTAB datasets loading needs proper implementation.")
        return dataset_dict

    def _load_fgvc(self):
        # Example with datasets library or manual loading
        # For instance: OxfordPets
        dataset = load_dataset('oxford_pets', split=self.train_split_name, cache_dir='./data')
        # Similarly, load validation and test splits
        return dataset

    def _load_glue(self):
        dataset = load_dataset('glue', 'mrpc')  # Example: MRPC task
        return dataset

    def _load_gsm8k(self):
        # For GSM-8K, load from Huggingface datasets
        dataset = load_dataset('gsm8k', split=self.train_split_name)
        return dataset

    def _load_imagenet(self):
        # Typically, load from torchvision or custom dataset
        # For a reproducible pipeline, you may cache dataset or load from local
        # placeholder implementation
        # Return a dummy dataset
        return None

    def _load_cifar(self):
        dataset = load_dataset('cifar10')
        return dataset

    def _get_transforms(self, dataset_name: str, is_training: bool):
        # Define dataset-specific transformations
        if 'imagenet' in dataset_name or 'cifar' in dataset_name:
            # Vision dataset transformations
            if is_training:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            return transform
        elif 'fgvc' in dataset_name:
            # Stronger augmentation (if specified)
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            return transform
        else:
            # For NLP datasets, tokenization handled separately
            # For vision-only, fallback
            return None

    def get_dataset(self, split: str, is_training: bool):
        """
        Load dataset for given split, applying preprocessing
        """
        ds = self.load_dataset()
        # Determine data and labels
        if ds is None:
            raise ValueError("Dataset loading failed.")
        if 'train' in split:
            # For datasets with splits
            subset = ds.get(split, ds) if isinstance(ds, dict) else ds
            data = self._extract_data(subset, is_training)
        elif 'validation' in split:
            subset = ds.get(split, ds) if isinstance(ds, dict) else ds
            data = self._extract_data(subset, is_training)
        elif 'test' in split:
            subset = ds.get(split, ds) if isinstance(ds, dict) else ds
            data = self._extract_data(subset, is_training)
        else:
            raise ValueError(f"Unknown split: {split}")
        transform = self._get_transforms(self.dataset_name, is_training)
        dataset_obj = CustomVisionDataset(data['images'], data['labels'], transform)
        return dataset_obj

    def _extract_data(self, dataset_split, is_training: bool):
        # Extract images and labels from dataset split
        images = []
        labels = []
        if hasattr(dataset_split, 'column_names'):
            for example in dataset_split:
                images.append(example['image'])
                labels.append(example['label'])
        elif isinstance(dataset_split, list):
            for example in dataset_split:
                images.append(example['image'])
                labels.append(example['label'])
        else:
            # fallback, handle as needed
            pass
        return {'images': images, 'labels': labels}
```

## evaluation.py

```python
## evaluation.py
import torch
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np
from datasets import Dataset

class Evaluation:
    """
    Class to evaluate a model on a given dataset split, computing specified metrics.
    Supports optional gradient norm computation for analysis of regularization effects.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        config: Dict,
        device: Optional[str] = 'cuda:0',
        verbose: bool = True
    ):
        """
        Initialize Evaluation instance.
        Args:
            model: The trained PyTorch model instance.
            dataset: Dataset object providing data loader method.
            config: Dict with evaluation settings, including metrics.
            device: String or torch.device for inference.
            verbose: If True, print detailed logs.
        """
        self.model = model
        self.dataset = dataset
        self.device = torch.device(device if isinstance(device, str) else device)
        self.verbose = verbose

        # Extract metrics configuration
        self.metrics = config.get('evaluation', {}).get('metrics', ['accuracy'])
        self.metric_target = config.get('evaluation', {}).get('metric_target', 'validation')
        self.save_best = config.get('evaluation', {}).get('save_best_model', True)
        self.eval_interval = config.get('evaluation', {}).get('evaluation_interval', 10)

        # Metrics storage
        self.best_score = None
        self.best_model_state_dict = None
        self.log_metrics_history = []

        # Set model in evaluation mode
        self.model.eval()
        self.model.to(self.device)

        # Handle optional gradient norm tracking
        self.compute_grad_norms = self._check_need_grad_norm()

        if self.compute_grad_norms:
            self.grad_norms = []

    def _check_need_grad_norm(self) -> bool:
        """
        Decide whether to compute gradient norms based on metrics or config.
        For simplicity, assume always false unless explicitly requested.
        """
        # For this implementation, as per guidelines, compute only if specified.
        # For demonstration, set False; can be extended to make conditional.
        return False

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the dataset split, compute metrics.
        Returns:
            Dictionary with metrics values.
        """
        dataloader = self.dataset.get_dataloader(split=self.metric_target, batch_size=128, shuffle=False)
        all_preds = []
        all_labels = []

        # For gradient norms, manage gradient context
        if self.compute_grad_norms:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)

        total_loss = 0.0
        total_samples = 0
        metrics_results = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs, labels = batch
                inputs = self._to_device(inputs)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model.forward(inputs, perturb_params={'apply_perturb': False})
                # outputs shape expected: for classification, logits (batch x classes)
                # for regression, continuous values, etc.

                # Collect predictions
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())

                # Compute per-batch loss if relevant
                # For classification: cross-entropy
                # For regression: MSE
                # For now, just accumulate for metrics
                # Could implement specific losses if needed in config

        # Concatenate all outputs and labels
        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Compute requested metrics
        for metric in self.metrics:
            if metric.lower() == 'accuracy' or metric.lower() == 'acc':
                acc = self.compute_accuracy(preds, labels)
                metrics_results['accuracy'] = acc
                if self.verbose:
                    print(f"Validation Accuracy: {acc:.4f}")
                # Check for best
                self._update_best(metric_score=acc, metric_name='accuracy')
            elif metric.lower() == 'mse':
                mse = self.compute_mse(preds, labels)
                metrics_results['mse'] = mse
                if self.verbose:
                    print(f"Validation MSE: {mse:.4f}")
            elif metric.lower() == 'correlation':
                corr = self.compute_correlation(preds, labels)
                metrics_results['correlation'] = corr
                if self.verbose:
                    print(f"Validation Corr: {corr:.4f}")
            else:
                if self.verbose:
                    print(f"Unknown metric '{metric}', skipping.")

        # Optional: compute gradient norms
        if self.compute_grad_norms:
            grad_norm = self._compute_model_gradient_norm()
            metrics_results['gradient_norm'] = grad_norm
            if self.verbose:
                print(f"Gradient Norm: {grad_norm:.4f}")

        return metrics_results

    def _to_device(self, inputs: Dict) -> Dict:
        """
        Transfer inputs, assuming inputs is a dict of tensors.
        """
        # For datasets with dict inputs (images, input_ids, etc.)
        device_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                device_inputs[k] = v.to(self.device)
            else:
                device_inputs[k] = v
        return device_inputs

    def compute_accuracy(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute classification accuracy.
        Assumes preds are logits.
        """
        if preds.ndim > 1 and preds.shape[1] > 1:
            pred_labels = torch.argmax(preds, dim=1)
        else:
            # For binary classification or regression, possibly threshold
            pred_labels = (preds > 0.5).long() if preds.ndim > 1 else (preds > 0).long()
        correct = (pred_labels == labels).sum().item()
        total = labels.numel()
        return correct / total

    def compute_mse(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute Mean Squared Error.
        """
        mse = F.mse_loss(preds.squeeze(), labels.float()).item()
        return mse

    def compute_correlation(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute Pearson correlation coefficient.
        """
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        if preds_np.ndim > 1 and preds_np.shape[1] == 1:
            preds_np = preds_np.squeeze()
        if labels_np.ndim > 1 and labels_np.shape[1] == 1:
            labels_np = labels_np.squeeze()
        if len(preds_np) == 0:
            return 0.0
        corr = np.corrcoef(preds_np, labels_np)[0,1]
        return corr

    def _compute_model_gradient_norm(self) -> float:
        """
        Compute L2 norm of gradients across all trainable model parameters.
        """
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_sq += torch.sum(p.grad.data ** 2).item()
        total_norm = np.sqrt(total_norm_sq)
        return total_norm

    def _update_best(self, metric_score: float, metric_name: str):
        """
        If current metric improves on best, save model state.
        """
        if self.best_score is None or metric_score > self.best_score:
            self.best_score = metric_score
            self.best_model_state_dict = self.model.state_dict()

    def _save_checkpoint(self, path: str):
        """
        Save model checkpoint to file.
        """
        torch.save(self.model.state_dict(), path)

```


## main.py

```python
# main.py
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
import logging

from dataset import DatasetLoader
from model import TransformerModel
from utils import set_random_seed, log_metrics
from trainer import Trainer
from evaluation import Evaluation

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    seed = int(config.get('misc', {}).get('seed', 42))
    set_random_seed(seed)

    # Setup device
    device_str = config.get('misc', {}).get('device', 'cuda:0')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # Verbose logging setup
    verbose = config.get('misc', {}).get('verbose_logging', True)
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # 1. Dataset Loading
    dataset_cfg = config.get('dataset', {})
    dataset_loader = DatasetLoader(dataset_cfg)

    print(f"Loading dataset: {dataset_cfg.get('dataset_name', 'Unknown')}")
    # Load train, validation, test datasets
    train_dataset = dataset_loader.get_dataset(split=dataset_cfg.get('train_split', 'train'), is_training=True)
    val_dataset = dataset_loader.get_dataset(split=dataset_cfg.get('validation_split', 'validation'), is_training=False)
    test_dataset = dataset_loader.get_dataset(split=dataset_cfg.get('test_split', 'test'), is_training=False)

    # DataLoader creation
    batch_size = int(config.get('training', {}).get('batch_size', 16))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model Initialization
    model_cfg = config.get('model', {})
    pretrained_name = model_cfg.get('pretrained_model_name', '')
    peft_method = model_cfg.get('peft_method', 'LoRA')
    peft_rank = int(model_cfg.get('peft_rank', 16))
    adapter_scale = float(model_cfg.get('adapter_params', 1.0))
    perturb_sigma = float(model_cfg.get('perturbation_sigma', 0.2))
    adapter_perturbation = bool(model_cfg.get('adapter_perturbation', True))
    output_regulation = bool(model_cfg.get('output_regularization', True))
    
    # Instantiate model
    model = TransformerModel(
        pretrained_model_name=pretrained_name,
        config={
            'peft_method': peft_method,
            'peft_rank': peft_rank,
            'adapter_params': adapter_scale,
            'perturbation_sigma': perturb_sigma,
            'adapter_perturbation': adapter_perturbation,
            'output_regularization': output_regulation
        }
    ).to(device)

    # 3. Optimizer setup
    learning_rate = float(config.get('training', {}).get('learning_rate', 2e-5))
    weight_decay = float(config.get('training', {}).get('weight_decay', 1e-4))
    optimizer_type = config.get('training', {}).get('optimizer', 'AdamW')
    # For simplicity, use AdamW
    params = list(model.get_parameters())  # Only PEFT + head params typically
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # 4. Trainer initialization
    trainer_cfg = {
        'epochs': int(config.get('training', {}).get('epochs', 300)),
        'batch_size': batch_size,
        'lambda_consistency': float(config.get('training', {}).get('lambda_consistency', 0.01)),
        'sigma_noise': float(config.get('training', {}).get('sigma_noise', 0.2)),
        'regularization_type': config.get('training', {}).get('regularization_type', 'standard'),
        'lazy_update_interval': int(config.get('training', {}).get('lazy_update_interval', 10)),
        'use_previous_epoch_outputs': bool(config.get('training', {}).get('use_previous_epoch_outputs', True))
    }
    trainer = Trainer(model, train_loader, optimizer, trainer_cfg, device=device_str)

    # 5. Training Loop
    for epoch in range(1, trainer_cfg['epochs'] + 1):
        print(f"Epoch {epoch}/{trainer_cfg['epochs']}")
        trainer.train_one_epoch()

        # Validation & Model Saving
        if epoch % int(config.get('evaluation', {}).get('evaluation_interval', 10)) == 0:
            val_metrics = trainer.evaluate(split='validation')
            # Check for best model
            val_acc = val_metrics.get('accuracy', 0.0)
            if trainer.best_val_metric is None or val_acc > trainer.best_val_metric:
                trainer.best_val_metric = val_acc
                # Save best model state dict
                trainer._save_best_model()

    # Load the best checkpoint after training
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load('best_model.pth') if os.path.exists('best_model.pth') else model.state_dict())

    # 6. Final Evaluation on Test Set
    print("Evaluating on test dataset...")
    test_metrics = trainer.evaluate(split='test')

    # Log final test metrics
    print(f"Final Test Metrics: {test_metrics}")

def save_best_model(trainer: 'Trainer', path='best_model.pth'):
    torch.save(trainer.model.state_dict(), path)

# This is the main entry point.
if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Union

class PEFTModule(nn.Module):
    """
    Abstract base class for PEFT modules.
    """
    def __init__(self):
        super().__init__()
        self.trainable_params = []

    def get_parameters(self):
        return self.parameters()

    def perturb_features(self, features: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply multiplicative Gaussian noise to features.
        """
        if sigma <= 0:
            return features
        noise = torch.normal(mean=1.0, std=sigma, size=features.shape, device=features.device)
        return features * noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override in subclasses. Processes input features.
        """
        raise NotImplementedError("PEFTModule subclasses must implement forward method.")

class LoRAModule(PEFTModule):
    """
    Implementation of LoRA: low-rank adaptation matrices decomposing weight updates.
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        in_dim = original_layer.in_features
        out_dim = original_layer.out_features

        # LoRA matrices
        self.W_d = nn.Parameter(torch.randn(out_dim, rank))
        self.W_u = nn.Parameter(torch.randn(rank, in_dim))
        # Optional bias
        if hasattr(original_layer, 'bias') and original_layer.bias is not None:
            self.bias = original_layer.bias
        else:
            self.bias = None

        self.trainable_params.extend([self.W_d, self.W_u])

    def forward(self, x: torch.Tensor, perturb: bool = False, sigma: float = 0.0) -> torch.Tensor:
        # Compute the low-rank delta
        delta_W = self.W_d @ self.W_u  # shape (out_dim, in_dim)
        weight = self.original_layer.weight + delta_W
        if perturb and sigma > 0:
            weight = self.perturb_features(weight, sigma)
        # Use the perturbed or original weight
        return nn.functional.linear(x, weight, self.bias)

class AdapterLayer(nn.Module):
    """
    Implementation of a residual adapter module with bottleneck architecture.
    """
    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.trainable_params = [self.down_proj, self.up_proj]

    def forward(self, x: torch.Tensor, perturb: bool = False, sigma: float = 0.0) -> torch.Tensor:
        residual = x
        delta = self.down_proj(x)
        delta = self.activation(delta)
        delta = self.up_proj(delta)
        if perturb and sigma > 0:
            delta = self.perturb_features(delta, sigma)
        return residual + delta

class VPTPrompt(nn.Module):
    """
    Implementation of Visual Prompt Tuning (VPT): learnable prompt tokens.
    """
    def __init__(self, prompt_length: int, embedding_dim: int):
        super().__init__()
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, embedding_dim))
        self.trainable_params = [self.prompt_embeddings]

    def forward(self, input_embeddings: torch.Tensor, perturb: bool = False, sigma: float = 0.0) -> torch.Tensor:
        if perturb and sigma > 0:
            noise = self.perturb_features(self.prompt_embeddings, sigma)
            prompt = self.prompt_embeddings + noise
        else:
            prompt = self.prompt_embeddings
        # Concatenate prompts to input embeddings
        return torch.cat([prompt.unsqueeze(0).expand(input_embeddings.size(0), -1, -1), input_embeddings], dim=1)

    def perturb_features(self, features: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return features
        noise = torch.normal(mean=1.0, std=sigma, size=features.shape, device=features.device)
        return features * noise

class TransformerModel(nn.Module):
    def __init__(self, pretrained_model_name: str, config: Dict):
        """
        Initialize the transformer backbone with PEFT modules.
        Args:
            pretrained_model_name: model identifier in transformers.
            config: configuration dictionary with keys:
                - peft_method: 'LoRA', 'Adapter', 'VPT'
                - peft_rank: int
                - adapter_params: float (scaling factor)
                - perturbation_sigma: float
                - adapter_perturbation: bool
                - output_regularization: bool
        """
        super().__init__()
        # Load the backbone
        self.pretrained_model_name = pretrained_model_name
        self.config = config
        self.peft_method = config.get('peft_method', 'LoRA')
        self.peft_rank = config.get('peft_rank', 16)
        self.adapter_params = config.get('adapter_params', 1.0)
        self.perturbation_sigma = config.get('perturbation_sigma', 0.2)
        self.adapter_perturbation = config.get('adapter_perturbation', True)
        self.output_regularization = config.get('output_regularization', True)

        # Depending on architecture, load model
        self.backbone_config = AutoConfig.from_pretrained(pretrained_model_name)
        self.backbone = AutoModel.from_pretrained(pretrained_model_name, config=self.backbone_config)
        # Check if backbone architecture is vision or language
        # For example, for vision models:
        self.is_vision = hasattr(self.backbone, 'embeddings') or 'vit' in pretrained_model_name.lower()
        self._init_peft_modules()

        # Placeholder for storing features if needed
        self._peft_feature_layers = []

    def _init_peft_modules(self):
        """
        Initialize the PEFT modules depending on the method.
        For vision transformers, insert adapters or LoRA in attention/MLP layers.
        For NLP models, similarly modify linear layers or attention.
        """
        self.peft_modules = nn.ModuleList()

        # For demonstration, assume we modify all linear layers in self.backbone
        # In practice, target specific layers such as query, key, value, MLP
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear):
                if self.peft_method == 'LoRA':
                    # Replace with LoRA module
                    lo_ra = LoRAModule(module, rank=self.peft_rank)
                    setattr(self.backbone, name, lo_ra)
                    self.peft_modules.append(lo_ra)
                elif self.peft_method == 'Adapter':
                    adapter = AdapterLayer(hidden_dim=module.out_features, bottleneck_dim=int(self.adapter_params * module.out_features))
                    setattr(self.backbone, name, adapter)
                    self.peft_modules.append(adapter)
                elif self.peft_method == 'VPT':
                    # For VPT, consider prompt tokens, handled elsewhere
                    pass
        # Additionally, for VPT, initialize prompt tokens if needed
        if 'VPT' in self.peft_method:
            # Example prompt length and embedding dims
            self.prompt_length = int(self.adapter_params * 10)  # arbitrary ratio
            self.embedding_dim = self.backbone.config.hidden_size
            self.vpt_prompt = VPTPrompt(prompt_length=self.prompt_length, embedding_dim=self.embedding_dim)

    def get_peft_module(self) -> nn.Module:
        """
        Return the list of PEFT modules for external access.
        """
        return self.peft_modules

    def perturb_features(self, features: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply multiplicative Gaussian noise to features if perturbation is enabled.
        """
        if not self.training or sigma <= 0:
            return features
        noise = torch.normal(mean=1.0, std=sigma, size=features.shape, device=features.device)
        return features * noise

    def extract_adapter_features(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward specific parts to get features passing through PEFT modules.
        For vision models, extract features after patch embedding or after adapter.
        For NLP, extract hidden states after PEFT modules.
        """
        # Forward input through backbone up to the PEFT parts
        x = input
        # Example: For vision, after patch embedding
        # For NLP, after embedding layer
        # To generalize, hook or override the forward method in subclasses
        # Here, assume backbone returns features as last hidden states
        # Return the features passing through PEFT modules
        # For simplicity, assume it's the output before final classification
        # In practice, hook into specific layers
        features = None
        # Forward with hook: we can implement hooks or modify the backbone
        # For demonstration:
        features = self.backbone(**x).last_hidden_state  # assumes dict input or tensor for vision
        return features

    def forward(self, inputs: Dict, perturb_params: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass with optional perturbation.
        Args:
            inputs: dict containing required inputs (images, token ids, etc.)
            perturb_params: optional dict, e.g.,
                - 'apply_perturb': True/False
                - 'sigma': float
                - 'perturb_features': True/False
        """
        apply_perturb = False
        sigma = 0.0
        perturb_features_flag = False
        if perturb_params:
            apply_perturb = perturb_params.get('apply_perturb', False)
            sigma = perturb_params.get('sigma', 0.0)
            perturb_features_flag = self.adapter_perturbation and apply_perturb

        # Forward input through the backbone
        if self.is_vision:
            # For vision, inputs could be images
            outputs = self.backbone(**inputs)
            features = outputs.last_hidden_state  # shape: batch x seq x hidden_dim
            # For classification, typically pooled or CLS token
            pooled_output = features[:, 0, :]  # assuming first token as CLS
        else:
            # For NLP, inputs could be tokenized dict
            outputs = self.backbone(**inputs, output_hidden_states=True)
            features = outputs.last_hidden_state
            pooled_output = features[:, 0, :]  # CLS token

        # Perturb features if required
        if perturb_features_flag:
            features = self.perturb_features(features, sigma)

        # Additional optional VPT prompt addition
        if self.peft_method == 'VPT' and hasattr(self, 'vpt_prompt'):
            # Assuming input embeddings are accessible
            # For vision, embedding is often initial patch embedding
            # For NLP, it's token embeddings
            # Here, assume we can get the input embeddings
            # For simplicity, only apply to NLP
            if 'input_ids' in inputs:
                input_ids = inputs['input_ids']
                embedding_layer = self.backbone.get_input_embeddings()
                input_embeddings = embedding_layer(input_ids)
                prompt_embeddings = self.vpt_prompt(input_embeddings, perturb=apply_perturb, sigma=sigma)
                # Re-encode with prompt concatenated
                # Note: For real implementation, need to pass these embeddings directly to the model
                return self.backbone(inputs_embeds=prompt_embeddings, attention_mask=inputs.get('attention_mask', None))
            # For vision, prompts may not be applied in this manner
            # placeholder handling
        # Forward the features through remaining of the model
        # For final classification head:
        # Assume the backbone has classifier or head attribute
        logits = self._classification_head(pooled_output)
        return logits

    def _classification_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        Simple linear layer head for classification/regression
        """
        # For example, a linear layer with number of classes
        # For demonstration, replace with actual head if available
        # Unless specified, define a dummy linear classifier
        if not hasattr(self, '_head'):
            # For MNIST/CIFAR, set num_classes
            self.num_classes = 1000  # placeholder, update as needed
            self._head = nn.Linear(features.shape[-1], self.num_classes).to(features.device)
        return self._head(features)

    def get_parameters(self):
        """
        Return trainable parameters (PEFT modules + output head)
        """
        params = []
        for module in self.peft_modules:
            params.extend(list(module.parameters()))
        # Add classification head
        params.extend(list(self._classification_head.parameters()))
        return params

    def save(self, save_path: str):
        """
        Save model and PEFT modules
        """
        torch.save(self.state_dict(), save_path)

    def load(self, load_path: str):
        """
        Load model state
        """
        self.load_state_dict(torch.load(load_path))
```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List
from utils import generate_gaussian_noise, compute_gradient_norm, log_metrics, set_random_seed

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset,
        optimizer,
        config: Dict,
        device: Optional[str] = 'cuda:0'
    ):
        """
        Initialize the Trainer with model, dataset, optimizer, and hyperparameters.
        Args:
            model: The transformer model with PEFT modules.
            dataset: Dataset object providing dataloaders for train, val, test.
            optimizer: Optimizer instance (AdamW, etc.).
            config: Dict containing hyperparameters and flags.
            device: Device string, e.g., 'cuda:0'.
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = device
        self.config = config

        # Extract training hyperparameters from config
        self.epochs = config.get('training', {}).get('epochs', 300)
        self.batch_size = config.get('training', {}).get('batch_size', 16)
        self.learning_rate = config.get('training', {}).get('learning_rate', 2e-5)
        self.lambda_consistency = config.get('training', {}).get('lambda_consistency', 0.01)
        self.sigma_noise = config.get('training', {}).get('sigma_noise', 0.2)
        self.use_prev_epoch_outputs = config.get('training', {}).get('use_previous_epoch_outputs', True)
        self.lazy_update_interval = config.get('training', {}).get('lazy_update_interval', 10)
        self.regularization_type = config.get('training', {}).get('regularization_type', 'standard')  # 'standard', 'lazy', 'fast'
        self.device = device

        # Set seed for reproducibility
        seed = self.config.get('misc', {}).get('seed', 42)
        set_random_seed(seed)

        # Initialize container for previous outputs (for lazy variant)
        self.prev_outputs_buffer = None  # will be Dict[split, List[Tensor]]
        self._initialize_prev_outputs()

        # Store gradient norms over training
        self.grad_norms = []

        # Additional setup
        self._initialize_lr_scheduler()

        # For evaluation
        self.best_val_metric = None
        self.best_model_state_dict = None
        self.device = device

    def _initialize_prev_outputs(self):
        """Prepare storage for previous epoch outputs if lazy method is used."""
        if self.use_prev_epoch_outputs and self.config['training'].get('regularization_type') == 'lazy':
            # Initialize buffer for validation and train set
            self.prev_outputs_buffer = {
                'train': None,
                'validation': None
            }
        else:
            self.prev_outputs_buffer = None

    def _initialize_lr_scheduler(self):
        """Set up learning rate scheduler if needed."""
        # Optional: Implement scheduler if desired
        pass

    def train(self):
        """Main training loop."""
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_samples = 0
            gradient_norms = []

            dataloader = self.dataset.get_dataloader(split='train', batch_size=self.batch_size, shuffle=True)
            for batch_idx, batch in enumerate(dataloader):
                inputs, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Prepare perturbation parameters
                apply_perturb = False
                sigma = 0.0
                if self.config['training']['regularization_type'] in ['standard', 'fast']:
                    apply_perturb = True
                    sigma = self.config['training'].get('sigma_noise', 0.2)
                elif self.config['training']['regularization_type'] == 'lazy':
                    # For lazy, update disturbance based on epoch
                    apply_perturb = True
                    sigma = self.config['training'].get('sigma_noise', 0.2)

                # Extract PEFT modules
                peft_modules = self.model.get_peft_module()

                # Compute adapter features before perturbation for all samples in batch
                adapter_features = self.model.extract_adapter_features(inputs)
                # shape: (batch_size, feature_dim)

                # Generate noisy adapter features
                if apply_perturb:
                    # For standard, generate independent noise samples for z1, z2
                    if self.config['training'].get('regularization_type', 'standard') in ['standard', 'fast']:
                        z1 = generate_gaussian_noise(adapter_features.shape, sigma).to(self.device)
                        z2 = generate_gaussian_noise(adapter_features.shape, sigma).to(self.device)
                    elif self.config['training']['regularization_type'] == 'lazy':
                        # For lazy: reuse previous epoch outputs if available
                        pass
                else:
                    z1 = z2 = torch.ones_like(adapter_features)

                # Define function to run model with perturbed adapter features
                def run_model_with_perturb(z: torch.Tensor):
                    """
                    Run model forward with adapter features multiplied elementwise by z.
                    """
                    # Save original peft modules parameters for perturbation
                    # Assumption: model architecture supports perturbation
                    # Apply perturbation to modules
                    # For simplicity: assume perturb applies internally during forward
                    # Else, we implement perturbation here
                    # For this implementation, we'll pass the perturbation params
                    perturb_params = {
                        'apply_perturb': True,
                        'sigma': sigma,
                        'perturb_features': True
                    }

                    # Override the model's perturb method for this forward
                    # For demonstration, assume model.forward accepts perturb_params
                    return self.model.forward(inputs, perturb_params=perturb_params)

                # Run two perturbed forward passes
                with torch.no_grad():
                    output_z1 = run_model_with_perturb(z1)
                with torch.no_grad():
                    output_z2 = run_model_with_perturb(z2)

                # Alternatively, for 'lazy' regularization, compare current output with stored previous outputs
                if self.config['training'].get('regularization_type') == 'lazy' and self.use_prev_epoch_outputs:
                    if self.prev_outputs_buffer['train'] is not None:
                        # Compute consistency loss with previous output
                        prev_output = self.prev_outputs_buffer['train']
                        consistency_loss = F.mse_loss(output_z1, prev_output)
                        # Use only one perturbation for lazy
                    else:
                        # Fallback: use difference between current perturbed outputs
                        consistency_loss = F.mse_loss(output_z1, output_z2)
                else:
                    # Standard PACE: compute consistency between two perturbed outputs
                    consistency_loss = F.mse_loss(output_z1, output_z2)

                # Compute main task loss
                task_logits = output_z1  # could be logits, ensure matching shape
                # For classification
                main_loss = F.cross_entropy(task_logits, labels)

                # Total loss
                total_batch_loss = main_loss + self.lambda_consistency * consistency_loss

                # Backpropagate
                total_batch_loss.backward()

                # Optional: gradient clipping for stability
                grad_norm = compute_gradient_norm(self.model)
                gradient_norms.append(grad_norm)
                max_grad_norm = self.config.get('misc', {}).get('clip_grad_norm', None)
                if max_grad_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                # Optimizer step
                self.optimizer.step()

                total_loss += total_batch_loss.item() * labels.size(0)
                total_samples += labels.size(0)

            # End of epoch: compute average gradient norm
            if len(gradient_norms) > 0:
                epoch_grad_norm = np.mean(gradient_norms)
            else:
                epoch_grad_norm = 0
            self.grad_norms.append(epoch_grad_norm)

            # Lazy update: store current epoch outputs if needed
            if self.config['training']['regularization_type'] == 'lazy' and self.use_prev_epoch_outputs:
                # Save current model outputs for next epoch
                self._save_epoch_outputs()

            # Logging
            avg_loss = total_loss / total_samples
            log_metrics(
                {
                    'epoch': epoch,
                    'loss': avg_loss,
                    'grad_norm': epoch_grad_norm
                },
                step=epoch,
                log_file=self.config.get('misc', {}).get('log_path', None)
            )

            # Evaluate periodically
            if epoch % self.config['evaluation'].get('evaluation_interval', 10) == 0:
                self.evaluate(split='validation')

        # After training, load best model if saved
        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)

    def _save_epoch_outputs(self):
        """
        Save model outputs for the validation/train set for lazy consistency.
        """
        self.model.eval()
        for split in ['validation', 'train']:
            dataloader = self.dataset.get_dataloader(split=split, batch_size=self.batch_size, shuffle=False)
            all_outputs = []
            with torch.no_grad():
                for batch in dataloader:
                    inputs, labels = batch
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model.forward(inputs, perturb_params={'apply_perturb': False})
                    all_outputs.append(outputs.detach().cpu())
            # Store concatenated outputs
            self.prev_outputs_buffer[split] = torch.cat(all_outputs, dim=0)

    def evaluate(self, split='validation'):
        """
        Run evaluation on a dataset split.
        """
        self.model.eval()
        dataloader = self.dataset.get_dataloader(split=split, batch_size=self.batch_size, shuffle=False)
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # Forward without perturbation for evaluation
                logits = self.model.forward(inputs, perturb_params={'apply_perturb': False})
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        # Calculate metrics (accuracy, etc.)
        outputs = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        accuracy = self._compute_accuracy(outputs, labels)
        # Save best model if improve
        if self.config['evaluation'].get('save_best_model', True):
            if self.best_val_metric is None or accuracy > self.best_val_metric:
                self.best_val_metric = accuracy
                self.best_model_state_dict = self.model.state_dict()
        # Log evaluation metrics
        log_metrics({'validation_accuracy': accuracy}, step='eval')
        return accuracy

    def _compute_accuracy(self, outputs, labels):
        preds = torch.argmax(outputs, dim=1)
        return (preds == labels).float().mean().item()

```

## utils.py

```python
## utils.py
import torch
import numpy as np
import logging
from typing import Tuple, Dict, Optional

def generate_gaussian_noise(shape: Tuple[int, ...], sigma: float, device: Optional[str] = None) -> torch.Tensor:
    """
    Generate multiplicative Gaussian noise with mean 1 and standard deviation sigma.
    Args:
        shape: Shape of the tensor to generate.
        sigma: Standard deviation of the Gaussian noise.
        device: Destination device ('cpu' or 'cuda'). Defaults to 'cpu' if None.
    Returns:
        torch.Tensor of shape 'shape' with entries sampled from N(1, sigma^2).
    """
    if sigma <= 0:
        return torch.ones(shape, device=device if device else 'cpu')
    device = device if device else 'cpu'
    noise = torch.normal(mean=1.0, std=sigma, size=shape, device=device)
    return noise

def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute the total L2 norm of gradients for all parameters in the model.
    Args:
        model: The torch.nn.Module with gradients computed (after backward()).
    Returns:
        float: The L2 norm of all gradients.
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += torch.sum(p.grad.data ** 2).item()
    total_norm = np.sqrt(total_norm_sq)
    return total_norm

def get_lambda(epoch: int, schedule_type: str='fixed', base_lambda: float=0.01, max_lambda: float=1.0, total_epochs: int=300) -> float:
    """
    Get the regularization coefficient lambda based on schedule.
    Args:
        epoch: Current epoch number.
        schedule_type: Type of schedule ('fixed', 'linear', 'exponential').
        base_lambda: Base lambda value.
        max_lambda: Maximum lambda for schedules if applicable.
        total_epochs: Total number of epochs (used for schedule computations).
    Returns:
        float: Current lambda value.
    """
    if schedule_type == 'fixed':
        return base_lambda
    elif schedule_type == 'linear':
        # Increase linearly from 0 to max_lambda
        return min(base_lambda + (max_lambda - base_lambda) * epoch / total_epochs, max_lambda)
    elif schedule_type == 'exponential':
        # Exponential schedule
        return base_lambda * (max_lambda / base_lambda) ** (epoch / total_epochs)
    else:
        # Default to fixed
        return base_lambda

def get_sigma(epoch: int, schedule_type: str='fixed', base_sigma: float=0.2, min_sigma: float=0.05, total_epochs: int=300) -> float:
    """
    Get the noise sigma for perturbation based on schedule.
    Args:
        epoch: Current epoch number.
        schedule_type: ('fixed', 'linear', 'anneal')
        base_sigma: The initial or max sigma.
        min_sigma: Minimum sigma if decreasing.
        total_epochs: For schedule computation.
    Returns:
        float: Sigma for current epoch.
    """
    if schedule_type == 'fixed':
        return base_sigma
    elif schedule_type == 'linear':
        # Decrease sigma linearly from base_sigma to min_sigma
        sigma_val = max(base_sigma - (base_sigma - min_sigma) * epoch / total_epochs, min_sigma)
        return sigma_val
    elif schedule_type == 'anneal':
        # Alternatively, exponential decay
        decay_rate = 0.95
        sigma_val = max(base_sigma * (decay_rate ** epoch), min_sigma)
        return sigma_val
    else:
        return base_sigma

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str):
    """
    Save model and optimizer checkpoint.
    Args:
        model: model state to save.
        optimizer: optimizer state.
        epoch: current epoch.
        path: file path for saving.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)

def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]=None):
    """
    Load model and optimizer state from checkpoint.
    Args:
        path: checkpoint file path.
        model: model to load state into.
        optimizer: optional optimizer to load state.
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', None)

def log_metrics(metrics: Dict[str, float], step: int, log_file: Optional[str]=None):
    """
    Log metrics to console and optionally to file.
    Args:
        metrics: dictionary of metric name to value.
        step: current training step or epoch.
        log_file: optional path to save logs.
    """
    message = f"Step/Epoch {step}: " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def set_random_seed(seed: int):
    """
    Set seed for reproducibility across torch, numpy, and cuda.
    Args:
        seed: seed integer.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\PACE\PACE_repo`
