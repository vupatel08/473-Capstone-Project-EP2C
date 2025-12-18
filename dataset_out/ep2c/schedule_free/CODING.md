# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset.py

```python
## dataset.py
import os
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class DatasetLoader:
    """
    Handles loading datasets, applying transformations, normalization, and creating data loaders.
    """
    def __init__(
        self,
        dataset_name: str = "CIFAR10",
        data_path: str = "./data",
        batch_size: int = 32,
        train: bool = True,
        normalize_mean: Optional[list] = None,
        normalize_std: Optional[list] = None,
        num_workers: int = 4,
        image_size: int = 32
    ):
        """
        Initialize DatasetLoader with dataset parameters.
        Args:
            dataset_name (str): Name of dataset ('CIFAR10', 'CIFAR100', 'SVHN', 'ImageNet').
            data_path (str): Path to dataset directory.
            batch_size (int): Batch size for data loader.
            train (bool): If True, load training data, else validation/test.
            normalize_mean (list): Mean for normalization.
            normalize_std (list): Std for normalization.
            num_workers (int): Number of worker threads for data loading.
            image_size (int): Target image size (used for resize, crop).
        """
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.train = train
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.num_workers = num_workers
        self.image_size = image_size

        # Set default normalization if not provided
        if self.normalize_mean is None or self.normalize_std is None:
            if self.dataset_name in ["CIFAR10", "CIFAR100"]:
                self.normalize_mean = [0.4914, 0.4822, 0.4465]
                self.normalize_std = [0.2023, 0.1994, 0.2010]
            elif self.dataset_name == "SVHN":
                self.normalize_mean = [0.4377, 0.4438, 0.4728]
                self.normalize_std = [0.1980, 0.2010, 0.1970]
            elif self.dataset_name == "ImageNet":
                self.normalize_mean = [0.485, 0.456, 0.406]
                self.normalize_std = [0.229, 0.224, 0.225]
            else:
                # Default to CIFAR10 stats
                self.normalize_mean = [0.4914, 0.4822, 0.4465]
                self.normalize_std = [0.2023, 0.1994, 0.2010]

        # Initialize dataset attribute
        self.dataset = None

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load dataset and create DataLoader objects for train and validation/test sets.
        Returns:
            tuple of (train_loader, val_loader)
        """
        if self.train:
            dataset = self._get_dataset(train=True)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return dataloader
        else:
            dataset = self._get_dataset(train=False)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return dataloader

    def _get_dataset(self, train: bool):
        """
        Instantiate dataset object based on dataset_name and train flag.
        """
        transform = self._build_transform(train)
        if self.dataset_name == "CIFAR10":
            dataset = datasets.CIFAR10(
                root=self.data_path,
                train=train,
                download=True,
                transform=transform
            )
        elif self.dataset_name == "CIFAR100":
            dataset = datasets.CIFAR100(
                root=self.data_path,
                train=train,
                download=True,
                transform=transform
            )
        elif self.dataset_name == "SVHN":
            split = "train" if train else "test"
            dataset = datasets.SVHN(
                root=self.data_path,
                split=split,
                download=True,
                transform=transform
            )
        elif self.dataset_name == "ImageNet":
            # Load ImageNet via ImageFolder with proper directory structure
            split_folder = "train" if train else "val"
            dataset = datasets.ImageFolder(
                root=os.path.join(self.data_path, split_folder),
                transform=transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return dataset

    def _build_transform(self, train: bool):
        """
        Build data transformation pipeline for training or validation.
        """
        transform_list = []
        if train:
            # Data augmentation for training
            # Reflection padding of 4, random crop 32x32
            transform_list.append(transforms.RandomCrop(self.image_size, padding=4, pad_if_needed=True))
            transform_list.append(transforms.RandomHorizontalFlip())
        else:
            # For validation, resize or center crop
            # If using ImageNet, apply resize + center crop
            if self.dataset_name == "ImageNet":
                transform_list.append(transforms.Resize(256))
                transform_list.append(transforms.CenterCrop(self.image_size))
            else:
                # For CIFAR and SVHN, no resize needed
                pass
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        # Normalize
        transform_list.append(transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std))
        return transforms.Compose(transform_list)
```

## evaluation.py

```python
## evaluation.py
import torch
from typing import Dict

class Evaluation:
    """
    Handles model evaluation on a dataset, computes specified metrics.
    
    Attributes:
        model (torch.nn.Module): The trained model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for validation/test data.
        device (torch.device): Device to run inference on.
        metrics (list of str): Metrics to compute, e.g., ['accuracy', 'loss'].
        loss_fn (callable): Loss function used for 'loss' metric.
        correct_sum (int): Accumulated number of correct predictions.
        total_samples (int): Total number of samples evaluated.
        loss_sum (float): Accumulated loss (if 'loss' in metrics).
    """
    def __init__(
        self,
        model,
        data_loader,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        metrics: list = ['accuracy']
    ):
        """
        Initialize Evaluation instance.
        Args:
            model (torch.nn.Module): The trained model.
            data_loader (DataLoader): DataLoader for evaluation dataset.
            device (torch.device): Device for inference.
            metrics (list): List of metrics to compute ('accuracy', 'loss', ...).
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.metrics = metrics
        # Use CrossEntropyLoss if 'loss' is requested
        if 'loss' in self.metrics:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = None
        # Initialize accumulators
        self.reset_metrics()
    
    def reset_metrics(self) -> None:
        """
        Reset all accumulator variables before evaluation.
        """
        self.correct_sum = 0
        self.total_samples = 0
        self.loss_sum = 0.0

    def evaluate(self) -> Dict[str, float]:
        """
        Run model on dataset, compute metrics, and return results.
        Returns:
            dict: {'accuracy': float, 'loss': float} depending on metrics
        """
        self.model.eval()
        with torch.no_grad():
            for batch in self.data_loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)

                # Compute accuracy if needed
                if 'accuracy' in self.metrics:
                    preds = torch.argmax(outputs, dim=1)
                    self.correct_sum += (preds == labels).sum().item()

                # Compute loss if needed
                if 'loss' in self.metrics:
                    loss = self.loss_fn(outputs, labels)
                    self.loss_sum += loss.item() * inputs.size(0)

                self.total_samples += inputs.size(0)

        results = {}
        if 'accuracy' in self.metrics:
            accuracy = self.correct_sum / self.total_samples if self.total_samples > 0 else 0.0
            results['accuracy'] = accuracy
        if 'loss' in self.metrics:
            mean_loss = self.loss_sum / self.total_samples if self.total_samples > 0 else 0.0
            results['loss'] = mean_loss
        return results
```

## main.py

```python
## main.py
import os
import time
import logging
import yaml
import torch
import numpy as np
from utils import set_seeds, get_device, estimate_bounds, save_checkpoint, plot_training_curve
from dataset import DatasetLoader
from model import Model
from optimizer import ScheduleFreeOptimizer
from evaluation import Evaluation

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set seeds for reproducibility
    seed = config.get('training', {}).get('seed', 42)
    set_seeds(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Load dataset
    dataset_cfg = config.get('dataset', {})
    dataset_name = dataset_cfg.get('name', 'CIFAR10')
    data_path = dataset_cfg.get('data_path', './data')
    batch_size = dataset_cfg.get('batch_size', 32)
    normalize_mean = dataset_cfg.get('normalize', {}).get('mean', None)
    normalize_std = dataset_cfg.get('normalize', {}).get('std', None)

    # Initialize data loaders
    train_loader_obj = DatasetLoader(
        dataset_name=dataset_name,
        data_path=data_path,
        batch_size=batch_size,
        train=True,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std
    )
    val_loader_obj = DatasetLoader(
        dataset_name=dataset_name,
        data_path=data_path,
        batch_size=batch_size,
        train=False,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std
    )
    train_loader = train_loader_obj.load_data()
    val_loader = val_loader_obj.load_data()

    # Initialize model
    model_cfg = config.get('model', {})
    architecture = model_cfg.get('architecture', 'ResNet50')
    hyperparams = {k: v for k, v in model_cfg.items() if k != 'architecture'}
    model = Model(architecture, hyperparams).to(device)
    model.initialize_weights()

    # Estimate bounds D and G
    # For simplicity, use first batch to estimate G and initial parameter norm for D
    G_estimate = utils.estimate_G(model, train_loader, device)
    D_estimate = utils.estimate_D(model, train_loader, device)

    # Set large fixed learning rate based on D/G ratio
    large_lr_flag = config['training'].get('large_learning_rate', True)
    base_lr = config['training'].get('learning_rate', 0.0025)
    if large_lr_flag:
        gamma = D_estimate / G_estimate
    else:
        gamma = base_lr  # fallback

    # Hyperparameters for optimizer
    # Use beta from config, default 0.9
    beta = config['training'].get('beta', 0.9)
    weight_decay = config['training'].get('weight_decay', 1e-4)
    # Instantiate the Schedule-Free optimizer
    optimizer = ScheduleFreeOptimizer(
        model_params=list(model.parameters()),
        optimizer_type=config['optimizer'].get('type', 'AdamW'),
        lr_scale=1.0,  # will set eta as gamma
        beta=beta,
        D=D_estimate,
        G=G_estimate,
        eta=gamma,
        weight_decay=weight_decay
    )

    # Training settings
    num_epochs = config['training'].get('epochs', 100)
    log_interval = config.get('logging', {}).get('log_interval', 50)
    checkpoint_dir = config.get('logging', {}).get('checkpoint_dir', './checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Loss function (can be modified as per task)
    criterion = torch.nn.CrossEntropyLoss()

    total_steps = 0
    metrics_history = {'loss': [], 'accuracy': [], 'lr': []}
    start_time = time.time()

    # Training loop
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        total_samples = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = utils.normalize_input(inputs).to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backprop
            loss.backward()

            # Update model parameters via Schedule-Free optimizer
            optimizer.step(gradient_eval_fn=lambda y_params, inp=inputs, lbls=labels: _compute_grads(model, y_params, criterion, inp, lbls))

            # Metrics
            batch_loss = loss.item()
            epoch_loss += batch_loss * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct = (preds == labels).sum().item()
            epoch_correct += correct
            total_samples += inputs.size(0)

            total_steps += 1

            if total_steps % log_interval == 0:
                avg_loss = epoch_loss / total_samples
                accuracy = epoch_correct / total_samples
                current_lr = optimizer.eta
                logging.info(f"Epoch {epoch} Step {total_steps}: Loss={avg_loss:.4f} "
                             f"Acc={accuracy:.4f} LR={current_lr:.6f}")
                metrics_history['loss'].append(avg_loss)
                metrics_history['accuracy'].append(accuracy)
                metrics_history['lr'].append(current_lr)
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{total_steps}.pt')
                save_checkpoint(model, optimizer.optimizer, checkpoint_path)

        # Epoch end metrics
        epoch_loss_avg = epoch_loss / total_samples
        epoch_acc = epoch_correct / total_samples
        logging.info(f"Epoch {epoch} complete: Loss={epoch_loss_avg:.4f} Accuracy={epoch_acc:.4f}")

    total_time = time.time() - start_time
    logging.info(f"Training finished in {total_time/60:.2f} minutes.")

    # Save final x_T parameters
    x_T_params = optimizer.get_current_x_params()
    # Load into model
    for p_model, p_x in zip(model.parameters(), x_T_params):
        p_model.data.copy_(p_x)
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_x_T.pt')
    torch.save(model.state_dict(), final_checkpoint_path)
    logging.info(f"Final model saved to {final_checkpoint_path}")

    # Final evaluation on validation set
    val_evaluator = Evaluation(model, val_loader, device=device, metrics=['accuracy', 'loss'])
    val_metrics = val_evaluator.evaluate()
    logging.info(f"Validation metrics: {val_metrics}")

    # Plot training curves
    plot_training_curve(metrics_history, save_path=os.path.join(checkpoint_dir, 'training_curves.png'))

def _compute_grads(model, y_params, criterion, inputs, labels):
    """
    Compute gradients at evaluation point y_params for the current batch.
    y_params: list of tensors representing the current evaluation parameters.
    """
    # Assign y_params to model
    for p, y_p in zip(model.parameters(), y_params):
        p.data.copy_(y_p)
    model.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.clone())
        else:
            grads.append(torch.zeros_like(p))
    return grads

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torchvision.models as models
import math

class Model(nn.Module):
    """
    This class encapsulates various neural network architectures used in experiments,
    including ResNet50, WideResNet, DenseNet, and others.
    It provides a unified interface for initialization, forward pass, and weight setup.
    """
    def __init__(self, model_class: str = "ResNet50", hyperparams: dict = None):
        """
        Initialize the model based on model_class and hyperparameters.
        Args:
            model_class (str): Type of architecture ('ResNet50', 'WideResNet', 'DenseNet', etc.)
            hyperparams (dict): Architecture-specific hyperparameters, e.g.,
                For WideResNet: {'depth': 16, 'width_multiplier': 8, 'dropout': 0.3}
                For DenseNet: {'growth_rate': 12, 'block_config': [6,12,24,16]}
        """
        super().__init__()
        self.model_class = model_class
        self.hyperparams = hyperparams if hyperparams is not None else {}

        # Select and instantiate architecture
        self.model = self.select_architecture()

        # Initialize weights explicitly to ensure reproducibility
        self.initialize_weights()

    def select_architecture(self) -> nn.Module:
        """
        Instantiate the neural network architecture based on self.model_class.
        Returns:
            nn.Module: the constructed model
        """
        if self.model_class == "ResNet50":
            # Standard ResNet50 from torchvision
            model = models.resnet50(pretrained=False)
        elif self.model_class == "WideResNet":
            # Build WideResNet with specified depth and width
            depth = self.hyperparams.get("depth", 16)
            widen_factor = self.hyperparams.get("width_multiplier",8)
            dropout = self.hyperparams.get("dropout", 0.3)
            model = self.build_wideresnet(depth, widen_factor, dropout)
        elif self.model_class == "DenseNet":
            # Use torchvision DenseNet121 as default; customize if needed
            growth_rate = self.hyperparams.get("growth_rate", 12)
            block_config = self.hyperparams.get("block_config", [6,12,24,16])
            model = models.densenet121(pretrained=False)
            # For customization, you can replace classifier or features accordingly
            # But for simplicity, we'll use the default as placeholder
        elif self.model_class == "ResNet":
            # General ResNet, e.g., for other depths if needed
            # default to ResNet50 for now
            model = models.resnet50(pretrained=False)
        else:
            raise ValueError(f"Unrecognized model class: {self.model_class}")
        return model

    def build_wideresnet(self, depth: int, widen_factor: int, dropout: float) -> nn.Module:
        """
        Build a WideResNet architecture.
        Args:
            depth (int): depth of the network, typically 16, 28, etc.
            widen_factor (int): width multiplier
            dropout (float): dropout rate
        Returns:
            nn.Module: WideResNet model
        """
        # Implementation of WideResNet based on typical structure
        # For simplicity, using a custom implementation
        # Note: in practice, replace with a well-tested implementation
        from models.wideresnet import WideResNet
        return WideResNet(depth=depth, widen_factor=widen_factor, dropout=dropout)

    def initialize_weights(self) -> None:
        """
        Initialize the network weights with He initialization for conv and linear layers.
        """
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output logits
        """
        return self.model(x)

    def get_model(self) -> nn.Module:
        """
        Return underlying model for compatibility with optimizer and evaluation.
        Returns:
            nn.Module: the network model
        """
        return self.model
```


## optimizer.py

```python
## optimizer.py
import torch
from torch.optim import Optimizer
from typing import List, Dict, Optional

class ScheduleFreeOptimizer(Optimizer):
    """
    Implements the Schedule-Free optimization strategy as described in the paper.
    Maintains sequences z_t (the optimizer trajectory) and x_t (interpolated average),
    updating z_t via a base optimizer (e.g., AdamW or SGD) at each step,
    and updating x_t with decreasing weights (c_t ~ 1/t).
    
    Attributes:
        params: Iterable of model parameters (must be references to model parameters).
        optimizer: The inner optimizer (e.g., AdamW) used for z_t updates.
        beta: Coupling parameter between x_t and z_t, typically around 0.9.
        D: Estimated initial distance bound (used to set large learning rate).
        G: Gradient norm bound (used to set large learning rate).
        eta: Fixed learning rate (approximate D / G), set during init.
        iteration: Keeps track of current iteration step.
        z_params: List of tensors representing z_t parameters.
        x_params: List of tensors representing x_t parameters.
        buffers for optimizer state if needed (AdamW's m_t and v_t).
    """

    def __init__(
        self,
        model_params: List[torch.nn.Parameter],
        optimizer_type: str = "AdamW",
        lr_scale: float = 1.0,  # user-defined scale for eta, to be set as D/G
        beta: float = 0.9,
        D: float = 1.0,
        G: float = 1.0,
        eta: Optional[float] = None,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        betas: tuple = (0.9, 0.999),
        max_grad_norm: Optional[float] = None,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize the ScheduleFreeOptimizer.
        Args:
            model_params (List[torch.nn.Parameter]): model parameters to optimize.
            optimizer_type (str): type of base optimizer ("AdamW" or "SGD").
            lr_scale (float): optional scale; default 1.0, usually set as D / G.
            beta (float): coupling parameter, e.g., 0.9.
            D (float): distance bound, estimate of initial parameter distance.
            G (float): gradient norm bound.
            eta (float): fixed large learning rate; if None, set to D / G.
            weight_decay (float): weight decay coefficient.
            eps (float): epsilon for AdamW.
            betas (tuple): betas for AdamW.
            max_grad_norm (float): optional, for gradient clipping.
            device (torch.device): device to run computations.
        """
        # Store hyperparameters
        self.params = model_params
        self.beta = beta
        self.D = D
        self.G = G
        # Set fixed learning rate
        self.eta = eta if eta is not None else D / G if G > 1e-8 else D
        self.iteration = 1  # start from 1 for 1-based indexing
        self.device = device

        # Initialize z_t as clone of model parameters
        self.z_params = [p.clone().detach().to(device) for p in model_params]
        # Initialize x_t as clone of model parameters (start same as initial)
        self.x_params = [p.clone().detach().to(device) for p in model_params]
        # Initialize optimizer for z_t
        if optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self._parameters_to_optimizer_params(),
                lr=self.eta,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
        elif optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(
                self._parameters_to_optimizer_params(),
                lr=self.eta,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        # Initialize optimizer's internal state
        self.optimizer.zero_grad()

    def _parameters_to_optimizer_params(self) -> List[torch.nn.Parameter]:
        """
        Convert z_params list to a list suitable for optimizer.
        """
        return self.z_params

    def step(self, data=None, target=None, gradient_eval_fn=None):
        """
        Perform a single inner update step:
        - Compute gradient at y_t
        - Update z_t according to optimizer
        - Update x_t as weighted average with c_t
        - Update y_t
        Args:
            data, target: optional, for gradient computation
            gradient_eval_fn: optional function to compute gradients, if external
        """
        # 1. Compute y_t
        y_t = []
        for x_p, z_p in zip(self.x_params, self.z_params):
            y_t.append((1.0 - self.beta) * x_p + self.beta * z_p)
        # 2. Evaluate gradients at y_t
        # User provides gradient_eval_fn, or we assume external gradient computation
        if gradient_eval_fn is None:
            raise RuntimeError("gradient_eval_fn must be provided to perform gradient evaluation.")
        # The eval fn should set gradients on model parameters, or return grads
        grads = gradient_eval_fn(y_t, data, target)

        # 3. Update z_t using optimizer step with grads at y_t
        # Assign computed grads to z_params
        for p, g in zip(self.z_params, grads):
            p.grad = g
        # Step optimizer to update z_t
        self.optimizer.step()
        # Save z_t parameters after update
        # Copy current z_t as clone
        for idx, p in enumerate(self.z_params):
            self.z_params[idx] = p.clone().detach()

        # 4. Update x_t following the decreasing weight schedule c_t
        c_t = 1.0 / self.iteration  # c_t ~ 1/t
        for idx, p in enumerate(self.x_params):
            # x_{t+1} = (1 - c_t) x_t + c_t z_{t+1}
            self.x_params[idx] = (1 - c_t) * p + c_t * self.z_params[idx]

        # Increment iteration count
        self.iteration += 1

        # The optimizer's internal state has been updated; no need to return
        return

    def get_current_x_params(self) -> List[torch.nn.Parameter]:
        """
        Return the current interpolated parameters x_t.
        """
        return self.x_params

    def load_state_dict(self, state_dict: Dict):
        """
        Load state dict for optimizer and internal z,x buffers.
        """
        self.optimizer.load_state_dict(state_dict['optimizer'])
        # Load z_params
        z_state = state_dict.get('z_params', None)
        if z_state is not None:
            self.z_params = [p.clone().detach() for p in z_state]
        # Load x_params
        x_state = state_dict.get('x_params', None)
        if x_state is not None:
            self.x_params = [p.clone().detach() for p in x_state]
        self.iteration = state_dict.get('iteration', self.iteration)

    def state_dict(self) -> Dict:
        """
        Return the state dict including optimizer state, z_params, x_params, iteration.
        """
        return {
            'optimizer': self.optimizer.state_dict(),
            'z_params': self.z_params,
            'x_params': self.x_params,
            'iteration': self.iteration
        }

    def zero_grad(self):
        """
        Zero out gradients of z_params (not X parameters).
        """
        self.optimizer.zero_grad()

    def clip_gradients(self, max_norm: float):
        """
        Optional gradient clipping for z_t parameters.
        """
        torch.nn.utils.clip_grad_norm_(self.z_params, max_norm)

    def to(self, device: torch.device):
        """
        Move optimizer's internal tensors to device as needed.
        """
        self.device = device
        for idx, p in enumerate(self.z_params):
            self.z_params[idx] = p.to(device)
        for idx, p in enumerate(self.x_params):
            self.x_params[idx] = p.to(device)
        self.optimizer.to(device)

```

## train.py

```python
## train.py
import torch
import time
from utils import set_seeds, save_checkpoint, plot_training_curve, normalize_input
from dataset import DatasetLoader
from model import Model
from optimizer import ScheduleFreeOptimizer

class TrainLoop:
    def __init__(self, config: dict):
        # Set random seed for reproducibility
        seed = config.get('training', {}).get('seed', 42)
        set_seeds(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load dataset
        dataset_cfg = config.get('dataset', {})
        dataset_name = dataset_cfg.get('name', 'CIFAR10')
        data_path = dataset_cfg.get('data_path', './data')
        batch_size = dataset_cfg.get('batch_size', 32)
        self.train_loader = DatasetLoader(
            dataset_name=dataset_name,
            data_path=data_path,
            batch_size=batch_size,
            train=True
        ).load_data()
        self.val_loader = DatasetLoader(
            dataset_name=dataset_name,
            data_path=data_path,
            batch_size=batch_size,
            train=False
        ).load_data()

        # Initialize model
        model_cfg = config.get('model', {})
        architecture = model_cfg.get('architecture', 'ResNet50')
        hyperparams = {k: v for k, v in model_cfg.items() if k != 'architecture'}
        self.model = Model(architecture, hyperparams).get_model().to(self.device)

        # Hyperparameters
        training_cfg = config.get('training', {})
        self.num_epochs = training_cfg.get('epochs', 100)
        self.batch_size = training_cfg.get('batch_size', 32)
        warmup_steps = training_cfg.get('warmup_steps', 4000)
        large_lr_flag = training_cfg.get('large_learning_rate', True)
        initial_lr = training_cfg.get('learning_rate', 0.0025)  # default fixed large lr

        # Estimate bounds D and G
        # Here, we set D and G as per prior knowledge or estimation.
        D = training_cfg.get('initial_D', 1.0)  # Placeholder, can estimate
        G = training_cfg.get('G_estimate', 1.0)  # Placeholder, can estimate from data

        # Compute fixed large learning rate based on D/G ratios
        if large_lr_flag:
            self.gamma = D / G  # constr. from theory D/G; fallback to manual if needed
        else:
            self.gamma = initial_lr

        # Hyperparameters for optimizer
        beta = training_cfg.get('beta', 0.9)
        weight_decay = training_cfg.get('weight_decay', 1e-4)
        # Initialize optimizer
        self.optimizer = ScheduleFreeOptimizer(
            model_params=list(self.model.parameters()),
            optimizer_type=training_cfg.get('optimizer', 'AdamW'),
            lr_scale=1.0,  # scale will be set as self.gamma
            beta=beta,
            D=D,
            G=G,
            eta=self.gamma,
            weight_decay=weight_decay
        )

        # Training tracking
        self.global_step = 0
        self.log_interval = config.get('logging', {}).get('log_interval', 50)  # steps
        self.checkpoint_dir = config.get('logging', {}).get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.metrics_history = {'loss': [], 'accuracy': [], 'lr': []}
        self.loss_fn = torch.nn.CrossEntropyLoss()  # default loss, modify if needed

    def train(self):
        total_steps = 0
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            epoch_correct = 0
            total_samples = 0
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs = normalize_input(inputs).to(self.device)
                labels = labels.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()

                # Step the Schedule-Free optimizer
                self.optimizer.step(gradient_eval_fn=self._compute_gradients_at_y)

                # Metrics
                batch_loss = loss.item()
                epoch_loss += batch_loss * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct = (preds == labels).sum().item()
                epoch_correct += correct
                total_samples += inputs.size(0)

                self.global_step += 1
                total_steps += 1

                # Logging
                if self.global_step % self.log_interval == 0:
                    avg_loss = epoch_loss / total_samples
                    accuracy = epoch_correct / total_samples
                    print(f"Epoch {epoch} Step {self.global_step}: Loss={avg_loss:.4f} "
                          f"Accuracy={accuracy:.4f} LR={self.optimizer.eta:.6f}")
                    self.metrics_history['loss'].append(avg_loss)
                    self.metrics_history['accuracy'].append(accuracy)
                    self.metrics_history['lr'].append(self.optimizer.eta)
                    # Save checkpoint
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{self.global_step}.pt')
                    save_checkpoint(self.model, self.optimizer.optimizer, checkpoint_path)

            # Epoch end metrics
            epoch_loss_avg = epoch_loss / total_samples
            epoch_accuracy = epoch_correct / total_samples
            print(f"Epoch {epoch} completed: Loss={epoch_loss_avg:.4f} "
                  f"Accuracy={epoch_accuracy:.4f}")

        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.2f} minutes.")

        # Final evaluation on validation set using the last interpolation x_T
        final_metrics = self._evaluate(self.val_loader)
        print(f"Validation results: {final_metrics}")

        # Save final model parameters (x_T)
        # Retrieve x_T parameters
        x_T_params = self.optimizer.get_current_x_params()
        self._load_params_into_model(self.model, x_T_params)
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'final_x_T.pt'))

        return self.metrics_history, final_metrics

    def _compute_gradients_at_y(self, y_params, inputs=None, labels=None):
        """
        Compute gradients at evaluation point y_t.
        y_params: list of tensors representing the evaluation point.
        """
        # Assign y_t to model parameters
        for p, y_p in zip(self.model.parameters(), y_params):
            p.data.copy_(y_p)
        # Zero gradients
        self.optimizer.optimizer.zero_grad()
        # Forward pass
        outputs = self.model(inputs) if inputs is not None else None
        # If no inputs/labels provided, need to pass data outside
        # but for simplicity, assume inputs/labels provided
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        # Collect gradients
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.clone())
            else:
                grads.append(torch.zeros_like(p))
        return grads

    def _load_params_into_model(self, model, params_list):
        """
        Assign parameters in params_list into model in-place.
        """
        for p, new_p in zip(model.parameters(), params_list):
            p.data.copy_(new_p)

# Example of instantiating and running training
if __name__ == '__main__':
    import yaml
    import os
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    trainer = TrainLoop(config)
    metrics_history, final_metrics = trainer.train()
    # Optionally plot training curves
    plot_training_curve(metrics_history)
```

## utils.py

```python
## utils.py
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility across torch, numpy, and random.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # For reproducibility in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """
    Return the available device: CUDA if available, else CPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def send_to_device(tensor: torch.Tensor) -> torch.Tensor:
    """
    Send a tensor to the appropriate device.
    """
    device = get_device()
    return tensor.to(device)

def estimate_bounds(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """
    Estimate initial bounds D and G from model weights and gradients.
    D: Frobenius norm of initial parameters (or reference point).
    G: Max gradient norm over a batch.
    """
    model = model.to(device)
    model.eval()
    # Initialize D as the norm of model parameters
    param_vector = []
    for param in model.parameters():
        param_vector.append(param.detach().cpu().view(-1))
    param_vec = torch.cat(param_vector)
    D = torch.norm(param_vec).item()

    # Use a single batch for gradient norm estimation
    for batch in dataloader:
        # Assume batch is a tuple: (inputs, labels)
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = compute_loss(outputs, labels)
        loss.backward()
        # Compute the gradient norms
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach())
        grad_norms = [torch.norm(g) for g in grads]
        G = max([gn.item() for gn in grad_norms]) if grads else 0.0
        break  # only one batch needed
    return D, G

def compute_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Placeholder loss function: uses CrossEntropyLoss by default.
    """
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, targets)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str) -> None:
    """
    Save model and optimizer state dictionaries to a file.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str) -> None:
    """
    Load model and optimizer state dictionaries from a file.
    """
    checkpoint = torch.load(filename, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def plot_training_curve(metrics: dict, save_path: str = None) -> None:
    """
    Plot training metrics over epochs or steps.
    metrics: dict with keys as {'loss', 'accuracy', ...} and values as lists or arrays.
    save_path: if provided, save plot to this path.
    """
    plt.figure(figsize=(8, 6))
    for metric_name, metric_values in metrics.items():
        plt.plot(metric_values, label=metric_name)
    plt.xlabel('Epochs/Steps')
    plt.ylabel('Metric')
    plt.title('Training Metrics')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def normalize_input(image: torch.Tensor, mean: list = [0.4914, 0.4822, 0.4465], std: list = [0.2023, 0.1994, 0.2010]) -> torch.Tensor:
    """
    Normalize input image tensor using dataset-specific mean and std.
    """
    device = get_device()
    mean_tensor = torch.tensor(mean, device=device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(-1, 1, 1)
    return (image - mean_tensor) / std_tensor

def estimate_gradient_norm(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Perform a forward and backward pass on a batch to estimate max gradient norm for G.
    """
    model = model.to(device)
    model.eval()
    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = compute_loss(outputs, labels)
        loss.backward()
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach())
        max_grad_norm = max([torch.norm(g).item() for g in grads]) if grads else 0.0
        return max_grad_norm
    return 0.0

def estimate_initial_parameter_distance(model: torch.nn.Module, reference: torch.nn.Module = None) -> float:
    """
    Compute the Euclidean norm of the initial model parameters relative to a reference point.
    If reference is None, use zero vector (or initial weights).
    """
    params = list(model.parameters())
    if reference is None:
        ref_params = [torch.zeros_like(p) for p in params]
    else:
        ref_params = list(reference.parameters())
    distance_vector = []
    for p, rp in zip(params, ref_params):
        distance_vector.append((p - rp).detach().cpu().view(-1))
    total_distance = torch.norm(torch.cat(distance_vector))
    return total_distance.item()

def prepare_device_and_seed(seed: int = 42):
    """
    Utility to set seed and get device once, for consistent setup.
    """
    set_seeds(seed)
    device = get_device()
    return device
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\schedule_free\schedule_free_repo`
