# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## bn_layers.py

# bn_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialBN(nn.Module):
    """
    Spatial Batch Normalization layer adapted for rate-based backpropagation.
    This BN normalizes across spatial dimensions per time step, with learnable parameters.
    
    Usage:
        - For single-step mode (rate_S), normalizing over spatial batch for each timestep.
        - During training, computes batch-wise mean and variance; during inference, uses stored parameters.
    """
    def __init__(self, num_features, epsilon=1e-5, affine=True, momentum=0.1):
        """
        Initializes the SpatialBN layer.
        Args:
            num_features (int): Number of input channels.
            epsilon (float): Small constant for numerical stability.
            affine (bool): If True, includes learnable affine parameters.
            momentum (float): Momentum for running stats (not used here, replaced with standard batch norm dynamics).
        """
        super(SpatialBN, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        # Register buffers for running mean and var for inference
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.training_mode = True  # control mode externally

    def forward(self, I_t):
        """
        Forward pass for spatial BN.
        Args:
            I_t (Tensor): Input tensor of shape [B, C, H, W]
        Returns:
            Tensor: Normalized tensor, shape same as input.
        """
        if self.training_mode:
            # Compute per batch mean and var over spatial dimensions
            mu = I_t.mean(dim=[0, 2, 3], keepdim=True)
            var = I_t.var(dim=[0, 2, 3], unbiased=False, keepdim=True)
            # Update running estimates (simulate online updates)
            self.running_mean = (1 - 0.1) * self.running_mean + 0.1 * mu.squeeze()
            self.running_var = (1 - 0.1) * self.running_var + 0.1 * var.squeeze()
        else:
            mu = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)

        # Normalize
        I_norm = (I_t - mu) / torch.sqrt(var + self.epsilon)
        if self.affine:
            # Apply learnable affine transformation
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
            I_norm = gamma * I_norm + beta
        return I_norm

    def set_training(self, mode=True):
        """
        Set training mode for BN.
        Args:
            mode (bool): True for training, False for inference.
        """
        self.training_mode = mode

    def backward(self, grad_output):
        """
        Backward pass for BN.
        Args:
            grad_output (Tensor): Gradient of loss w.r.t. output, shape same as input.
        Returns:
            grad_input (Tensor): Gradient of loss w.r.t. input.
        """
        # Manual backward to match the update rules and for clarity.
        # For the implementation, better to rely on autograd, but here we do explicit for transparency.
        # We provide gradient w.r.t. I_t, gamma, beta for parameter update.
        # Note: For simplicity, during training, batch stats are used, so autograd handles gradients.
        pass  # Implementation of backward is optional; rely on autograd for simplicity.

class TemporalBN(nn.Module):
    """
    Temporal Batch Normalization layer adapted for multi-step (rate_M) mode.
    This BN normalizes over entire sequence (time dimension + batch) for each feature.
    
    Usage:
        - During training, computes global mean and variance over all time steps and batch.
        - During inference, uses stored parameters for normalization.
    """
    def __init__(self, num_features, epsilon=1e-5, affine=True, momentum=0.1):
        """
        Initializes the TemporalBN layer.
        Args:
            num_features (int): Number of input channels.
            epsilon (float): Small constant for numerical stability.
            affine (bool): Enables learnable scale and shift.
            momentum (float): For running statistics.
        """
        super(TemporalBN, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        # Running global mean and var over entire sequences
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.training_mode = True

    def forward(self, I_seq):
        """
        Forward pass for temporal BN.
        Args:
            I_seq (Tensor): Input tensor of shape [B, T, C, H, W]
        Returns:
            Tensor: Normalized input, shape same as input.
        """
        if self.training_mode:
            # Compute mean and variance over batch and time (sequence)
            mu = I_seq.mean(dim=[0, 1, 3, 4], keepdim=True)
            var = I_seq.var(dim=[0, 1, 3, 4], unbiased=False, keepdim=True)
            # Update running estimates
            self.running_mean = (1 - 0.1) * self.running_mean + 0.1 * mu.squeeze()
            self.running_var = (1 - 0.1) * self.running_var + 0.1 * var.squeeze()
        else:
            mu = self.running_mean.view(1, 1, -1, 1, 1)
            var = self.running_var.view(1, 1, -1, 1, 1)
        # Normalize each sequence
        I_norm = (I_seq - mu) / torch.sqrt(var + self.epsilon)
        if self.affine:
            gamma = self.gamma.view(1, 1, -1, 1, 1)
            beta = self.beta.view(1, 1, -1, 1, 1)
            I_norm = gamma * I_norm + beta
        return I_norm

    def set_training(self, mode=True):
        """
        Set whether BN operates in training or inference mode.
        Args:
            mode (bool): True for training, False for inference.
        """
        self.training_mode = mode

    def backward(self, grad_output):
        """
        Backward pass for BN.
        Args:
            grad_output (Tensor): Gradient of loss w.r.t. output.
        Returns:
            grad_input (Tensor): Gradient of loss w.r.t. input.
        """
        # As with SpatialBN, rely on autograd for gradient calculations.
        pass  # Custom backward implementation optional. Rely on autograd.

# Note:
# - In practical training, we would set the models in training mode (set_training(True))
#   or eval mode (set_training(False)) depending on the phase.
# - For explicit gradient calculations (e.g., for stability bounds), additional code
#   can be added to extract and manipulate intermediate variables as needed.
# - For cross-mode switching, instantiate either SpatialBN or TemporalBN according to config.
# - The implementation simplifies the typical BN forward/backward, focusing on the core logic
#   suitable for rate code approximation and training. During actual backprop, rely on autograd.
```

## dataset_loader.py

```python
## dataset_loader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import h5py

# Utility for custom pre-processing, e.g., CIFAR10-DVS
def load_cifar10_dvs_data(path, split='train'):
    """
    Load CIFAR10-DVS data from provided HDF5 or numpy files.
    Args:
        path (str): Path to preprocessed CIFAR10-DVS dataset.
        split (str): 'train' or 'test'.
    Returns:
        images (np.ndarray): Array of shape [num_samples, T, C, H, W]
        labels (np.ndarray): Labels array.
    """
    # For illustration: assume preprocessed data stored as HDF5 with datasets 'images', 'labels'
    filename = os.path.join(path, f'cifar10_dvs_{split}.h5')
    with h5py.File(filename, 'r') as f:
        images = np.array(f['images'])  # shape: [num_samples, T, H, W, C] or similar
        labels = np.array(f['labels'])
    # Convert to [num_samples, T, C, H, W]
    # Depending on storage format; adapt accordingly
    if images.shape[-1] != 3:
        raise ValueError("Expected last dimension to be channels=3")
    # Permute if needed
    images = np.transpose(images, (0, 1, 4, 2, 3))  # [N, T, C, H, W]
    return images, labels

def encode_pixel_to_spike(pixel_value, T, method='bernoulli'):
    """
    Convert pixel intensity (0-1) to spike train over T timesteps.
    Args:
        pixel_value (float): normalized pixel value [0,1]
        T (int): number of timesteps
        method (str): 'bernoulli' or 'poisson'
    Returns:
        spikes (np.ndarray): binary array shape [T], 0/1
    """
    if method == 'bernoulli':
        return np.random.rand(T) < pixel_value
    elif method == 'poisson':
        return np.random.poisson(pixel_value, size=T)
    else:
        raise ValueError("Unsupported encoding method.")

class EncodedDataset(Dataset):
    """
    Dataset that provides input as spike sequences obtained from images or raw data.
    """
    def __init__(self, data, labels, T=4, encoding_method='bernoulli', mode='static'):
        """
        Args:
            data (np.ndarray): images or raw data, shape depends on dataset
            labels (np.ndarray): labels
            T (int): number of time steps
            encoding_method (str): 'bernoulli' or 'poisson'
            mode (str): 'static' or 'dynamic' (for datasets like CIFAR10-DVS)
        """
        self.data = data
        self.labels = labels
        self.T = T
        self.mode = mode
        self.encoding_method = encoding_method

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.labels[idx]
        # For static datasets like CIFAR/ImageNet:
        image = self.data[idx]
        # Normalize pixel to [0,1]
        if isinstance(image, np.ndarray):
            img_norm = image.astype(np.float32) / 255.0
        elif isinstance(image, torch.Tensor):
            img_norm = image.float() / 255.0
        else:
            # For PIL Image
            img_norm = np.array(image).astype(np.float32) / 255.0
        # Encode to spikes
        input_seq = np.zeros((self.T, *img_norm.shape))
        for t in range(self.T):
            if self.mode == 'static':
                # For static images, sample spike train for each pixel
                for c in range(img_norm.shape[0]):
                    input_seq[t, c] = encode_pixel_to_spike(img_norm[c], self.T, self.encoding_method)
            elif self.mode == 'dynamic':
                # Placeholder: for datasets like CIFAR10-DVS, data is already spike sequences
                pass  # For dynamic, load directly
        # Convert to torch tensor
        input_tensor = torch.from_numpy(input_seq).float()  # shape: [T, C, H, W]
        label_tensor = torch.tensor(label).long()
        return input_tensor, label_tensor

class CIFAR10_DVS_Dataset(Dataset):
    """
    Dataset handler for CIFAR10-DVS preprocessed sequences.
    Assumes data stored as h5 files with 'images' and 'labels'.
    """
    def __init__(self, data_path, split='train', T=4):
        self.images, self.labels = load_cifar10_dvs_data(data_path, split)
        self.T = T

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_seq = self.images[idx]  # shape: [T, C, H, W]
        label = self.labels[idx]
        input_tensor = torch.from_numpy(img_seq).float()
        label_tensor = torch.tensor(label).long()
        return input_tensor, label_tensor

class DatasetLoader:
    """
    Main class to load datasets, apply transformations, and provide DataLoaders.
    """
    def __init__(self, dataset_name='CIFAR-10', batch_size=128, T=4,
                 input_encoding='direct', augmentation=None,
                 normalization_mean=None, normalization_std=None,
                 train_split_ratio=0.8, dataset_path=None,
                 num_workers=4, mode='static'):
        """
        Args:
            dataset_name (str): 'CIFAR-10', 'CIFAR-100', 'ImageNet', 'CIFAR10-DVS'
            batch_size (int)
            T (int): sequence length for spike encoding
            input_encoding (str): 'direct', 'rate', etc.
            augmentation (callable or None): Data augmentation transforms
            normalization_mean (list): normalization mean per channel
            normalization_std (list): std per channel
            train_split_ratio (float): for datasets needing split
            dataset_path (str): path for storing/loading datasets (especially DVS)
            num_workers (int): DataLoader workers
            mode (str): 'static' for CIFAR/ImageNet, 'dynamic' for CIFAR10-DVS
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.T = T
        self.input_encoding = input_encoding
        self.augmentation = augmentation
        self.norm_mean = normalization_mean
        self.norm_std = normalization_std
        self.train_split_ratio = train_split_ratio
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.mode = mode

    def load_data(self):
        if self.dataset_name.lower() == 'cifar-10':
            # Load CIFAR-10
            train_transform, test_transform = self._get_cifar_transforms()
            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        elif self.dataset_name.lower() == 'cifar-100':
            train_transform, test_transform = self._get_cifar_transforms()
            train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        elif self.dataset_name.lower() == 'imagenet':
            # Placeholder: user should prepare ImageNet dataset in appropriate structure
            train_transform, test_transform = self._get_imagenet_transforms()
            train_dataset = datasets.ImageFolder(root='./imagenet/train', transform=train_transform)
            test_dataset = datasets.ImageFolder(root='./imagenet/val', transform=test_transform)
        elif self.dataset_name.lower() == 'cifar10-dvs':
            # Load preprocessed DVS data
            train_dataset = CIFAR10_DVS_Dataset(dataset_path, split='train', T=self.T)
            test_dataset = CIFAR10_DVS_Dataset(dataset_path, split='test', T=self.T)
        else:
            raise ValueError(f'Unsupported dataset: {self.dataset_name}')

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=True)
        return train_loader, test_loader

    def _get_cifar_transforms(self):
        """
        Compose training/test transforms for CIFAR datasets.
        """
        transform_list = []
        if self.augmentation:
            # Apply augmentation during training
            transform_list.append(self.augmentation)
        transform_list.append(transforms.ToTensor())  # convert to tensor
        if self.norm_mean and self.norm_std:
            transform_list.append(transforms.Normalize(self.norm_mean, self.norm_std))
        return transforms.Compose(transform_list), transforms.Compose(transform_list)

    def _get_imagenet_transforms(self):
        """
        Compose transforms for ImageNet.
        """
        # Training transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # Validation transforms
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return train_transform, test_transform
```

## evaluation.py

```python
## evaluation.py
import torch
import numpy as np
import os
import yaml
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ResNet, VGG, SEW_ResNet
from dataset_loader import DatasetLoader
from surrogate_gradients import sigmoid
from utils import load_config, initialize_logger
import torch.nn.functional as F

class RateBasedEvaluation:
    """
    Evaluates a trained spiking neural network model on the test dataset,
    computing accuracy and spike rate statistics, following the rate-based
    inference paradigm described in the paper.
    """

    def __init__(self, config: dict, device: torch.device):
        """
        Initialize evaluation with configuration.
        Args:
            config: Configuration dictionary.
            device: Torch device ('cuda' or 'cpu').
        """
        self.config = config
        self.device = device
        self.model = self._build_model()
        self.checkpoint_path = self._get_checkpoint_path()
        self._load_checkpoint()
        self.model.eval()
        self.T = self._get_evaluation_timesteps()
        self.dataset_name = self.config['dataset']['name']
        self.batch_size = self.config['training']['batch_size']
        self.test_loader = self._load_test_dataset()
        self.spike_rate_stats = {}

    def _get_checkpoint_path(self):
        """
        Determines checkpoint path from configuration.
        """
        ckpt_dir = self.config.get('checkpoint_dir', './checkpoints')
        checkpoint_filename = self.config.get('checkpoint_file', 'best_model.pt')
        checkpoint_path = os.path.join(ckpt_dir, checkpoint_filename)
        return checkpoint_path

    def _build_model(self):
        """
        Instantiate model architecture as per configuration.
        """
        arch = self.config['model']['architecture'].lower()
        if arch == 'resnet18':
            model = ResNet(architecture='resnet18', config=self.config, training_mode='rate_M')
        elif arch == 'vgg11':
            # assuming 10 classes for CIFAR, modify for other datasets as needed
            model = VGG('VGG11', num_classes=10)
        elif arch == 'sew-resnet34':
            model = SEW_ResNet('sew-resnet34', config=self.config, training_mode='rate_M')
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        return model.to(self.device)

    def _load_checkpoint(self):
        """
        Load pretrained weights from checkpoint.
        """
        assert os.path.exists(self.checkpoint_path), f"Checkpoint not found: {self.checkpoint_path}"
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def _get_evaluation_timesteps(self):
        """
        Get T used during evaluation based on config; fallback to default
        """
        return self.config['training'].get('T', 4)

    def _load_test_dataset(self):
        """
        Load test dataset with appropriate transforms.
        """
        dataset_name = self.dataset_name.lower()
        batch_size = self.batch_size
        # Compose transforms: normalization
        norm_mean = self.config['dataset'].get('normalization_mean', [0.4914, 0.4822, 0.4465])
        norm_std = self.config['dataset'].get('normalization_std', [0.2023, 0.1994, 0.2010])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

        if dataset_name == 'cifar-10':
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'cifar-100':
            test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'imagenet':
            # Assuming images stored in 'imagenet/val'
            test_dataset = datasets.ImageFolder(root='./imagenet/val', transform=transform)
        elif dataset_name == 'cifar10-dvs':
            # Load preprocessed DVS data (assumed to be prepared as tensors)
            # Placeholder: user should implement appropriate loader
            from dataset_loader import CIFAR10_DVS_Dataset
            test_dataset = CIFAR10_DVS_Dataset('./data_cifar10_dvs', split='test', T=self.T)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def evaluate(self):
        """
        Run inference over the test dataset, compute accuracy and spike rate statistics.
        """
        total_samples = 0
        correct_predictions = 0
        # For per-layer spike rate stats
        layer_spike_accum = {}
        layer_neurons_count = {}
        for batch_idx, (inputs, labels) in enumerate(self.test_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            batch_size = inputs.shape[0]
            # Forward pass with rate-based inference
            outputs, layer_rates = self._inference_forward(inputs)
            # Compute accuracy
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            correct_predictions += correct
            total_samples += batch_size

            # Collect spike rates for statistics
            self._accumulate_spike_rates(layer_rates, batch_size)

        accuracy = correct_predictions / total_samples
        # Finalize spike rate stats
        self._compute_layer_rate_stats(total_samples)

        # Log results
        print(f"Evaluation Accuracy (Top-1): {accuracy*100:.2f}%")
        self._log_spike_rate_stats()

    def _inference_forward(self, inputs):
        """
        Run the model with inputs, collect per-layer spike mean rates.
        """
        # Depending on mode, use either the network's forward or emulate rate calcs
        # For simplicity, use model's forward. The model should output logits.
        with torch.no_grad():
            output_logits = self.model(inputs, mode='rate_S', T=self.T)
            # Compute probability predictions
            probs = F.softmax(output_logits, dim=1)
            predictions = probs
            # To calculate firing rates, we need the spike activity per layer.
            # Since the model doesn't return intermediate states, assume:
            # 1. The model setup allows access to internal spike counts.
            # 2. Or, we directly compute the firing rate based on the spike sequences if stored.
        # Placeholder: Assume we have a method to get spike counts per layer
        # For estimation, generate dummy firing rates: in reality, get from custom model forward
        layer_rate_estimates = self._collect_layer_spike_rates(inputs)
        return predictions, layer_rate_estimates

    def _collect_layer_spike_rates(self, inputs):
        """
        Placeholder: In actual implementation, this should access internal spike counts,
        or, for static datasets, compute rates directly from stored spike sequences,
        or, if only rate approximation is used, generate or calculate estimates here.
        """
        # For illustration, assume the model provides a method `get_layer_spike_rates()`
        # which returns dict: {layer_name: tensor of shape [batch, neurons]}
        # Here, we simulate by generating random rates as placeholder
        layer_rates = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'neuron1'):
                # Simulate firing rate: 
                # In actual code, replace with real spike counts.
                # For now, set to averaged over the batch
                # For real application, implement hooks or update the model to store spike activity
                dummy_rate = torch.rand(inputs.shape[0], 100).to(self.device)  # 100 neurons as example
                layer_rates[name] = dummy_rate
        return layer_rates

    def _accumulate_spike_rates(self, layer_rates, batch_size):
        """
        Accumulate sum of spike activities across batches for statistics.
        """
        for layer_name, rate_tensor in layer_rates.items():
            if layer_name not in self.spike_rate_stats:
                self.spike_rate_stats[layer_name] = {'sum_rates': None}
            sum_r = self.spike_rate_stats[layer_name].get('sum_rates', None)
            if sum_r is None:
                self.spike_rate_stats[layer_name]['sum_rates'] = torch.sum(rate_tensor, dim=0)
            else:
                self.spike_rate_stats[layer_name]['sum_rates'] += torch.sum(rate_tensor, dim=0)

    def _compute_layer_rate_stats(self, total_samples):
        """
        Compute mean firing rate per neuron per layer over entire dataset.
        """
        for layer_name, stats_dict in self.spike_rate_stats.items():
            total_rate = stats_dict['sum_rates'] / total_samples  # mean rate
            self.spike_rate_stats[layer_name]['mean_rate'] = total_rate.cpu().numpy()

    def _log_spike_rate_stats(self):
        """
        Log or save spike rate statistics for analysis.
        """
        print("Layer-wise Average Firing Rates:")
        for layer_name, stats in self.spike_rate_stats.items():
            mean_rate = stats['mean_rate']
            print(f"{layer_name}: mean rate per neuron: {np.mean(mean_rate):.4f}, std: {np.std(mean_rate):.4f}")

```

## main.py

```python
## main.py
import argparse
import os
import time
import datetime
import torch
import numpy as np
import yaml

from utils import load_config, initialize_logger, set_random_seed, save_config
from dataset_loader import DatasetLoader
from model import ResNet, VGG, SEW_ResNet
from trainer import RateBasedTrainer
from evaluation import RateBasedEvaluation

def main():
    # ------------------------------
    # 1. Load and override configs 
    # ------------------------------
    args = argparse.ArgumentParser().parse_args()
    # Load default config
    config = load_config('config.yaml')
    # Override with command-line args
    args_overrides = vars(args)
    config = override_config_with_args(config, args_overrides)
    
    # Set seed for reproducibility
    seed = config['training'].get('seed', 42)
    set_random_seed(seed)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup logging
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = initialize_logger(log_dir)
    # Save final config for reproducibility
    save_config(config, os.path.join(log_dir, 'final_config.yaml'))

    # ------------------------------
    # 2. Load datasets
    # ------------------------------
    batch_size = config['training']['batch_size']
    T = config['training_mode'].get('sequence_length', config['training'].get('T', 4))
    dataset_loader = DatasetLoader(
        dataset_name=config['dataset']['name'],
        batch_size=batch_size,
        T=T,
        input_encoding=config['dataset'].get('input_encoding', 'direct'),
        augmentation=config['dataset'].get('augmentation', None),
        normalization_mean=config['dataset'].get('normalization_mean', None),
        normalization_std=config['dataset'].get('normalization_std', None),
        train_split_ratio=0.8,
        dataset_path=None,  # add path if needed
        num_workers=4,
        mode='static' if config['dataset']['name'].lower() != 'cifar10-dvs' else 'dynamic'
    )
    train_loader, test_loader = dataset_loader.load_data()

    # ------------------------------
    # 3. Instantiate model
    # ------------------------------
    architecture = config['model']['architecture']
    neuron_type = config['model'].get('neuron_type', 'LIF')  # Not used explicitly here
    if architecture.lower() == 'resnet18':
        model = ResNet('resnet18', config=config, training_mode=config['training_mode']['mode'])
    elif architecture.lower() == 'vgg11':
        model = VGG('VGG11', num_classes=10)
    elif architecture.lower() == 'sew-resnet34':
        model = SEW_ResNet('sew-resnet34', config=config, training_mode=config['training_mode']['mode'])
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    model.to(device)

    # ------------------------------
    # 4. Setup optimizer and scheduler
    # ------------------------------
    lr = config['training']['learning_rate']
    opt_type = config['training'].get('optimizer', 'Adam')
    weight_decay = config['training'].get('weight_decay', 5e-4)
    if opt_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'SGD':
        momentum = config['training'].get('momentum', 0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_type}")
    # Learning rate decay schedule
    decay_rate = config['training'].get('decay_rate', 0.95)

    # ------------------------------
    # 5. Initialize trainer
    # ------------------------------
    trainer = RateBasedTrainer(config, device)
    trainer.model = model
    trainer.optimizer = optimizer

    # ------------------------------
    # 6. Run training
    # ------------------------------
    total_epochs = config['training']['epochs']
    save_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.time()
    for epoch in range(1, total_epochs + 1):
        epoch_start_time = time.time()
        trainer._reset_traces()  # prepare traces
        trainer.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            stage_time = time.time()
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass with eligibility trace update
            outputs, rate_outputs, neuron_states = trainer._forward_pass(inputs, batch_size=inputs.shape[0])
            loss = trainer.criterion(outputs, targets)
            trainer._lr_schedule(epoch)  # update LR if needed
            optimizer.zero_grad()
            # Compute gradient approximation via rate backprop
            d_L_d_rate = trainer._compute_output_gradient(rate_outputs, targets)
            # Backward pass
            trainer._backward(rate_outputs, neuron_states, d_L_d_rate)
            # Update weights
            optimizer.step()
            # Metrics
            preds = outputs.argmax(dim=1)
            correct = (preds == targets).sum().item()
            total_loss += loss.item() * inputs.shape[0]
            total_correct += correct
            total_samples += inputs.shape[0]

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Epoch [{epoch}/{total_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                            f"Loss: {loss.item():.4f} Acc: {correct/inputs.shape[0]:.4f} "
                            f"Time: {time.time() - stage_time:.2f}s")
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        logger.info(f"Epoch [{epoch}/{total_epochs}] completed in {epoch_time:.2f}s, "
                    f"Avg Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f'epoch_{epoch}.pt')
        torch.save({'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    checkpoint_path)

        # Validate
        trainer._validate(test_loader)
        # Decay LR
        trainer._lr_schedule(epoch)

    total_training_time = time.time() - start_time
    logger.info(f"Training complete in {total_training_time/60:.2f} minutes.")

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from bn_layers import SpatialBN, TemporalBN
from neuron import LIFNeuron

class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-style architecture with rate-based neurons.
    Consists of two conv+BN+neuron layers with residual addition.
    """
    def __init__(self, in_channels, out_channels, stride=1, config=None, training_mode='rate_M'):
        super(BasicBlock, self).__init__()
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.training_mode = training_mode

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # Select BN type based on mode
        if self.training_mode == 'rate_M':
            self.bn1 = TemporalBN(out_channels)
        else:
            self.bn1 = SpatialBN(out_channels)
        self.neuron1 = LIFNeuron(V_th=self.config['model'].get('V_th', 1.0),
                                 decay_lambda=self.config['model'].get('decay_lambda', 0.95),
                                 surrogate_type='sigmoid', alpha=self.config['model'].get('alpha', 4.0))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if self.training_mode == 'rate_M':
            self.bn2 = TemporalBN(out_channels)
        else:
            self.bn2 = SpatialBN(out_channels)
        self.neuron2 = LIFNeuron(V_th=self.config['model'].get('V_th', 1.0),
                                 decay_lambda=self.config['model'].get('decay_lambda', 0.95),
                                 surrogate_type='sigmoid', alpha=self.config['model'].get('alpha', 4.0))
        # Downsample if needed
        self.downsample = None
        if stride !=1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                self.bn1 if self.training_mode=='rate_S' else self.bn1
            )

    def forward(self, x, mode='rate_M', T=4):
        """
        x: input tensor of shape [B, C, H, W]
        mode: 'rate_M' or 'rate_S'
        T: number of timesteps
        """
        residual = x
        # First conv bn neuron
        x_conv1 = self.conv1(x)
        x_bn1 = self.bn1(x_conv1)  # BN variant
        # Initialize membrane potentials and spikes
        u1 = torch.zeros_like(x_bn1)
        s1 = torch.zeros_like(x_bn1)
        spikes_layer1 = []
        for t in range(T):
            s_pre1 = s1 if t > 0 else torch.zeros_like(s1)
            s1, u1 = self.neuron1(u1, s_pre1, s_pre1, W=self.conv1.weight, lambda_=self.neuron1.decay_lambda, V_th=self.neuron1.V_th)
            spikes_layer1.append(s1)
        # Aggregate rate if needed
        if mode == 'rate_M':
            # compute firing rate over T
            rate1 = torch.mean(torch.stack(spikes_layer1), dim=0)
        else:
            # For 'rate_S', possibly just last timestep
            rate1 = s1
        # Residual path
        if self.downsample:
            residual = self.downsample(x)
        # Second layer
        x2 = self.conv2(rate1)  # use rate-based input
        x_bn2 = self.bn2(x2)
        u2 = torch.zeros_like(x_bn2)
        s2 = torch.zeros_like(x_bn2)
        spikes_layer2 = []
        for t in range(T):
            s_pre2 = s2 if t > 0 else torch.zeros_like(s2)
            s2, u2 = self.neuron2(u2, s_pre2, s_pre2, W=self.conv2.weight, lambda_=self.neuron2.decay_lambda, V_th=self.neuron2.V_th)
            spikes_layer2.append(s2)
        if mode == 'rate_M':
            rate2 = torch.mean(torch.stack(spikes_layer2), dim=0)
        else:
            rate2 = s2
        # Residual addition
        out = rate2 + residual
        return out

class Bottleneck(nn.Module):
    """
    Bottleneck residual block with three conv layers, for deeper architectures.
    """
    def __init__(self, in_channels, out_channels, stride=1, config=None, training_mode='rate_M'):
        super(Bottleneck, self).__init__()
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.training_mode = training_mode

        width = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        if self.training_mode=='rate_M':
            self.bn1 = TemporalBN(width)
        else:
            self.bn1 = SpatialBN(width)
        self.neuron1 = LIFNeuron(V_th=self.config['model'].get('V_th', 1.0),
                                 decay_lambda=self.config['model'].get('decay_lambda', 0.95),
                                 surrogate_type='sigmoid', alpha=self.config['model'].get('alpha', 4.0))
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        if self.training_mode=='rate_M':
            self.bn2 = TemporalBN(width)
        else:
            self.bn2 = SpatialBN(width)
        self.neuron2 = LIFNeuron(V_th=self.config['model'].get('V_th', 1.0),
                                 decay_lambda=self.config['model'].get('decay_lambda', 0.95),
                                 surrogate_type='sigmoid', alpha=self.config['model'].get('alpha', 4.0))
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        if self.training_mode=='rate_M':
            self.bn3 = TemporalBN(out_channels)
        else:
            self.bn3 = SpatialBN(out_channels)
        self.neuron3 = LIFNeuron(V_th=self.config['model'].get('V_th', 1.0),
                                 decay_lambda=self.config['model'].get('decay_lambda', 0.95),
                                 surrogate_type='sigmoid', alpha=self.config['model'].get('alpha', 4.0))
        self.downsample = None
        if stride !=1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                self.bn1 if self.training_mode=='rate_S' else self.bn1
            )

    def forward(self, x, mode='rate_M', T=4):
        residual = x
        x1 = self.conv1(x)
        x1_bn = self.bn1(x1)
        u1 = torch.zeros_like(x1_bn)
        s1 = torch.zeros_like(x1_bn)
        for t in range(T):
            s_pre1 = s1 if t>0 else torch.zeros_like(s1)
            s1, u1 = self.neuron1(u1, s_pre1, s_pre1, W=self.conv1.weight, lambda_=self.neuron1.decay_lambda, V_th=self.neuron1.V_th)
        rate1 = torch.mean(torch.stack([s1]), dim=0) if mode=='rate_M' else s1

        if self.downsample:
            residual = self.downsample(x)

        x2 = self.conv2(rate1)
        x2_bn = self.bn2(x2)
        u2 = torch.zeros_like(x2_bn)
        s2 = torch.zeros_like(x2_bn)
        for t in range(T):
            s_pre2 = s2 if t>0 else torch.zeros_like(s2)
            s2, u2 = self.neuron2(u2, s_pre2, s_pre2, W=self.conv2.weight, lambda_=self.neuron2.decay_lambda, V_th=self.neuron2.V_th)
        rate2 = torch.mean(torch.stack([s2]), dim=0) if mode=='rate_M' else s2

        x3 = self.conv3(rate2)
        x3_bn = self.bn3(x3)
        u3 = torch.zeros_like(x3_bn)
        s3 = torch.zeros_like(x3_bn)
        for t in range(T):
            s_pre3 = s3 if t>0 else torch.zeros_like(s3)
            s3, u3 = self.neuron3(u3, s_pre3, s_pre3, W=self.conv3.weight, lambda_=self.neuron3.decay_lambda, V_th=self.neuron3.V_th)
        rate3 = torch.mean(torch.stack([s3]), dim=0) if mode=='rate_M' else s3
        out = rate3 + residual
        return out

class ResNet(nn.Module):
    """
    General ResNet class supporting different depths and configurations,
    integrating rate-based neurons and BN layers in accordance with mode.
    """
    def __init__(self, architecture='resnet18', config=None, training_mode='rate_M'):
        super(ResNet, self).__init__()
        self.config = config
        self.architecture = architecture.lower()
        self.training_mode = training_mode
        # Define initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.training_mode=='rate_M':
            self.bn1 = TemporalBN(64)
        else:
            self.bn1 = SpatialBN(64)
        self.neuron1 = LIFNeuron(V_th=self.config['model'].get('V_th', 1.0),
                                 decay_lambda=self.config['model'].get('decay_lambda', 0.95),
                                 surrogate_type='sigmoid', alpha=self.config['model'].get('alpha', 4.0))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define layer configurations based on architecture
        if self.architecture=='resnet18':
            layers = [2,2,2,2]
            block = BasicBlock
        elif self.architecture=='resnet34':
            # can be added similarly
            raise NotImplementedError("ResNet34 not implemented yet")
        elif self.architecture=='resnet19':
            layers = [3,3,3]
            block = BasicBlock
        elif self.architecture=='resnet50':
            # for completeness
            raise NotImplementedError("ResNet50 not implemented")
        else:
            raise ValueError("Unsupported architecture: {}".format(architecture))

        # Create residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256 * (1 if self.architecture=='resnet18' else 1), 10)  # assuming 10 classes

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.config, self.training_mode))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 1, self.config, self.training_mode))
        return nn.Sequential(*layers)

    def forward(self, x, mode='rate_M', T=4):
        """
        Forward pass supporting different modes.
        Args:
            x: input tensor [B, 3, H, W]
            mode: 'rate_M' or 'rate_S'
            T: sequence length
        Returns:
            logits: output predictions
        """
        # Initial convolution + BN + neuron
        x = self.conv1(x)
        if self.training_mode=='rate_M':
            x = self.bn1(x)
        else:
            x = self.bn1(x)
        u = torch.zeros_like(x)
        s = torch.zeros_like(x)
        spikes = []
        for t in range(T):
            s_pre = s if t>0 else torch.zeros_like(s)
            s, u = self.neuron1(u, s_pre, s_pre, self.conv1.weight, lambda_=self.neuron1.decay_lambda, V_th=self.neuron1.V_th)
            spikes.append(s)
        if mode=='rate_M':
            out = torch.mean(torch.stack(spikes), dim=0)
        else:
            out = s
        out = self.maxpool(out)

        # Layer1
        out = self.layer1(out, mode, T)

        # Layer2
        out = self.layer2(out, mode, T)

        # Layer3
        out = self.layer3(out, mode, T)

        # Final pooling and classifier
        feat = self.avgpool(out)
        feat = feat.view(feat.size(0), -1)
        logits = self.fc(feat)
        return logits

    def get_stats(self):
        """
        Return network statistics, e.g., spike rates, for monitoring.
        """
        pass # could be implemented as needed
```

## neuron.py

```python
## neuron.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from surrogate_gradients import SurrogateFunction

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron model with surrogate gradient support, 
    adapted for rate-based backpropagation training in deep SNNs.
    Implements membrane potential updates and surrogate spike generation.

    Attributes:
        V_th (float): Spike threshold (from config).
        decay_lambda (float): Decay factor for membrane potential (from config).
        surrogate_fn (nn.Module): Surrogate function for spike gradient approximation.
        device (torch.device): Computation device.
        u (torch.Tensor): Membrane potential state, shape: [batch_size, neuron_num].
        s (torch.Tensor): Spike output, shape: [batch_size, neuron_num].
    """
    def __init__(self, V_th=1.0, decay_lambda=0.95, surrogate_type='sigmoid', alpha=4.0, device=None):
        """
        Initialize the neuron model with parameters.

        Args:
            V_th (float): Spike threshold.
            decay_lambda (float): Decay factor.
            surrogate_type (str): Type of surrogate ('sigmoid', etc.).
            alpha (float): Steepness parameter for surrogate.
            device (torch.device): Device to run computations on.
        """
        super(LIFNeuron, self).__init__()
        self.V_th = V_th
        self.decay_lambda = decay_lambda
        self.surrogate_fn = SurrogateFunction(surrogate_type, alpha)
        self.device = device if device is not None else torch.device('cpu')
        self.u = None  # Membrane potential state
        self.s = None  # Spike output

    def reset(self, batch_size):
        """
        Reset neuron states before a new sequence or epoch.

        Args:
            batch_size (int): Batch size for resetting states.
        """
        self.u = torch.zeros(batch_size, 1, device=self.device)
        self.s = torch.zeros(batch_size, 1, device=self.device)

    def forward(self, u_prev, s_prev, s_pre, W, lambda_=None, V_th=None):
        """
        Perform membrane potential update and generate spike.

        Args:
            u_prev (torch.Tensor): Previous membrane potential, shape [batch, neurons].
            s_prev (torch.Tensor): Previous spike (for decay correction), shape [batch, neurons].
            s_pre (torch.Tensor): Presynaptic input spikes, shape [batch, neurons].
            W (torch.Tensor): Synaptic weights, shape [neurons, presynaptic_neurons].
            lambda_ (float): Decay factor; if None, use class attribute.
            V_th (float): Threshold; if None, use class attribute.

        Returns:
            s (torch.Tensor): Spike output (binary, 0/1), shape [batch, neurons].
            u (torch.Tensor): Updated membrane potential.
        """
        if lambda_ is None:
            lambda_ = self.decay_lambda
        if V_th is None:
            V_th = self.V_th

        # Membrane potential update
        u = lambda_ * (u_prev - V_th * s_prev) + torch.matmul(s_pre, W.t())

        # Generate spike using surrogate gradient
        s = self.surrogate_fn.apply(u - V_th)

        return s, u

class SurrogateFunction(nn.Module):
    """
    Surrogate function with customizable types (e.g., sigmoid).
    Implements forward as a smooth approximation to Heaviside step,
    with a non-zero gradient for backpropagation.
    """
    def __init__(self, surrogate_type='sigmoid', alpha=4.0):
        """
        Initialize the surrogate.

        Args:
            surrogate_type (str): Type of surrogate ('sigmoid', 'gaussian', etc.).
            alpha (float): Steepness/slant parameter.
        """
        super(SurrogateFunction, self).__init__()
        self.surrogate_type = surrogate_type
        self.alpha = alpha

    def forward(self, u):
        """
        Forward pass: apply the surrogate function.

        Args:
            u (torch.Tensor): Membrane potential minus threshold.

        Returns:
            torch.Tensor: Surrogate spike value (continuous) for backprop.
        """
        if self.surrogate_type == 'sigmoid':
            return torch.sigmoid(self.alpha * u)
        else:
            # Defaults to sigmoid if unknown
            return torch.sigmoid(self.alpha * u)

    def apply(self, u):
        """
        Apply surrogate function and ensure differentiability.

        Args:
            u (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Surrogate output with gradient support.
        """
        return self.forward(u)

    def surrogate_derivative(self, u):
        """
        Compute surrogate gradient (derivative) for custom backward.
        Not used explicitly here, as PyTorch autograd handles it via .apply()

        Args:
            u (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Derivative of surrogate at u.
        """
        if self.surrogate_type == 'sigmoid':
            sig = torch.sigmoid(self.alpha * u)
            return self.alpha * sig * (1 - sig)
        else:
            return torch.sigmoid(self.alpha * u)

# Note: For simplicity, in this implementation, the backward pass
# uses autograd through the apply() method, which handles gradients
# automatically. If needed, custom backward functions can be implemented
# for more control. This class supports extendability to other surrogates.

```

## surrogate_gradients.py

```python
## surrogate_gradients.py
import torch

def sigmoid(x: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    """
    Surrogate activation function: sigmoid approximation of the Heaviside step.
    This function provides a smooth approximation suitable for gradient backpropagation.
    
    Args:
        x (torch.Tensor): The input tensor, representing membrane potential or pre-activation.
        alpha (float): Steepness parameter controlling the slope of the sigmoid function.
            Default is 4.0 as per configuration; higher values make the sigmoid steeper.
    
    Returns:
        torch.Tensor: The surrogate output values, same shape as input.
    """
    return torch.sigmoid(alpha * x)

def surrogate_gradient(x: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    """
    Compute the derivative (gradient) of the surrogate sigmoid function.
    This is used during backpropagation to approximate \(\frac{\partial H}{\partial x}\).
    
    Args:
        x (torch.Tensor): The input tensor (membrane potential or pre-activation).
        alpha (float): Steepness parameter matching the surrogate function.
    
    Returns:
        torch.Tensor: The gradient values, same shape as input.
    """
    sig = torch.sigmoid(alpha * x)
    return alpha * sig * (1 - sig)

# Optional: encapsulate in a class if needed for flexibility in the future
# Here, providing only functions for simplicity and modularity.

```

## trainer.py

```python
# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging
import time
from utils import load_config, set_random_seed, save_config, initialize_logger
from dataset_loader import DatasetLoader
from model import ResNet, VGG, SEW_ResNet
from surrogate_gradients import sigmoid
from torch.utils.data import DataLoader

class RateBasedTrainer:
    """
    Implements the core training loop for deep SNNs with rate-based backpropagation,
    following the methodology described in the paper, including eligibility traces,
    surrogate gradients, and simplified gradient computation.
    """
    def __init__(self, config, device):
        """
        Initialize trainer with configuration and device.
        Args:
            config (dict): configuration parameters from YAML
            device (torch.device): CPU or CUDA
        """
        self.config = config
        self.device = device
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        self.T = self.config['training']['T']
        self.training_mode = self.config['training_mode']['mode']
        self.sequence_length = self.T
        self.num_layers = self._count_layers()
        self._init_training_state()
        self.logger = None  # will be initialized outside

    def _build_model(self):
        """
        Instantiate the neural network based on specified architecture and config.
        """
        arch = self.config['model']['architecture']
        if arch.lower() == 'resnet18':
            model = ResNet(architecture='resnet18', config=self.config, training_mode=self.training_mode)
        elif arch.lower() == 'vgg11':
            model = VGG('VGG11', num_classes=10)  # adapt size for dataset if needed
        elif arch.lower() == 'sew-resnet34':
            model = SEW_ResNet('sew-resnet34', config=self.config, training_mode=self.training_mode)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        model = model.to(self.device)
        return model

    def _build_optimizer(self):
        """
        Build optimizer with parameters from config.
        """
        optim_type = self.config['training'].get('optimizer', 'Adam')
        lr = self.config['training'].get('learning_rate', 0.1)
        wd = self.config['training'].get('weight_decay', 5e-4)
        if optim_type == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif optim_type == 'SGD':
            momentum = self.config['training'].get('momentum', 0.9)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {optim_type}")
        return optimizer

    def _count_layers(self):
        """
        Count trainable layers, mainly the number of weight layers for gradient accumulation.
        """
        count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                count += 1
        return count

    def _init_training_state(self):
        """
        Initialize eligibility traces and auxiliary variables for each layer.
        """
        self.e_trace = {}  # e_t^l: eligibility trace
        self.g_trace = {}  # g_t^l: gradient estimators
        self.rho = {}      # rho_t^l: neuron dynamics influence
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                size = param.shape
                self.e_trace[name] = torch.zeros(size, device=self.device)
                self.g_trace[name] = torch.zeros(size, device=self.device)
                # For rho, shape matches the neuron layer output
                self.rho[name] = 0.0  # scalar or tensor as appropriate
        # Might initialize more if needed for batch norm statistics, etc.

    def set_logger(self, log_path):
        """
        Initialize logger for logging training metrics.
        """
        self.logger = initialize_logger(log_path)

    def train(self, train_loader, val_loader, num_epochs, save_dir):
        """
        Main training loop over epochs and data.
        Args:
            train_loader (DataLoader): training data loader
            val_loader (DataLoader): validation data loader
            num_epochs (int): total epochs
            save_dir (str): directory for saving models and logs
        """
        os.makedirs(save_dir, exist_ok=True)
        start_time = time.time()
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            self.model.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                batch_time_start = time.time()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Reset eligibility traces and neuron states
                self._reset_traces()
                # Forward pass
                output, rate_activations, neuron_states = self._forward_pass(inputs, batch_size=inputs.size(0))
                # Compute loss
                loss = self.criterion(output, targets)
                total_loss += loss.item() * inputs.size(0)
                # Backward for rate-based gradient
                self.optimizer.zero_grad()
                # Compute rate gc
                d_L_d_rate_prop = self._compute_output_gradient(rate_activations, targets)
                # Backpropagation through the network
                self._backward(rate_activations, neuron_states, d_L_d_rate_prop)
                # Optimizer step
                self.optimizer.step()
                # Compute accuracy
                preds = output.argmax(dim=1)
                correct = (preds == targets).sum().item()
                total_correct += correct
                total_samples += inputs.size(0)
                batch_time_end = time.time()

                if self.logger and batch_idx % 50 == 0:
                    self.logger.info(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                                     f"Loss: {loss.item():.4f} Acc: {correct/inputs.size(0):.4f} "
                                     f"Time: {batch_time_end - batch_time_start:.2f}s")
            epoch_time = time.time() - epoch_start
            train_loss_avg = total_loss / total_samples
            train_acc = total_correct / total_samples
            self._validate(val_loader)
            self._save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"))
            if self.logger:
                self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s, "
                                 f"Avg Loss: {train_loss_avg:.4f}, Accuracy: {train_acc:.4f}")
            # decay schedule if any
            self._lr_schedule(epoch)
        total_time = time.time() - start_time
        if self.logger:
            self.logger.info(f"Training completed in {total_time/60:.2f} minutes.")

    def _reset_traces(self):
        """
        Zero out eligibility and auxiliary traces for the new batch.
        """
        for key in self.e_trace:
            self.e_trace[key].zero_()
            self.g_trace[key].zero_()
            if isinstance(self.rho[key], torch.Tensor):
                self.rho[key].fill_(0.0)
            else:
                self.rho[key] = 0.0

    def _forward_pass(self, inputs, batch_size):
        """
        Implement the forward pass with rate-based approximation, eligibility trace updates.
        Args:
            inputs (Tensor): input data tensor with shape [B, T, C, H, W] or [B, C, H, W] for rate_S
            batch_size (int): batch size
        Returns:
            output (Tensor): class logits
            rate_activations (dict): per-layer rate estimates
            neuron_states (dict): stored neuron membrane potentials and spikes for backward
        """
        # Initialize the rate dictionaries
        rate_activations = {}
        neuron_states = {}  # to store u, s, e_t, g_t, rho during the sequence
        # Prepare initial inputs for the network
        if self.training_mode == 'rate_M':
            # inputs shape: [B, T, C, H, W]
            # Initialize containers
            batch = inputs.shape[0]
            # initialize memory states
            # For each layer, we keep u, s states, eligibility traces etc.
            # For simplicity, here we implement a minimal version. More detailed per-layer storage is recommended.
            # --- Start forward pass for T steps ---
            # For illustration, assume inputs are [B, T, C, H, W], sequence dimension T
            outputs_list = []
            # Initialize neuron states and eligibility traces for all layers
            neuron_state_buffers = {}  # e.g., {'layer1': {'u': ..., 's': ...}}
            for name, layer in self.model.named_modules():
                if hasattr(layer, 'neuron1'):
                    neuron_state_buffers[name] = {
                        'u': torch.zeros(batch_size, layer.neuron1.V_th.shape[1], device=self.device),
                        's': torch.zeros(batch_size, layer.neuron1.V_th.shape[1], device=self.device)
                    }
            # Loop over T to process each timestep
            for t in range(self.T):
                # Get input for time t
                x_t = inputs[:, t, ...]
                # Forward through network
                out, layer_states = self._forward_single_step(x_t, neuron_state_buffers)
                # Save/accumulate outputs for loss calculation
                outputs_list.append(out)
                # Update neuron states for next timestep
                # Store states internally, or in buffer
            # Compute average output
            output_logits = torch.stack(outputs_list, dim=1).mean(dim=1)
            # Prepare the rate activations dictionary (if needed for loss)
            rate_activations['layer_outputs'] = output_logits
            neuron_states['layer_neurons'] = layer_states
        else:
            # Rate_S: single step, process one timestep
            x_t = inputs  # shape: [B, C, H, W]
            out, layer_states = self._forward_single_step(x_t, None)
            output_logits = out
            rate_activations['layer_outputs'] = output_logits
            neuron_states['layer_neurons'] = layer_states

        return output_logits, rate_activations, neuron_states

    def _forward_single_step(self, x_t, neuron_state_buffers):
        """
        Forward function for a single timestep.
        Args:
            x_t (Tensor): current input
            neuron_state_buffers (dict): neuron states for each layer, if any
        Returns:
            out: network output at this timestep
            layer_states: store neuron states for backprop
        """
        # Forward through initial conv + BN + neuron
        # Placeholder: use model defined in 'model.py'
        # For illustration:
        # Replace with actual calls to model modules and neuron update steps
        out = x_t
        layer_states = {}
        # Example: process through model layers, updating neuron states and eligibility traces
        # For each layer, implement update of u, s
        # one must integrate surrogate gradients, eligibility trace update, etc.
        # Since this code is highly schematic, assume the model handles internal states
        # In actual implementation, call model forward with appropriate mode
        # For now, return the raw input as output (to be replaced with real model forward)
        return out, layer_states

    def _compute_output_gradient(self, rate_outputs, targets):
        """
        Compute the gradient of the loss w.r.t. the rate outputs (e.g., via loss derivative)
        Placeholder: this depends on actual loss implementation.
        Args:
            rate_outputs (dict): network output estimates
            targets (Tensor): true labels
        Returns:
            d_L_d_rate (Tensor): derivative of loss with respect to rate estimates
        """
        # For simplicity, assuming loss is cross-entropy, so:
        # Compute softmax and derivative
        logits = rate_outputs['layer_outputs']
        probs = nn.functional.softmax(logits, dim=1)
        grad = probs
        grad.scatter_(1, targets.unsqueeze(1), 0)
        grad = -grad / logits.shape[0]  # normalized derivative
        return grad

    def _backward(self, rate_activations, neuron_states, d_L_d_rate):
        """
        Perform backward pass for rate-based gradients using eligibility traces and surrogate derivatives.
        Args:
            rate_activations (dict): stored rate estimates
            neuron_states (dict): stored neuron mem potentials and spikes
            d_L_d_rate (Tensor): gradient of loss w.r.t. rate output
        """
        # Implement a simplified backward as per rate-based derivation in paper
        # E.g., compute delta terms (δ) for each layer
        # Update gradients of W^l as: ΔW^l = δ^l * (r^{l-1})^T
        # For each layer, compute δ and gradients
        # Here, we demonstrate a minimal example;
        # the actual implementation should match the gradient derivations and variable storage.
        for name, param in self.model.named_parameters():
            # Skip bias or non-weight parameters
            if 'weight' not in name:
                continue
            # For illustration, generate dummy delta
            delta = torch.ones_like(param.data)  # Placeholder
            # Compute gradient as outer product with previous layer rate
            prev_rate = rate_activations.get('layer_outputs', None)
            if prev_rate is None:
                continue
            grad_W = torch.mm(delta.view(-1,1), prev_rate.view(1,-1))
            param.grad = grad_W

        # After computing gradients, optimizer steps will update weights

    def _save_checkpoint(self, filepath):
        """
        Save model parameters, optimizer state, and current trace state.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def _load_checkpoint(self, filepath):
        """
        Load previous checkpoint.
        """
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def _lr_schedule(self, epoch):
        """
        Implement decay schedule, e.g., exponential per epoch.
        """
        base_lr = self.config['training'].get('learning_rate', 0.1)
        decay_rate = self.config['training'].get('decay_rate', 0.95)
        new_lr = base_lr * (decay_rate ** (epoch - 1))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _validate(self, val_loader):
        """
        Run validation on the validation set.
        """
        total_correct = 0
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self._forward_inference(inputs)
                preds = output.argmax(dim=1)
                total_correct += (preds == targets).sum().item()
                total_samples += inputs.size(0)
        accuracy = total_correct / total_samples
        if self.logger:
            self.logger.info(f"Validation accuracy: {accuracy:.4f}")

    def _forward_inference(self, inputs):
        """
        Forward pass during inference with the standard mode.
        """
        # Simplified: just run the model's direct inference
        return self.model(inputs, mode='rate_S', T=1)

# Usage example outside this class:
# if __name__ == '__main__':
#     config = load_config('config.yaml')
#     set_random_seed(config['training'].get('seed', 42))
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     trainer = RateBasedTrainer(config, device)
#     train_loader, val_loader = DatasetLoader(...).load_data()
#     trainer.set_logger(os.path.join(config['logging']['log_dir'], 'train_log.txt'))
#     trainer.train(train_loader, val_loader, num_epochs=config['training']['epochs'], save_dir='./checkpoints')
```

## utils.py

## utils.py
```python
import argparse
import os
import yaml
import logging
import time
import datetime
import random
import numpy as np
import torch


def parse_args():
    """
    Parse command-line arguments to override configuration parameters.
    Returns:
        args (argparse.Namespace): Parsed arguments with default values set from config.yaml.
    """
    parser = argparse.ArgumentParser(description='Utility functions for reproducibility and experiment control.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file.')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name (CIFAR-10, CIFAR-100, ImageNet, CIFAR10-DVS).')
    parser.add_argument('--architecture', type=str, default=None, help='Model architecture (ResNet-18, VGG-11, etc.)')
    parser.add_argument('--training_mode', type=str, choices=['rate_M', 'rate_S'], default=None, help='Training mode.')
    parser.add_argument('--sequence_length', type=int, default=None, help='Sequence length T for rate approximation.')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=None, help='Initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size.')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD'], default=None, help='Optimizer choice.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--mode', type=str, choices=['rate_M', 'rate_S'], default=None, help='Training mode mode override.')
    parser.add_argument('--T', type=int, default=None, help='Number of timesteps T.')
    parser.add_argument('--online', action='store_true', help='Use online training mode.')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs and outputs.')
    args = parser.parse_args()
    return args


def load_config(config_path='config.yaml'):
    """
    Load the configuration from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        config (dict): Dictionary with configuration parameters.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"Empty config file: {config_path}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing YAML config: {e}")


def override_config_with_args(config, args):
    """
    Override configuration dictionary with command-line arguments.
    Args:
        config (dict): Original configuration dictionary.
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        new_config (dict): Updated configuration with command-line overrides.
    """
    new_config = dict(config)  # shallow copy
    # Map arguments to config keys
    if args.dataset is not None:
        new_config['dataset']['name'] = args.dataset
    if args.architecture is not None:
        new_config['model']['architecture'] = args.architecture
    if args.training_mode is not None:
        new_config['training_mode']['mode'] = args.training_mode
    if args.sequence_length is not None:
        new_config['training']['sequence_length'] = args.sequence_length
        new_config['training']['T'] = args.sequence_length
    if args.epochs is not None:
        new_config['training']['epochs'] = args.epochs
    if args.learning_rate is not None:
        new_config['training']['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        new_config['training']['batch_size'] = args.batch_size
    if args.optimizer is not None:
        new_config['training']['optimizer'] = args.optimizer
    if args.seed is not None:
        new_config['training']['seed'] = args.seed
    if args.mode is not None:
        new_config['training_mode']['mode'] = args.mode
    if args.T is not None:
        new_config['training']['T'] = args.T
    if args.online:
        new_config['training_mode']['online'] = True
    if args.log_dir:
        new_config['logging']['log_dir'] = args.log_dir
    return new_config


def save_config(config, filepath):
    """
    Save the final configuration to a YAML file for reproducibility.
    Args:
        config (dict): Configuration dictionary.
        filepath (str): Path to save the YAML configuration.
    """
    with open(filepath, 'w') as f:
        yaml.dump(config, f)


def initialize_logger(log_dir, log_filename='log.txt'):
    """
    Initialize a logger that logs messages to stdout and a log file.
    Args:
        log_dir (str): Directory where log file will be saved.
        log_filename (str): Log file name.
    Returns:
        logger (logging.Logger): Configured logger object.
    """
    # Create log directory if not exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Stream handler for stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler for log file
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def log_time(start_time, message="Elapsed Time"):
    """
    Log the elapsed time since start_time.
    Args:
        start_time (float): Start timestamp.
        message (str): Description to prepend.
    Returns:
        elapsed_seconds (float): Duration since start_time.
    """
    elapsed_seconds = time.time() - start_time
    formatted_time = str(datetime.timedelta(seconds=elapsed_seconds))
    logging.info(f"{message}: {formatted_time}")
    return elapsed_seconds


def set_random_seed(seed):
    """
    Set seed for reproducibility across torch, numpy, and random.
    Args:
        seed (int): Seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_output_dir(path):
    """
    Create output directory if not exists.
    Args:
        path (str): Directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\rate-based-backpropagation\rate-based-backpropagation_repo`
