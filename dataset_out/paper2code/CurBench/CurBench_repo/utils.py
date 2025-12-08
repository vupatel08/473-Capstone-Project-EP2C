## utils.py
import os
import time
import logging
import json
import yaml
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from typing import Any, Dict, Optional

class ConfigManager:
    """Handles loading and accessing nested configuration parameters from YAML file."""
    def __init__(self, filepath: str = 'config.yaml'):
        self.config = self.load_config(filepath)

    @staticmethod
    def load_config(filepath: str) -> Dict[str, Any]:
        """Load YAML configuration file into a dictionary."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        with open(filepath, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg

    def get(self, key_path: str, default: Optional[Any] = None) -> Any:
        """Retrieve nested configuration value using dot notation, e.g., 'dataset.noise_ratio'."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enforce deterministic behavior (can impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ResourceLogger:
    """Handles timing and GPU memory tracking."""
    def __init__(self, enable_time: bool = True, enable_memory: bool = True):
        self.enable_time = enable_time
        self.enable_memory = enable_memory
        self.start_time = None
        self.max_memory_bytes = 0

    def log_time_start(self) -> float:
        """Record start time."""
        if self.enable_time:
            self.start_time = time.perf_counter()
            return self.start_time
        return 0.0

    def log_time_end(self, start_time: float) -> float:
        """Calculate elapsed time."""
        if self.enable_time and self.start_time is not None:
            elapsed = time.perf_counter() - start_time
            return elapsed
        return 0.0

    def reset_memory(self):
        """Reset max memory counter."""
        if self.enable_memory and torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()

    def log_gpu_memory(self) -> float:
        """Return current maximum GPU memory allocated in MB."""
        if self.enable_memory and torch.cuda.is_available():
            max_mem_bytes = torch.cuda.max_memory_allocated()
            # Update overall max if needed
            if max_mem_bytes > self.max_memory_bytes:
                self.max_memory_bytes = max_mem_bytes
            return max_mem_bytes / (1024 ** 2)  # Convert to MB
        return 0.0

    def get_max_memory_MB(self) -> float:
        """Get maximum GPU memory used during monitoring in MB."""
        if self.enable_memory and torch.cuda.is_available():
            return self.max_memory_bytes / (1024 ** 2)
        return 0.0

def setup_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure global logger."""
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=log_format, handlers=handlers)

def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics dictionary into a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_metrics(metrics: Dict[str, list], metric_name: str = 'Score', save_path: str = 'metrics_plot.png') -> None:
    """Plot training/validation metrics over epochs."""
    plt.figure(figsize=(8, 6))
    for label, values in metrics.items():
        plt.plot(values, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Metrics over epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_data_transform(dataset_name: str, phase: str = 'train'):
    """Provide data transformation pipeline based on dataset and phase."""
    from torchvision import transforms
    # Example for CV datasets; extend for NLP/graphs accordingly
    if dataset_name.lower().startswith('cifar') or dataset_name.lower().startswith('tinyimagenet'):
        if phase == 'train':
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.247, 0.243, 0.261))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.247, 0.243, 0.261))
            ])
    else:
        # Placeholder: extend for NLP or graph datasets
        return None

def set_deterministic(seed: int = 42):
    """Apply deterministic settings for reproducibility, if needed."""
    set_seed(seed)
    # Additional deterministic backend configs can be added here
