## dataset_loader.py
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Optional: import for specific neuromorphic datasets if available
# For example purposes, placeholders are used for neuromorphic data loaders
# In practice, replace with actual data loading code for event datasets

class DatasetLoader:
    def __init__(self, config: dict):
        """
        Initialize the DatasetLoader with configuration parameters.

        Args:
            config (dict): Configuration dictionary, expecting keys:
                - dataset: dict with dataset info (name, path, batch_size, etc.)
                - training: dict with training params
        """
        self.dataset_name = config['dataset'].get('name', 'CIFAR10')
        self.dataset_path = config['dataset'].get('dataset_path', './data')
        self.batch_size = config['dataset'].get('batch_size', 128)
        self.num_workers = config['dataset'].get('num_workers', 4)
        self.norm_mean = config['dataset'].get('normalization_mean', [0.4914, 0.4822, 0.4465])
        self.norm_std = config['dataset'].get('normalization_std', [0.2023, 0.1994, 0.2010])
        self.encoding_scheme = config['dataset'].get('encoding_scheme', 'direct_spike_encoding')
        self.seed = config['training'].get('seed', 2022)

        self.train_transform = None
        self.test_transform = None

    def load_data(self):
        """
        Load and preprocess datasets based on dataset name.

        Returns:
            train_dataset, test_dataset: datasets ready for DataLoader
        """
        # Fix seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.dataset_name.lower() in ['cifar10', 'cifar100']:
            return self._load_static_dataset()
        elif self.dataset_name.lower() == 'tinyimagenet':
            return self._load_tinyimagenet()
        elif self.dataset_name.lower() == 'dvs-gesture':
            return self._load_dvs_gesture()
        elif self.dataset_name.lower() == 'dvs-cifar10':
            return self._load_dvs_cifar10()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _load_static_dataset(self):
        """
        Load static image datasets like CIFAR-10, CIFAR-100, TinyImageNet.
        
        Applies preprocessing and creates spike-encoded datasets.

        Returns:
            train_dataset, test_dataset
        """
        # Define basic normalization transform
        normalize = transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        # Data augmentation transforms
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        test_transforms = [
            transforms.ToTensor(),
            normalize
        ]

        # Compose transforms
        self.train_transform = transforms.Compose(train_transforms)
        self.test_transform = transforms.Compose(test_transforms)

        # Load datasets
        if self.dataset_name.lower() == 'cifar10':
            train_dataset = datasets.CIFAR10(root=self.dataset_path, train=True, download=True, transform=self.train_transform)
            test_dataset = datasets.CIFAR10(root=self.dataset_path, train=False, download=True, transform=self.test_transform)
        elif self.dataset_name.lower() == 'cifar100':
            train_dataset = datasets.CIFAR100(root=self.dataset_path, train=True, download=True, transform=self.train_transform)
            test_dataset = datasets.CIFAR100(root=self.dataset_path, train=False, download=True, transform=self.test_transform)
        elif self.dataset_name.lower() == 'tinyimagenet':
            # Placeholder: implement custom loader for TinyImageNet
            train_dataset = self._load_tinyimagenet_dataset(split='train')
            test_dataset = self._load_tinyimagenet_dataset(split='val')
        else:
            raise ValueError(f"Unsupported static dataset: {self.dataset_name}")

        # Wrap datasets to produce spike sequences
        train_dataset = SpikeDatasetWrapper(train_dataset, self.encoding_scheme)
        test_dataset = SpikeDatasetWrapper(test_dataset, self.encoding_scheme)
        return train_dataset, test_dataset

    def _load_tinyimagenet_dataset(self, split='train'):
        """
        Placeholder for TinyImageNet dataset loading.
        In practice, load from local extracted folder.

        Args:
            split (str): 'train' or 'val'
        Returns:
            dataset
        """
        # Implement custom dataset loading for TinyImageNet
        # For simplicity, assuming data is in 'tiny-imagenet-200' folder
        # with standard structure.
        from torchvision.datasets.folder import ImageFolder
        dataset_path = os.path.join(self.dataset_path, 'tiny-imagenet-200', split)
        return ImageFolder(root=dataset_path, transform=self.test_transform)

    def _load_dvs_gesture(self):
        """
        Placeholder for DVS Gesture dataset loader.
        In practice, load from event data files and convert to frame sequences.
        """
        # Replace with actual loader for DVS-Gesture dataset
        return DVSGestureDataset(self.dataset_path, split='train'), DVSGestureDataset(self.dataset_path, split='test')

    def _load_dvs_cifar10(self):
        """
        Placeholder for DVS-CIFAR10 dataset loader.
        """
        # Replace with actual DVS-CIFAR10 loader
        return DVS_CIFAR10Dataset(self.dataset_path, split='train'), DVS_CIFAR10Dataset(self.dataset_path, split='test')


class SpikeDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, encoding_scheme: str, num_timesteps: int = 6):
        """
        Wraps a dataset to return spike-encoded sequences.

        Args:
            dataset (Dataset): Original dataset (images or data)
            encoding_scheme (str): Algorithm for encoding ('direct_spike_encoding')
            num_timesteps (int): Number of time steps T
        """
        self.dataset = dataset
        self.encoding_scheme = encoding_scheme
        self.num_timesteps = 6  # default, can be parameterized

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve an item and encode into spike sequence.

        Returns:
            input_tensor: shape (T, C, H, W) with spikes
            label
        """
        data, label = self.dataset[idx]

        if isinstance(data, torch.Tensor):
            # Static image: data shape (C, H, W)
            # Normalize and encode to spikes over T timesteps
            spike_seq = self._encode_static_image(data)
            return spike_seq, label
        elif isinstance(data, np.ndarray): 
            # If raw numpy array, convert to tensor
            tensor_data = torch.from_numpy(data)
            spike_seq = self._encode_static_image(tensor_data)
            return spike_seq, label
        elif hasattr(data, '__getitem__'):
            # For dataset items like PIL Images
            tensor_data = transforms.ToTensor()(data)
            spike_seq = self._encode_static_image(tensor_data)
            return spike_seq, label
        else:
            # For other data formats (event streams), assume preprocessing elsewhere
            # Here, simply pass through
            return data, label

    def _encode_static_image(self, image_tensor: torch.Tensor):
        """
        Encode a static image into spike train(s).

        Args:
            image_tensor (Tensor): shape (C, H, W), values in [0,1]
        Returns:
            spike_tensor: shape (T, C, H, W), dtype torch.float32 (binary spikes)
        """
        # Ensure pixel values are in [0,1]
        image_tensor = image_tensor.clamp(0, 1)
        C, H, W = image_tensor.shape
        T = self.num_timesteps
        spike_tensor = torch.zeros((T, C, H, W), dtype=torch.float32)

        # For 'direct_spike_encoding', implement rate-based encoding
        # For each pixel, generate a spike at each timestep with probability = pixel value
        for t in range(T):
            rand_mask = torch.rand((C, H, W))
            spikes = (rand_mask < image_tensor).float()
            spike_tensor[t] = spikes

        return spike_tensor


# Placeholder classes for neuromorphic datasets
class DVSGestureDataset(Dataset):
    def __init__(self, dataset_path: str, split: str = 'train'):
        """
        Load DVS-Gesture event data and convert to frame sequences.
        """
        # Implement actual loading and conversion
        self.data = []  # list of preprocessed tensors
        self.labels = []
        # Example: load from files
        # For this placeholder, create dummy data
        super().__init__()
        # For practice, assume small dummy dataset
        for _ in range(100):  # dummy size
            self.data.append(torch.randn(1, 128, 128))
            self.labels.append(random.randint(0, 10))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DVS_CIFAR10Dataset(Dataset):
    def __init__(self, dataset_path: str, split: str = 'train'):
        """
        Load DVS-CIFAR10 event data and convert to frame sequences.
        """
        # Implement actual loading
        self.data = []
        self.labels = []
        # Dummy data
        for _ in range(100):
            self.data.append(torch.randn(1, 128, 128))
            self.labels.append(random.randint(0, 9))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
