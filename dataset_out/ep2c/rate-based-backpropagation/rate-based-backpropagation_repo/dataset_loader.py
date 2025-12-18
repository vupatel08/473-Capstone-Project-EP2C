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
