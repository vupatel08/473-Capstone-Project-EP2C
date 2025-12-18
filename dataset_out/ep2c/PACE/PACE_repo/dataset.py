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
