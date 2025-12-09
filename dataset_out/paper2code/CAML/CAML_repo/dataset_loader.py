## dataset_loader.py

import os
import random
from typing import List, Tuple, Dict, Optional
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets

# We assume that the datasets are available locally or can be downloaded.
# For datasets not supported by torchvision, custom loaders will be implemented here.

class GenericImageFolder(Dataset):
    """
    Generic dataset loader based on directory structure.
    Assumes images are stored in:
        root/class_x/xxx.png
        root/class_x/xxy.png
        ...
    """
    def __init__(self, root_dir: str, classes: Optional[List[str]] = None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List of (image_path, label)
        self.class_name_to_idx = {}
        self.idx_to_class_name = {}
        self.classes = classes  # Optional: subset of classes to use

        # Load class directories
        all_classes = sorted(os.listdir(root_dir))
        if classes is not None:
            all_classes = [c for c in all_classes if c in classes]
        self.class_name_to_idx = {c: i for i, c in enumerate(all_classes)}
        self.idx_to_class_name = {i: c for c, i in self.class_name_to_idx.items()}

        for class_name in all_classes:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append((os.path.join(class_dir, fname), self.class_name_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class CustomPaintingsDataset(Dataset):
    """
    Placeholder for a custom paintings dataset loader.
    Assuming dataset in form of a CSV or annotations with image paths and labels.
    """
    def __init__(self, data_dir: str, split: str, transform=None):
        # Placeholder: Implement actual loading logic with annotations or directory structure
        # For now, assume images are in data_dir/split/class_name/*.jpg
        self.samples = []
        self.class_name_to_idx = {}
        self.idx_to_class_name = {}
        self.transform = transform

        split_dir = os.path.join(data_dir, split)
        all_classes = sorted(os.listdir(split_dir))
        self.class_name_to_idx = {c: i for i, c in enumerate(all_classes)}
        self.idx_to_class_name = {i: c for c, i in self.class_name_to_idx.items()}

        for class_name in all_classes:
            class_dir = os.path.join(split_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append((os.path.join(class_dir, fname), self.class_name_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class ChestXRayDataset(Dataset):
    """
    Placeholder for medical ChestX-ray dataset loader.
    Assumes images and labels are stored similarly.
    """
    def __init__(self, data_dir: str, split: str, transform=None):
        # Implement actual dataset loading here
        # For now, mimic similar structure to above
        self.samples = []
        self.class_name_to_idx = {}
        self.idx_to_class_name = {}
        self.transform = transform

        split_dir = os.path.join(data_dir, split)
        all_classes = sorted(os.listdir(split_dir))
        self.class_name_to_idx = {c: i for i, c in enumerate(all_classes)}
        self.idx_to_class_name = {i: c for c, i in self.class_name_to_idx.items()}

        for class_name in all_classes:
            class_dir = os.path.join(split_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, fname), self.class_name_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class DatasetWrapper:
    """
    Wrapper for datasets to handle dataset-specific loading and sampling.
    """
    def __init__(self, name: str, split: str, base_path: str, transform):
        """
        Initialize dataset based on name.
        """
        self.name = name
        self.split = split
        self.base_path = base_path
        self.transform = transform
        self.dataset_obj = None
        self.class_to_indices = {}  # class_idx: list of indices in dataset

        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset according to its name.
        """
        if self.name.lower() in ['mini-imagenet', 'tiered-imagenet', 'imagenet', 'cifar-fs', 'cifar10', 'cifar100', 'pascal voc', 'paintings', 'cub', 'aircraft', 'chestx']:
            # Use generic loader by directory structure
            self.dataset_obj = GenericImageFolder(self.base_path, transform=self.transform)
        else:
            # For unsupported datasets, raise error for now
            raise ValueError(f"Dataset {self.name} not supported for automatic loading.")
        # Build class to indices mapping
        self._build_class_to_indices()

    def _build_class_to_indices(self):
        """
        For the loaded dataset, build map from class labels to list of indices.
        """
        self.class_to_indices = {}
        for idx in range(len(self.dataset_obj)):
            _, label = self.dataset_obj[idx]
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

    def get_available_classes(self) -> List[int]:
        """
        Return list of class labels available in this dataset.
        """
        return list(self.class_to_indices.keys())

    def sample_classes(self, num_classes: int) -> List[int]:
        """
        Randomly sample 'num_classes' classes from available classes.
        """
        available = self.get_available_classes()
        assert len(available) >= num_classes, \
            f"Not enough classes to sample {num_classes} classes, only {len(available)} available."
        selected = random.sample(available, num_classes)
        return selected

    def sample_images_from_class(self, class_label: int, num_images: int) -> List[Tuple[torch.Tensor, int]]:
        """
        Sample 'num_images' images from the specified class.
        Returns list of (image_tensor, label).
        """
        indices = self.class_to_indices[class_label]
        assert len(indices) >= num_images, \
            f"Not enough images in class {class_label}; requested {num_images}, available {len(indices)}."
        selected_indices = random.sample(indices, num_images)
        images_list = []
        for idx in selected_indices:
            img, lbl = self.dataset_obj[idx]
            images_list.append((img, lbl))
        return images_list

    def get_dataset(self):
        """
        Return the underlying dataset object (for potential data loader).
        """
        return self.dataset_obj


class DatasetLoader:
    """
    Core class to manage loading of multiple datasets and episodic sampling.
    """
    def __init__(self, config: dict):
        """
        Initialize dataset loader based on configuration.
        """
        self.datasets_config = config['dataset']['datasets']
        self.transform = self._build_transform()
        self.datasets: List[DatasetWrapper] = []

        # Set seed for reproducibility if needed
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize dataset objects
        for ds_conf in self.datasets_config:
            name = ds_conf['name']
            split = ds_conf['split']
            base_path = ds_conf.get('path', './')  # Default path if not specified
            # For demo purposes, treat all datasets as directory-based
            dataset_obj = DatasetWrapper(name, split, base_path, self.transform)
            self.datasets.append(dataset_obj)

    def _build_transform(self):
        """
        Build image transforms according to CLIP model's preprocessing.
        """
        # CLIP requires resize to 224, center crop, normalize
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def load_data(self):
        """
        Placeholder for compatibility; datasets are loaded in init,
        so nothing more is needed here.
        """
        pass

    def sample_task(self, way: int, shot: int) -> Dict:
        """
        Sample one task: select one dataset and sample 'way' classes,
        with 'shot' support images per class, and query images from remaining.
        Returns a dict with support images/labels and query images/labels.
        """
        # Select a dataset at random
        dataset = random.choice(self.datasets)

        available_classes = dataset.get_available_classes()
        # Ensure enough classes
        assert len(available_classes) >= way, \
            f"Not enough classes in dataset {dataset.name} to sample {way} classes."

        # Sample classes
        selected_classes = random.sample(available_classes, way)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for class_idx, class_label in enumerate(selected_classes):
            # Sample support images
            support_samples = dataset.sample_images_from_class(class_label, shot)
            for img, lbl in support_samples:
                support_images.append(img)
                support_labels.append(class_idx)  # relabel classes 0..way-1

            # For query, sample a fixed number of images, e.g., 15
            # Here, for simplicity, use same number as support
            query_samples = dataset.sample_images_from_class(class_label, max(1, 15))
            # Remove support images to avoid duplication
            # But since we sample randomly, duplicates are unlikely; otherwise, handle explicitly
            for img, lbl in query_samples:
                query_images.append(img)
                query_labels.append(class_idx)

        # Convert lists to tensors
        support_images = support_images
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_images = query_images
        query_labels = torch.tensor(query_labels, dtype=torch.long)

        # Return as a dict
        return {
            'support_images': support_images,
            'support_labels': support_labels,
            'query_images': query_images,
            'query_labels': query_labels,
            'dataset_name': dataset.name,
            'class_mapping': selected_classes
        }
