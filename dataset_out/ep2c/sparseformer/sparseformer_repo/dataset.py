## dataset.py
import os
from typing import Optional, Callable, List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageNetDataset(Dataset):
    """
    Dataset class for ImageNet-1K, supporting train and validation splits,
    with configurable augmentation and preprocessing pipelines.
    """
    def __init__(self, config: Dict, split: str = 'train'):
        """
        Args:
            config (Dict): Configuration dictionary parsed from config.yaml.
            split (str): 'train' or 'val'. Determines dataset split.
        """
        self.data_dir = config['dataset']['data_dir']
        self.image_size = config['dataset'].get('image_size', 224)
        self.num_workers = config['dataset'].get('num_workers', 8)
        self.augmentation = config['dataset'].get('augmentation', [])
        self.split = split.lower()

        # Determine root directory based on split
        if self.split == 'train':
            root_dir = os.path.join(self.data_dir, config['dataset']['train_split'])
            is_training = True
        elif self.split == 'val' or self.split == 'test':
            root_dir = os.path.join(self.data_dir, config['dataset']['val_split'])
            is_training = False
        else:
            raise ValueError(f"Unknown dataset split: {split}")

        # Setup transforms based on augmentation config
        self.transform = self._build_transform(is_training)

        # Initialize underlying dataset
        self.dataset = torchvision.datasets.ImageFolder(root=root_dir, transform=self.transform)

    def _build_transform(self, is_training: bool) -> Callable:
        """
        Build torchvision transforms pipeline based on augmentation configuration.
        """
        transform_list: List[Callable] = []

        # Parse augmentation settings
        augmentation_cfg = self.augmentation

        # For training, apply augmentation: random resized crop, flip, normalization
        if is_training:
            # Check if 'random_resized_crop' specified
            if any('random_resized_crop' in str(step) for step in augmentation_cfg):
                transform_list.append(transforms.RandomResizedCrop(self.image_size))
            else:
                transform_list.append(transforms.Resize(256))
                transform_list.append(transforms.RandomCrop(self.image_size))
            # Horizontal flip
            if any('horizontal_flip' in str(step) for step in augmentation_cfg):
                transform_list.append(transforms.RandomHorizontalFlip())
        else:
            # For validation/test
            transform_list.append(transforms.Resize(256))
            transform_list.append(transforms.CenterCrop(self.image_size))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalize
        norm_cfg = {}
        for step in augmentation_cfg:
            if isinstance(step, dict) and 'normalization' in step:
                norm_cfg = step['normalization']
                break
        if not norm_cfg:
            # Default normalization if not specified
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = norm_cfg.get('mean', [0.485, 0.456, 0.406])
            std = norm_cfg.get('std', [0.229, 0.224, 0.225])

        transform_list.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Return:
            image (Tensor): Transformed image tensor, shape [3, H, W]
            label (int): Class index label
        """
        image, label = self.dataset[index]
        return image, label
