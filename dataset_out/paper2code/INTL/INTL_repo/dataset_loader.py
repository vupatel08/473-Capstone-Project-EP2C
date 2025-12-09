## dataset_loader.py
import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

class MultiViewTransform(Dataset):
    """
    Dataset wrapper that, given an underlying dataset, applies data augmentations
    to generate multiple views per sample, suitable for SSL training.
    """
    def __init__(self, base_dataset: Dataset, num_views: int, augmentation_params: dict):
        """
        Args:
            base_dataset (Dataset): dataset object from torchvision.datasets
            num_views (int): number of augmented views to generate per sample
            augmentation_params (dict): dictionary containing augmentation parameters
        """
        self.base_dataset = base_dataset
        self.num_views = num_views
        self.aug_params = augmentation_params

        # Build the augmentation pipeline based on params
        self.transform = self._build_transform()

    def _build_transform(self):
        # Compose transforms based on augmentation_params
        transform_list = []
        # RandomResizedCrop with scale range
        crop_size = self.aug_params.get('crop_size', 224)
        scale_min = self.aug_params.get('crop_scale_min', 0.08)
        scale_max = self.aug_params.get('crop_scale_max', 1.0)
        transform_list.append(
            transforms.RandomResizedCrop(crop_size, scale=(scale_min, scale_max))
        )
        # Horizontal Flip
        p_flip = self.aug_params.get('horizontal_flip_prob', 0.5)
        transform_list.append(transforms.RandomHorizontalFlip(p=p_flip))
        # Color jitter
        brightness = self.aug_params.get('brightness', 0.4)
        contrast = self.aug_params.get('contrast', 0.4)
        saturation = self.aug_params.get('saturation', 0.2)
        hue = self.aug_params.get('hue', 0.1)
        color_jitter_prob = self.aug_params.get('color_jitter_prob', 0.8)
        if color_jitter_prob > 0:
            transform_list.append(
                transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                       saturation=saturation, hue=hue)
            )
        # ToTensor
        transform_list.append(transforms.ToTensor())

        # Additional augmentations: Gaussian noise, solarization, if specified
        # For simplicity, implement Gaussian noise as a modality
        class GaussianNoise:
            def __init__(self, mean=0.0, std=0.1):
                self.mean = mean
                self.std = std

            def __call__(self, tensor):
                return tensor + torch.randn_like(tensor) * self.std + self.mean

        gaussian_prob = self.aug_params.get('gaussian_prob', 0.0)
        if gaussian_prob > 0:
            transform_list.append(GaussianNoise())

        # Solarization (Invert pixel values above threshold), optional
        solar_prob = self.aug_params.get('solarization_prob', 0.0)
        if solar_prob > 0:
            class Solarization:
                def __init__(self, p=1.0, threshold=128):
                    self.p = p
                    self.threshold = threshold

                def __call__(self, img):
                    if torch.rand(1).item() < self.p:
                        img_temp = TF.to_pil_image(img)
                        img_temp = TF.invert(img_temp)
                        return TF.to_tensor(img_temp)
                    return img

            transform_list.append(Solarization(p=solar_prob))
        # Compose the transforms
        return transforms.Compose(transform_list)

    def __getitem__(self, index):
        """
        For each sample, generate 'num_views' augmented versions.
        Returns a list of tensors.
        """
        original_img, label = self.base_dataset[index]
        views = []
        for _ in range(self.num_views):
            augmented_img = self.transform(original_img)
            views.append(augmented_img)
        # Return all views as a tuple (view1, view2, ...)
        return tuple(views)

    def __len__(self):
        return len(self.base_dataset)

class DatasetLoader:
    """
    Responsible for loading datasets according to configuration, applying
    augmentation pipelines, and providing datasets for DataLoader.
    """
    def __init__(self, dataset_name: str = 'ImageNet-100', dataset_params: dict = None):
        """
        Args:
            dataset_name (str): String identifier of dataset ('CIFAR10', 'CIFAR100', 'ImageNet-100')
            dataset_params (dict): Parameters including crop sizes, augmentation params
        """
        self.dataset_name = dataset_name
        self.dataset_params = dataset_params if dataset_params is not None else {}
        # Placeholder for datasets
        self.train_dataset = None
        self.val_dataset = None

    def load_data(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        Loads dataset based on the configuration and applies data augmentation.
        Returns:
            train_dataset (Dataset): Dataset object with multi-view augmentation if needed
            val_dataset (Dataset): Validation dataset
        """
        dataset_type = self.dataset_params.get('dataset_type', 'image_classification')
        # Build transforms for training
        train_transform = self._build_transform(is_train=True)
        # Build transforms for validation (usually just ToTensor and normalization)
        val_transform = self._build_transform(is_train=False)

        if self.dataset_name.lower() == 'cifar10':
            root = os.path.expanduser('~/.cache/torch/datasets')
            base_train = datasets.CIFAR10(root=root, train=True, download=True)
            base_val = datasets.CIFAR10(root=root, train=False, download=True)
            # Wrap with MultiViewTransform for training (e.g., generate 2 views)
            num_views = self.dataset_params.get('total_crops', 2)
            self.train_dataset = MultiViewTransform(base_train, num_views, self.dataset_params.get('augmentation_params', {}))
            self.val_dataset = datasets.CIFAR10(root=root, train=False, transform=val_transform, download=False)
        elif self.dataset_name.lower() == 'cifar100':
            root = os.path.expanduser('~/.cache/torch/datasets')
            base_train = datasets.CIFAR100(root=root, train=True, download=True)
            base_val = datasets.CIFAR100(root=root, train=False, download=True)
            num_views = self.dataset_params.get('total_crops', 2)
            self.train_dataset = MultiViewTransform(base_train, num_views, self.dataset_params.get('augmentation_params', {}))
            self.val_dataset = datasets.CIFAR100(root=root, train=False, transform=val_transform, download=False)
        elif self.dataset_name.lower() == 'imagenet-100':
            # Assume dataset is organized in folders in a path
            # For small datasets, datasets.ImageFolder is common
            root_path = self.dataset_params.get('dataset_path', './imagenet-100/')
            # For training, apply augmentation
            self.train_dataset = datasets.ImageFolder(root=os.path.join(root_path, 'train'), transform=train_transform)
            self.val_dataset = datasets.ImageFolder(root=os.path.join(root_path, 'val'), transform=val_transform)
        else:
            raise ValueError(f"Unsupported dataset {self.dataset_name}")

        return self.train_dataset, self.val_dataset

    def _build_transform(self, is_train: bool):
        """
        Build validation or training transform
        """
        crop_size = self.dataset_params.get('crop_size', 224)
        augmentation_params = self.dataset_params.get('augmentation_params', {})

        if is_train:
            # Augmentation pipeline
            transform_list = []
            scale_min = self.dataset_params.get('crop_scale_min', 0.08)
            scale_max = self.dataset_params.get('crop_scale_max', 1.0)
            transform_list.append(
                transforms.RandomResizedCrop(crop_size, scale=(scale_min, scale_max))
            )
            p_flip = augmentation_params.get('horizontal_flip_prob', 0.5)
            transform_list.append(transforms.RandomHorizontalFlip(p=p_flip))
            brightness = augmentation_params.get('brightness', 0.4)
            contrast = augmentation_params.get('contrast', 0.4)
            saturation = augmentation_params.get('saturation', 0.2)
            hue = augmentation_params.get('hue', 0.1)
            transform_list.append(
                transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                       saturation=saturation, hue=hue)
            )
            # Add normalization for ImageNet if applicable
            if self.dataset_name.lower().startswith('imagenet'):
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                transform_list.append(transforms.ToTensor())
                transform_list.append(transforms.Normalize(mean=mean, std=std))
            else:
                transform_list.append(transforms.ToTensor())

            return transforms.Compose(transform_list)
        else:
            # Validation transform: just resize/crop and ToTensor
            transform_list = []
            transform_list.append(transforms.Resize(crop_size + 32))
            transform_list.append(transforms.CenterCrop(crop_size))
            transform_list.append(transforms.ToTensor())
            if self.dataset_name.lower().startswith('imagenet'):
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                transform_list.append(transforms.Normalize(mean=mean, std=std))
            return transforms.Compose(transform_list)
