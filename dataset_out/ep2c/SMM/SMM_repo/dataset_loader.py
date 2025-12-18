## dataset_loader.py
import os
import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

class DatasetLoader:
    """
    DatasetLoader loads datasets with specified configurations, applies resizing,
    normalization, and returns DataLoader objects for training and testing.
    """
    def __init__(self,
                 dataset_name: str = "CIFAR10",
                 input_size: Tuple[int, int] = (32, 32),
                 batch_size: int = 128,
                 seed: int = 42,
                 root_dir: str = "./data"):
        """
        Initialize DatasetLoader with dataset name, input size, batch size, seed, and root directory.
        """
        self.dataset_name = dataset_name
        self.input_size = input_size
        self.batch_size = batch_size
        self.seed = seed
        self.root_dir = root_dir

        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # Map dataset names to dataset classes and normalization params
        self._dataset_info = {
            "CIFAR10": {
                "class": datasets.CIFAR10,
                "normalization": ([0.4914, 0.4822, 0.4465],
                                  [0.2471, 0.2435, 0.2616]),
            },
            "CIFAR100": {
                "class": datasets.CIFAR100,
                "normalization": ([0.4914, 0.4822, 0.4465],
                                  [0.2471, 0.2435, 0.2616]),
            },
            "SVHN": {
                "class": datasets.SVHN,
                "normalization": ([0.4377, 0.4438, 0.4728],
                                  [0.1980, 0.2010, 0.1970]),
            },
            "GTSRB": {
                "class": datasets.ImageFolder,
                "normalization": ([0.3404, 0.3126, 0.3087],
                                  [0.2762, 0.2722, 0.2662]),
            },
            "Flowers102": {
                "class": datasets.ImageFolder,
                "normalization": ([0.4850, 0.4560, 0.4060],
                                  [0.2460, 0.2420, 0.2510]),
            },
            "DTD": {
                "class": datasets.ImageFolder,
                "normalization": ([0.5160, 0.4680, 0.4290],
                                  [0.2760, 0.2620, 0.2680]),
            },
            "UCF101": {
                "class": datasets.ImageFolder,
                "normalization": ([0.4321, 0.4165, 0.3859],
                                  [0.2466, 0.2384, 0.2432]),
            },
            "Food101": {
                "class": datasets.ImageFolder,
                "normalization": ([0.485, 0.456, 0.406],
                                  [0.246, 0.242, 0.251]),
            },
            "EuroSAT": {
                "class": datasets.ImageFolder,
                "normalization": ([0.436, 0.442, 0.448],
                                  [0.214, 0.209, 0.204]),
            },
            "OxfordPets": {
                "class": datasets.ImageFolder,
                "normalization": ([0.769, 0.672, 0.644],
                                  [0.232, 0.224, 0.229]),
            },
            "SUN397": {
                "class": datasets.ImageFolder,
                "normalization": ([0.473, 0.472, 0.437],
                                  [0.239, 0.239, 0.259]),
            }
        }

    def _get_transforms(self, normalization: Tuple[list, list], is_train: bool = True):
        """
        Compose transforms: resize, to Tensor, normalize.
        """
        mean, std = normalization
        transform_list = [
            transforms.Resize(self.input_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        return transforms.Compose(transform_list)

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load the dataset, apply transforms, and return train and test DataLoaders.
        """
        if self.dataset_name not in self._dataset_info:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")

        dataset_class = self._dataset_info[self.dataset_name]["class"]
        normalization = self._dataset_info[self.dataset_name]["normalization"]

        # Set dataset-specific root directory
        dataset_root = os.path.join(self.root_dir, self.dataset_name)

        # Define transforms
        transform_train = self._get_transforms(normalization, is_train=True)
        transform_test = self._get_transforms(normalization, is_train=False)

        # Load training dataset
        if self.dataset_name in ["CIFAR10", "CIFAR100", "SVHN"]:
            train_dataset = dataset_class(
                root=dataset_root,
                train=True,
                transform=transform_train,
                download=True
            )
            test_dataset = dataset_class(
                root=dataset_root,
                train=False,
                transform=transform_test,
                download=True
            )
        elif self.dataset_name in ["GTSRB", "Flowers102", "DTD", "UCF101", "Food101", "EuroSAT", "OxfordPets", "SUN397"]:
            # Assuming datasets are organized in a standard folder structure:
            train_dataset = dataset_class(
                root=dataset_root,
                split='train',
                transform=transform_train
            )
            test_dataset = dataset_class(
                root=dataset_root,
                split='test',
                transform=transform_test
            )
        else:
            # Default to ImageFolder for datasets not explicitly supported
            train_dataset = dataset_class(
                root=dataset_root,
                train=True,
                transform=transform_train
            )
            test_dataset = dataset_class(
                root=dataset_root,
                train=False,
                transform=transform_test
            )

        # DataLoader for train
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True
        )

        # DataLoader for test
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True
        )

        return train_loader, test_loader
