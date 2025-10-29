"""dataset_loader.py

This module provides the DatasetLoader class responsible for downloading and preprocessing the
MNIST dataset as per the experimental setup. It applies the necessary transforms to normalize
the images to [0, 1] and flatten them into 784-dimensional vectors, and creates DataLoader objects
for both training and testing sets.
"""

import os
import logging
from typing import Tuple, Any, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DatasetLoader:
    """DatasetLoader handles downloading, preprocessing, and loading the MNIST dataset.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary with training and dataset parameters.
        batch_size (int): Batch size for DataLoader.
        dataset_name (str): Name of the dataset (default "MNIST").
        data_dir (str): Directory path to store/download the dataset.
        transform (transforms.Compose): Transformation pipeline to apply on the dataset.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the DatasetLoader with provided configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary (typically loaded from config.yaml).
        """
        self.config = config
        self.batch_size: int = config.get("training", {}).get("batch_size", 128)
        self.dataset_name: str = config.get("dataset", {}).get("name", "MNIST")
        # Optional: allow user to specify a custom data directory, defaulting to "./data"
        self.data_dir: str = config.get("dataset", {}).get("data_dir", "./data")
        self.transform = self._build_transforms()

        # Set up logging
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
        logging.info("DatasetLoader initialized with batch_size: %d, dataset: %s, data_dir: %s",
                     self.batch_size, self.dataset_name, self.data_dir)

    def _build_transforms(self) -> transforms.Compose:
        """Constructs the transformation pipeline for the dataset.

        Returns:
            transforms.Compose: A composed transform that converts images to tensors and flattens them.
        """
        # Convert image to tensor (automatically scales pixel values to [0,1]),
        # then flatten the 28x28 image tensor to a 784-dimensional vector.
        flatten_transform = transforms.Lambda(lambda x: x.view(-1))
        transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
            flatten_transform
        ])
        return transform_pipeline

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Loads the MNIST dataset and returns DataLoader objects for training and testing.

        Returns:
            Tuple[DataLoader, DataLoader]: (train_dataloader, test_dataloader)
        """
        # Download and instantiate the MNIST training and test datasets with the defined transform.
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=True
        )

        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=True
        )

        # Create DataLoader objects with the specified batch size
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        logging.info("MNIST dataset loaded: %d training samples, %d test samples",
                     len(train_dataset), len(test_dataset))
        return train_loader, test_loader
