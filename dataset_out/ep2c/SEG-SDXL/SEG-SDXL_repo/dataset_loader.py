## dataset_loader.py
import os
from typing import Optional, List, Tuple, Union, Dict
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import random

class DatasetLoader(Dataset):
    """
    DatasetLoader loads datasets for training/evaluation of diffusion models.
    Supports unconditional and conditional datasets with optional prompts.
    """
    def __init__(
        self,
        dataset_path: str = "/path/to/dataset",
        image_size: Tuple[int, int] = (512, 512),
        dataset_type: str = "unconditional",  # 'unconditional' or 'conditional'
        dataset_name: str = "laion",  # 'cifar', 'ffhq', 'laion', etc.
        prompts_list: Optional[List[str]] = None,  # List of prompts if available
        conditioning_files: Optional[List[str]] = None,  # For masks, labels
        prompt_tokenizer=None,  # Optional tokenizer for text prompts
        split: str = "train"  # 'train' or 'test'
    ):
        """
        Initialize DatasetLoader.

        Args:
            dataset_path (str): Path to dataset folder or configuration.
            image_size (tuple): Target image size (H, W).
            dataset_type (str): 'unconditional' or 'conditional'.
            dataset_name (str): Dataset identifier ('cifar', 'ffhq', 'laion', etc.).
            prompts_list (list, optional): List of prompts for conditional datasets.
            conditioning_files (list, optional): List of file paths for conditioning data.
            prompt_tokenizer (callable, optional): Tokenizer to process prompts.
            split (str): Which split to load ('train' or 'test').

        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.dataset_type = dataset_type.lower()
        self.dataset_name = dataset_name.lower()
        self.prompts_list = prompts_list
        self.conditioning_files = conditioning_files
        self.prompt_tokenizer = prompt_tokenizer
        self.split = split

        # Define normalization transform (assuming model needs [-1, 1])
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Normalize to [-1, 1]
        ])

        # Load dataset based on dataset_name
        if self.dataset_name in ["cifar10", "cifar100"]:
            # Use torchvision datasets
            self.raw_dataset = datasets.CIFAR10(
                root=self.dataset_path,
                train=(split == "train"),
                download=True,
                transform=None  # We'll transform later
            )
        elif self.dataset_name == "ffhq":
            # Assuming images in a folder structure
            # e.g., dataset_path/ffhq_images/...
            image_folder = os.path.join(self.dataset_path, "ffhq_images")
            self.raw_dataset = datasets.ImageFolder(
                root=image_folder,
                transform=None
            )
        elif self.dataset_name == "laion":
            # For LAION: load image file paths and prompts if available
            # For simplicity, assume directory of images
            image_folder = os.path.join(self.dataset_path, "images")
            self.raw_dataset = datasets.ImageFolder(
                root=image_folder,
                transform=None
            )
        else:
            # Custom dataset: Expect image list and optional prompts
            # For extensibility, assume images are in dataset_path/images/
            image_dir = os.path.join(self.dataset_path, "images")
            if not os.path.exists(image_dir):
                raise ValueError(f"Image directory not found: {image_dir}")
            self.raw_dataset = datasets.ImageFolder(
                root=image_dir,
                transform=None
            )

        # For prompt and conditioning data, prepare lists
        if self.dataset_type == "conditional":
            if self.prompts_list is None:
                # If prompts are not provided, create dummy prompts
                self.prompts_list = [""] * len(self.raw_dataset)
            if self.conditioning_files is None:
                self.conditioning_files = [None] * len(self.raw_dataset)
        # Save dataset length
        self.dataset_size = len(self.raw_dataset)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Fetch preprocessed image and optional conditioning data.
        Returns:
            image_tensor: torch.FloatTensor of shape [3, H, W], normalized [-1,1]
            condition: Optional[str or tensor], if dataset is conditional
        """
        # Load image
        img_path, _ = self.raw_dataset.imgs[idx] if hasattr(self.raw_dataset, 'imgs') else (self.raw_dataset.samples[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)  # Resize, ToTensor, Normalize

        # Prepare output
        if self.dataset_type == "conditional":
            # Fetch prompt if available
            prompt = None
            if self.prompts_list:
                prompt = self.prompts_list[idx]
            # Fetch conditioning data if available
            cond_file = None
            if self.conditioning_files:
                cond_file = self.conditioning_files[idx]
            # For example, cond_file can be a segmentation mask path or label
            condition = None
            if cond_file:
                # Try load condition (e.g., mask)
                if cond_file.endswith(('.png', '.jpg', '.jpeg')):
                    condition_img = Image.open(cond_file).convert('L')  # grayscale mask
                    condition = self.transform(condition_img)
                elif cond_file.endswith('.pt'):
                    condition = torch.load(cond_file)
                else:
                    condition = cond_file  # fallback to path or string
            # Return image with prompt or condition
            return image, prompt if prompt is not None else condition
        else:
            # Unconditional: only image
            return image
