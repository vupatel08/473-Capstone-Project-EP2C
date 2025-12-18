## dataset_loader.py
import os
import glob
import numpy as np
from PIL import Image
from typing import List
import random

class Dataset:
    """
    Dataset class for loading and preprocessing DIV2K validation images.

    Attributes:
        image_paths (List[str]): List of file paths for images.
        image_size (int): Target size (both height and width) for images.
        normalize (bool): Whether to normalize pixel values to [0, 1].
        augment (bool): Whether to apply data augmentation during loading.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self,
                 dataset_path: str,
                 split: str = 'validation',
                 image_size: int = 128,
                 normalize: bool = True,
                 augment: bool = False,
                 seed: int = 42):
        """
        Initialize the Dataset object with dataset parameters.

        Args:
            dataset_path (str): Path to the root directory of the dataset.
            split (str): Dataset split to use ('validation' or others). Default='validation'.
            image_size (int): Desired image size (height and width). Default=128.
            normalize (bool): Whether to normalize images to [0,1]. Default=True.
            augment (bool): Whether to apply data augmentation. Default=False.
            seed (int): Random seed for reproducibility. Default=42.
        """
        self.dataset_path = dataset_path
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        self.seed = seed
        self.image_paths = []  # Will be populated by load_data()

        # Set seed for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Load file paths
        self.load_data()

    def load_data(self):
        """
        Load the list of image file paths from dataset directory.
        Supports common image formats.
        """
        split_dir = os.path.join(self.dataset_path, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

        # Search for common image files
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(split_dir, ext)))

        # Sort for consistency
        self.image_paths = sorted(self.image_paths)

        if not self.image_paths:
            raise RuntimeError(f"No images found in {split_dir} with extensions {extensions}")

    def __len__(self):
        """
        Return total number of images in dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """
        Load and preprocess image at given index.

        Args:
            index (int): Index of the image.

        Returns:
            np.ndarray: Preprocessed image as a NumPy array with shape (H, W, C).
        """
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')  # Ensure RGB

        # Resize image to target size
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        img_np = np.array(img).astype(np.float32)

        # Normalize if required
        if self.normalize:
            img_np /= 255.0

        # Apply augmentation if enabled
        if self.augment:
            img_np = self.apply_augmentation(img_np)

        return img_np

    def apply_augmentation(self, img: np.ndarray):
        """
        Apply random augmentation: flip and rotation.

        Args:
            img (np.ndarray): Image array.

        Returns:
            np.ndarray: Augmented image.
        """
        # Set seed for reproducibility during augmentation in this instance
        random.seed(self.seed)

        # Random horizontal flip
        if random.random() < 0.5:
            img = np.fliplr(img)

        # Random vertical flip
        if random.random() < 0.5:
            img = np.flipud(img)

        # Random rotation: 0°, 90°, 180°, 270°
        rotations = [0, 1, 2, 3]
        k = random.choice(rotations)
        if k != 0:
            img = np.rot90(img, k)

        return img
