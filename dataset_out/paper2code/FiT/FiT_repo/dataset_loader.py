## dataset_loader.py
import os
from PIL import Image
import torchvision.transforms as T
import torchvision.datasets as datasets
from typing import Tuple, Optional

class DatasetLoader:
    def __init__(
        self,
        dataset_path: str = "./imagenet",
        resolution: Tuple[int, int] = (256, 256),
        split: str = "train",
        short_side: int = 256,
        max_resolution_constraint: Tuple[int, int] = (256, 256),
        is_training: bool = True
    ):
        """
        Initializes the DatasetLoader for ImageNet.

        Args:
            dataset_path (str): Root directory of ImageNet dataset.
            resolution (Tuple[int, int]): Target resolution (H, W) for evaluation; used for center cropping.
            split (str): 'train' or 'val' (validation).
            short_side (int): The size the shortest side will be resized to during training.
            max_resolution_constraint (Tuple[int, int]): Max allowed H x W for images, used during evaluation.
            is_training (bool): Whether dataset is for training or evaluation.
        """
        self.dataset_path = dataset_path
        self.resolution = resolution
        self.split = split
        self.short_side = short_side
        self.max_res = max_resolution_constraint
        self.is_training = is_training

        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(os.path.join(self.dataset_path, "ILSVRC") or ""):
            # For torchvision.datasets.ImageNet, the structure is /train, /val
            split = self.split
        else:
            split = self.split

        # Select dataset split
        dataset = datasets.ImageNet(
            root=self.dataset_path,
            split=split,
            transform=None  # We set transforms later
        )
        return dataset

    def _get_transform(self):
        """
        Defines the transformation pipeline based on whether it's training or evaluation.
        """
        transforms_list = []

        if self.is_training:
            # For training:
            # Resize so that the shortest side == self.short_side, aspect ratio preserved
            resize_transform = T.Resize(
                size=self._get_resize_size(train=True),
                interpolation=Image.BICUBIC
            )
            transforms_list.append(resize_transform)

            # Random crop to self.resolution (H, W)
            transforms_list.append(T.RandomCrop(self.resolution))
            # Random horizontal flip
            transforms_list.append(T.RandomHorizontalFlip())
        else:
            # For evaluation:
            # Resize to meet max resolution constraints while maintaining aspect ratio
            resize_transform = T.Resize(
                size=self._get_resize_size(train=False),
                interpolation=Image.BICUBIC
            )
            transforms_list.append(resize_transform)

            # Center crop to self.resolution
            transforms_list.append(T.CenterCrop(self.resolution))
        
        # Convert to tensor
        transforms_list.append(T.ToTensor())
        # Normalize to [-1, 1] (assuming diffusion model normalization)
        transforms_list.append(T.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5]))
        return T.Compose(transforms_list)

    def _get_resize_size(self, train: bool) -> Tuple[int, int]:
        """
        Computes the resize size for aspect ratio preservation.

        For training:
            - Resize so that the shortest side == self.short_side
        For evaluation:
            - Resize so that the longest side <= max_res, keeping aspect ratio
        """
        # Access original image size via dataset
        # For batch processing, we do this per sample; but in dataset init, we cannot.
        # Instead, we assume images are kept at full size, and transforms handle resizing dynamically.
        # Thus, return a size tuple for resize.

        # However, in torchvision, Resize's size argument can be int or Tuple[int, int].
        # For aspect ratio preservation, setting size=int (short side) scales accordingly.
        # Alternatively, we can set size=int for Resize, which scales so that shorter side == size.

        # Because in torchvision.transforms.Resize, size can be:
        # - int: resize the shorter side to size, keep aspect ratio
        # - tuple: exact size (may distort aspect ratio)
        # We will use int, which keeps aspect ratio.

        # For training:
        if train:
            # Resize so that shortest side == self.short_side
            return self.short_side
        else:
            # Resize so that the longer side <= max_res[0]/[1]
            # For evaluation, set size to match max resolution constraint if needed
            # We want to ensure that resized images do not exceed max resolution
            # For simplicity, set size to max_res's maximum dimension
            max_dim = max(self.max_res)
            return max_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        """
        Returns a dict containing the preprocessed image tensor and label.
        """
        sample = self.dataset[index]
        image, label = sample

        # Apply the selected transforms
        transform = self._get_transform()
        image = transform(image)

        return {
            'image': image,
            'label': label
        }
