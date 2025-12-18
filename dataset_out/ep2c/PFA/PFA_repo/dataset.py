## dataset.py

import os
import glob
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A

# Disable warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

class PascalVOCScribbleDataset(Dataset):
    """
    Pascal VOC 2012 Dataset with Scribble Annotations for Weakly Supervised Semantic Segmentation.
    Loads images, scribble masks, and (optionally) ground truth labels.
    Supports data augmentation suitable for training, and can serve validation data without augmentation.
    """

    def __init__(
        self,
        data_dir: str = "path/to/VOC2012",
        scribble_dir: str = "path/to/ScribbleAnnotations",
        split: str = "train",
        image_size: int = 512,
        transforms: Optional[Callable] = None,
        is_train: bool = True,
        return_mask: bool = False,
        ignore_index: int = 255,
    ):
        """
        Args:
            data_dir (str): Root directory of VOC dataset.
            scribble_dir (str): Directory containing scribble annotation masks.
            split (str): 'train', 'val', or 'test'.
            image_size (int): Resize/crop size for images.
            transforms (callable, optional): Albumentations augmentation pipeline.
            is_train (bool): Flag indicating train or eval mode.
            return_mask (bool): During validation, whether to return ground truth labels.
            ignore_index (int): The label value to ignore during loss computation.
        """
        super().__init__()
        self.data_dir = data_dir
        self.scribble_dir = scribble_dir
        self.split = split
        self.image_size = image_size
        self.transforms = transforms
        self.is_train = is_train
        self.return_mask = return_mask
        self.ignore_index = ignore_index

        # Initialize list of image IDs based on Pascal VOC structure
        self.image_ids = self._init_image_ids()

        # Define image and annotation paths
        self.image_paths = self._get_image_paths()
        self.scribble_paths = self._get_scribble_paths()
        self.gt_paths = self._get_gt_paths() if return_mask else None

        assert len(self.image_paths) == len(self.scribble_paths), \
            "Mismatch between images and scribble annotations"

        if self.return_mask:
            assert len(self.image_paths) == len(self.gt_paths), \
                "Mismatch between images and ground truth labels"

        # Define albumentations pipeline for training data
        if self.is_train:
            self.augmentation = A.Compose([
                A.Resize(height=self.image_size, width=self.image_size),
                A.RandomScale(scale_limit=(0.5, 2.0), p=1.0),
                A.Rotate(limit=10, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.RandomCrop(height=self.image_size, width=self.image_size),
            ])
        else:
            # For validation or inference, only resize for consistency
            self.augmentation = A.Compose([
                A.Resize(height=self.image_size, width=self.image_size),
            ])

    def _init_image_ids(self) -> List[str]:
        """
        Initialize list of image IDs based on the dataset split.
        Assumes standard Pascal VOC directory structure.
        """
        split_file = os.path.join(self.data_dir, "ImageSets", "Segmentation", f"{self.split}.txt")
        with open(split_file, 'r') as f:
            ids = [line.strip() for line in f.readlines()]
        return ids

    def _get_image_paths(self) -> List[str]:
        """
        Get list of image file paths for the dataset split.
        Assumes images stored under 'JPEGImages' folder.
        """
        image_paths = []
        for img_id in self.image_ids:
            img_path = os.path.join(self.data_dir, "JPEGImages", f"{img_id}.jpg")
            if os.path.exists(img_path):
                image_paths.append(img_path)
            else:
                # Alternatively, handle other image extensions if needed
                img_path = os.path.join(self.data_dir, "JPEGImages", f"{img_id}.png")
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                else:
                    raise FileNotFoundError(f"Image file not found for ID {img_id}")
        return image_paths

    def _get_scribble_paths(self) -> List[str]:
        """
        Get list of scribble mask paths.
        Assumes scribble masks stored under scribble_dir with same filenames as images.
        """
        scribble_paths = []
        for img_path in self.image_paths:
            filename = os.path.basename(img_path).replace('.jpg', '').replace('.png', '')
            scribble_path = os.path.join(self.scribble_dir, self.split, f"{filename}.png")
            if os.path.exists(scribble_path):
                scribble_paths.append(scribble_path)
            else:
                raise FileNotFoundError(f"Scribble annotation not found for {filename}")
        return scribble_paths

    def _get_gt_paths(self) -> List[str]:
        """
        Get ground truth label paths for validation.
        Assumes stored under 'SegmentationClass' folder.
        """
        gt_paths = []
        for img_path in self.image_paths:
            filename = os.path.basename(img_path).replace('.jpg', '').replace('.png', '')
            gt_path = os.path.join(self.data_dir, 'SegmentationClass', f"{filename}.png")
            if os.path.exists(gt_path):
                gt_paths.append(gt_path)
            else:
                raise FileNotFoundError(f"Ground truth not found for {filename}")
        return gt_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # Load image
        img_path = self.image_paths[index]
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load scribble mask
        scribble_path = self.scribble_paths[index]
        scribble_mask_img = np.array(Image.open(scribble_path))
        # Convert scribble mask image to class labels
        # Assuming scribble masks are stored as PNG with class IDs per pixel
        # For unlabeled pixels, assume pixel value == 255 (common practice)
        label_mask = np.full_like(scribble_mask_img, self.ignore_index, dtype=np.int64)
        # Mask out ignored pixels
        label_mask[scribble_mask_img != 255] = scribble_mask_img[scribble_mask_img != 255]

        # For validation, load full ground truth labels if needed
        gt_label = None
        if self.return_mask:
            gt_path = self.gt_paths[index]
            gt_mask_np = np.array(Image.open(gt_path))
            gt_label = gt_mask_np.astype(np.int64)

        # Convert to dict for augmentation
        augmented_data = self.augmentation(image=image, mask=label_mask)

        img_tensor = augmented_data['image'].astype(np.float32) / 255.0  # normalize to [0,1]
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).contiguous()  # (C,H,W)

        label_tensor = torch.from_numpy(augmented_data['mask']).long()

        sample = {
            'image': img_tensor,
            'label': label_tensor,
            'image_path': img_path,
        }

        if self.return_mask:
            sample['ground_truth'] = torch.from_numpy(gt_label)

        return sample

# Note:
# - The class expects that your dataset directory has the standard Pascal VOC structure with
#   'JPEGImages/', 'ImageSets/Segmentation/', 'SegmentationClass/', and separate scribble annotation directories.
# - The file paths and labels should be consistent with the experiment setup.
# - For actual use, consider adding __repr__() and optional caching for efficiency.

