## dataset_loader.py
import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
import torch

class DatasetLoader(Dataset):
    def __init__(self, dataset_path: str, dataset_name: str = 'ExDark', 
                 input_size: int = 608, 
                 train_split_ratio: float = 0.8, 
                 val_split_ratio: float = 0.2, 
                 synthetic_illumination: bool = True,
                 augmentation: dict = None,
                 mode: str = 'train'):
        """
        Initialize DatasetLoader.
        Args:
            dataset_path (str): Path to dataset directory.
            dataset_name (str): Name of dataset ('ExDark', 'DarkFace', 'LIS').
            input_size (int): Size to resize images (e.g., 608).
            train_split_ratio (float): Split ratio for train.
            val_split_ratio (float): Split ratio for validation.
            synthetic_illumination (bool): Whether to generate paired images for II Loss.
            augmentation (dict): Augmentation parameters.
            mode (str): 'train', 'val', or 'test'.
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.input_size = input_size
        self.synthetic_illumination = synthetic_illumination
        self.augmentation = augmentation if augmentation is not None else {}
        self.mode = mode

        # Storage for file paths and labels
        self.image_paths = []
        self.annotation_data = []

        # Load dataset based on name
        self._load_dataset()

        # Generate indices for split
        self._split_dataset(train_split_ratio, val_split_ratio)

        # For quick access
        self.data_indices = self.train_indices if mode=='train' else self.val_indices

        # Basic augmentation defaults if not present
        self.flip = self.augmentation.get('flip', True)
        self.scale_range = self.augmentation.get('scale', [0.8, 1.2])
        self.crop = self.augmentation.get('crop', True)
        self.brightness_range = self.augmentation.get('brightness_adjustment', [0.5, 1.5])

    def _load_dataset(self):
        """
        Loads dataset image paths and annotations depending on dataset_name.
        Assumes standardized folder structures.
        """
        if self.dataset_name == 'ExDark':
            self._load_exdark()
        elif self.dataset_name == 'DarkFace' or self.dataset_name == 'UG2+DARK FACE':
            self._load_darkface()
        elif self.dataset_name == 'LIS':
            self._load_lis()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _load_exdark(self):
        """
        Load ExDark dataset: expects images and annotation files.
        Assumes dataset structure:
          - images: in dataset_path/images/
          - annotations: in dataset_path/annotations/
        """
        images_dir = os.path.join(self.dataset_path, 'images')
        annotations_dir = os.path.join(self.dataset_path, 'annotations')
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg','.png','.bmp'))])
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            ann_path = os.path.join(annotations_dir, img_file.replace('.jpg','.txt').replace('.png','.txt').replace('.bmp','.txt'))
            if os.path.exists(ann_path):
                self.image_paths.append(img_path)
                self.annotation_data.append(self._parse_annotation_exdark(ann_path))
            else:
                # Skip images without annotations
                continue

    def _load_darkface(self):
        """
        Load DarkFace / UG2+DARK FACE datasets: expects annotations in standard format.
        """
        images_dir = os.path.join(self.dataset_path, 'images')
        annotations_dir = os.path.join(self.dataset_path, 'annotations')
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg','.png'))])
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            ann_path = os.path.join(annotations_dir, img_file.replace('.jpg','.txt').replace('.png','.txt'))
            if os.path.exists(ann_path):
                self.image_paths.append(img_path)
                self.annotation_data.append(self._parse_annotation_darkface(ann_path))
            else:
                continue

    def _load_lis(self):
        """
        Load LIS dataset: masks and detection annotations.
        Expect directory structure with images and mask files.
        """
        images_dir = os.path.join(self.dataset_path, 'images')
        mask_dir = os.path.join(self.dataset_path, 'masks')
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg','.png'))])
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file.replace('.jpg','.png').replace('.png','.png'))
            if os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.annotation_data.append(self._parse_annotation_lis(mask_path))
            else:
                continue

    def _parse_annotation_exdark(self, ann_path):
        """
        Parse annotation file for ExDark dataset:
        Assumes a format: each line: class_idx xmin ymin xmax ymax
        """
        boxes = []
        classes = []
        with open(ann_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_idx = int(parts[0])
                xmin = float(parts[1])
                ymin = float(parts[2])
                xmax = float(parts[3])
                ymax = float(parts[4])
                boxes.append([xmin, ymin, xmax, ymax])
                classes.append(cls_idx)
        return {'boxes': np.array(boxes), 'labels': np.array(classes)}

    def _parse_annotation_darkface(self, ann_path):
        """
        Parse annotation for DarkFace/UG2 detection:
        Similar format as above.
        """
        boxes = []
        classes = []
        with open(ann_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_idx = int(parts[0])
                xmin = float(parts[1])
                ymin = float(parts[2])
                xmax = float(parts[3])
                ymax = float(parts[4])
                boxes.append([xmin, ymin, xmax, ymax])
                classes.append(cls_idx)
        return {'boxes': np.array(boxes), 'labels': np.array(classes)}

    def _parse_annotation_lis(self, mask_path):
        """
        For LIS, annotations include segmentation masks and bounding boxes.
        For simplicity, we'll only parse bounding boxes from mask contours.
        """
        # Placeholder implementation; assumes mask is binary.
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x + w, y + h])
        return {'boxes': np.array(boxes), 'labels': np.ones(len(boxes), dtype=int)}  # single class

    def _split_dataset(self, train_ratio, val_ratio):
        """
        Shuffle and split dataset indices into train and validation.
        """
        total_indices = list(range(len(self.image_paths)))
        random.shuffle(total_indices)
        train_end = int(len(total_indices) * train_ratio)
        val_end = train_end + int(len(total_indices) * val_ratio)
        self.train_indices = total_indices[:train_end]
        self.val_indices = total_indices[train_end:val_end]

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        """
        Fetch data point at index; apply preprocessing and augmentation.
        Returns dictionary with image tensor, labels, optional paired image, and metadata.
        """
        real_index = self.data_indices[index]
        img_path = self.image_paths[real_index]
        ann = self.annotation_data[real_index]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Load annotations
        boxes = ann['boxes']  # Nx4 numpy array
        labels = ann['labels']  # N numpy array

        # Apply augmentations
        image, boxes = self._apply_augmentations(image, boxes)

        # Resize image, adjust boxes accordingly
        original_size = image.shape[:2]
        image, scale_x, scale_y = self._resize_image(image, self.input_size)
        boxes = self._scale_boxes(boxes, scale_x, scale_y)

        # Convert image to tensor
        image_tensor = torch.from_numpy(image).permute(2,0,1).float()

        # Generate synthetic illumination variation if needed
        if self.mode=='train' and self.synthetic_illumination:
            pair_image = self._apply_synthetic_brightness(image)
        else:
            pair_image = None

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'original_size': original_size,
            'filename': os.path.basename(img_path),
        }

        return {
            'image': image_tensor,
            'targets': target,
            'pair_image': pair_image,
            'metadata': {
                'original_size': original_size,
                'filename': os.path.basename(img_path)
            }
        }

    def _apply_augmentations(self, image, boxes):
        """
        Randomly apply geometric and photometric augmentations.
        """
        # Geometric augmentations
        if self.flip and random.random() < 0.5:
            image = cv2.flip(image, 1)
            boxes[:, [0,2]] = image.shape[1] - boxes[:, [2,0]]
        scale_factor = random.uniform(*self.scale_range)
        if scale_factor != 1.0:
            h, w = image.shape[:2]
            new_w, new_h = int(w*scale_factor), int(h*scale_factor)
            image = cv2.resize(image, (new_w, new_h))
            boxes *= scale_factor
        # Random crop to original size
        if self.crop:
            image, boxes = self._random_crop(image, boxes)
        # Photometric augmentation: brightness
        brightness_factor = random.uniform(*self.brightness_range)
        image = self._adjust_brightness(image, brightness_factor)
        return image, boxes

    def _resize_image(self, image, size):
        """
        Resize image to size x size, scale boxes accordingly.
        """
        h, w = image.shape[:2]
        image_resized = cv2.resize(image, (size, size))
        scale_x = size / w
        scale_y = size / h
        return image_resized, scale_x, scale_y

    def _scale_boxes(self, boxes, scale_x, scale_y):
        """
        Scale bounding boxes according to resize.
        """
        boxes[:, [0,2]] *= scale_x
        boxes[:, [1,3]] *= scale_y
        return boxes

    def _random_crop(self, image, boxes):
        """
        Random crop within the image, maintaining bounding boxes if possible.
        """
        h, w = image.shape[:2]
        crop_ratio = random.uniform(0.8, 1.0)
        new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
        x_start = random.randint(0, w - new_w)
        y_start = random.randint(0, h - new_h)
        crop_img = image[y_start:y_start+new_h, x_start:x_start+new_w]
        # Adjust boxes
        boxes[:, [0,2]] -= x_start
        boxes[:, [1,3]] -= y_start
        # Clip boxes
        boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, new_w)
        boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, new_h)
        return crop_img, boxes

    def _adjust_brightness(self, image, factor):
        """
        Adjust image brightness via gamma correction.
        """
        # avoid division by zero
        gamma = factor if factor > 0 else 0.1
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype(np.uint8)
        image_uint8 = (image * 255).astype(np.uint8)
        adjusted = cv2.LUT(image_uint8, table)
        return adjusted.astype(np.float32) / 255.0

    def _apply_synthetic_brightness(self, image):
        """
        Generate a complementary image with randomized gamma for II Loss.
        """
        gamma_factor = random.uniform(0.5, 1.5)
        return self._adjust_brightness(image, gamma_factor)

