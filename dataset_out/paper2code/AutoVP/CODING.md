# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
from typing import List, Dict, Optional, Callable, Union
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import json
import csv
import zipfile
import urllib.request
import io

# Optional: For custom datasets
# from torchvision.datasets import VisionDataset

# Define custom dataset class for segmentation
class SegmentationDataset(Dataset):
    def __init__(self, image_paths: List[str], mask_paths: List[str],
                 transform: Optional[Callable]=None, target_transform: Optional[Callable]=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        # Load mask
        mask = Image.open(self.mask_paths[idx])
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
        return {'image': img, 'mask': mask}

# Define custom dataset class for detection (with bounding boxes)
class DetectionDataset(Dataset):
    def __init__(self, image_paths: List[str], annotations: List[dict],
                 transform: Optional[Callable]=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        ann = self.annotations[idx]
        # Assume ann contains 'boxes' and 'labels'
        if self.transform:
            img = self.transform(img)
        boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.tensor(ann['labels'], dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}
        return {'image': img, 'target': target}

# Utility function to download and parse dataset annotations (placeholder)
def download_annotations(url: str, file_type: str) -> List[dict]:
    # Implementation depends on dataset annotation format
    # For simplicity, return empty list
    return []

# Main DatasetLoader class
class DatasetLoader:
    def __init__(self, datasets_list: List[str], batch_size: int,
                 transforms_fn: Optional[Callable]=None,
                 dataset_root: Optional[str]=None,
                 seed: int=42):
        self.datasets_list = datasets_list
        self.batch_size = batch_size
        self.transforms_fn = transforms_fn
        self.dataset_root = dataset_root or './datasets'
        self.seed = seed
        # Store datasets after loading
        self.loaded_datasets = {}

        # Set torch seed for reproducibility
        torch.manual_seed(self.seed)

        # Mapping dataset names to loader functions
        self.dataset_map = {
            'CIFAR10': self._load_cifar10,
            'CIFAR100': self._load_cifar100,
            'Flowers102': self._load_flowers102,
            'ISIC': self._load_isic,
            'SVHN': self._load_svhn,
            'GTSRB': self._load_gtsrb,
            'Food101': self._load_food101,
            'EuroSAT': self._load_eurosat,
            'OxfordIIITPet': self._load_oxford_pet,
            'UCF101': self._load_ucf101,
            'FMoW': self._load_fmow,
            'DTD': self._load_dtd,
            # Add more datasets here
        }

    def load_data(self) -> Dict[str, Dict[str, DataLoader]]:
        """
        Load all specified datasets, create train/val/test DataLoaders.
        Returns:
            dict: {dataset_name: {'train': loader, 'val': loader, 'test': loader}}
        """
        data_dict = {}
        for ds_name in self.datasets_list:
            if ds_name not in self.dataset_map:
                print(f"Dataset {ds_name} not supported.")
                continue
            print(f"Loading dataset: {ds_name}")
            train_ds, val_ds, test_ds = self.dataset_map[ds_name]()
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
            data_dict[ds_name] = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }
        return data_dict

    def _load_cifar10(self):
        root = os.path.join(self.dataset_root, 'CIFAR10')
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        train_transform = transforms.Compose([
            transforms.Resize(int(32 * self.get_scale_factor())),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        test_transform = train_transform  # same for simplicity
        train_ds = datasets.CIFAR10(root, train=True, download=True, transform=train_transform)
        val_ds = datasets.CIFAR10(root, train=False, download=True, transform=test_transform)
        test_ds = datasets.CIFAR10(root, train=False, download=True, transform=test_transform)
        return train_ds, val_ds, test_ds

    def _load_cifar100(self):
        root = os.path.join(self.dataset_root, 'CIFAR100')
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        transform = transforms.Compose([
            transforms.Resize(int(32 * self.get_scale_factor())),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.507, 0.507],
                                 std=[0.226, 0.226, 0.226])
        ])
        train_ds = datasets.CIFAR100(root, train=True, download=True, transform=transform)
        val_ds = datasets.CIFAR100(root, train=False, download=True, transform=transform)
        test_ds = datasets.CIFAR100(root, train=False, download=True, transform=transform)
        return train_ds, val_ds, test_ds

    def _load_flowers102(self):
        root = os.path.join(self.dataset_root, 'Flowers102')
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        # torchvision.datasets.Flowers102 doesn't exist in torchvision; assume custom or downloadable
        # For placeholder, use ImageFolder with dataset available locally
        # Here we assume dataset is downloaded and unzipped in root
        transform = transforms.Compose([
            transforms.Resize(int(224 * self.get_scale_factor())),  # default size 224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(root=os.path.join(root, 'images'), transform=transform)
        # Split into train/val/test
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split1 = int(0.8 * dataset_size)
        train_idx, val_idx, test_idx = indices[:split1], indices[split1:int(0.9*dataset_size)], indices[int(0.9*dataset_size):]

        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)
        test_ds = torch.utils.data.Subset(dataset, test_idx)
        return train_ds, val_ds, test_ds

    def _load_isic(self):
        # Custom dataset load: assuming images and masks are stored in specific directories
        root = os.path.join(self.dataset_root, 'ISIC')
        images_dir = os.path.join(root, 'images')
        masks_dir = os.path.join(root, 'masks')
        image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')])
        mask_paths = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.png') or f.endswith('.jpg')])

        # For simplicity, split based on sorted list
        total = len(image_paths)
        train_end = int(0.8 * total)
        train_images = image_paths[:train_end]
        val_images = image_paths[train_end:]
        train_masks = mask_paths[:train_end]
        val_masks = mask_paths[train_end:]

        # Define transforms
        resize_scale = self.get_scale_factor() * 224  # assuming original size
        img_transform = transforms.Compose([
            transforms.Resize((int(resize_scale), int(resize_scale))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((int(resize_scale), int(resize_scale)), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        train_ds = SegmentationDataset(train_images, train_masks, transform=img_transform, target_transform=mask_transform)
        val_ds = SegmentationDataset(val_images, val_masks, transform=img_transform, target_transform=mask_transform)
        # For test, can replicate or load separately
        # Placeholder: use validation as test
        test_ds = val_ds
        return train_ds, val_ds, test_ds

    def _load_svhn(self):
        root = os.path.join(self.dataset_root, 'SVHN')
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        transform = transforms.Compose([
            transforms.Resize(int(32 * self.get_scale_factor())),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                 std=[0.1980, 0.2010, 0.1970])
        ])
        train_ds = datasets.SVHN(root, split='train', download=True, transform=transform)
        test_ds = datasets.SVHN(root, split='test', download=True, transform=transform)
        val_ds = test_ds  # Placeholder; better to do proper split
        return train_ds, val_ds, test_ds

    def _load_gtsrb(self):
        root = os.path.join(self.dataset_root, 'GTSRB')
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        # Placeholder: use torchvision's GTSRB if available
        transform = transforms.Compose([
            transforms.Resize(int(40 * self.get_scale_factor())),  # 40x40 default
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                                 std=[0.2720, 0.2603, 0.2669])
        ])
        dataset = datasets.GTSRB(root, split='train', download=True, transform=transform)
        total = len(dataset)
        split_idx = int(0.8 * total)
        train_ds = torch.utils.data.Subset(dataset, list(range(split_idx)))
        val_ds = torch.utils.data.Subset(dataset, list(range(split_idx, total)))
        test_ds = val_ds  # placeholder
        return train_ds, val_ds, test_ds

    def _load_food101(self):
        root = os.path.join(self.dataset_root, 'Food101')
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        # Use torchvision's Food101
        transform = transforms.Compose([
            transforms.Resize(int(224 * self.get_scale_factor())),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.527, 0.491, 0.447],
                                 std=[0.253, 0.247, 0.261])
        ])
        train_ds = datasets.Food101(root, split='train', download=True, transform=transform)
        test_ds = datasets.Food101(root, split='test', download=True, transform=transform)
        val_ds = test_ds  # placeholder, usually better to split from train
        return train_ds, val_ds, test_ds

    def _load_eurosat(self):
        root = os.path.join(self.dataset_root, 'EuroSAT')
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        # Placeholder: assuming EuroSAT is downloaded as ImageFolder
        transform = transforms.Compose([
            transforms.Resize(int(64 * self.get_scale_factor())),  # imagery size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.344, 0.380, 0.371],
                                 std=[0.129, 0.124, 0.123])
        ])
        dataset = datasets.ImageFolder(root=os.path.join(root, 'images'), transform=transform)
        total = len(dataset)
        split1 = int(0.8 * total)
        train_idx, val_idx = list(range(split1)), list(range(split1, total))
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)
        test_ds = val_ds  # placeholder
        return train_ds, val_ds, test_ds

    def _load_oxford_pet(self):
        root = os.path.join(self.dataset_root, 'OxfordIIITPet')
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        # Use torchvision datasets or custom logic
        # Placeholder: assume dataset is prepared; map class labels
        # For simplicity, reusing ImageFolder
        transform = transforms.Compose([
            transforms.Resize(int(224 * self.get_scale_factor())),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.504, 0.486, 0.426],
                                 std=[0.263, 0.248, 0.251])
        ])
        dataset = datasets.ImageFolder(root=os.path.join(root, 'images'), transform=transform)
        total = len(dataset)
        split_idx = int(0.8 * total)
        train_idx, val_idx = list(range(split_idx)), list(range(split_idx, total))
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)
        test_ds = val_ds
        return train_ds, val_ds, test_ds

    def _load_ucf101(self):
        root = os.path.join(self.dataset_root, 'UCF101')
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        # Placeholder: UCF101 is videos; need frame extraction or pre-extracted frames
        # For simplicity, assume dataset is image frames stored locally
        # Not implemented: return empty or mock dataset
        # Here, just placeholders:
        # Create dummy datasets
        dummy_image = Image.new('RGB', (224, 224))
        dummy_dataset = [(dummy_image, 0)] * 1000
        class DummyDataset(Dataset):
            def __len__(self): return len(dummy_dataset)
            def __getitem__(self, idx):
                return {'image': transforms.ToTensor()(dummy_dataset[idx][0]), 'label': dummy_dataset[idx][1]}
        # Split into train/val/test
        total = len(dummy_dataset)
        train_end = int(0.8 * total)
        dataset = DummyDataset()
        train_ds = torch.utils.data.Subset(dataset, list(range(train_end)))
        val_ds = torch.utils.data.Subset(dataset, list(range(train_end, total)))
        test_ds = val_ds
        return train_ds, val_ds, test_ds

    def _load_fmow(self):
        root = os.path.join(self.dataset_root, 'FMoW')
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        # Placeholder: Load FMoW images and labels accordingly
        # For simplicity, create dummy dataset
        dummy_image = Image.new('RGB', (224, 224))
        dummy_dataset = [(dummy_image, 0)] * 1000
        class DummyDS(Dataset):
            def __len__(self): return len(dummy_dataset)
            def __getitem__(self, idx):
                return {'image': transforms.ToTensor()(dummy_dataset[idx][0]), 'label': dummy_dataset[idx][1]}
        total = len(dummy_dataset)
        train_end = int(0.8 * total)
        dataset = DummyDS()
        train_ds = torch.utils.data.Subset(dataset, list(range(train_end)))
        val_ds = torch.utils.data.Subset(dataset, list(range(train_end, total)))
        test_ds = val_ds
        return train_ds, val_ds, test_ds

    def _load_dtd(self):
        root = os.path.join(self.dataset_root, 'DTD')
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        # Placeholder: load dataset similarly
        dummy_image = Image.new('RGB', (224, 224))
        dummy_dataset = [(dummy_image, 0)] * 1000
        class DummyDS(Dataset):
            def __len__(self): return len(dummy_dataset)
            def __getitem__(self, idx):
                return {'image': transforms.ToTensor()(dummy_dataset[idx][0]), 'label': dummy_dataset[idx][1]}
        total = len(dummy_dataset)
        train_end = int(0.8 * total)
        dataset = DummyDS()
        train_ds = torch.utils.data.Subset(dataset, list(range(train_end)))
        val_ds = torch.utils.data.Subset(dataset, list(range(train_end, total)))
        test_ds = val_ds
        return train_ds, val_ds, test_ds

    def get_scale_factor(self) -> float:
        # Return scale factor from config; default to 1.0 here
        # Real implementation: read from config file
        return 1.0
```

## evaluation.py

```python
## evaluation.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple

# Import dataset loaders, models, prompt, label mapping, and configuration
from dataset_loader import DatasetLoader
from model import PretrainedModel
from prompt_module import PromptGenerator
from label_mapping import LabelMapper

# Optional: For segmentation IoU computation
def compute_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor, num_classes: int) -> float:
    """
    Compute average IoU for batch predictions.
    Args:
        pred_mask: (N, H, W) long tensor with predicted classes
        target_mask: (N, H, W) long tensor with ground truth classes
        num_classes: total number of classes
    Returns:
        average IoU score over batch
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        target_cls = (target_mask == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            # No ground truth and no prediction for this class
            continue
        ious.append(intersection / union)
    if len(ious) == 0:
        return 1.0  # If no classes present, assume perfect
    return sum(ious) / len(ious)

class Evaluation:
    def __init__(self,
                 model: PretrainedModel,
                 prompts: PromptGenerator,
                 dataset_loader: DatasetLoader,
                 label_mapper: LabelMapper,
                 config: Dict[str, Any]):
        """
        Initialize evaluation with model, prompts, dataset loader, label mapper, and config.
        """
        self.model = model
        self.prompts = prompts
        self.dataset_loader = dataset_loader
        self.label_mapper = label_mapper
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        # Dataset info
        self.dataset_name = self.config['dataset']['name']
        self.num_classes = self.config['dataset'].get('num_classes', 10)
        # Visualization directory
        self.vis_dir = self.config.get('logging', {}).get('log_dir', './logs')
        os.makedirs(self.vis_dir, exist_ok=True)

    def evaluate(self):
        """
        Run inference on test dataset, compute metrics, generate visualizations.
        Returns:
            dict: metrics results and optional visualizations info
        """
        # Prepare data loader
        test_loader = self.dataset_loader['test']
        # Initialize metrics accumulators
        total_samples = 0
        correct_preds = 0
        total_iou = 0.0
        total_batches = 0
        total_robust_acc = 0.0  # if robustness test included
        total_confidence = 0.0  # average confidence
        total_confidence_samples = 0

        # For segmentation, collect predicted and target masks
        pred_masks = []
        target_masks = []

        # Set model in eval mode
        self.model.model.eval()
        self.prompts.prompt_tensor.eval()
        if hasattr(self.prompts, 'real_coeffs'):
            self.prompts.real_coeffs.eval()
        if hasattr(self.prompts, 'imag_coeffs'):
            self.prompts.imag_coeffs.eval()

        with torch.no_grad():
            for batch in tqdm(test_loader):
                # Handle different dataset types
                imgs = batch.get('image', None)
                labels = batch.get('label', None)
                masks = batch.get('mask', None)  # For segmentation
                targets = batch.get('target', None)  # For detection
                # Move to device
                if imgs is not None:
                    imgs = imgs.to(self.device)
                # For segmentation/detection, may need special handling
                # For classification:
                if imgs is None:
                    continue  # skip if no images
                
                # Resize images as per config
                imgs_resized = self._resize_images(imgs)

                # Get prompt
                prompt = self.prompts.get_prompt()
                # Apply prompt and resize
                prompted_imgs = self._apply_prompt(imgs_resized, prompt)

                # Forward pass
                preds = self.model.forward(prompted_imgs)
                # Get predictions (logits or features)
                if self.model.model_name == 'clip':
                    # CLIP: compute text similarity
                    # preds: image features (normalized)
                    # Compute cosine similarity with class text embeddings
                    class_embeddings = self._get_class_text_embeddings()
                    # Normalize image features
                    img_embs = self.model.extract_features(prompted_imgs)
                    # Compute cosine similarity (N, T)
                    sims = torch.matmul(img_embs, class_embeddings.T)
                    pred_logit = sims
                else:
                    # Vision model: logits or features
                    pred_logit = preds

                # Map to target classes
                mapped_preds = self.label_mapper.map(pred_logit)

                # Compute classification accuracy
                if labels is not None:
                    pred_labels = torch.argmax(mapped_preds, dim=1)
                    correct_preds += (pred_labels == labels).sum().item()
                    total_samples += labels.shape[0]

                # For segmentation: compute IoU
                if masks is not None:
                    pred_mask = torch.argmax(mapped_preds, dim=1)  # shape (N, H, W)
                    total_iou += compute_iou(pred_mask, masks, self.num_classes)
                    pred_masks.append(pred_mask.cpu())
                    target_masks.append(masks.cpu())

                # For detection: could extend with mAP calculation (not shown here)

                # Optional: robustness evaluation
                if 'corrupted' in batch:
                    # Evaluate on corrupted images if provided
                    corrupted_imgs = batch['corrupted'].to(self.device)
                    with torch.no_grad():
                        prompted_corr = self._apply_prompt(self._resize_images(corrupted_imgs), prompt)
                        preds_corr = self.model.forward(prompted_corr)
                        if self.model.model_name == 'clip':
                            sims_corr = torch.matmul(self.model.extract_features(prompted_corr), class_embeddings.T)
                            preds_corr_logits = sims_corr
                        else:
                            preds_corr_logits = preds_corr
                        mapped_preds_corr = self.label_mapper.map(preds_corr_logits)
                        pred_labels_corr = torch.argmax(mapped_preds_corr, dim=1)
                        correct_corr = (pred_labels_corr == batch['label'].to(self.device)).sum().item()
                        total_robust_acc += correct_corr
                        total_confidence += torch.max(F.softmax(mapped_preds_corr, dim=1), dim=1).sum().item()
                        total_confidence_samples += batch['label'].size(0)

        # Final metrics
        accuracy = 100.0 * correct_preds / total_samples if total_samples > 0 else 0.0
        avg_iou = total_iou / max(1, len(pred_masks)) if pred_masks else 0.0
        robustness_acc = (total_robust_acc / total_confidence_samples * 100.0) if total_confidence_samples > 0 else None
        avg_confidence = (total_confidence / total_confidence_samples) if total_confidence_samples > 0 else None

        results = {
            'accuracy': accuracy,
            'iou': avg_iou,
            'robust_accuracy': robustness_acc,
            'average_confidence': avg_confidence,
        }

        # Generate visualizations
        self._visualize_prompts()
        self._visualize_label_mapping()

        return results

    def _resize_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Resize images internally if needed per current config.
        Currently assuming fixed scale; extend with differentiable resize if needed.
        """
        # Placeholder: do nothing, return images directly
        return imgs

    def _apply_prompt(self, imgs: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        Apply pixel prompts or insert prompts into images.
        For pixel prompts: overlay or concatenate.
        """
        # For simplicity, assuming prompts are padding: pad images
        p = prompt.shape[1]
        # Padding with zeros (or prompts), adjust as needed
        batch_size, C, H, W = imgs.shape
        padded_imgs = F.pad(imgs, pad=(p, p, p, p), mode='constant', value=0)
        return padded_imgs

    def _get_class_text_embeddings(self):
        """
        Return class text embeddings for semantic similarity.
        Only relevant for CLIP.
        """
        # Assumes class names are available globally or in self
        # For simplicity, assume self._class_text_embeddings exists
        if hasattr(self, '_class_text_embeddings'):
            return self._class_text_embeddings
        else:
            # Needs to be initialized before
            # Placeholder: random embeddings
            return torch.randn((self.num_classes, 512), device=self.device)

    def _visualize_prompts(self):
        """
        Visualize current prompts (pixel or frequency) for inspection.
        """
        if hasattr(self.prompts, 'visualize'):
            try:
                prompt_img = self.prompts.visualize()
                plt.figure(figsize=(4,4))
                plt.imshow(prompt_img)
                plt.axis('off')
                save_path = os.path.join(self.vis_dir, 'prompt_visualization.png')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
            except Exception:
                pass

    def _visualize_label_mapping(self):
        """
        Visualize label mapping matrices or semantic similarities.
        """
        if hasattr(self.label_mapper, 'visualize_mapping'):
            try:
                self.label_mapper.visualize_mapping()
            except Exception:
                pass
```

## label_mapping.py

```python
## label_mapping.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from typing import List, Dict, Optional, Union
from tqdm import tqdm  # for progress bar during iterative updates
import matplotlib.pyplot as plt

class LabelMapper:
    def __init__(self,
                 strategy: str,  # 'FreqMap', 'IterMap', 'SemanticMap', 'FullyMap'
                 source_class_names: List[str],
                 target_class_names: List[str],
                 map_params: Dict,
                 device: torch.device):
        """
        Initialize the label mapping object according to strategy and parameters.
        Args:
            strategy (str): Mapping strategy.
            source_class_names (List[str]): List of source class names.
            target_class_names (List[str]): List of target dataset class names.
            map_params (dict): Additional parameters, e.g., n classes, init weights, etc.
            device (torch.device): Device to run computations on.
        """
        self.strategy = strategy
        self.source_class_names = source_class_names
        self.target_class_names = target_class_names
        self.device = device

        # Store parameters
        self.params = map_params

        # Initialize data structures depending on strategy
        num_source = len(source_class_names)
        num_target = len(target_class_names)

        # For strategies that require a mapping matrix
        if self.strategy in ['FreqMap', 'IterMap', 'FullyMap']:
            # Initialize mapping matrix: source x target
            # For FreqMap/IterMap: 1 indicates mapped, else 0
            self.M = torch.zeros((num_source, num_target), device=self.device)
            # For FullyMap, initialize linear weights later
        elif self.strategy == 'SemanticMap':
            # Compute embeddings for class names
            self.clip_model = None
            self.clip_processor = None
            self._init_clip_embeddings()
            # Similarity matrix between source and target classes
            self.semantic_similarity = None
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # For FullyMap, define linear layer for learned mapping
        if self.strategy == 'FullyMap':
            # Map source logits to target logits
            self.linear_mapping = nn.Linear(num_source, len(target_class_names))
            # Initialize linear layer weights
            self._init_fullymap_weights()
            self.linear_mapping.to(self.device)

        # For IterMap, store current mapping (initially FreqMap or default)
        if self.strategy == 'IterMap':
            # Initialize as empty, will be updated via update_mapping()
            self.iter_mapping = None

    def _init_clip_embeddings(self):
        """
        Initialize CLIP model and get class name embeddings for source and target.
        """
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            # Freeze CLIP model
            for param in self.clip_model.parameters():
                param.requires_grad = False
        except ImportError:
            # Fallback: use transformers CLIP
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Encode class names
        self.source_embeddings = self._compute_text_embeddings(self.source_class_names)
        self.target_embeddings = self._compute_text_embeddings(self.target_class_names)

        # Compute cosine similarity matrix: target x source
        # Result: shape (target_classes, source_classes)
        self.semantic_similarity = torch.zeros((len(self.target_class_names), len(self.source_class_names)), device=self.device)
        for i in range(len(self.target_class_names)):
            sim = F.cosine_similarity(self.target_embeddings[i].unsqueeze(0), self.source_embeddings, dim=-1)
            self.semantic_similarity[i] = sim

        # For mapping target class index to source class index
        # Will be used to assign source classes to target
        self.target_to_source_mapping = torch.argmax(self.semantic_similarity, dim=1)

    def _compute_text_embeddings(self, class_names: List[str]) -> torch.Tensor:
        """
        Compute normalized text embeddings for a list of class names via CLIP.
        """
        if hasattr(self, 'clip_processor'):
            inputs = self.clip_processor(text=class_names, return_tensors='pt', padding=True).to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats
        elif hasattr(self, 'clip_model'):
            # Alternative method if using clip from 'clip' package
            import clip
            tokens = clip.tokenize(class_names).to(self.device)
            with torch.no_grad():
                feats = self.clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats
        else:
            # Should not happen
            raise RuntimeError("No CLIP model available for text embeddings.")

    def map(self, predictions: torch.Tensor, train_data_preds: Optional[Dict]=None):
        """
        Map raw model predictions to target classes according to strategy.
        Args:
            predictions (Tensor): shape (N, K_s), source model logits or predictions.
            train_data_preds (optional): Used for FreqMap, dict of {target_class_idx: count}
        Returns:
            mapped_preds (Tensor): shape (N,), target class indices
        """
        if self.strategy == 'FreqMap':
            # Use frequency counts to assign target class
            # train_data_preds: dict or tensor with counts
            # For online predictions, we assume the tally is already computed
            # The mapping matrix self.M indicates source->target
            # Predictions: shape (N, K_s)
            # Get source class predictions
            source_preds = torch.argmax(predictions, dim=1)  # shape (N,)
            # Map source class to target class based on the mapping matrix
            # For each source class, find assigned target class
            target_indices = torch.zeros_like(source_preds)
            for s_idx in range(self.M.shape[0]):
                tgt_idx = torch.argmax(self.M[s_idx])  # target class assigned to source s_idx
                source_mask = (source_preds == s_idx)
                target_indices[source_mask] = tgt_idx
            return target_indices
        elif self.strategy == 'IterMap':
            # Recompute mapping at current epoch/step
            # Call update_mapping() externally to refresh self.M
            # After update_mapping(), use same logic as FreqMap
            source_preds = torch.argmax(predictions, dim=1)
            target_indices = torch.zeros_like(source_preds)
            for s_idx in range(self.M.shape[0]):
                tgt_idx = torch.argmax(self.M[s_idx])
                source_mask = (source_preds == s_idx)
                target_indices[source_mask] = tgt_idx
            return target_indices
        elif self.strategy == 'SemanticMap':
            # Use class name embeddings similarity
            # predictions: source class indices
            # Map prediction to source class embedding, then find closest target class
            # Usually, predictions are class indices (or logits). Here, assume predictions are (N,)
            # For predictions in logits, take argmax
            pred_source_indices = torch.argmax(predictions, dim=1)
            source_embs = self.source_embeddings[pred_source_indices]  # (N, D)
            # Compute cosine similarity with target embeddings
            # target_embeddings shape: (T, D)
            # similarity: (N, T)
            sim = F.cosine_similarity(source_embs.unsqueeze(1), self.target_embeddings.unsqueeze(0), dim=-1)
            # For each, pick target class with max similarity
            target_preds = torch.argmax(sim, dim=1)
            return target_preds
        elif self.strategy == 'FullyMap':
            # Pass source logits (predictions) through linear layer
            # predictions shape: (N, K_s)
            final_logits = self.linear_mapping(predictions)  # (N, T)
            target_preds = torch.argmax(final_logits, dim=1)
            return target_preds
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def update_mapping(self, training_dataset=None, source_model=None):
        """
        Update class correspondence or weights during training for strategies like IterMap and FullyMap.
        Args:
            training_dataset (Dataset): dataset to compute mappings from.
            source_model (PretrainedModel): model for predictions if needed.
        """
        if self.strategy == 'IterMap':
            # Recompute frequency-based mapping from training dataset
            if training_dataset is None or source_model is None:
                raise ValueError("training_dataset and source_model required for IterMap update.")
            count_matrix = torch.zeros((len(self.source_class_names), len(self.target_class_names)), device=self.device)
            dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=self.params.get('batch_size', 32),
                                                     shuffle=False)
            with torch.no_grad():
                for batch in dataloader:
                    imgs = batch['image'].to(self.device)
                    # Obtain predictions from source model
                    preds = source_model.forward(imgs)
                    pred_labels = torch.argmax(preds, dim=1)
                    # For each sample, get source and target labels
                    # Assuming batch includes target labels in batch['label'] for training set
                    target_labels = batch.get('label', None)
                    if target_labels is None:
                        # If no explicit target labels provided, infer from dataset order or metadata
                        # For placeholder, assume source labels correspond to class indices
                        break  # cannot update without target class info
                    # Here, for practical purposes, require dataset to provide target labels
                    # For simplicity, we skip actual update unless dataset provides 'label'
                    # Better to implement using ground truth labels if available
            # After computing frequency, update self.M accordingly
            # But as placeholder, we here just re-initialize mapping based on max class predictions
            # For actual implementation, this requires dataset ground truth
            # or prediction counts per class
            # For simplicity, set to default: assign target classes using semantic similarity
            # Alternatively, could do:
            # For each target class t, find source class s that predicts most of t
            # But requires dataset labels
            pass
        elif self.strategy == 'FullyMap':
            # Update linear layer weights possibly with weights derived from semantic similarity
            # Here, as per paper, initialize weights based on semantic similarity or keep fixed
            # For illustration, do a simple semantic initialization if desired
            # Otherwise, keep weights fixed
            pass

    def visualize_mapping(self):
        """
        Generate a visualization of the class correspondence.
        For FreqMap and IterMap: show mapping counts or confusion
        For SemanticMap: plot similarity matrix
        For FullyMap: visualize linear weights
        """
        if self.strategy in ['FreqMap', 'IterMap']:
            # Plot the mapping matrix as a heatmap
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,8))
            plt.imshow(self.M.cpu().numpy(), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.xlabel('Target Classes')
            plt.ylabel('Source Classes')
            plt.title(f'Mapping matrix heatmap: {self.strategy}')
            plt.show()
        elif self.strategy == 'SemanticMap':
            # Plot similarity matrix
            import seaborn as sns
            plt.figure(figsize=(8,6))
            sns.heatmap(self.semantic_similarity.cpu().numpy(), annot=True, cmap='coolwarm')
            plt.xlabel('Source Classes')
            plt.ylabel('Target Classes')
            plt.title('Semantic similarity between classes')
            plt.show()
        elif self.strategy == 'FullyMap':
            # Visualize linear layer weights as a heatmap
            weights = self.linear_mapping.weight.data.cpu()
            plt.figure(figsize=(10,8))
            sns.heatmap(weights, cmap='viridis', xticklabels=self.target_class_names, yticklabels=self.source_class_names)
            plt.xlabel('Target Classes')
            plt.ylabel('Source Classes')
            plt.title('FullyMap Linear Layer Weights')
            plt.show()
        else:
            print("No visualization available for this strategy.")
```


## main.py

```python
```python
## main.py
import os
import yaml
import random
import numpy as np
import torch
import logging

from dataset_loader import DatasetLoader
from model import PretrainedModel
from prompt_module import PromptGenerator
from label_mapping import LabelMapper
from trainer import Trainer
from tuner import HyperparameterTuner
from evaluation import Evaluation

def main():
    # Load configuration from YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set logging verbosity
    verbosity_str = config.get('logging', {}).get('verbosity', 'INFO').upper()
    import logging
    logging_level = getattr(logging, verbosity_str, logging.INFO)
    logging.basicConfig(level=logging_level)

    seed = config.get('misc', {}).get('seed', 42)
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset_name = config['dataset']['name']
    dataset_root = config['dataset'].get('root_dir', None)
    dataset_loader = DatasetLoader(
        datasets_list=[dataset_name],
        batch_size=config['training'].get('batch_size', 32),
        dataset_root=dataset_root
    )
    datasets_dict = dataset_loader.load_data()
    train_loader = datasets_dict[dataset_name]['train']
    val_loader = datasets_dict[dataset_name]['val']
    test_loader = datasets_dict[dataset_name]['test']
    dataset_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # Initialize pre-trained backbone model
    backbone_name = config['model']['backbone']
    backbone = PretrainedModel(model_name=backbone_name, freeze=True).to(device)
    backbone.eval()

    # Initialize PromptGenerator
    prompt_size = config['model']['prompt_size']
    prompt_type = config['model']['prompt_type']
    prompt_init_type = config['model'].get('prompt_init_type', 'zeros')
    prompts = PromptGenerator(prompt_size=prompt_size,
                              prompt_type=prompt_type,
                              prompt_init_type=prompt_init_type)

    # Get class names for source and target
    # For source classes, assume ImageNet classes or mock list
    # For target, use dataset specific classes
    def get_source_class_names(model_name):
        # Placeholder: list of 1000 ImageNet classes
        return [f'class_{i}' for i in range(1000)]
    def get_target_class_names(dataset_name):
        # Placeholder: load dataset class names from dataset info
        num_classes = config['dataset'].get('num_classes', 10)
        return [f'class_{i}' for i in range(num_classes)]

    source_class_names = get_source_class_names(backbone_name)
    target_class_names = get_target_class_names(dataset_name)

    # Initialize LabelMapper based on hyperparameter choice
    mapping_strategy = None
    mapping_choice = None
    if 'label_mapping' in config:
        mapping_choice = config['label_mapping']
    else:
        mapping_choice = 'FreqMap'  # default
    map_params = {
        'batch_size': config['training'].get('batch_size', 32),
        'num_source_classes_per_target': config['dataset'].get('num_classes', None)
    }
    label_mapper = LabelMapper(
        strategy=mapping_choice,
        source_class_names=source_class_names,
        target_class_names=target_class_names,
        map_params=map_params,
        device=device
    )

    # Hyperparameter search space
    hp_search_space = {
        'prompt_size': config['hyperparameters'].get('prompt_size_options', [16]),
        'input_scale': config['hyperparameters'].get('input_scale_options', [1.0]),
        'model_choice': config['hyperparameters'].get('model_choices', ['resnet18']),
        'label_mapping': config['hyperparameters'].get('label_mapping_strategies', ['FreqMap'])
        # Additional hyperparameters like number of source classes per target can be added
    }

    # Initialize HyperparameterTuner
    tuner = HyperparameterTuner(
        config={
            'dataset_name': dataset_name,
            'dataset_root': dataset_root,
            'training': config['training'],
            'model': config['model'],
            'hyperparameters': config['hyperparameters'],
            'logging': config.get('logging', {}),
            'misc': config.get('misc', {})
        }
    )

    # Run hyperparameter search to find best config
    best_hyperparams = tuner.run()

    # Instantiate objects with best hyperparameters
    # Unpack best hyperparameters
    best_prompt_size = best_hyperparams['prompt_size']
    best_input_scale = best_hyperparams['input_scale']
    best_model_choice = best_hyperparams['model_choice']
    best_mapping_strategy = best_hyperparams['label_mapping']

    # Re-initialize model
    backbone_best = PretrainedModel(model_name=best_model_choice, freeze=True).to(device)
    backbone_best.eval()

    # Initialize prompts
    prompts_best = PromptGenerator(
        prompt_size=best_prompt_size,
        prompt_type=prompt_type,
        prompt_init_type=prompt_init_type
    )

    # Recompute class names if needed
    source_class_names_best = get_source_class_names(best_model_choice)
    target_class_names_best = get_target_class_names(dataset_name)

    label_mapper_best = LabelMapper(
        strategy=best_mapping_strategy,
        source_class_names=source_class_names_best,
        target_class_names=target_class_names_best,
        map_params=map_params,
        device=device
    )

    # Final full training with selected hyperparameters
    # Initialize optimizer for prompts and label mapping
    optimizer_params = list(prompts_best.prompt_tensor.parameters())
    if hasattr(prompts_best, 'real_coeffs'):
        optimizer_params += list(prompts_best.real_coeffs.parameters())
    if hasattr(prompts_best, 'imag_coeffs'):
        optimizer_params += list(prompts_best.imag_coeffs.parameters())
    if best_mapping_strategy == 'FullyMap':
        optimizer_params += list(label_mapper_best.linear_mapping.parameters())

    optimizer = torch.optim.Adam(optimizer_params,
                                 lr=config['training']['learning_rate'],
                                 weight_decay=config['training'].get('weight_decay', 0))
    # Optional LR scheduler
    lr_scheduler = None
    if config['training'].get('lr_scheduler', None) == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    # Initialize trainer and train over full epochs
    trainer = Trainer(
        model=backbone_best,
        prompts=prompts_best,
        dataset=dataset_dict,
        label_mapper=label_mapper_best,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config
    )
    # Run full training
    trainer.train()

    # Final evaluation on test set
    evaluator = Evaluation(model=backbone_best,
                           prompts=prompts_best,
                           dataset_loader=dataset_loader,
                           label_mapper=label_mapper_best,
                           config=config)
    results = evaluator.evaluate()

    # Save final model and prompts
    save_dir = config.get('logging', {}).get('log_dir', './logs')
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': backbone_best.model.state_dict(),
        'prompts_state_dict': prompts_best.prompt_tensor.state_dict()
        # Save label mapper if needed
    }, os.path.join(save_dir, 'final_model_prompts.pth'))

    # Print summary
    print("Final Evaluation Results:")
    print(f"Accuracy: {results.get('accuracy', 'N/A'):.2f}%")
    print(f"IoU: {results.get('iou', 'N/A'):.4f}")
    if results.get('robust_accuracy', None) is not None:
        print(f"Robust Accuracy: {results['robust_accuracy']:.2f}%")
    if results.get('average_confidence', None) is not None:
        print(f"Average Confidence: {results['average_confidence']:.4f}")

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
from torchvision import models
from transformers import CLIPModel, CLIPProcessor

class PretrainedModel:
    def __init__(self, model_name: str = 'clip', freeze: bool = True):
        """
        Initialize the pre-trained backbone model based on the given model_name.
        Supports 'resnet18', 'resnext101-ig', 'swin-t', 'clip'.
        Args:
            model_name (str): Name of the backbone model.
            freeze (bool): If True, freeze all model parameters.
        """
        self.model_name = model_name.lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None  # For CLIP text processing
        self._load_model(freeze)
    
    def _load_model(self, freeze: bool):
        if self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        elif self.model_name == 'resnext101-ig':
            # Using timm for ResNeXt-101-IG pretraining
            import timm
            self.model = timm.create_model('resnext101_32x8d', pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        elif self.model_name == 'swin-t':
            # Using timm for Swin-T
            import timm
            self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        elif self.model_name == 'clip':
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.model.eval()
            # Initialize CLIP tokenizer (processor)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        # Freeze model parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model, outputs prediction logits.
        Args:
            x (torch.Tensor): Input images tensor (N, C, H, W)
        Returns:
            torch.Tensor: logits or prediction scores
        """
        if self.model_name == 'clip':
            # CLIP: produce image features
            with torch.no_grad():
                image_features = self.model.get_image_features(pixel_values=x)
            return image_features
        else:
            # Vision models
            return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings from input images (before classification layers).
        For CLIP, use get_image_features; for other models, use penultimate features.
        Args:
            x (torch.Tensor): Input images tensor.
        Returns:
            torch.Tensor: Normalized feature embeddings.
        """
        if self.model_name == 'clip':
            with torch.no_grad():
                features = self.model.get_image_features(pixel_values=x)
            # Normalize embeddings
            features = features / features.norm(dim=-1, keepdim=True)
            return features
        elif self.model_name.startswith('resnet'):
            # For ResNet, hook or custom method to get features
            return self._get_resnet_features(x)
        elif self.model_name == 'swin-t':
            return self._get_swin_features(x)
        else:
            # Default: just return model output
            return self.forward(x)

    def _get_resnet_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from ResNet (before final FC)
        """
        # ResNet: get features from average pooling layer
        # Temporarily replace fc with identity if necessary
        # Or use hooks; here, assume access via the penultimate layer
        # For simplicity, here we remove last fc layer
        resnet_feat_extractor = nn.Sequential(*list(self.model.children())[:-1])  # all layers except fc
        with torch.no_grad():
            feat = resnet_feat_extractor(x)  # shape (N, 512, 1, 1)
        feat = feat.squeeze(-1).squeeze(-1)  # shape (N, 512)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def _get_swin_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from Swin Transformer
        """
        # Timm Swin models have a 'forward_features' method
        with torch.no_grad():
            feat = self.model.forward_features(x)
        # Depending on model, output shape may vary
        feat = feat.mean([-2, -1])  # global average pooling if necessary
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def extract_text_embeddings(self, class_names: list) -> torch.Tensor:
        """
        For CLIP: generate normalized text embeddings for class labels.
        Args:
            class_names (list of str): List of class label strings.
        Returns:
            torch.Tensor: embeddings of shape (num_classes, embedding_dim)
        """
        if self.model_name != 'clip':
            raise RuntimeError("Text embeddings are supported only for CLIP models.")
        # Prepare inputs
        inputs = self.processor(text=class_names, return_tensors='pt', padding=True)
        # Move inputs to same device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_feats = self.model.get_text_features(**inputs)
        # Normalize to unit sphere
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats
```

## prompt_module.py

```python
## prompt_module.py
import torch
import torch.nn as nn
import torch.fft as fft
import matplotlib.pyplot as plt
import numpy as np

class PromptGenerator:
    def __init__(self, prompt_size: int = 16, prompt_type: str = 'pixel',
                 prompt_init_type: str = 'zeros', input_channels: int = 3,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initialize the prompt generator.
        Args:
            prompt_size (int): size of the prompt (pixels).
            prompt_type (str): 'pixel' or 'frequency'.
            prompt_init_type (str): 'zeros', 'random', 'learned'
            input_channels (int): e.g., 3 for RGB.
            device (torch.device): device to store parameters.
        """
        self.prompt_size = prompt_size
        self.prompt_type = prompt_type.lower()
        self.input_channels = input_channels
        self.device = device

        if self.prompt_type == 'pixel':
            # Pixel prompts: trainable tensor of shape (C, p, p)
            if prompt_init_type == 'zeros':
                init_tensor = torch.zeros((self.input_channels, self.prompt_size, self.prompt_size), device=self.device)
            elif prompt_init_type == 'random':
                init_tensor = 0.01 * torch.randn((self.input_channels, self.prompt_size, self.prompt_size), device=self.device)
            else:
                init_tensor = torch.zeros((self.input_channels, self.prompt_size, self.prompt_size), device=self.device)
            self.prompt_tensor = nn.Parameter(init_tensor)
        elif self.prompt_type == 'frequency':
            # Frequency prompts: trainable FFT coefficients (real)
            # Store real and imaginary parts separately for interpretability
            real_part = torch.zeros((self.input_channels, self.prompt_size, self.prompt_size), device=self.device)
            imag_part = torch.zeros_like(real_part)
            if prompt_init_type == 'random':
                real_part = 0.01 * torch.randn_like(real_part)
                imag_part = 0.01 * torch.randn_like(imag_part)
            elif prompt_init_type == 'zeros':
                real_part = torch.zeros_like(real_part)
                imag_part = torch.zeros_like(imag_part)
            # Parameters: real and imaginary parts
            self.real_coeffs = nn.Parameter(real_part)
            self.imag_coeffs = nn.Parameter(imag_part)
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")
        
    def get_prompt(self) -> torch.Tensor:
        """
        Return the current prompt tensor as a spatial image.
        For pixel prompts, return directly.
        For frequency prompts, perform inverse FFT.
        Output shape: (C, p, p)
        """
        if self.prompt_type == 'pixel':
            return self.prompt_tensor
        elif self.prompt_type == 'frequency':
            # Reconstruct complex FFT tensor
            complex_fft = torch.complex(self.real_coeffs, self.imag_coeffs)
            # Inverse FFT to spatial domain
            # Note: torch.fft.ifft2 output is complex, take real part
            spatial_prompt = fft.ifft2(complex_fft, norm='forward')
            spatial_prompt = spatial_prompt.real
            # Clamp or normalize for visualization if needed
            spatial_prompt = spatial_prompt.clamp(0, 1)
            return spatial_prompt
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")

    def update(self, grads: torch.Tensor, step_size: float=1.0):
        """
        Update the prompt parameters based on provided gradients.
        Args:
            grads (Tensor): gradient tensor of same shape as prompt tensor.
            step_size (float): learning rate for update.
        """
        if self.prompt_type == 'pixel':
            # manual update
            if self.prompt_tensor.grad is None:
                raise RuntimeError("Gradients must be computed before calling update.")
            # Use stored gradient
            self.prompt_tensor.data -= step_size * self.prompt_tensor.grad.data
            self.prompt_tensor.grad.zero_()
        elif self.prompt_type == 'frequency':
            # grads should be same shape as real_coeffs and imag_coeffs
            if hasattr(grads, 'real') and hasattr(grads, 'imag'):
                # grads as complex tensor
                self.real_coeffs.data -= step_size * grads.real
                self.imag_coeffs.data -= step_size * grads.imag
            else:
                # assume grads contains real and imaginary parts
                self.real_coeffs.data -= step_size * grads['real']
                self.imag_coeffs.data -= step_size * grads['imag']
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")

    def visualize(self):
        """
        Visualize the current prompt as an image.
        Returns:
            image (numpy array): shape (H, W, C), for plotting.
        """
        prompt_img = self.get_prompt()  # shape (C, p, p)
        # Convert to [0,255] image array for visualization
        np_img = prompt_img.detach().cpu().permute(1, 2, 0).numpy()
        # Normalize for visualization
        np_img = np.clip(np_img, 0, 1)
        return np_img

    def save_visualization(self, path: str):
        """
        Save the visualization as an image file.
        """
        import matplotlib.pyplot as plt
        np_img = self.visualize()
        plt.imshow(np_img)
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()
```

## trainer.py

```python
# trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import copy
from typing import Optional, Dict, Any
from dataset_loader import DatasetLoader
from model import PretrainedModel
from prompt_module import PromptGenerator
from label_mapping import LabelMapper
from evaluation import Evaluator

class Trainer:
    def __init__(
        self,
        model: PretrainedModel,
        prompts: PromptGenerator,
        dataset: Dict[str, Dict[str, Any]],
        label_mapper: LabelMapper,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None,
        config: Dict[str, Any]=None,
    ):
        """
        Initialize the trainer with model, prompts, datasets, label mapper, optimizer, and configs.
        Args:
            model: PretrainedModel instance (frozen backbone).
            prompts: PromptGenerator instance with trainable prompts.
            dataset: dict with 'train' and 'val' DataLoader objects.
            label_mapper: LabelMapper instance for label conversion.
            optimizer: optimizer for prompt and label mapper parameters.
            lr_scheduler: optional LR scheduler.
            config: configuration dictionary from YAML.
        """
        self.model = model
        self.prompts = prompts
        self.label_mapper = label_mapper
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_train = dataset['train']
        self.dataset_val = dataset['val']
        self.epochs = config['training'].get('epochs', 50)
        self.batch_size = config['training'].get('batch_size', 32)
        self.early_stop_patience = config['training'].get('early_stop_patience', 3)
        self.total_iterations = config['training'].get('total_iterations', None)
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.early_stop_counter = 0
        self.checkpoint_path = os.path.join(config['logging'].get('log_dir', './logs'), 'best_model.pth')
        self.validation_metrics = {}
        # Prepare DataLoaders
        self.train_loader = self.dataset_train
        self.val_loader = self.dataset_val
        # Set prompts and label mapper trainable parameters
        self._prepare_train_params()

    def _prepare_train_params(self):
        """
        Collect parameters for optimization: prompts and label_mapper if trainable.
        Backbone is frozen.
        """
        params = []
        # Prompts prompts are trainable tensors
        params += list(self.prompts.prompt_tensor.parameters()) if hasattr(self.prompts.prompt_tensor, 'parameters') else []
        # For frequency prompts, include real and imaginary parts if trainable
        if hasattr(self.prompts, 'real_coeffs'):
            params += list(self.prompts.real_coeffs.parameters()) if hasattr(self.prompts.real_coeffs, 'parameters') else []
        if hasattr(self.prompts, 'imag_coeffs'):
            params += list(self.prompts.imag_coeffs.parameters()) if hasattr(self.prompts.imag_coeffs, 'parameters') else []
        # Label mapping parameters
        if self.label_mapper.strategy == 'FullyMap':
            params += list(self.label_mapper.linear_mapping.parameters())
        # Initialize optimizer with these only
        # Assuming optimizer is already constructed outside and passed in
        # No need to re-initialize here, but confirm optimizer's params
        pass

    def train(self):
        """
        Main training loop with early stopping.
        """
        best_epoch_state = None
        train_loss_history = []
        val_acc_history = []

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            train_loss, train_acc = self._train_one_epoch()
            val_metrics = self._validate()

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Validation Accuracy: {val_metrics['accuracy']:.2f}%")
            
            # Check for improvement
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.early_stop_counter = 0
                # Save best model state
                best_epoch_state = {
                    'prompts': copy.deepcopy(self.prompts),
                    'label_mapper': copy.deepcopy(self.label_mapper),
                    'model_state_dict': copy.deepcopy(self.model.model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                    'epoch': epoch,
                    'val_acc': val_metrics['accuracy']
                }
                self._save_checkpoint(self.checkpoint_path)
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_patience:
                    print("Early stopping triggered.")
                    break

            # Step LR scheduler if provided
            if self.lr_scheduler:
                self.lr_scheduler.step()

        # Load best model
        if best_epoch_state:
            self._load_checkpoint(self.checkpoint_path)
            self.prompts = best_epoch_state['prompts']
            self.label_mapper = best_epoch_state['label_mapper']
            self.model.model.load_state_dict(best_epoch_state['model_state_dict'])

    def _train_one_epoch(self):
        """
        Run a single epoch of training.
        """
        self.model.model.eval()  # freeze backbone
        self.prompts.prompt_tensor.train()
        if hasattr(self.prompts, 'real_coeffs'):
            self.prompts.real_coeffs.train()
        if hasattr(self.prompts, 'imag_coeffs'):
            self.prompts.imag_coeffs.train()
        self._set_optimizer_params()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        for batch in tqdm(self.train_loader):
            imgs = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            # Resize images according to scale (if learnable, differentiable)
            imgs_resized = self._resize_images(imgs)

            # Get current prompts
            prompt = self.prompts.get_prompt()  # shape (C, p, p)
            # Apply prompts to images
            prompted_imgs = self._apply_prompt(imgs_resized, prompt)

            # Forward pass with backbone
            with torch.no_grad():
                preds = self.model.forward(prompted_imgs)  # shape (N, K_s) or features
            # For classification, assume preds are logits
            # Map logits (or features) to target labels
            mapped_preds = self.label_mapper.map(preds, train_data_preds=None)
            loss = self.loss_fn(mapped_preds, labels)
            total_loss += loss.item() * imgs.shape[0]
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            pred_labels = torch.argmax(mapped_preds, dim=1)
            correct += (pred_labels == labels).sum().item()
            total_samples += labels.shape[0]
        epoch_loss = total_loss / total_samples
        epoch_acc = 100.0 * correct / total_samples
        return epoch_loss, epoch_acc

    def _validate(self):
        """
        Run validation epoch
        """
        self.model.model.eval()
        self.prompts.prompt_tensor.eval()
        if hasattr(self.prompts, 'real_coeffs'):
            self.prompts.real_coeffs.eval()
        if hasattr(self.prompts, 'imag_coeffs'):
            self.prompts.imag_coeffs.eval()

        correct = 0
        total_samples = 0
        for batch in tqdm(self.val_loader):
            imgs = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            imgs_resized = self._resize_images(imgs)
            with torch.no_grad():
                prompt = self.prompts.get_prompt()
                prompted_imgs = self._apply_prompt(imgs_resized, prompt)
                preds = self.model.forward(prompted_imgs)
                mapped_preds = self.label_mapper.map(preds)
                pred_labels = torch.argmax(mapped_preds, dim=1)
                correct += (pred_labels == labels).sum().item()
                total_samples += labels.shape[0]
        accuracy = 100.0 * correct / total_samples
        return {'accuracy': accuracy}

    def _save_checkpoint(self, path: str):
        """
        Save the current best model, prompts, label mapper
        """
        checkpoint = {
            'prompts': self.prompts,
            'label_mapper': self.label_mapper,
            'model_state_dict': self.model.model.state_dict(),
        }
        torch.save(checkpoint, path)

    def _load_checkpoint(self, path: str):
        """
        Load saved checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        # Load model backbone if needed
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        # Prompts and label mapper are deep copies
        # (assuming they are serializable or have state_dict methods)
        self.prompts = checkpoint['prompts']
        self.label_mapper = checkpoint['label_mapper']

    def _resize_images(self, imgs: torch.Tensor):
        """
        Resize images according to current scale factor and differentiable if needed.
        For simplicity, assumes fixed scale; can be extended.
        """
        # Placeholder: if scale is fixed, return imgs directly
        # For learnable scale, integrate kornia.transform
        return imgs

    def _apply_prompt(self, imgs: torch.Tensor, prompt: torch.Tensor):
        """
        Add pixel prompts or frequency prompts as per prompt_module design.
        For pixel prompts: overlay/pad prompts onto images.
        """
        # Assume prompt is (C, p, p), images are (N, C, H, W)
        # For simplicity, padding images with prompts (simulate Eq.1)
        p = prompt.shape[1]
        # Use padding to embed prompts
        # This is a simplified example; in practice, the method depends on prompt strategy
        batch_size, C, H, W = imgs.shape
        pad = p
        # For pixel prompts, just overlay or concatenate as needed
        # Placeholder: simple padding with zero
        padded_imgs = F.pad(imgs, (pad, pad, pad, pad), mode='constant', value=0)
        return padded_imgs

    def _set_optimizer_params(self):
        """
        Ensure optimizer only updates prompts and label mapping params.
        """
        params = []
        if hasattr(self.prompts, 'prompt_tensor'):
            params += list(self.prompts.prompt_tensor.parameters()) if hasattr(self.prompts.prompt_tensor, 'parameters') else []
        if hasattr(self.prompts, 'real_coeffs'):
            params += list(self.prompts.real_coeffs.parameters())
        if hasattr(self.prompts, 'imag_coeffs'):
            params += list(self.prompts.imag_coeffs.parameters())
        if self.label_mapper.strategy == 'FullyMap':
            params += list(self.label_mapper.linear_mapping.parameters())
        # Reinitialize optimizer with only these params
        # Assuming optimizer is passed from outside, but if not, can do:
        # self.optimizer = torch.optim.Adam(params, lr=...)
        # For robustness, do not alter externally created optimizer here.
        pass
```

## tuner.py

```python
## tuner.py
import itertools
import torch
import random
import numpy as np

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from dataset_loader import DatasetLoader
from model import PretrainedModel
from prompt_module import PromptGenerator
from label_mapping import LabelMapper

class HyperparameterTuner:
    def __init__(self, config: dict):
        """
        Initialize the hyperparameter tuner with configuration.
        Args:
            config (dict): configuration dictionary, from 'config.yaml'
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load dataset info - assume dataset info is provided in config
        self.dataset_name = config['dataset']['name']
        self.dataset_root = config['dataset']['root_dir']
        
        # Define default settings for the search grid
        self.prompt_size_options = config['hyperparameters'].get('prompt_size_options', [16])
        self.input_scale_options = config['hyperparameters'].get('input_scale_options', [1.0])
        self.model_choices = config['hyperparameters'].get('model_choices', ['resnet18'])
        self.label_mapping_strategies = config['hyperparameters'].get('label_mapping_strategies', ['FreqMap'])
        # Optional: define total_trials limit
        self.max_trials = config.get('max_trials', None)  # None for unlimited

        # Number of top trials to retain
        self.top_k = 2

        # Early stopping patience
        self.early_stop_patience = config['training'].get('early_stop_patience', 3)

        # For reproducibility
        seed = config['misc'].get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Initialize Ray Tune scheduler
        self.scheduler = ASHAScheduler(
            max_t= self.config['training'].get('epochs', 50),
            grace_period=1,
            reduction_factor=2,
            brackets=1,
            stop_last_trials=True
        )

        # Setup search space as a dictionary for Ray Tune
        self.search_space = {
            'prompt_size': tune.choice(self.prompt_size_options),
            'input_scale': tune.choice(self.input_scale_options),
            'model_choice': tune.choice(self.model_choices),
            'label_mapping': tune.choice(self.label_mapping_strategies)
        }

    def run(self):
        """
        Run the hyperparameter search using Ray Tune.
        """
        analysis = tune.run(
            self._trial_fn,
            resources_per_trial={'cpu': 4, 'gpu': 1},
            config=self.search_space,
            num_samples=None,  # For grid search, set to None and define trials explicitly
            scheduler=self.scheduler,
            stop={"training_iteration": self.config['training'].get('epochs', 50)},
            mode='max'  # maximize validation accuracy
        )

        # Retrieve best hyperparameters
        best_trial = analysis.get_best_trial(metric='validation_accuracy', mode='max', scope='all')
        best_config = best_trial.config
        best_result = analysis.get_best_metric_analysis(metric='validation_accuracy', mode='max').best_checkpoint
        print(f"Best hyperparameters: {best_config}")
        print(f"Best validation accuracy: {best_trial.last_result['validation_accuracy']:.2f}%")
        return best_config

    def _trial_fn(self, config: dict):
        """
        Ray Tune trial function: sets up model, dataset, prompts, label mapping, trains, evaluate.
        Receives a dict with hyperparameters sampled.
        """
        hyperparams = config
        prompt_size = hyperparams['prompt_size']
        input_scale = hyperparams['input_scale']
        model_choice = hyperparams['model_choice']
        label_map_strategy = hyperparams['label_mapping']

        # Load dataset
        dataset_loader = DatasetLoader(
            datasets_list=[self.dataset_name],
            batch_size=self.config['training'].get('batch_size', 32),
            dataset_root=self.dataset_root
        )
        datasets_dict = dataset_loader.load_data()
        train_loader = datasets_dict[self.dataset_name]['train']
        val_loader = datasets_dict[self.dataset_name]['val']

        # Initialize model
        backbone_model = PretrainedModel(model_name=model_choice, freeze=True)

        # Initialize prompts
        prompt_init_type = self.config['model'].get('prompt_init_type', 'zeros')
        prompt_type = self.config['model'].get('prompt_type', 'pixel')
        prompts = PromptGenerator(
            prompt_size=prompt_size,
            prompt_type=prompt_type,
            prompt_init_type=prompt_init_type
        )

        # Class names for source and target
        source_class_names = self._get_source_class_names(model_choice)
        target_class_names = self._get_target_class_names(self.dataset_name)

        # Initialize label mapper
        map_params = {
            'batch_size': self.config['training'].get('batch_size', 32),
            'num_source_classes_per_target': self.config['dataset'].get('num_classes', None)  # Will be used for strategies if needed
        }
        label_mapper = LabelMapper(
            strategy=label_map_strategy,
            source_class_names=source_class_names,
            target_class_names=target_class_names,
            map_params=map_params,
            device=self.device
        )

        # Initialize optimizer for prompts and label mapping parameters only
        optimizer_params = list(prompts.prompt_tensor.parameters())
        if hasattr(prompts, 'real_coeffs'):
            optimizer_params += list(prompts.real_coeffs.parameters())
        if hasattr(prompts, 'imag_coeffs'):
            optimizer_params += list(prompts.imag_coeffs.parameters())
        if label_map_strategy == 'FullyMap':
            optimizer_params += list(label_mapper.linear_mapping.parameters())

        optimizer = torch.optim.Adam(optimizer_params, lr=self.config['training']['learning_rate'],
                                     weight_decay=self.config['training'].get('weight_decay', 0))
        # Optional LR scheduler
        lr_scheduler = None
        if self.config['training'].get('lr_scheduler', None) == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.config['training'].get('epochs', 50))

        # Setup training
        trainer = Trainer(
            model=backbone_model,
            prompts=prompts,
            dataset={'train': train_loader, 'val': val_loader},
            label_mapper=label_mapper,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=self.config
        )

        # Run prompt tuning for specified epochs
        tuning_epochs = self.config['model'].get('tuning_epochs', 5)
        for epoch in range(tuning_epochs):
            trainer._set_optimizer_params()  # to ensure optimizer updates only prompt and label mapping
            loss, acc = trainer._train_one_epoch()
            # Validation
            val_metrics = trainer._validate()
            val_acc = val_metrics['accuracy']

            # Ray Tune reports metrics
            tune.report(validation_accuracy=val_acc)

            # Early stopping within trial isn't directly supported; rely on Ray's ASHA scheduler
            # As an extra, can implement manual early stopping based on a patience counter
            # (Ray handles early stopping via the scheduler's stop condition)

        # After tuning epochs, evaluate and record final validation accuracy
        final_eval = trainer._validate()
        tune.report(final_validation_accuracy=final_eval['accuracy'])

        # Save artifacts if needed (optional)
        # For the purpose of hyperparameter search, only validation accuracy is needed
        # The best configuration is obtained externally

    def _get_source_class_names(self, model_choice: str):
        """
        Retrieve the source class names (assumed to be ImageNet classes or from the model)
        For simplicity, use a fixed list or mock list.
        """
        # Placeholder: in real code, load actual classes if available
        return [f'class_{i}' for i in range(1000)]

    def _get_target_class_names(self, dataset_name: str):
        """
        Load dataset-specific class names for target dataset.
        """
        # Placeholder: load from dataset info or assume generic classes
        num_classes = self.config['dataset'].get('num_classes', 10)
        return [f'class_{i}' for i in range(num_classes)]
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\AutoVP\AutoVP_repo`
