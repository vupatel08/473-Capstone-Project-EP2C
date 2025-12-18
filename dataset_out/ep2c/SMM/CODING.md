# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## config.py

```python
# config.py

from typing import List, Tuple, Dict, Union

class Config:
    # Dataset configuration: defines dataset specifics for loading and preprocessing
    dataset: Dict[str, Union[str, List[int], int]] = {
        'name': 'CIFAR10',                     # Dataset name (e.g., CIFAR10, SVHN, etc.)
        'input_size': [32, 32],                # Size images are resized to before feeding into the model
        'train_split': 50000,                  # Total training set size
        'test_split': 10000,                   # Total testing set size
        'batch_size': 128                      # Batch size for training and evaluation
    }

    # Model configuration: specifies backbone architecture and pretrained setting
    model: Dict[str, Union[str, bool]] = {
        'name': 'ResNet50',                     # Model backbone: 'ResNet50' or 'ViT-B32'
        'pretrained': True                      # Whether to load ImageNet-pretrained weights
    }

    # Mask generator architecture hyperparameters
    mask_generator: Dict[str, Union[int, str]] = {
        'architecture_depth': 5,                # Number of convolutional + pooling layers (e.g., 5)
        'kernel_size': 3,                       # Kernel size for conv layers (3x3)
        'filters': 64,                          # Number of filters per conv layer
        'pooling_layers': 2,                     # Number of MaxPooling layers (e.g., 2)
        'output_ratio': '1/8'                   # Downsampling ratio of mask relative to input size
    }

    # Training hyperparameters for LR schedules, epochs, and regularization
    training: Dict[str, Union[str, float, List[int]]] = {
        'optimizer': 'Adam',                    # Optimizer type: 'Adam'
        'learning_rate': 0.01,                   # Initial LR for mask generator parameters (phi)
        'lr_decay_epochs': [100],               # Epochs at which LR decays
        'lr_decay_factor': 0.1,                  # Multiplicative decay factor
        'epochs': 200,                          # Total training epochs
        'pattern_lr': 0.001,                     # LR for the pattern delta
        'pattern_lr_decay_epochs': [100],       # LR decay epochs for pattern lr
        'pattern_lr_decay_factor': 0.1,         # LR decay factor for delta
        'weight_decay': 1e-4,                   # L2 regularization coefficient
        'pattern_init': 'zeros'                  # Pattern initialization method ('zeros' or 'random')
    }

    # Sampling and interpolation configuration
    sampling: Dict[str, Union[List[int], int]] = {
        'image_resize': [32, 32],                # Resize images to given size before processing
        'patch_size': 8                         # Patch size for patch-wise upsampling (e.g., 8)
    }

    # Evaluation settings and visualization toggle
    evaluation: Dict[str, Union[str, bool]] = {
        'metrics': 'accuracy',                  # Metric for evaluation
        'visualize': True                       # Whether to generate visualizations post-training
    }

    # Reproducibility seed
    reproducibility: Dict[str, int] = {
        'seed': 42                              # Random seed for reproducibility
    }

    # Additional method to load all configurations as dictionary (optional)
    def as_dict(self) -> Dict:
        return {
            'dataset': self.dataset,
            'model': self.model,
            'mask_generator': self.mask_generator,
            'training': self.training,
            'sampling': self.sampling,
            'evaluation': self.evaluation,
            'reproducibility': self.reproducibility
        }
```

## dataset_loader.py

```python
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
```

## evaluation.py

```python
## evaluation.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

class Evaluation:
    """
    Evaluation class for testing the fixed pre-trained classifier on reprogrammed images.
    Computes metrics such as accuracy and optionally visualizes reprogrammed images and masks.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        mask_generator=None,
        pattern=None,
        config: dict = None,
        device: torch.device = None
    ):
        """
        Initialize Evaluation with the classifier, optional reprogramming components, configuration.
        Args:
            model (torch.nn.Module): Fixed pre-trained classifier (f_P).
            mask_generator (object): Optional, for visualizing masks if available.
            pattern (torch.nn.Parameter): Optional, for visualizing pattern delta.
            config (dict): Configuration dict (from YAML).
            device (torch.device): Computation device.
        """
        self.model = model
        self.mask_generator = mask_generator
        self.pattern = pattern
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        self.visualize = self.config.get('evaluation', {}).get('visualize', False)
        self.num_visualize = 8  # Number of images to visualize if needed
        self.device = self.device

        # Set model to eval mode
        self.model.eval()
        self.model.to(self.device)

    def evaluate(self, data_loader, mask_generator=None, pattern=None):
        """
        Run inference on the dataset, compute accuracy, optionally visualize.
        Args:
            data_loader (torch.utils.data.DataLoader): Loader for test/validation dataset.
            mask_generator (object): Optional, for visualization.
            pattern (torch.nn.Parameter): Optional, for visualization.
        Returns:
            metrics (dict): Contains 'accuracy' (float).
        """
        total_samples = 0
        correct_predictions = 0

        # Accumulators for visualization if enabled
        vis_images, vis_reprogrammed, vis_masks = [], [], []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Run reprogramming step: resize, generate masks, add pattern
                # The evaluation can re-use the pipeline externally if needed,
                # here we assume no additional reprogramming, or define an internal method
                
                # For visualization, generate reprogrammed images
                reprogrammed_images = None
                masks_batch = None
                if self.mask_generator or mask_generator:
                    # Use provided mask_generator or class attribute
                    mg = mask_generator or self.mask_generator
                    delta = pattern or self.pattern
                    images_resized = self._resize(images, self.config['sampling'].get('image_resize', [32, 32]))
                    masks_batch = self._generate_masks(images_resized, mg)
                    delta_exp = delta.unsqueeze(0).to(self.device)
                    pattern_masked = delta_exp * masks_batch
                    reprogrammed_images = images_resized + pattern_masked
                else:
                    # No reprogramming; just use images
                    reprogrammed_images = images

                # Obtain logits
                logits = self.model(reprogrammed_images)
                _, predicted = torch.max(logits, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                # For visualization, store first batch images
                if self.config.get('evaluation', {}).get('visualize', False) and batch_idx == 0:
                    # Save original and reprogrammed images for visualization
                    # Denormalize images if normalization applied
                    if hasattr(reprogrammed_images, 'cpu'):
                        reprogrammed_images_vis = reprogrammed_images.cpu()
                        images_vis = images.cpu()
                        if masks_batch is not None:
                            masks_vis = masks_batch.cpu()
                        else:
                            masks_vis = None
                        vis_images.extend(images_vis[:self.num_visualize])
                        vis_reprogrammed.extend(reprogrammed_images_vis[:self.num_visualize])
                        if masks_vis is not None:
                            vis_masks.extend(masks_vis[:self.num_visualize])

        accuracy = correct_predictions / total_samples
        metrics = {'accuracy': accuracy}

        # Visualize if requested
        if self.config.get('evaluation', {}).get('visualize', False):
            self._visualize_results(vis_images, vis_reprogrammed, vis_masks)

        return metrics

    def _resize(self, images, size: list):
        """
        Resize images to target size using bilinear interpolation.
        """
        size_tuple = tuple(size)
        return torch.nn.functional.interpolate(images, size=size_tuple, mode='bilinear', align_corners=False)

    def _generate_masks(self, images_resized, mask_generator):
        """
        Generate sample-specific masks batch for input images.
        """
        batch_size = images_resized.size(0)
        H, W = images_resized.shape[2], images_resized.shape[3]
        masks_list = []

        # Generate singleton masks one by one due to patch-wise upsampling constraints
        for i in range(batch_size):
            img = images_resized[i].unsqueeze(0)  # shape [1, 3, H, W]
            mask_low_res = mask_generator.generate_mask(img)  # shape [1, 3, H', W']
            mask_upsampled = self._patch_upsample(mask_low_res, (H, W))
            masks_list.append(mask_upsampled)
        masks_batch = torch.cat(masks_list, dim=0)
        return masks_batch

    def _patch_upsample(self, mask, size: Tuple[int, int]):
        """
        Upsample mask via patch-wise (pixel) repetition to match original size.
        """
        H_in, W_in = size
        _, C, H', W' = mask.shape
        # Calculate patch size (tiles per pixel)
        patch_size_h = max(1, H_in // H')
        patch_size_w = max(1, W_in // W')
        # Repeat each pixel patch-wise
        upsampled = mask.repeat_interleave(patch_size_h, dim=2)
        upsampled = upsampled.repeat_interleave(patch_size_w, dim=3)
        # Crop to exact size
        upsampled_cropped = upsampled[:, :, :H_in, :W_in]
        return upsampled_cropped

    def _visualize_results(self, original_images, reprogrammed_images, masks):
        """
        Generate visualizations for the first few images: original, reprogrammed, masks, overlays.
        """
        num_images = min(self.num_visualize, len(original_images))
        plt.figure(figsize=(15, 5))
        for i in range(num_images):
            # Original image
            plt.subplot(3, num_images, i+1)
            self._imshow_tensor(original_images[i])
            plt.title('Original')
            plt.axis('off')

            # Reprogrammed image
            plt.subplot(3, num_images, i+1+num_images)
            self._imshow_tensor(reprogrammed_images[i])
            plt.title('Reprogrammed')
            plt.axis('off')

            # Mask overlay
            plt.subplot(3, num_images, i+1+2*num_images)
            mask_vis = masks[i] if masks is not None else None
            if mask_vis is not None:
                overlay = self._overlay_mask(reprogrammed_images[i], mask_vis)
                self._imshow_tensor(overlay)
            else:
                self._imshow_tensor(reprogrammed_images[i])
            plt.title('Mask Overlay')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _imshow_tensor(self, tensor_img):
        """
        Helper to show tensor image after denormalization.
        """
        img = tensor_img.clone()
        # If normalization was applied, unnormalize
        # Assuming normalization mean,std from dataset
        # For simplicity, assume images are in [0,1]
        np_img = img.permute(1, 2, 0).numpy()
        np_img = np.clip(np_img, 0, 1)
        plt.imshow(np_img)

    def _overlay_mask(self, image_tensor, mask_tensor, alpha: float=0.3):
        """
        Overlay mask on image for visualization.
        """
        image = image_tensor.permute(1, 2, 0).cpu().numpy()
        mask = mask_tensor.permute(1, 2, 0).cpu().numpy()
        mask_normalized = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        overlay = image * (1 - alpha) + mask_normalized * alpha
        overlay = np.clip(overlay, 0, 1)
        return overlay
```

## main.py

```python
## main.py
import os
import sys
import yaml
import torch
import numpy as np
import random
import argparse
from datetime import datetime

from dataset_loader import DatasetLoader
from model import Model
from mask_generator import MaskGenerator
from pattern import Pattern
from trainer import Trainer
from evaluation import Evaluation
from visualization import Visualizer

def main():
    # 1. Load configuration from YAML file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Set random seeds for reproducibility
    seed = config.get('reproducibility', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 3. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 4. Load dataset
    dataset_cfg = config['dataset']
    dataset_name = dataset_cfg.get('name', 'CIFAR10')
    input_size = tuple(dataset_cfg.get('input_size', [32, 32]))
    batch_size = dataset_cfg.get('batch_size', 128)
    train_split_size = dataset_cfg.get('train_split', None)
    test_split_size = dataset_cfg.get('test_split', None)

    data_loader_obj = DatasetLoader(
        dataset_name=dataset_name,
        input_size=input_size,
        batch_size=batch_size,
        seed=seed
    )
    train_loader, val_loader = data_loader_obj.load_data()
    print(f"Loaded dataset: {dataset_name}")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    # Log batch shapes
    for batch in train_loader:
        imgs, labels = batch
        print(f"Sample batch input shape: {imgs.shape}")
        break

    # 5. Load pre-trained classifier model
    model_cfg = config['model']
    model_name = model_cfg.get('name', 'ResNet50')
    pretrained = model_cfg.get('pretrained', True)
    classifier = Model(model_name=model_name, pretrained=pretrained).to(device)
    classifier.eval()
    print(f"Loaded pre-trained model: {model_name}")

    # 6. Instantiate mask generator
    mg_cfg = config['mask_generator']
    arch_depth = mg_cfg.get('architecture_depth', 5)
    kernel_size = mg_cfg.get('kernel_size', 3)
    filters = mg_cfg.get('filters', 64)
    pooling_layers = mg_cfg.get('pooling_layers', 2)
    output_ratio_str = mg_cfg.get('output_ratio', '1/8')
    # Parse ratio string to float
    output_ratio = float(eval(output_ratio_str))
    mask_generator = MaskGenerator(
        input_size=input_size,
        architecture_depth=arch_depth,
        kernel_size=kernel_size,
        filters=filters,
        pooling_layers=pooling_layers,
        output_ratio=output_ratio
    ).to(device)

    # 7. Initialize learnable pattern delta
    # Determine shape: channels * height * width
    channels = 3  # For RGB
    height, width = input_size
    pattern_shape = (channels, height, width)
    init_method = config['training'].get('pattern_init', 'zeros')
    pattern = Pattern(shape=pattern_shape, init_type=init_method).to(device)

    # 8. Setup optimizer: optimize mask generator and pattern
    pattern_lr = config['training'].get('pattern_lr', 0.001)
    lr_decay_epochs = config['training'].get('lr_decay_epochs', [100])
    lr_decay_factor = config['training'].get('lr_decay_factor', 0.1)
    optimizer = torch.optim.Adam(
        [{'params': mask_generator.parameters(), 'lr': pattern_lr},
         {'params': pattern.parameters(), 'lr': pattern_lr}],
        weight_decay=1e-4
    )

    # 9. Setup training parameters
    total_epochs = config['training'].get('epochs', 200)
    # Learning rate scheduler for mask generator and pattern (step decay)
    lr_schedule_steps = lr_decay_epochs
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_schedule_steps, gamma=lr_decay_factor)

    # 10. Instantiate trainer
    trainer = Trainer(
        model=classifier,
        mask_generator=mask_generator,
        pattern=pattern,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # 11. Training loop
    print(f"Starting training for {total_epochs} epochs.")
    for epoch in range(1, total_epochs + 1):
        trainer.train_epoch()
        # Step LR scheduler
        scheduler.step()

        # Optionally evaluate on validation set each epoch
        if (epoch % 10 == 0) or (epoch == total_epochs):
            val_results = trainer.evaluate(val_loader)
            print(f"Epoch {epoch} validation accuracy: {val_results['accuracy']:.4f}")

    # 12. Final evaluation on test set
    test_eval = Evaluation(
        model=classifier,
        mask_generator=mask_generator,
        pattern=pattern,
        config=config,
        device=device
    )
    test_metrics = test_eval.evaluate(train_loader=None, data_loader=val_loader)  # Using val_loader as test
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    # 13. Visualization of some reprogrammed images and masks, if enabled
    if config.get('evaluation', {}).get('visualize', False):
        print("Visualizing sample reprogrammed images and masks...")
        # Generate some samples from validation set
        images_list = []
        reprogrammed_list = []
        masks_list = []
        batch_for_vis = next(iter(val_loader))
        images_vis, labels_vis = batch_for_vis
        images_vis = images_vis.to(device)
        # Generate masks for visualization
        masks_vis = mask_generator(images_vis)
        delta_tensor = pattern.get_pattern().to(device).unsqueeze(0)
        masks_upsampled = masks_vis
        pattern_masked = delta_tensor * masks_upsampled
        reprogrammed_images = images_vis + pattern_masked

        # Save original, reprogrammed, masks
        visu = Visualizer(config)
        for i in range(min(8, len(images_vis))):
            orig_img = images_vis[i]
            reprogram_img = reprogrammed_images[i]
            mask_img = masks_upsampled[i]
            # Denormalize images for display if needed
            visu.show_original_image(orig_img, title="Original")
            visu.show_reprogrammed_image(reprogram_img, title="Reprogrammed")
            visu.show_mask(mask_img, title="Generated Mask")
            # Overlay visualization
            visu.show_mask_overlay(orig_img, mask_img, alpha=0.3, title="Overlay Mask")
        print("Visualization complete.")

    # 14. Save final pattern and mask generator parameters
    save_dir = "./smm_results"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(pattern.get_pattern().detach().cpu(), os.path.join(save_dir, "final_pattern.pt"))
    torch.save(mask_generator.state_dict(), os.path.join(save_dir, "mask_generator.pt"))
    print(f"Saved pattern and generator weights to {save_dir}")

if __name__ == "__main__":
    main()
```

## mask_generator.py

```python
## mask_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MaskGenerator(nn.Module):
    """
    Lightweight CNN-based mask generator for sample-specific multi-channel masks.
    Generates a mask of reduced resolution from input images and performs patch-wise
    upsampling via pixel repetition to match input image size.
    """
    def __init__(self,
                 input_size: Tuple[int, int],
                 architecture_depth: int = 5,
                 kernel_size: int = 3,
                 filters: int = 64,
                 pooling_layers: int = 2,
                 output_ratio: float = 1/8):
        """
        Initialize the MaskGenerator.
        Args:
            input_size (Tuple[int, int]): Size of input images (H, W).
            architecture_depth (int): Number of convolutional layers.
            kernel_size (int): Kernel size for Conv layers.
            filters (int): Number of filters/channels in each conv layer.
            pooling_layers (int): Number of MaxPool layers.
            output_ratio (float): Ratio of output mask size to input size.
        """
        super(MaskGenerator, self).__init__()
        self.input_size = input_size  # (H, W)
        self.architecture_depth = architecture_depth
        self.kernel_size = kernel_size
        self.filters = filters
        self.pooling_layers = pooling_layers
        self.output_ratio = output_ratio

        # Compute the size of the intermediate feature map after pooling
        # Calculate number of pooling steps to determine output size
        H, W = self.input_size
        for _ in range(self.pooling_layers):
            H = H // 2
            W = W // 2
        self.reduced_size = (H, W)  # Size after pooling

        # Build convolutional layers
        layers = []

        in_channels = 3  # Input image has 3 channels
        for layer_idx in range(self.architecture_depth):
            conv_layer = nn.Conv2d(in_channels, self.filters, kernel_size=self.kernel_size, padding=1)
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
            nn.init.zeros_(conv_layer.bias)
            layers.append(conv_layer)
            # Optional: batch normalization for training stability
            layers.append(nn.BatchNorm2d(self.filters))
            layers.append(nn.ReLU(inplace=True))
            # Add pooling layers as specified
            if layer_idx < self.pooling_layers:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = self.filters

        # Final convolution to produce 3-channel mask
        final_conv = nn.Conv2d(self.filters, 3, kernel_size=3, padding=1)
        nn.init.kaiming_uniform_(final_conv.weight, nonlinearity='relu')
        nn.init.zeros_(final_conv.bias)
        layers.append(final_conv)

        self.net = nn.Sequential(*layers)

        # Determine output size of CNN: (batch_size, 3, H', W')
        self.output_size = self.reduced_size
        # The patch size for upsampling (integer)
        H_in, W_in = self.input_size
        H_out, W_out = self.output_size

        # Compute patch size (assumes integral division)
        self.patch_size_h = max(1, H_in // H_out)
        self.patch_size_w = max(1, W_in // W_out)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generate the sample-specific mask for input images.
        Args:
            image (torch.Tensor): Batch of images, shape (B, 3, H, W).

        Returns:
            torch.Tensor: Masks of shape (B, 3, H, W), upsampled via patch-wise repetition.
        """
        # Pass through CNN to get low-res masks
        mask_low_res = self.net(image)  # shape (B, 3, H', W')
        # Patch-wise upsampling via pixel repetition
        masks_upsampled = self._patch_upsample(mask_low_res)
        return masks_upsampled

    def _patch_upsample(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Upsample the mask from reduced resolution to original size
        by repeating each pixel in spatial dimensions.
        Args:
            mask (torch.Tensor): shape (B, 3, H', W').

        Returns:
            torch.Tensor: shape (B, 3, H, W), the upsampled mask.
        """
        B, C, H', W' = mask.shape
        # Repeat each pixel patch size times both vertically and horizontally
        upsampled = mask.repeat_interleave(self.patch_size_h, dim=2)
        upsampled = upsampled.repeat_interleave(self.patch_size_w, dim=3)
        # The resulting size may be larger than input due to integer division
        # Trim to match input size
        H_in, W_in = self.input_size
        return upsampled[:, :, :H_in, :W_in]

```

## model.py

```python
## model.py
import torch
import torchvision.models as models
from typing import Optional

class Model:
    """
    Encapsulates a pre-trained backbone model (e.g., ResNet or ViT) with frozen parameters.
    Supports loading the specified architecture with pretrained weights and provides
    an inference interface.
    """

    def __init__(self, model_name: str = "ResNet50", pretrained: bool = True):
        """
        Loads the specified pre-trained model architecture, freezes its parameters,
        and prepares it for inference.

        Args:
            model_name (str): Name of the model architecture. Supported values:
                              "ResNet50", "ResNet18", "ViT-B32".
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        self.model_name = model_name
        self.pretrained = pretrained

        # Load the model based on architecture name
        if self.model_name == "ResNet50":
            self.model = models.resnet50(pretrained=self.pretrained)
        elif self.model_name == "ResNet18":
            self.model = models.resnet18(pretrained=self.pretrained)
        elif self.model_name == "ViT-B32":
            # Assuming torchvision supports ViT-B32 (if not, replace with appropriate loading)
            # As of torchvision 0.11.1, ViT is not supported, so add custom loading or handle accordingly.
            # For this code, we attempt to load from torchvision if available
            try:
                self.model = models.vit_b_32(pretrained=self.pretrained)
            except AttributeError:
                raise NotImplementedError(
                    "ViT-B32 not available in torchvision models for this version. "
                    "Please implement a custom loader or update torchvision."
                )
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        # Freeze parameters to prevent updates during training
        for param in self.model.parameters():
            param.requires_grad = False

        # Set model to evaluation mode
        self.model.eval()

        # Store device info
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the frozen pre-trained model.

        Args:
            x (torch.Tensor): Batch of input images, shape [B, C, H, W].

        Returns:
            torch.Tensor: Model output logits, shape [B, num_classes].
        """
        with torch.no_grad():
            logits = self.model(x)
        return logits
```

## pattern.py

```python
## pattern.py
import torch
import torch.nn as nn
from typing import Tuple

class Pattern(nn.Module):
    """
    Represents the learnable prompt pattern (delta) in the reprogramming framework.
    This tensor is shared across all samples and is optimized during training.
    """

    def __init__(self,
                 shape: Tuple[int, int, int],
                 init_type: str = 'zeros'):
        """
        Initializes the Pattern tensor.

        Args:
            shape (Tuple[int, int, int]): (channels, height, width) of the pattern.
            init_type (str): Initialization method; default is 'zeros'.
                             Can be extended to other methods if needed.
        """
        super(Pattern, self).__init__()
        c, h, w = shape

        if init_type == 'zeros':
            pattern_tensor = torch.zeros(c, h, w)
        elif init_type == 'random':
            pattern_tensor = torch.randn(c, h, w)
        else:
            raise ValueError(f"Unsupported init_type: {init_type}")

        # Register as a trainable parameter
        self.pattern = nn.Parameter(pattern_tensor)

    def get_pattern(self) -> torch.Tensor:
        """
        Returns the current pattern tensor.

        Returns:
            torch.Tensor: The learnable pattern tensor of shape (C, H, W).
        """
        return self.pattern

    def reset(self, init_type: str = 'zeros') -> None:
        """
        Reinitializes the pattern tensor.

        Args:
            init_type (str): Reinitialization method; default is 'zeros'.
        """
        with torch.no_grad():
            if init_type == 'zeros':
                self.pattern.copy_(torch.zeros_like(self.pattern))
            elif init_type == 'random':
                self.pattern.copy_(torch.randn_like(self.pattern))
            else:
                raise ValueError(f"Unsupported init_type: {init_type}")
```

## requirements.txt

# requirements.txt

torch==1.10.0
torchvision==0.11.1
numpy==1.21.0
matplotlib==3.4.3
Pillow==8.4.0
scipy==1.7.0

## trainer.py

```python
# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import Optional
from tqdm import tqdm
import numpy as np

class Trainer:
    """
    Trainer class for training the SMM-based visual reprogramming framework.
    Handles the training loop updating only the mask generator parameters and pattern delta.
    """
    def __init__(
        self,
        model,
        mask_generator,
        pattern,
        train_loader,
        val_loader,
        config,
        device: Optional[torch.device]=None,
    ):
        """
        Initialize the trainer with model, dataset, configs, etc.
        Args:
            model (torch.nn.Module): fixed pre-trained classifier (f_P).
            mask_generator (MaskGenerator): the mask generator module with parameters phi.
            pattern (Pattern): shared learnable pattern delta.
            train_loader (DataLoader): training data loader.
            val_loader (DataLoader): validation data loader, optional.
            config (dict): configurations loaded from config.yaml.
            device (torch.device): computation device.
        """
        self.model = model
        self.mask_generator = mask_generator
        self.pattern = pattern
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        # Set model to eval mode, fix parameters
        self.model.eval()
        self.model.to(self.device)

        # Only optimize phi (mask generator params) and pattern delta
        self.optimizer = optim.Adam(
            list(self.mask_generator.parameters()) + [self.pattern.pattern],
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Setup scheduler for phi and pattern LR decays
        self.lr_decay_epochs = self.config['training'].get('lr_decay_epochs', [])
        self.lr_decay_factor = self.config['training'].get('lr_decay_factor', 1.0)
        self.pattern_lr = self.config['training'].get('pattern_lr', 0.001)
        self.pattern_lr_decay_epochs = self.config['training'].get('pattern_lr_decay_epochs', [])
        self.pattern_lr_decay_factor = self.config['training'].get('pattern_lr_decay_factor', 1.0)

        # Optimizers for pattern delta and mask generator separately can be used if needed, 
        # but here combined for simplicity.
        self.current_lr = self.config['training']['learning_rate']
        self.pattern_lr_value = self.pattern_lr

        # Training parameters
        self.epochs = self.config['training']['epochs']
        self.pattern_lr_epochs = self.config['training'].get('pattern_lr_decay_epochs', [])
        self.pattern_lr_decay_epochs = set(self.pattern_lr_epochs)
        self.learn_rate = self.config['training']['learning_rate']
        self.pattern_lr_value = self.config['training']['pattern_lr']
        self.smoothed_train_loss = None

        # For logging
        self.best_val_acc = 0.0
        self.best_model_state = None

        # For verbose progress bar
        self.use_tqdm = True

        # Setup loss criterion
        self.criterion = nn.CrossEntropyLoss()

    def adjust_learning_rate(self, epoch: int):
        """
        Decay learning rate at scheduled epochs.
        """
        if epoch in self.lr_decay_epochs:
            self.current_lr *= self.lr_decay_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
        if epoch in self.pattern_lr_decay_epochs:
            self.pattern_lr_value *= self.pattern_lr_decay_factor
            # Update the learning rate for pattern delta specifically if needed
            # (In this implementation, both are in same optimizer, so we keep as is)
            # Could separate optimizers for more control

    def resize_image(self, images):
        """
        Resize batch of images to input size specified in config.
        Uses bilinear interpolation.
        """
        size = self.config['sampling']['image_resize']
        # images: tensor shape [B, C, H, W]
        return nn.functional.interpolate(images, size=size, mode='bilinear', align_corners=False)

    def generate_masks(self, images):
        """
        For each image in batch, generate sample-specific mask via mask generator,
        perform patch-wise upsampling to match input size.
        Returns batch of masks [B, C, H, W].
        """
        batch_size = images.size(0)
        H, W = images.shape[2], images.shape[3]
        masks = []

        # Generate masks sample-wise
        with torch.set_grad_enabled(True):
            for i in range(batch_size):
                img = images[i].unsqueeze(0)  # shape [1, 3, H, W]
                mask_low_res = self.mask_generator.generate_mask(img)  # shape [1, 3, H', W']
                mask_upsampled = self._patch_upsample(mask_low_res, (H, W))
                masks.append(mask_upsampled)
        masks = torch.cat(masks, dim=0)  # shape [B, 3, H, W]
        return masks

    def _patch_upsample(self, mask, size):
        """
        Upsample mask from low resolution to target size via patch-wise repetition.
        Args:
            mask: tensor [1, 3, H', W']
            size: (H, W) tuple
        """
        H_in, W_in = size
        B, C, H', W' = mask.shape
        # Calculate patch size
        patch_size_h = max(1, H_in // H')
        patch_size_w = max(1, W_in // W')
        # Repeat each pixel patch-wise
        upsampled = mask.repeat_interleave(patch_size_h, dim=2)
        upsampled = upsampled.repeat_interleave(patch_size_w, dim=3)
        # Crop to exact size
        upsampled = upsampled[:, :, :H_in, :W_in]
        return upsampled

    def train(self):
        """
        Main training loop
        """
        pattern_lr = self.pattern_lr_value
        print("Starting training...")
        for epoch in range(1, self.epochs + 1):
            self.adjust_learning_rate(epoch)
            epoch_loss = 0.0
            epoch_acc = 0.0
            total_samples = 0

            # Prepare tqdm progress bar if enabled
            pbar = tqdm(self.train_loader) if self.use_tqdm else self.train_loader

            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Resize images to input size
                images_resized = self.resize_image(images)  # shape [B, C, H_in, W_in]

                # Generate sample-specific masks
                masks = self.generate_masks(images_resized)  # shape [B, C, H_in, W_in]

                # Expand delta pattern to batch, shape [1, C, H, W]
                delta = self.pattern.pattern
                delta_expanded = delta.unsqueeze(0)  # shape [1, C, H, W]
                delta_expanded = delta_expanded.to(self.device)

                # Element-wise multiply masks with delta
                pattern_masked = delta_expanded * masks

                # Add pattern to the resized images
                reprogrammed_images = images_resized + pattern_masked

                # Obtain logits from fixed model
                logits = self.model(reprogrammed_images)

                # Compute loss
                loss = self.criterion(logits, labels)

                # Zero gradients
                self.optimizer.zero_grad()

                # Backpropagate loss
                loss.backward()

                # Update only phi and delta; since only those are in optimizer, here just step
                self.optimizer.step()

                # Track epoch loss and accuracy
                batch_loss = loss.item()
                epoch_loss += batch_loss * images.size(0)
                _, predicted = torch.max(logits, 1)
                correct = (predicted == labels).sum().item()
                epoch_acc += correct
                total_samples += images.size(0)

                if self.use_tqdm:
                    pbar.set_description(f"Epoch {epoch}/{self.epochs} | Loss: {batch_loss:.4f} | Acc: {correct / images.size(0):.4f}")

            # Compute average loss and accuracy
            avg_loss = epoch_loss / total_samples
            avg_acc = epoch_acc / total_samples

            print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")

            # Optional: LR decay
            if epoch in self.lr_decay_epochs:
                self.current_lr *= self.lr_decay_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                print(f"Learning rate decayed to {self.current_lr}")

            # Optional: Save best model based on validation accuracy, if validation is provided
            if self.val_loader is not None:
                val_acc = self.evaluate(self.val_loader)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    # Save relevant state
                    self.best_model_state = {
                        'phi': self.mask_generator.state_dict(),
                        'delta': self.pattern.pattern.data.clone()
                    }

        print("Training finished.")
        # Load best model if validation was used
        if self.best_model_state is not None:
            self.mask_generator.load_state_dict(self.best_model_state['phi'])
            self.pattern.pattern.data.copy_(self.best_model_state['delta'])

    def evaluate(self, data_loader):
        """
        Evaluate the model on validation/test set; compute accuracy.
        """
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                images_resized = self.resize_image(images)
                # Generate masks
                masks = self.generate_masks(images_resized)
                delta = self.pattern.pattern
                delta_exp = delta.unsqueeze(0).to(self.device)

                pattern_masked = delta_exp * masks
                reprogrammed_images = images_resized + pattern_masked

                logits = self.model(reprogrammed_images)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        self.model.train()
        print(f"Validation Accuracy: {accuracy:.4f}")
        return accuracy
```

## visualization.py

```python
## visualization.py
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Optional, List, Tuple

class Visualizer:
    """
    Provides visualization utilities for original images, reprogrammed images, masks, and overlay visualizations.
    """

    def __init__(self, config: dict):
        """
        Initialize the visualizer with optional configuration.
        Args:
            config (dict): Configuration dictionary, used to set visualization defaults. 
                           Expected keys: 'evaluation'->'visualize' (bool).
        """
        self.config = config
        # Number of images to visualize in overlay (set to 8 or less)
        self.num_visualize = 8
        self.use_vis = self.config.get('evaluation', {}).get('visualize', False)

    def show_original_image(self, image_tensor: torch.Tensor, title: Optional[str] = None) -> None:
        """
        Display an original input image.
        Args:
            image_tensor (torch.Tensor): Tensor of shape (C, H, W), assumed normalized.
            title (str): Optional figure title.
        """
        np_img = self._tensor_to_np(image_tensor)
        plt.figure()
        plt.imshow(np_img, interpolation='nearest')
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()

    def show_reprogrammed_image(self, reprogrammed_image: torch.Tensor, title: Optional[str] = None) -> None:
        """
        Display a reprogrammed (pattern + image) image.
        Args:
            reprogrammed_image (torch.Tensor): Tensor shape (C, H, W), normalized.
            title (str): Optional title.
        """
        np_img = self._tensor_to_np(reprogrammed_image)
        plt.figure()
        plt.imshow(np_img, interpolation='nearest')
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()

    def show_mask(self, mask_tensor: torch.Tensor, title: Optional[str] = None) -> None:
        """
        Display a mask tensor as a heatmap.
        Args:
            mask_tensor (torch.Tensor): Tensor shape (H, W, 3) with values typically [0,1].
            title (str): Optional figure title.
        """
        np_mask = self._tensor_to_np(mask_tensor)
        plt.figure()
        # For masks, using 'viridis' colormap for better contrast
        plt.imshow(np_mask, cmap='viridis')
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()

    def show_mask_overlay(
        self,
        original_image: torch.Tensor,
        mask: torch.Tensor,
        alpha: float = 0.3,
        title: Optional[str] = None
    ) -> None:
        """
        Overlay a mask onto an original image for visual assessment.
        Args:
            original_image (torch.Tensor): (C, H, W), normalized.
            mask (torch.Tensor): (H, W, 3), values in [0,1].
            alpha (float): Transparency factor for overlay.
            title (str): Optional figure title.
        """
        img_np = self._tensor_to_np(original_image)
        mask_np = self._tensor_to_np(mask)
        overlay = self._blend_images(img_np, mask_np, alpha)
        plt.figure()
        plt.imshow(overlay, interpolation='nearest')
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()

    def _tensor_to_np(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a tensor to a numpy image scaled between 0 and 1.
        Handles (C, H, W) with normalization assumed.
        """
        if tensor.ndim == 3:
            img = tensor.detach().cpu()
            # If normalization mean/std are applied, unnormalize here if known.
            # For generality, assume inputs are normalized; for display, clip to [0,1].
            img = torch.clamp(img, 0, 1)
            np_img = img.permute(1, 2, 0).numpy()
        elif tensor.ndim == 2:
            # For masks or grayscale images
            np_img = tensor.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
        return np_img

    def _blend_images(self, img_np: np.ndarray, mask_np: np.ndarray, alpha: float) -> np.ndarray:
        """
        Blend the original image with mask heatmap using alpha transparency.
        """
        blended = img_np * (1 - alpha) + mask_np * alpha
        blended = np.clip(blended, 0, 1)
        return blended
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\SMM\SMM_repo`
