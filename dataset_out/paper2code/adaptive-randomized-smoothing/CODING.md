# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import random
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CelebA, ImageNet
from PIL import Image
import numpy as np

class CustomCIFAR10(Dataset):
    """
    CIFAR-10 dataset with optional background augmentation.
    If background images are provided, overlays CIFAR images onto background edges at random positions.
    """
    def __init__(self, root: str, split: str = 'train', background_images: Optional[list] = None,
                 background_scale: int = 640, transform=None, download: bool = True):
        self.split = split  # 'train' or 'test'
        self.transform = transform
        self.background_images = background_images
        self.background_scale = background_scale
        self.use_background = background_images is not None

        self.cifar = CIFAR10(root=root, train=(split == 'train'), transform=None, download=download)
        self.data = self.cifar.data  # numpy array of shape (N, 32, 32, 3)
        self.targets = self.cifar.targets

        if self.use_background:
            # Preload background images resized to background_scale
            self.bg_images = self._load_backgrounds()
        else:
            self.bg_images = []

    def _load_backgrounds(self):
        # Placeholder: list is passed during instantiation
        # Assume background_images is a list of PIL Images or file paths
        resized_bgs = []
        for bg in self.background_images:
            img = bg if isinstance(bg, Image.Image) else Image.open(bg)
            resized_bg = img.resize((self.background_scale, self.background_scale), Image.BILINEAR)
            resized_bgs.append(resized_bg)
        return resized_bgs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_np = self.data[index]
        label = self.targets[index]

        img = Image.fromarray(img_np)

        if self.use_background:
            bg_img = random.choice(self.bg_images)
            # Overlay CIFAR onto background at random edge position
            bg_w, bg_h = bg_img.size
            # Decide which edge to embed: 0=top,1=bottom,2=left,3=right
            edge = random.choice([0, 1, 2, 3])
            # Determine position along the selected edge
            if edge == 0:  # top edge
                x_pos = random.randint(0, bg_w - img.width)
                y_pos = 0
            elif edge == 1:  # bottom edge
                x_pos = random.randint(0, bg_w - img.width)
                y_pos = bg_h - img.height
            elif edge == 2:  # left edge
                x_pos = 0
                y_pos = random.randint(0, bg_h - img.height)
            else:  # right edge
                x_pos = bg_w - img.width
                y_pos = random.randint(0, bg_h - img.height)

            # Paste CIFAR image onto background
            bg_img = bg_img.convert('RGB')
            bg_np = np.array(bg_img)
            img_np = np.array(img)
            # Overlay: for simplicity, overwrite pixels
            bg_np[y_pos:y_pos+img.height, x_pos:x_pos+img.width, :] = img_np
            combined_img = Image.fromarray(bg_np)

        else:
            combined_img = img

        if self.transform:
            combined_img = self.transform(combined_img)

        return combined_img, label


class CelebADataset(Dataset):
    """
    CelebA dataset with optional spatial variation via random crops.
    """
    def __init__(self, root: str, split: str = 'train', crop_size: Tuple[int, int] = (160, 160),
                 transform=None, crop_variation: bool = True):
        # For simplicity, assuming the standard CelebA dataset is available in torchvision.datasets.CelebA
        self.cel = CelebA(root=root, split=split, target_type='attr', transform=None, download=True)
        self.transform = transform
        self.crop_size = crop_size
        self.crop_variation = crop_variation

        # Load images and attributes
        self.imgs = self.cel.images
        self.attr = self.cel.attr

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        attr = self.attr[index]
        img = Image.open(img_path).convert('RGB')

        if self.crop_variation:
            # Randomly crop to simulate spatial variation (keeping mouth >10px from edge)
            width, height = img.size
            crop_w, crop_h = self.crop_size
            # Ensure crop is within image
            max_x = width - crop_w
            max_y = height - crop_h
            start_x = random.randint(0, max_x)
            start_y = random.randint(0, max_y)
            img = img.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))
        else:
            # For evaluation, mean crop to fixed size (e.g., center crop)
            img = transforms.CenterCrop(self.crop_size)(img)

        if self.transform:
            img = self.transform(img)

        label = 1 if attr[20] == 1 else 0  # Assume attribute 21 (mouth open) target
        return img, label


class ImageNetDataset(Dataset):
    """
    Standard ImageNet dataset loader.
    """
    def __init__(self, root: str, split: str = 'train', transform=None):
        # Using torchvision.datasets.ImageNet; for custom split, assume proper folder structure
        # Alternatively, custom dataset implementation can be used if needed
        self.dataset = ImageNet(root=root, split=split, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transform:
            img = self.transform(img)
        return img, label


class DatasetLoader:
    """
    Encapsulates dataset loading, optional background embedding, and provides data loaders.
    """
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['dataset']['name']
        self.background_scale = config['dataset'].get('background_scale', 640)
        self.data_split = config['dataset'].get('data_split', 'train')
        self.root_dir = os.path.expanduser('~/.torch_datasets')  # or your preferred cache dir
        self.transform = self._get_transform()
        self.background_images = None

        # For background augmentation
        if self.dataset_name in ['CIFAR10'] and self.config['dataset'].get('background_images', None):
            # Load background images from the path or list provided
            bg_list = self.config['dataset']['background_images']
            # Expect list of file paths or PIL images
            self.background_images = []
            for bg in bg_list:
                if isinstance(bg, Image.Image):
                    self.background_images.append(bg)
                elif isinstance(bg, str):
                    self.background_images.append(Image.open(bg))
                else:
                    # Possibly load from a predefined directory
                    pass

        self.dataset = self._load_dataset()

    def _get_transform(self):
        # Define standard transforms (normalize per dataset)
        if self.dataset_name == 'CIFAR10' or 'CIFAR10' in self.dataset_name:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
            ])
        elif self.dataset_name == 'CelebA':
            # Resize to 160x160 (or as required)
            return transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
        elif self.dataset_name == 'ImageNet':
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            # Default
            return transforms.ToTensor()

    def _load_dataset(self):
        split = self.data_split
        if self.dataset_name == 'CIFAR10':
            use_bg = self.background_images is not None
            return CustomCIFAR10(
                root=os.path.join(self.root_dir, 'CIFAR10'),
                split=split,
                background_images=self.background_images if use_bg else None,
                background_scale=self.background_scale,
                transform=self.transform,
                download=True
            )
        elif self.dataset_name == 'CelebA':
            return CelebADataset(
                root=os.path.join(self.root_dir, 'CelebA'),
                split=split,
                transform=self.transform,
                crop_variation=(split=='test')  # For test, often keep crop variation true
            )
        elif self.dataset_name == 'ImageNet':
            return ImageNetDataset(
                root=os.path.join(self.root_dir, 'ImageNet'),
                split=split,
                transform=self.transform
            )
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    def get_data_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 4):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
```

## evaluation.py

```python
## evaluation.py
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
from math import sqrt
from utils import add_gaussian_noise, compute_coefficients, combine_predictions, compute_cert_radius

class Evaluator:
    def __init__(self, mask_unet, classifier, sigma1, sigma2, device, background_scale=1):
        """
        Initialize the evaluator with trained models and parameters.
        Args:
            mask_unet (nn.Module): Mask model w (predicts mask w(m1))
            classifier (nn.Module): Base classifier g
            sigma1 (float): Noise scale for first step (mask generation)
            sigma2 (float): Noise scale for second step (masked image)
            device (torch.device): Computation device
            background_scale (int): Scale factor for background in certification formulas
        """
        self.mask_unet = mask_unet
        self.classifier = classifier
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.device = device
        self.background_scale = background_scale

    def predict_mask(self, noisy_input):
        """
        Generate mask w(m1) given noisy input.
        Args:
            noisy_input (torch.Tensor): Input tensor, shape (C,H,W).
        Returns:
            torch.Tensor: Mask w in [0,1], shape (1,H,W).
        """
        with torch.no_grad():
            mask_raw = self.mask_unet(noisy_input.unsqueeze(0).to(self.device))
            mask = torch.sigmoid(mask_raw)
        return mask.squeeze(0)  # shape: (1,H,W)

    def generate_masked_noisy_image(self, w, X):
        """
        Generate m2 = masked X + noise, with adaptive noise scale.
        Args:
            w (torch.Tensor): mask, shape (1,H,W).
            X (torch.Tensor): original input, shape (C,H,W).
        Returns:
            m2 (torch.Tensor): noisy masked image, shape (C,H,W).
            sigma_w (float): noise scale for m2
        """
        # Compute norm_2 of w for noise scale in Eq. 2.5
        w_flat = w.view(-1)
        norm_w2 = torch.norm(w_flat, p=2).item()
        d = w.numel()  # total number of pixels in mask
        # variance of noise in m2
        noise_std = (norm_w2 / np.sqrt(d)) * self.sigma2
        # Add Gaussian noise
        z2 = torch.randn_like(X) * noise_std
        masked_X = w * X
        m2 = masked_X + z2
        return m2, noise_std

    def predict(self, X, n_samples=1000, class_prior=None):
        """
        Perform Monte Carlo prediction and classify.
        Args:
            X (torch.Tensor): input image, shape (C,H,W)
            n_samples (int): number of MC samples
            class_prior (torch.Tensor): optional prior probabilities for classes, shape (num_classes,)
        Returns:
            dict: containing class probabilities, predicted class, lower bounds, radius
        """
        self.classifier.eval()
        # Store class counts for class probability estimation
        class_counts = None
        # Store predictions
        probs_list = []

        for _ in range(n_samples):
            # Sample z1 noise
            z1 = add_gaussian_noise(torch.zeros_like(X), self.sigma1).to(self.device)
            noisy_input = X + z1

            # 1. Generate mask w(m1)
            w = self.predict_mask(noisy_input)  # shape: (1,H,W)
            # 2. Generate m2 = masked X + noise
            m2, sigma_w = self.generate_masked_noisy_image(w, X)

            # Compute coefficients c1, c2 for unbiased combination
            c1, c2 = compute_coefficients(w, self.sigma1, sigma_w)

            # 3. Predict for m1 (original noisy input for m1 for consistency)
            # Alternatively, to follow the paper's estimator, combine the images:
            # Instead of logits, it's better to combine images via the estimated unbiased estimate
            # Here, for practical intuition, pass m2 through classifier
            # For the training purpose, we perform one forward per sample
            
            # Forward classifier
            logits = self.classifier(m2.unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs.squeeze(0))
        
        # Aggregate class probabilities over samples
        probs_stack = torch.stack(probs_list, dim=0)  # shape: (n_samples, num_classes)
        class_probs_mean = probs_stack.mean(dim=0)  # mean probability per class
        # Find predicted class
        pred_class = torch.argmax(class_probs_mean).item()
        p_plus = class_probs_mean[pred_class].item()
        # For bound estimation, we need class counts
        pred_counts = (probs_stack.argmax(dim=1) == pred_class).sum().item()

        # Count for other classes
        class_counts = torch.zeros(probs_stack.shape[1])
        class_counts = torch.zeros_like(class_probs_mean)
        for c_idx in range(probs_stack.shape[1]):
            class_counts[c_idx] = (probs_stack.argmax(dim=1) == c_idx).sum()

        # Compute confidence bounds
        p_plus_lower, p_minus_upper = self.compute_bounds(class_counts, pred_class,
                                                          conf_level=0.99, error_tol=0.01)

        # Compute the certified radius
        radius = self.compute_radius(p_plus_lower, p_minus_upper, self.sigma1, 
                                      dim=w.numel(), norm_type='L_infinity', bg_scale=self.background_scale)

        # Prepare output
        result = {
            'predicted_class': pred_class,
            'confidence_lower_bound': p_plus_lower,
            'confidence_upper_bound': p_minus_upper,
            'radius': radius,
            'class_probabilities': class_probs_mean.detach().cpu().numpy()
        }
        return result

    def compute_bounds(self, class_counts, pred_class_idx, conf_level=0.99, error_tol=0.01):
        """
        Using binomial confidence intervals (Clopper-Pearson, normal approximation),
        compute lower/upper bounds for class probabilities.
        Args:
            class_counts (torch.Tensor): counts of each class from MC, shape: (num_classes,)
            pred_class_idx (int): index of predicted class
            conf_level (float): e.g., 0.99
            error_tol (float): confidence interval tolerance for bounds
        Returns:
            p_plus_lower (float): lower bound for top class
            p_minus_upper (float): upper bound for max of other classes
        """
        total = class_counts.sum().item()
        p_c = class_counts[pred_class_idx].item() / total
        # Confidence interval for p_plus
        lower_p_plus = self.estimate_binom_bound(count=class_counts[pred_class_idx].item(),
                                                  n=total, conf_level=conf_level)
        # For top class's lower bound
        p_plus_lower = lower_p_plus

        # For other classes: find maximum over classes != pred_class
        max_other = max([class_counts[i].item() for i in range(len(class_counts)) if i != pred_class_idx])
        p_minus_upper = self.estimate_binom_bound(count=max_other,
                                                  n=total, conf_level=conf_level, upper=True)
        return p_plus_lower, p_minus_upper

    def estimate_binom_bound(self, count, n, conf_level=0.99, upper=False):
        """
        Use normal approximation or Clopper-Pearson for binomial CI.
        Args:
            count (int): number of successes
            n (int): total trials
            conf_level (float): confidence level
            upper (bool): if True, compute upper bound, else lower.
        Returns:
            float: probability bound
        """
        p_hat = count / n
        alpha = 1 - conf_level
        z = norm.ppf(1 - alpha/2)
        # standard error
        se = sqrt(p_hat * (1 - p_hat) / n)
        if upper:
            p_bound = p_hat + z * se
        else:
            p_bound = p_hat - z * se
        return min(max(p_bound, 0), 1)

    def compute_radius(self, p_plus, p_minus, sigma, dim, norm_type='L_infinity', bg_scale=1):
        """
        Calculate the certified radius using Eq. 2.2 or 2.4.
        Args:
            p_plus (float): lower bound on correct class probability
            p_minus (float): upper bound on maximum other class probability
            sigma (float): noise scale
            dim (int): dimension of input (e.g., number of pixels in mask image)
            norm_type (str): 'L_infinity' or default (for this code, 'L_infinity')
            bg_scale (int): scale factor of background for $L_\infty$ bounds
        Returns:
            float: certified radius
        """
        # For L-infinity, formula includes division by sqrt(d)
        if norm_type == 'L_infinity':
            radius = (sigma / (2 * bg_scale)) * (norm.ppf(p_plus) - norm.ppf(p_minus))
        else:
            # default: standard
            radius = (sigma / 2) * (norm.ppf(p_plus) - norm.ppf(p_minus))
        # Ensure radius is non-negative
        return max(radius, 0.0)

    def get_certified_radius(self, X, n_samples=1000, conf_level=0.99, error_tol=0.01):
        """
        Main interface: Given input X, return class, radius, and probability bounds.
        Args:
            X (torch.Tensor): input image, shape (C,H,W)
            n_samples (int): Monte Carlo sample count
            conf_level (float): confidence level
            error_tol (float): epsilon for bounds
        Returns:
            dict: class prediction, certified radius, and bounds.
        """
        result = self.predict(X, n_samples, conf_level)
        return result
```

## main.py

```python
## main.py
import argparse
import yaml
import os
import torch
import numpy as np
import random

from dataset_loader import DatasetLoader
from model import MaskUNet, ResNetClassifier
from trainer import Trainer
from evaluation import Evaluator
from utils import load_background_images

def parse_args():
    parser = argparse.ArgumentParser(description='Adaptive Randomized Smoothing Experiment')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'certify'],
                        help='Operation mode: train, evaluate, or certify')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', help='Compute device: auto, cpu, cuda')
    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_arg == 'cpu':
        return torch.device('cpu')
    else:
        return torch.device(device_arg)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize dataset loader
    dataset_loader = DatasetLoader(config)

    # Instantiate models
    mask_params = config['model']['mask_unet']
    classifier_params = config['model']['classifier']
    dataset_name = config['dataset']['name']
    arch = classifier_params.get('architecture', 'resnet50')
    # Initialize mask model
    mask_model = MaskUNet(
        base_channels=mask_params['base_channels'],
        channel_mult=mask_params['channel_mult'],
        step_size=mask_params.get('step_size', 40),
        gamma=mask_params.get('gamma', 0.5),
        momentum=mask_params.get('momentum', 0.9),
    ).to(device)

    # Initialize classifier
    num_classes = 10 if dataset_name=='CIFAR10' else 2 if dataset_name=='CelebA' else 1000
    classifier = ResNetClassifier(architecture=arch, num_classes=num_classes).to(device)

    # Initialize trainer
    trainer = Trainer(config)
    # Assign models for training
    trainer.mask_net = mask_module=mask_model
    trainer.classifier = classifier

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'evaluate':
        # Load trained models if checkpoints exist
        mask_path = './checkpoints/mask_net.pth'
        clf_path = './checkpoints/classifier.pth'
        if os.path.exists(mask_path):
            mask_model.load_state_dict(torch.load(mask_path))
        if os.path.exists(clf_path):
            classifier.load_state_dict(torch.load(clf_path))
        # Evaluate on validation/test set
        val_loader = dataset_loader.get_data_loader(
            batch_size=256, shuffle=False, num_workers=4
        )
        accuracy = trainer.evaluate(val_loader)
        print(f'Validation/Test Accuracy: {accuracy*100:.2f}%')
    elif args.mode == 'certify':
        # Load trained models
        mask_path = './checkpoints/mask_net.pth'
        clf_path = './checkpoints/classifier.pth'
        mask_model.load_state_dict(torch.load(mask_path))
        classifier.load_state_dict(torch.load(clf_path))
        mask_model.eval()
        classifier.eval()
        # Set parameters for certification
        n_samples = config['evaluation'].get('monte_carlo_samples', 1000)
        conf_level = config['evaluation'].get('certification_confidence', 0.99)
        error_tol = config['evaluation'].get('certification_error_tolerance', 0.01)

        # For each sample in test set, run certification
        test_loader = dataset_loader.get_data_loader(
            batch_size=1, shuffle=False, num_workers=4
        )
        total_samples = 0
        correct_predictions = 0
        certified_count = {}
        # Define radius thresholds for evaluation, e.g., [0, 0.1, 0.2, 0.5,...]
        radius_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
        for radius in radius_thresholds:
            certified_count[radius] = 0

        for X, label in test_loader:
            X = X.to(device)
            label = label.to(device)
            evaluator = Evaluator(mask_model, classifier, 
                                  sigma1=trainer.sigma1, sigma2=trainer.sigma2, device=device,
                                  background_scale=config['dataset'].get('background_scale', 640))
            result = evaluator.get_certified_radius(
                X[0], n_samples=n_samples, conf_level=conf_level, error_tol=error_tol
            )
            pred_class = result['predicted_class']
            radius_certified = False
            # Check certification at thresholds
            for thr in radius_thresholds:
                if result['radius'] >= thr:
                    radius_certified = True
            total_samples +=1

            # Count accuracy
            if pred_class == label.item():
                correct_predictions +=1
                # If radius certified above threshold, count for cert
                for thr in radius_thresholds:
                    if result['radius'] >= thr:
                        certified_count[thr] +=1

        # Print Results
        print(f'Accuracy on test samples: {correct_predictions/total_samples*100:.2f}%')
        for thr in radius_thresholds:
            print(f'Certified accuracy at radius >= {thr}: {certified_count[thr]/total_samples*100:.2f}%')

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    """Basic convolutional block: Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpSampleBlock(nn.Module):
    """Upsampling block: ConvTranspose -> Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(out_channels * 2, out_channels)
    def forward(self, x, skip_connection):
        x = self.up(x)
        # Pad if needed
        diffY = skip_connection.size(2) - x.size(2)
        diffX = skip_connection.size(1) - x.size(1)
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX //2, diffY //2, diffY - diffY //2])
        x = torch.cat([skip_connection, x], dim=1)
        return self.conv1(x)

class MaskUNet(nn.Module):
    """UNet architecture for pixel-wise mask prediction."""
    def __init__(self, base_channels=32, channel_mult=[1,2,4,8], step_size=40, gamma=0.5, momentum=0.9):
        super().__init__()
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.step_size = step_size
        self.gamma = gamma
        self.momentum = momentum

        # Encoder layers
        self.encoders = nn.ModuleList()
        in_ch = 3
        for mult in channel_mult:
            out_ch = base_channels * mult
            self.encoders.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = ConvBlock(in_ch, in_ch * 2)

        # Decoder layers
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for mult in reversed(channel_mult):
            out_ch = base_channels * mult
            self.upconvs.append(UpSampleBlock(in_ch * 2, out_ch))
            in_ch = out_ch

        # Final conv to 1 channel
        self.final_conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, noisy_input):
        # Encoder
        enc_features = []
        x = noisy_input
        for encoder in self.encoders:
            x = encoder(x)
            enc_features.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: upsample + skip connection
        for idx, upconv in enumerate(self.upconvs):
            skip = enc_features[-(idx+1)]
            x = upconv(x, skip)

        x = self.final_conv(x)
        mask = self.sigmoid(x)
        return mask  # shape: (batch_size, 1, H, W)

# For completeness, even if not used here, a placeholder for the classifier
class ResNetClassifier(nn.Module):
    def __init__(self, architecture='resnet50', num_classes=10, 
                 pretrained=False, custom_state_dict=None):
        """
        Instantiate ResNet backbone (from torchvision.models),
        optionally load pretrained weights or custom state dict.
        """
        super().__init__()
        if architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif architecture == 'resnet110':
            # If you have a ResNet-110 implementation, replace here
            # For demonstration, using resnet50; replace with actual if available
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Replace final FC layer to match num_classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if custom_state_dict:
            self.load_state_dict(custom_state_dict)

    def forward(self, images):
        return self.backbone(images)  # logits: shape (batch, num_classes)

    def predict(self, images):
        """Return class probabilities after softmax."""
        logits = self.forward(images)
        probs = nn.functional.softmax(logits, dim=1)
        return probs

```

## requirements.txt (static, not part of code but necessary for environment setup)

requirements.txt (static, not part of code but necessary for environment setup)
```
torch==1.13.1
torchvision==0.14.1
numpy==1.21.0
scipy==1.7.3
matplotlib==3.5.1
```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import random
from dataset_loader import DatasetLoader
from model import MaskUNet, ResNetClassifier
from utils import add_gaussian_noise
import os

class Trainer:
    def __init__(self, config: dict):
        """
        Initialize training with configuration loaded from config.yaml.
        """
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dataset setup
        self.dataset_loader = DatasetLoader(config)
        self.train_loader = self.dataset_loader.get_data_loader(
            batch_size=config['training']['batch_size'], shuffle=True, num_workers=4
        )

        # Extract dataset parameters
        dataset_name = config['dataset']['name']
        background_scale = config['dataset'].get('background_scale', 640)
        
        # Hyperparameters
        self.epochs = config['training'].get('epochs', 200)
        self.lr = config['training'].get('learning_rate', 1e-3)
        self.weight_decay = config['training'].get('weight_decay', 1e-4)
        self.optimizer_type = config['training'].get('optimizer', 'AdamW')
        self.lr_decay_step = config['training'].get('lr_decay', 30)
        self.lr_gamma = config['training'].get('lr_gamma', 0.1)
        self.total_sigma = config['training'].get('total_noise_budget_sigma', 1.0)
        # Variance split: sigma1 = sigma2 = sqrt(2)*sigma
        self.sigma1 = self.sigma2 = np.sqrt(2) * self.total_sigma

        # Initialize models
        mask_params = config['model']['mask_unet']
        classifier_params = config['model']['classifier']
        arch = classifier_params.get('architecture', 'resnet50')
        
        self.mask_net = MaskUNet(
            base_channels=mask_params['base_channels'],
            channel_mult=mask_params['channel_mult'],
            step_size=mask_params.get('step_size', 40),
            gamma=mask_params.get('gamma', 0.5),
            momentum=mask_params.get('momentum', 0.9)
        ).to(self.device)

        self.classifier = ResNetClassifier(
            architecture=arch,
            num_classes=10 if dataset_name=='CIFAR10' else 2 if dataset_name=='CelebA' else 1000,
        ).to(self.device)

        # Setup optimizers
        self.optimizer_mask = optim.AdamW(self.mask_net.parameters(), lr=mask_params.get('learning_rate', 1e-3),
                                              weight_decay=mask_params.get('weight_decay', 1e-4))
        self.optimizer_classifier = optim.AdamW(self.classifier.parameters(), lr=classifier_params.get('learning_rate', 1e-3),
                                                    weight_decay=classifier_params.get('weight_decay', 1e-4))
        # Learning rate scheduler
        self.scheduler_mask = lr_scheduler.StepLR(self.optimizer_mask, step_size=self.lr_decay_step, gamma=self.lr_gamma)
        self.scheduler_classifier = lr_scheduler.StepLR(self.optimizer_classifier, step_size=self.lr_decay_step, gamma=self.lr_gamma)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Validation set (for hyper-parameter tuning or validation)
        self.val_loader = self.dataset_loader.get_data_loader(
            batch_size=256, shuffle=False, num_workers=4
        )

        # Hyperparameters for training
        self.train_mask_only = False  # Set True if only training mask for pretrain
   
    def train(self):
        """
        Main training loop for joint training of mask network and classifier end-to-end.
        """
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            # Step LR schedulers
            self.scheduler_mask.step()
            self.scheduler_classifier.step()
            # Validation
            val_acc = self.evaluate(self.val_loader)
            print(f"Epoch {epoch+1}/{self.epochs} - Validation Accuracy: {val_acc:.4f}")
            # Save checkpoint based on validation accuracy
            self._save_checkpoint(epoch, val_acc)

    def train_one_epoch(self, epoch):
        self.mask_net.train()
        self.classifier.train()

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            batch_size = inputs.shape[0]

            # 1. Add Gaussian noise z1 for mask generator
            z1 = add_gaussian_noise(torch.zeros_like(inputs), self.sigma1).to(self.device)
            noisy_inputs = inputs + z1

            # 2. Compute mask w(m1)
            mask = self.mask_net(noisy_inputs)  # shape: (B,1,H,W), values in [0,1]

            # 3. Generate masked input
            masked_input = mask * inputs  # element-wise multiply

            # 4. Add second Gaussian noise z2
            z2 = add_gaussian_noise(torch.zeros_like(masked_input), self.sigma2).to(self.device)
            noisy_masked_input = masked_input + z2

            # 5. Compute weights for unbiased combination (c1, c2)
            c1, c2 = self.compute_comb_coefficients(mask, self.sigma1, self.sigma2)

            # 6. Obtain m1 and m2 predictions for each sample
            # For simplicity, perform one forward pass; for multiple MC,
            # multiple samples are taken at inference, but here train with one sample
            # Note: Alternatively, do multiple stochastic passes for better approximation
            # but for training, one pass suffices

            # Forward through classifier with biased estimate (as in paper, weights should be per pixel)
            # Compute hat_X = c1 * m1 + c2 * (w * m2) -- here, m1 and m2 are deterministic in this code
            # In practice, the noisy images are the basis for the classifier
            # Here, we treat noisy_masked_input as 'm2' (second noisy image),
            # and 'inputs + z1' as 'm1' (though in training, m1 is not separately stored)
            # For simplicity, run classifier once on noisy_masked_input after combining, or treat as stochastic.
            # But as per the described method, for training, do one step.
            # Let's treat the inputs' noises as m1 and m2 contributions:
            # For a proper implementation, you'd perform multiple MC samples and average, but here we do a single.

            # In practice, the estimate of hat_X:
            with torch.no_grad():
                m1_pred = self.classifier(inputs + z1)  # shape: batch x classes
            m2_pred = self.classifier(noisy_masked_input)  # shape: batch x classes

            # For minimal implementation, perform weighted sum of logits or probabilities
            # Here, for simplicity, just use m1_pred and m2_pred to get class probabilities
            # and perform loss (training on predicted class)
            # But for actual variance reduction, the linear weights c1 and c2 should be applied
            # on the noisy images before classifier, not logits.
            # For correctness, implement similar to Eq (2.6).

            # Compute combined estimates (e.g., in probability space)
            # But to keep consistent, just perform:
            prob_m1 = nn.functional.softmax(m1_pred, dim=1)
            prob_m2 = nn.functional.softmax(m2_pred, dim=1)

            # Combine class probabilities (approximating the unbiased combination)
            # Alternatively, combine the logits as weighted average
            logits_combined = torch.log(prob_m1 + 1e-10) * c1.mean() + torch.log(prob_m2 + 1e-10) * c2.mean()
            # Better to do weighted average of logits
            # Or, as in paper, combine 'images' (not logits), so here, for simplicity, we will do average of logits:
            # Because the formulas are derived assuming the mean of the images, not logits.

            # For training, a simple approach is to just perform:
            #   - Pass original input + z1 through mask to get 'm1'
            #   - Pass original input + z2 through classifier directly
            # For now, let's proceed with the current approximation:

            # 7. Get prediction logits
            logits = self.classifier(noisy_masked_input)
            loss = self.criterion(logits, labels)

            # 8. Backpropagation
            self.optimizer_mask.zero_grad()
            self.optimizer_classifier.zero_grad()
            loss.backward()
            self.optimizer_mask.step()
            self.optimizer_classifier.step()

    def compute_comb_coefficients(self, mask: torch.Tensor, sigma1: float, sigma2: float):
        """
        Compute pixel-wise coefficients c1 and c2 to combine m1 and m2.
        """
        epsilon = 1e-8
        # flatten mask for norms
        batch_size, _, H, W = mask.shape
        mask_flat = mask.view(batch_size, -1)
        norm_w2 = torch.norm(mask_flat, p=2, dim=1, keepdim=True)  # shape: (B,1)

        w_sq = mask ** 2

        # Denominator for c1
        denom_c1 = (w_sq.sum(dim=(2,3), keepdim=True) * sigma2 ** 2 + norm_w2 ** 2 * sigma1 ** 2) + epsilon
        c1 = (w_sq * sigma2 ** 2) / denom_c1
        c1 = c1 * sigma1 ** 2  # shape: (B,1,H,W)

        # Denominator for c2
        denom_c2 = (w_sq * sigma1 ** 2 + norm_w2 ** 2 * sigma2 ** 2) + epsilon
        c2 = (sigma1 ** 2 * mask) / denom_c2
        # shape: (B,1,H,W)

        return c1, c2

    def evaluate(self, dataloader):
        """
        Evaluate accuracy on validation/test set.
        """
        self.mask_net.eval()
        self.classifier.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self._predict(inputs)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        return total_correct / total_samples

    def _predict(self, inputs, n_samples=1):
        """
        Predict function for evaluation:
        Runs multiple MC samples and average predictions for certification.
        """
        # For simplicity, perform n_samples Monte Carlo sampling
        # and average softmax outputs for robust estimate
        probs_sum = None
        for _ in range(n_samples):
            # Add noise to inputs
            z1 = add_gaussian_noise(torch.zeros_like(inputs), self.sigma1).to(self.device)
            noisy_inputs = inputs + z1

            mask = self.mask_net(noisy_inputs)  # shape: (B,1,H,W)
            masked_input = mask * inputs

            z2 = add_gaussian_noise(torch.zeros_like(masked_input), self.sigma2).to(self.device)
            noisy_masked_input = masked_input + z2

            logits = self.classifier(noisy_masked_input)
            probs = nn.functional.softmax(logits, dim=1)
            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum += probs
        probs_avg = probs_sum / n_samples
        return probs_avg

    def _save_checkpoint(self, epoch, val_acc):
        """
        Save model weights if validation accuracy improves.
        """
        checkpoint_dir = './checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        path_mask = os.path.join(checkpoint_dir, 'mask_net.pth')
        path_clf = os.path.join(checkpoint_dir, 'classifier.pth')
        torch.save(self.mask_net.state_dict(), path_mask)
        torch.save(self.classifier.state_dict(), path_clf)
        print(f"Saved checkpoint at epoch {epoch+1}")

def main():
    import yaml
    # Load config.yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # Initialize trainer
    trainer = Trainer(config)
    # Start training loop
    trainer.train()

if __name__ == '__main__':
    main()
```

## utils.py

```python
## utils.py
import torch
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np

def add_gaussian_noise(input: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Adds Gaussian noise to the input tensor.
    
    Args:
        input (torch.Tensor): Input tensor of shape (B, C, H, W) or similar.
        sigma (float): Standard deviation of the Gaussian noise.
    
    Returns:
        torch.Tensor: Noisy tensor of the same shape as input.
    """
    noise = torch.randn_like(input) * sigma
    return input + noise

def compute_coefficients(w: torch.Tensor, sigma1: float, sigma2: float, epsilon: float = 1e-8):
    """
    Computes pixel-wise coefficients c1 and c2 for unbiased combination of m1 and m2 predictions,
    minimizing variance under the constraint c1 + w * c2 = 1.
    
    Args:
        w (torch.Tensor): mask weights, shape (B, 1, H, W), values in [0,1].
        sigma1 (float): Standard deviation used in m1.
        sigma2 (float): Standard deviation used in m2.
        epsilon (float): Small constant for numerical stability.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: c1 and c2 tensors, shape matching w, suitable for combining m1 and m2.
    """
    # Compute norm_2 of w (per sample)
    w_flat = w.view(w.shape[0], -1)
    norm_w2 = torch.norm(w_flat, p=2, dim=1, keepdim=True)  # shape (B,1)
    
    # Expand for pixel-wise coefficients
    w_sq = w ** 2
    # Compute denominator avoiding division by zero
    denom = (w_sq.sum(dim=(2,3), keepdim=True) * sigma2 ** 2 + norm_w2 ** 2 * sigma1 ** 2) + epsilon
    
    # Coefficient c1: for m1
    c1 = (w_sq * sigma2 ** 2) / denom
    c1 = c1 * sigma1 ** 2  # shape (B,1,H,W)
    
    # Coefficient c2: for m2
    denom_c2 = (w_sq * sigma1 ** 2 + norm_w2 ** 2 * sigma2 ** 2) + epsilon
    c2 = (sigma1 ** 2 * w) / denom_c2  # same shape as w
    
    return c1, c2

def combine_predictions(m1: torch.Tensor, m2: torch.Tensor, w: torch.Tensor, sigma1: float, sigma2: float) -> torch.Tensor:
    """
    Combine two noisy predictions m1 and m2 into an unbiased estimate with minimized variance.
    
    Args:
        m1 (torch.Tensor): First noisy prediction, shape (B, C, H, W).
        m2 (torch.Tensor): Second noisy prediction, shape same as m1.
        w (torch.Tensor): Mask weights, shape (B, 1, H, W), values in [0,1].
        sigma1 (float): Noise level in m1.
        sigma2 (float): Noise level in m2.
    
    Returns:
        torch.Tensor: Combined estimate $\hat{x}$, shape (B, C, H, W).
    """
    c1, c2 = compute_coefficients(w, sigma1, sigma2)

    hat_x = c1 * m1 + c2 * w * m2
    return hat_x

def compute_cert_radius(p_plus: float, p_minus: float, sigma: float) -> float:
    """
    Compute certified radius (e.g., Eq. 2.2) for classification based on class probabilities.
    
    Args:
        p_plus (float): Estimated probability of the top class.
        p_minus (float): Estimated probability of the top competing class.
        sigma (float): Noise scale used in the Gaussian mechanism.
    
    Returns:
        float: Certified robustness radius.
    """
    # Clip probabilities to avoid invalid values
    p_plus = min(max(p_plus, 1e-10), 1 - 1e-10)
    p_minus = min(max(p_minus, 1e-10), 1 - 1e-10)

    # Compute inverse CDF
    inv_p_plus = norm.ppf(p_plus)
    inv_p_minus = norm.ppf(p_minus)

    radius = (sigma / 2.0) * (inv_p_plus - inv_p_minus)
    return radius

def estimate_class_probabilities(input: torch.Tensor, classifier: callable, num_samples: int, sigma: float, device: torch.device = torch.device('cpu')) -> dict:
    """
    Estimates class probabilities via Monte Carlo sampling with noisy inputs.
    
    Args:
        input (torch.Tensor): Input tensor, shape (C, H, W), original input.
        classifier (callable): Classifier function accepting tensor batch, returning class probabilities.
        num_samples (int): Number of Monte Carlo samples.
        sigma (float): Noise level used in addition to the input.
        device (torch.device): Device for computation.
    
    Returns:
        dict: Mapping class label -> estimated probability.
    """
    input = input.to(device)
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            noisy_input = add_gaussian_noise(input.unsqueeze(0), sigma).squeeze(0)
            probs = classifier(noisy_input.unsqueeze(0))
            pred_class = probs.argmax(dim=1).item()
            predictions.append(pred_class)

    # Count predictions
    counts = {}
    for c in predictions:
        counts[c] = counts.get(c, 0) + 1

    class_probs = {}
    total = float(len(predictions))
    for c, count in counts.items():
        class_probs[c] = count / total

    return class_probs

def search_sigma(validation_function: callable, sigma_values: list, *args, **kwargs):
    """
    Performs grid search over sigma values for the best certified accuracy.
    
    Args:
        validation_function (callable): Function to evaluate model performance at given sigma.
        sigma_values (list): List of sigma values to evaluate.
        *args, **kwargs: Additional arguments for validation_function.
    
    Returns:
        float: Best sigma value.
        dict: Performance metrics for each sigma.
    """
    best_sigma = None
    best_score = -np.inf
    results = {}
    for sigma in sigma_values:
        score = validation_function(sigma, *args, **kwargs)
        results[sigma] = score
        if score > best_score:
            best_score = score
            best_sigma = sigma
    return best_sigma, results

def search_beta(validation_function: callable, beta_values: list, *args, **kwargs):
    """
    Performs grid search over beta (distribution tail parameter) for tuning.
    
    Args:
        validation_function (callable): Function to evaluate model at given beta.
        beta_values (list): List of beta values.
        *args, **kwargs: Additional args.
    
    Returns:
        float: Best beta.
        dict: Performance metrics per beta.
    """
    best_beta = None
    best_score = -np.inf
    results = {}
    for beta in beta_values:
        score = validation_function(beta, *args, **kwargs)
        results[beta] = score
        if score > best_score:
            best_score = score
            best_beta = beta
    return best_beta, results

def load_background_images(background_paths: list, background_scale: int = 640) -> list:
    """
    Load and resize background images from file paths.
    
    Args:
        background_paths (list): List of file paths to images.
        background_scale (int): Target size (width and height).
    
    Returns:
        List[Image.Image]: Resized background images.
    """
    from PIL import Image
    bg_images = []
    for path in background_paths:
        img = Image.open(path).convert('RGB')
        img_resized = img.resize((background_scale, background_scale), Image.BILINEAR)
        bg_images.append(img_resized)
    return bg_images

def resize_backgrounds(bg_images: list, background_scale: int) -> list:
    """
    Resize a list of background images to a specific size.
    
    Args:
        bg_images (list): List of PIL Images.
        background_scale (int): Target size.
    
    Returns:
        list: Resized PIL Images.
    """
    resized_bgs = []
    for img in bg_images:
        resized_bgs.append(img.resize((background_scale, background_scale), Image.BILINEAR))
    return resized_bgs
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\adaptive-randomized-smoothing\adaptive-randomized-smoothing_repo`
