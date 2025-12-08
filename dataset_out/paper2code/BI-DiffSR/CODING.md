# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import random
import glob
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image

class SRDataset(Dataset):
    """
    Super-resolution dataset for paired HR and LR images.
    Loads images, applies synchronised augmentation, and outputs pairs.
    """
    def __init__(
        self,
        hr_image_paths: List[str],
        crop_size: int = 64,
        upscale_factor: int = 2,
        augment_flip: bool = True,
        augment_rotation_angles: Optional[List[int]] = None,
        is_train: bool = True
    ):
        """
        Args:
            hr_image_paths (List[str]): List of file paths to HR images.
            crop_size (int): Size of cropped patches for training.
            upscale_factor (int): Degrees of downsampling (e.g., 2 or 4).
            augment_flip (bool): Whether to apply horizontal flip augmentation.
            augment_rotation_angles (Optional[List[int]]): list of allowed rotation angles.
            is_train (bool): Flag indicating train (True) or test (False).
        """
        self.hr_image_paths = hr_image_paths
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.augment_flip = augment_flip
        self.augment_rotation_angles = augment_rotation_angles
        self.is_train = is_train

    def __len__(self):
        return len(self.hr_image_paths)

    def __getitem__(self, index):
        hr_path = self.hr_image_paths[index]
        hr_image = Image.open(hr_path).convert('RGB')  # Ensure RGB

        # Generate LR image via bicubic downsampling
        lr_image = self._get_lr_image(hr_image, self.upscale_factor)

        # Convert images to tensors
        hr_tensor = TF.to_tensor(hr_image)
        lr_tensor = TF.to_tensor(lr_image)

        # During training, apply augmentation
        if self.is_train:
            hr_tensor, lr_tensor = self._augment(hr_tensor, lr_tensor)
        else:
            # For evaluation, no augmentation, only center crop if needed
            pass

        return {
            'HR': hr_tensor,
            'LR': lr_tensor
        }

    def _get_lr_image(self, hr_img: Image.Image, scale: int) -> Image.Image:
        """
        Downsample the HR image to create LR image with bicubic interpolation.
        """
        W, H = hr_img.size
        new_W, new_H = W // scale, H // scale
        lr_img = hr_img.resize((new_W, new_H), Image.BICUBIC)
        return lr_img

    def _augment(self, hr_tensor: torch.Tensor, lr_tensor: torch.Tensor):
        """
        Apply the same random crop, rotation, flip to HR and LR tensors.
        """
        # Convert tensors back to PIL for augmentation
        hr_img = TF.to_pil_image(hr_tensor)
        lr_img = TF.to_pil_image(lr_tensor)

        # 1. Random crop
        hr_crop, lr_crop = self._random_crop_pair(hr_img, lr_img, self.crop_size, self.upscale_factor)

        # 2. Random rotation
        if self.augment_rotation_angles:
            angle = random.choice(self.augment_rotation_angles)
            hr_crop = hr_crop.rotate(angle)
            lr_crop = lr_crop.rotate(angle)

        # 3. Random flip
        if self.augment_flip:
            if random.random() < 0.5:
                hr_crop = hr_crop.transpose(Image.FLIP_LEFT_RIGHT)
                lr_crop = lr_crop.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert back to tensors
        hr_tensor_aug = TF.to_tensor(hr_crop)
        lr_tensor_aug = TF.to_tensor(lr_crop)

        return hr_tensor_aug, lr_tensor_aug

    def _random_crop_pair(self, hr_img: Image.Image, lr_img: Image.Image, crop_size: int, scale: int):
        """
        Randomly crop HR and LR image pair with size crop_size and crop_size/scale.
        Ensures the patches correspond spatially.
        """
        W_hr, H_hr = hr_img.size
        crop_size_lr = crop_size // scale

        if W_hr < crop_size or H_hr < crop_size:
            raise ValueError("HR image size is smaller than crop size.")
        if W_hr < crop_size or H_hr < crop_size:
            raise ValueError("HR image size is smaller than crop size.")

        # Choose random top-left coordinate for HR crop
        top = random.randint(0, H_hr - crop_size)
        left = random.randint(0, W_hr - crop_size)

        # Corresponding LR crop
        top_lr = top // scale
        left_lr = left // scale

        hr_crop = hr_img.crop((left, top, left + crop_size, top + crop_size))
        lr_crop = lr_img.crop((left_lr, top_lr, left_lr + crop_size_lr, top_lr + crop_size_lr))

        return hr_crop, lr_crop


class DatasetLoader:
    """
    Handles loading dataset splits and creating DataLoaders.
    """
    def __init__(self, dataset_paths: Dict[str, str], batch_size: int = 16):
        """
        Args:
            dataset_paths (Dict[str, str]): Mapping dataset names to root directories.
            batch_size (int): Batch size for DataLoader.
        """
        self.dataset_paths = dataset_paths
        self.batch_size = batch_size

        # For simplicity, hold datasets for train and test
        self.train_dataset = None
        self.test_datasets = {}

    def load_train_datasets(self):
        """
        Load training datasets (e.g., DIV2K, Flick2K) with augmentation enabled.
        """
        train_list = []
        for name, path in self.dataset_paths.items():
            if name.lower() in ['div2k', 'flickr2k', 'flickr2k-y', 'div2k-train']:
                image_paths = self._get_image_paths(path)
                train_list.extend(image_paths)
        self.train_dataset = SRDataset(
            hr_image_paths=train_list,
            crop_size=64,
            upscale_factor=2,
            augment_flip=True,
            augment_rotation_angles=[90, 180, 270],
            is_train=True
        )

    def load_test_datasets(self):
        """
        Load test datasets without augmentation.
        """
        for name, path in self.dataset_paths.items():
            if name.upper() in ['SET5', 'B100', 'URBAN100', 'MANGA109']:
                image_paths = self._get_image_paths(path)
                self.test_datasets[name] = image_paths

    def get_train_loader(self):
        """
        Return DataLoader for training.
        """
        if self.train_dataset is None:
            self.load_train_datasets()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def get_test_loader(self, dataset_name: str):
        """
        Return DataLoader for specified test dataset.
        """
        if dataset_name not in self.test_datasets:
            raise ValueError(f"Test dataset {dataset_name} not loaded.")
        dataset = SRTestDataset(self.test_datasets[dataset_name])
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def _get_image_paths(self, root_path: str) -> List[str]:
        """
        Retrieve list of image file paths from a directory.
        """
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
        image_files = []
        for ext in valid_extensions:
            image_files.extend(glob.glob(os.path.join(root_path, '**', '*' + ext), recursive=True))
        return image_files

class SRTestDataset(Dataset):
    """
    Dataset for test images (no augmentation).
    Loads full images, applies bicubic downsampling for LR.
    """
    def __init__(self, image_paths: List[str], upscale_factor: int=2):
        self.image_paths = image_paths
        self.upscale_factor = upscale_factor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        hr_path = self.image_paths[index]
        hr_image = Image.open(hr_path).convert('RGB')
        W, H = hr_image.size

        # To match training, take center crop of expected size or full image
        # Here, for evaluation, we will just resize/convene as whole image
        # Alternatively, you can crop or resize to fixed size as needed
        # For simplicity, use entire image, assuming the test images are suitable size
        # if desired, you can add center crop or resize here for consistent sizes

        lr_image = self._get_lr_image(hr_image, self.upscale_factor)

        hr_tensor = TF.to_tensor(hr_image)
        lr_tensor = TF.to_tensor(lr_image)

        return {
            'HR': hr_tensor,
            'LR': lr_tensor
        }

    def _get_lr_image(self, hr_img: Image.Image, scale: int) -> Image.Image:
        """
        Downsample HR image via bicubic interpolation.
        """
        W, H = hr_img.size
        new_W, new_H = W // scale, H // scale
        lr_img = hr_img.resize((new_W, new_H), Image.BICUBIC)
        return lr_img
```

## evaluation.py

```python
## evaluation.py
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import torchvision.transforms.functional as TF

try:
    import lpips
except ImportError:
    lpips = None

from dataset_loader import DatasetLoader
from model import UNet
from utils import (
    get_timestep_embedding,
    tensor_to_image,
)

# Load configuration
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Extract evaluation parameters
scale_factor = config['evaluation'].get('scale', 2)
datasets_names = config['evaluation']['datasets']
metrics_list = config['evaluation'].get('metrics', ['PSNR', 'SSIM', 'LPIPS'])
inference_T = config['diffusion'].get('inference_timesteps', 50)
T_total = config['diffusion'].get('total_timesteps', 2000)

# Initialize test data loaders
dataset_paths = {}
# Assuming dataset_paths mapping is known; modify as needed
# e.g., dataset_paths = {'Set5':'path_to_Set5/', ...}
# Here, we set placeholders; replace with actual paths
for name in datasets_names:
    dataset_paths[name] = f'path_to_{name}'

loader_obj = DatasetLoader(dataset_paths, batch_size=1)
test_loaders = {}
for name in datasets_names:
    test_loaders[name] = loader_obj.get_test_loader(name)

# Initialize LPIPS if needed
if 'LPIPS' in metrics_list and lpips is not None:
    lpips_model = lpips.LPIPS(net='alex').to(device)
else:
    lpips_model = None

# Load model
model_config = {
    'channels': config['model'].get('channels', 64),
    'encoder_levels': config['model'].get('encoder_levels', 4),
    'res_blocks_per_level': config['model'].get('res_blocks_per_level', 2),
    'decoder_res_blocks': config['model'].get('decoder_res_blocks', 3),
    'total_timesteps': config['diffusion'].get('total_timesteps', 2000),
    'timestep_encoding_K': config['model'].get('timestep_encoding_K', 5),
}
model = UNet(model_config).to(device)
model_ckpt = 'checkpoints/model_final.pth'  # modify as needed
model.load_state_dict(torch.load(model_ckpt, map_location='cpu'))
model.eval()

# Diffusion schedule function (cosine schedule or linear as in training)
def get_diffusion_schedule(T):
    beta_start, beta_end = 0.0001, 0.02
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return torch.tensor(betas, dtype=torch.float32).to(device), torch.tensor(alpha_bars, dtype=torch.float32).to(device)

betas, alpha_bars = get_diffusion_schedule(T_total)

# Sampling function using DDIM
def ddim_sample(condition_lr, condition_hr_size, batch_size=1, seed=None):
    """
    condition_lr: tensor [B, 3, H, W], in [0,1]
    condition_hr_size: tuple (H, W)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    H, W = condition_hr_size
    # Initialize noisy x_T
    x_t = torch.randn(batch_size, 3, H, W, device=device)

    # Prepare scheduling
    total_T = T_total
    schedule = get_diffusion_schedule(total_T)

    # Create array of t_indices for inference steps
    t_steps = np.linspace(T_total-1, 0, inference_T, dtype=int)

    for t_idx in tqdm(t_steps, desc='Sampling'):
        t_b = torch.tensor([t_idx]*batch_size, device=device)
        t_emb = get_timestep_embedding(t_b, model_config['channels']).to(device)

        # Prepare model input: concatenate LR with current noisy HR
        model_input = torch.cat([condition_lr, x_t], dim=1)

        with torch.no_grad():
            epsilon_pred = model(model_input, t_b.float())

        alpha_t = schedule['alphas_cumprod'][t_idx]
        alpha_prev = schedule['alphas_cumprod'][t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=device)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        sqrt_alpha_prev = torch.sqrt(alpha_prev)

        # Compute predicted x_0
        x0_pred = (x_t - sqrt_one_minus_alpha_t * epsilon_pred) / sqrt_alpha_t

        # Compute mean of x_{t-1}
        mean_x_prev = x0_pred * torch.sqrt(alpha_prev) + torch.sqrt(1 - alpha_prev) * epsilon_pred

        # For stochastic sampling with eta>0 (not used here, for deterministic set eta=0)
        eta = 0.0
        if eta > 0:
            sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            noise = torch.randn_like(x_t)
            x_prev = mean_x_prev + sigma_t * noise
        else:
            x_prev = mean_x_prev

        x_t = x_prev

    return x_t

# Standardize data range for evaluation: images in [0,1]
def to_01(image_tensor):
    return torch.clamp((image_tensor + 1) / 2, 0, 1)

# Metrics functions
def compute_psnr(im_pred, im_true):
    im_pred_np = im_pred.astype(np.float32)
    im_true_np = im_true.astype(np.float32)
    return psnr_fn(im_true_np, im_pred_np, data_range=1.0)

def compute_ssim(im_pred, im_true):
    im_pred_np = im_pred.astype(np.float32)
    im_true_np = im_true.astype(np.float32)
    return ssim_fn(im_true_np, im_pred_np, data_range=1.0)

def compute_lpips(im_pred, im_true):
    if lpips_model is None:
        return None
    # expec images in [0,1], shape (H,W,C)
    im_pred_pt = torch.from_numpy(im_pred).permute(2,0,1).unsqueeze(0).to(device)*2 - 1
    im_true_pt = torch.from_numpy(im_true).permute(2,0,1).unsqueeze(0).to(device)*2 - 1
    with torch.no_grad():
        dist = lpips_model(im_pred_pt, im_true_pt)
    return dist.item()

# Main evaluation function
def evaluate_dataset(dataset_loader, dataset_name, save_dir='evaluation_results'):
    os.makedirs(save_dir, exist_ok=True)
    psnr_list = []
    ssim_list = []
    lpips_list = []

    for batch_idx, batch in enumerate(tqdm(dataset_loader, desc=f'Evaluating {dataset_name}')):
        with torch.no_grad():
            # batch['LR']: [1, 3, H, W], [0,1]
            # batch['HR']: [1, 3, H, W], [0,1]
            lr_img = batch['LR'].to(device)
            hr_img = batch['HR'].to(device)

            _, _, H, W = hr_img.shape

            # Sample SR image
            sr_img = ddim_sample(lr_img, (H, W), batch_size=1, seed=42+batch_idx)

            # Convert tensors to numpy images
            hr_np = tensor_to_image(hr_img.squeeze(0).cpu())
            sr_np = tensor_to_image(sr_img.squeeze(0).cpu())

            # Also get LR image for visualization
            lr_np = tensor_to_image(lr_img.squeeze(0).cpu())

            # Save images
            base_name = f"{dataset_name}_img{batch_idx}"
            TF.to_pil_image(hr_np).save(os.path.join(save_dir, f"{base_name}_HR.png"))
            TF.to_pil_image(sr_np).save(os.path.join(save_dir, f"{base_name}_SR.png"))
            TF.to_pil_image(lr_np).save(os.path.join(save_dir, f"{base_name}_LR.png"))

            # Prepare images in [0,1] for metric calculation
            hr_for_metrics = hr_np
            sr_for_metrics = sr_np

            # Extract luminance channel
            hr_y = rgb2y(hr_for_metrics)
            sr_y = rgb2y(sr_for_metrics)

            # Compute metrics
            p_val = compute_psnr(sr_y, hr_y)
            s_val = compute_ssim(sr_y, hr_y)
            if 'LPIPS' in metrics_list and lpips_model is not None:
                lp_val = compute_lpips(sr_for_metrics, hr_for_metrics)
            else:
                lp_val = None

            psnr_list.append(p_val)
            ssim_list.append(s_val)
            if lp_val is not None:
                lpips_list.append(lp_val)

    # Report averages
    print(f"Results for {dataset_name}:")
    print(f" PSNR: {np.mean(psnr_list):.4f} dB (+/- {np.std(psnr_list):.4f})")
    print(f" SSIM: {np.mean(ssim_list):.4f} (+/- {np.std(ssim_list):.4f})")
    if 'LPIPS' in metrics_list and lpips_model is not None:
        print(f" LPIPS: {np.mean(lpips_list):.4f} (+/- {np.std(lpips_list):.4f})")
    print("-"*50)

# Helper: RGB to Y (luminance) in [0,1]
def rgb2y(rgb_img):
    # rgb_img shape (H, W, 3), dtype float in [0,1]
    R = rgb_img[:,:,0]
    G = rgb_img[:,:,1]
    B = rgb_img[:,:,2]
    Y = 0.299*R + 0.587*G + 0.114*B
    return Y

# Run evaluation for all datasets
for dataset_name, loader in test_loaders.items():
    evaluate_dataset(loader, dataset_name)
```

## main.py

```python
## main.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import (
    get_timestep_embedding,
    sign_bin,
)
from dataset_loader import DatasetLoader
from model import UNet
from evaluation import evaluate_dataset
from sampling import ddim_sample

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract training parameters with defaults
    learning_rate = float(config['training'].get('learning_rate', 1e-4))
    batch_size = int(config['training'].get('batch_size', 16))
    total_iterations = int(config['training'].get('total_iterations', 10**6))
    iter_per_epoch = int(config['training'].get('iterations_per_epoch', 5000))
    crop_size = int(config['training'].get('image_crop_size', 64))
    augment_flip = bool(config['training'].get('augmentation', {}).get('flip', True))
    rotations = list(config['training'].get('augmentation', {}).get('rotations', [90,180,270]))

    # Diffusion parameters
    T = int(config['diffusion'].get('total_timesteps', 2000))
    inference_T = int(config['diffusion'].get('inference_timesteps', 50))
    eta = 0.0  # deterministic DDIM

    # Model hyperparameters
    channels = int(config['model'].get('channels', 64))
    encoder_levels = int(config['model'].get('encoder_levels', 4))
    res_blocks_per_level = int(config['model'].get('res_blocks_per_level', 2))
    decoder_res_blocks = int(config['model'].get('decoder_res_blocks',3))
    timestep_K = int(config['model'].get('timestep_encoding_K', 5))
    bias_pairs_num = int(config['model'].get('binarization', {}).get('bias_pairs', 5))
    scale_weights = bool(config['model'].get('binarization', {}).get('scale_weights', True))

    # Dataset paths: update these paths to your dataset locations
    dataset_paths = {
        'DIV2K': 'path_to_DIV2K/',     # <-- replace with actual paths
        'Flickr2K': 'path_to_Flickr2K/'
    }
    test_dataset_names = ['Set5', 'B100', 'Urban100', 'Manga109']

    # Instantiate DataLoader for training
    loader_obj = DatasetLoader(dataset_paths, batch_size)
    train_loader = loader_obj.get_train_loader()

    # Instantiate DataLoaders for test datasets
    test_loaders = {}
    for name in test_dataset_names:
        test_loaders[name] = loader_obj.get_test_loader(name)

    # Initialize the UNet model
    model_kwargs = {
        'channels': channels,
        'encoder_levels': encoder_levels,
        'res_blocks_per_level': res_blocks_per_level,
        'decoder_res_blocks': decoder_res_blocks,
        'total_timesteps': T,
        'timestep_encoding_K': timestep_K,
    }
    model = UNet(model_kwargs).to(device)
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    criterion = nn.L1Loss()

    # Diffusion schedule: betas, alphas, alpha_bars
    def get_diffusion_schedule(T):
        betas = np.linspace(0.0001, 0.02, T)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        return torch.tensor(betas, dtype=torch.float32).to(device), torch.tensor(alpha_bars, dtype=torch.float32).to(device)

    betas, alpha_bars = get_diffusion_schedule(T)

    # Optional: load checkpoint if resuming
    # ckpt_path = 'checkpoints/model_final.pth'
    # if os.path.exists(ckpt_path):
    #     model.load_state_dict(torch.load(ckpt_path))
    #     print(f"Loaded checkpoint {ckpt_path}")

    total_iters = total_iterations
    pbar = tqdm(total=total_iters)
    global_iter = 0

    # Training loop
    while global_iter < total_iters:
        for batch in train_loader:
            # Get data: HR and LR tensors
            hr_images = batch['HR'].to(device)    # shape: (B,C,H,W), [0,1]
            lr_images = batch['LR'].to(device)    # shape: (B,C,H,W), [0,1]

            B = hr_images.shape[0]
            # Sample random integer timestep t for each sample
            t = torch.randint(1, T+1, (B,), device=device).long()
            t_emb = get_timestep_embedding(t, channels).to(device)  # shape: (B, channels)

            # Add noise according to diffusion schedule
            epsilon = torch.randn_like(hr_images)
            alpha_t = alpha_bars[t-1].view(-1,1,1,1)  # shape: (B,1,1,1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            x_t = sqrt_alpha_t * hr_images + sqrt_one_minus_alpha_t * epsilon

            # Prepare model input: concatenate LR + noisy HR
            model_input = torch.cat([lr_images, x_t], dim=1)  # 6 channels

            # Forward pass
            pred_epsilon = model(model_input, t.float())  # model expects float t (same shape as t_emb)
            loss = criterion(pred_epsilon, epsilon)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)
            if global_iter % 1000 == 0:
                # Optional validation or logging
                pass
            # Save checkpoint periodically
            if global_iter > 0 and global_iter % 10000 == 0:
                save_dir = 'checkpoints'
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'model_{global_iter}.pth')
                torch.save(model.state_dict(), save_path)
            global_iter += 1
            if global_iter >= total_iters:
                break
    pbar.close()

    # Save final model (if not already saved)
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/model_final.pth')
    print("Training complete. Model saved.")

    # After training, optionally run inference/evaluation:
    # for name, loader in test_loaders.items():
    #     evaluate_dataset(loader, name, save_dir='results')  # or implement accordingly

if __name__ == "__main__":
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import get_timestep_embedding, sign_bin, STESign

# =========================
# Binarized Convolution Layer
# =========================
class BinarizedConv(nn.Module):
    """
    Binarized convolution layer with weight scaling and STE.
    Uses binarized weights (sign), scaled by mean absolute value.
    Supports 3x3 and 1x1 convolutions as needed.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1,
                 padding: int=1, bias: bool=False, scale_weights: bool=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale_weights = scale_weights
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Full-precision weight
        self.weight_fp = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # Binarized weight placeholder
        self.register_buffer('weight_b', torch.zeros_like(self.weight_fp))

    def binarize_weight(self):
        """
        Binarize weights with scaling and STE.
        """
        weight_abs_mean = self.weight_fp.abs().mean()
        weight_sign = sign_bin(self.weight_fp)
        if self.scale_weights:
            self.weight_b = weight_sign * weight_abs_mean
        else:
            self.weight_b = weight_sign

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Binarize weights for this forward pass
        self.binarize_weight()
        # Use binarized weights for convolution
        out = F.conv2d(x, self.weight_b, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

# =========================
# Residual Block
# =========================
class ResBlock(nn.Module):
    """
    Residual Block with binarized convolutions, optional Timestep conditioning.
    """
    def __init__(self, channels: int, K: int=5, use_taR: bool=True, use_taA: bool=True):
        super().__init__()
        self.channels = channels
        self.use_taR = use_taR
        self.use_taA = use_taA
        # TaR/TaA modules will be subclasses
        # For simplicity of code, instantiate placeholders; actual will be created in main
        # Internally, will be set via property or method later
        self.b1 = BinarizedConv(channels, channels)
        self.b2 = BinarizedConv(channels, channels)
        self.act1 = nn.Identity()  # placeholder; replaced with TaR in forward
        self.act2 = nn.Identity()  # placeholder; replaced with TaA in forward

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, taR_module=None, taA_module=None):
        """
        x: input feature
        t_emb: timestep embedding for modulation
        taR_module, taA_module: optional modules for timestep conditioning
        """
        # First conv + activation
        out = self.b1(x)
        if self.use_taR and taR_module:
            out = taR_module(out, t_emb)
        out = F.relu(out)  # can replace with TaA if needed
        # Second conv + activation
        out = self.b2(out)
        if self.use_taA and taA_module:
            out = taA_module(out, t_emb)
        # Add skip connection
        out = out + x
        return out

# =========================
# Cycle Pixel Shuffle (Downsampling)
# =========================
class CPDownModule(nn.Module):
    """
    Consistent Pixel-Downsample module:
    - Splits input channels into two halves
    - Processes each half with binarized conv
    - Combines and applies PixelUnShuffle (scale=2)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv1 = BinarizedConv(channels//2, channels//2)
        self.conv2 = BinarizedConv(channels//2, channels//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        c_half = self.channels // 2
        x1, x2 = torch.split(x, c_half, dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        combined = x1 + x2  # simple fusion
        # PixelUnshuffle with scale=2 -> output height/width doubled, channel doubled
        out = F.pixel_unshuffle(combined, downscale_factor=2)
        return out

# =========================
# Cycle Pixel Shuffle (Upsampling)
# =========================
class CPUpModule(nn.Module):
    """
    Consistent Pixel-Upsample module:
    - Process with two binarized convs
    - Concatenate and apply PixelShuffle (scale=2)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv1 = BinarizedConv(channels, channels)
        self.conv2 = BinarizedConv(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        concat = torch.cat([out1, out2], dim=1)
        # PixelShuffle with upscale factor 2
        out = F.pixel_shuffle(concat, upscale_factor=2)
        return out

# =========================
# Channel-Shuffle Fusion (Skip Connection)
# =========================
class CSSFusion(nn.Module):
    """
    Channel-shuffle fusion:
    - Split each feature into odd/even channels
    - Pair odd/even and interleave
    - Concatenate and process with 2 binarized convs
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        c_half = channels // 2
        self.conv_sh1 = BinarizedConv(channels, channels)
        self.conv_sh2 = BinarizedConv(channels, channels)

    def channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shuffle the channels of x.
        Here, we do simple re-interleaving: split into odd/even indices and concatenate.
        """
        # x shape: (B, C, H, W)
        C = x.shape[1]
        odd_idx = torch.tensor([i for i in range(C) if i % 2 == 1], device=x.device)
        even_idx = torch.tensor([i for i in range(C) if i % 2 == 0], device=x.device)
        odd_channels = torch.index_select(x, 1, odd_idx)
        even_channels = torch.index_select(x, 1, even_idx)
        shuf = torch.cat([even_channels, odd_channels], dim=1)  # interleave
        return shuf

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Fuse features x1 and x2.
        """
        # Channel shuffle
        x1_sh = self.channel_shuffle(x1)
        x2_sh = self.channel_shuffle(x2)
        # Concatenate (they are already shuffled for balanced range)
        # Apply binarized convolutions
        out1 = self.conv_sh1(x1_sh)
        out2 = self.conv_sh2(x2_sh)
        out = out1 + out2
        return out

# =========================
# Timestep Encoding (Sinusoidal)
# =========================
class TimestepEncoding(nn.Module):
    """
    Embed timestep scalar into sinusoidal embedding.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: tensor of shape (B,)
        returns: embedding tensor shape (B, channels)
        """
        return get_timestep_embedding(t, self.channels)

# =========================
# Timestep-aware Redistribution (TaR)
# =========================
class TaR(nn.Module):
    """
    Timestep-aware redistribution:
    - K learnable biases (b_i)
    - Select bias based on timestep group
    """
    def __init__(self, channels: int, total_timesteps: int, K: int=5):
        super().__init__()
        self.channels = channels
        self.K = K
        self.total_timesteps = total_timesteps
        # Bias parameters
        self.b_list = nn.ParameterList([
            nn.Parameter(torch.zeros(channels)) for _ in range(K)
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        t: (B,)
        """
        # Determine group index based on t
        t_idx = torch.clamp((t.float() / self.total_timesteps * self.K).long(), 0, self.K -1)
        bias = torch.stack([b for b in self.b_list], dim=0)[t_idx]  # shape (B, C)
        bias = bias.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x + bias

# =========================
# Timestep-aware Activation (TaA)
# =========================
class TaA(nn.Module):
    """
    Timestep-aware activation:
    - K RPReLU modules.
    - Select activation based on timestep group.
    """
    def __init__(self, channels: int, total_timesteps: int, K: int=5):
        super().__init__()
        self.channels = channels
        self.K = K
        self.total_timesteps = total_timesteps
        # Create K RPReLU instances with learnable biases (if needed)
        self.rprelu_list = nn.ModuleList([
            nn.ReLU() for _ in range(K)
        ])  # For simplicity, use ReLU; can replace with custom RPReLU if desired.

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_idx = torch.clamp((t.float() / self.total_timesteps * self.K).long(), 0, self.K -1)
        activation_fn = self.rprelu_list[t_idx]
        # Apply selected activation
        return activation_fn(x)

# =========================
# Main UNet Model
# =========================
class UNet(nn.Module):
    """
    UNet architecture optimized for binarization, with CP modules, CS-Fusion, TaR, and TaA.
    """
    def __init__(self, config: dict):
        super().__init__()
        # Read config parameters
        self.ch = config.get("channels", 64)
        self.num_levels = config.get("encoder_levels",4)
        self.res_blocks_per_level = config.get("res_blocks_per_level",2)
        self.decoder_res_blocks = config.get("decoder_res_blocks",3)
        self.total_timesteps = config.get("total_timesteps",2000)
        self.K = config.get("timestep_encoding_K",5)

        # Timestep embedding
        self.timestep_enc = TimestepEncoding(self.ch)

        # Input conv
        self.input_conv = BinarizedConv(6, self.ch)  # 2 images concatenated: y (LR) + noise image or condition, assumed input shape: 6 channels

        # Encoder levels
        self.encoder_levels = nn.ModuleList()
        for lvl in range(self.num_levels):
            blocks = nn.ModuleList()
            for _ in range(self.res_blocks_per_level):
                blocks.append(ResBlock(self.ch))
            down = CPDownModule(self.ch)
            self.encoder_levels.append(nn.ModuleDict({
                "res_blocks": blocks,
                "down": down
            }))

        # Bottleneck residual blocks
        self.bottleneck = nn.ModuleList()
        for _ in range(self.res_blocks_per_level):
            self.bottleneck.append(ResBlock(self.ch))

        # Decoder levels
        self.decoder_levels = nn.ModuleList()
        for lvl in range(self.num_levels):
            blocks = nn.ModuleList()
            for _ in range(self.decoder_res_blocks):
                blocks.append(ResBlock(self.ch))
            up = CPUpModule(self.ch)
            self.decoder_levels.append(nn.ModuleDict({
                "res_blocks": blocks,
                "up": up
            }))

        # Skip connection fusions (CS-Fusion)
        self.cs_fusions = nn.ModuleList()
        for _ in range(self.num_levels):
            self.cs_fusions.append(CSSFusion(self.ch))

        # Final convolution
        self.output_conv = BinarizedConv(self.ch, 3, kernel_size=3, padding=1)

        # Timestep modules
        self.taR_modules = nn.ModuleList()
        self.taA_modules = nn.ModuleList()
        for _ in range(self.num_levels * 2 + len(self.bottleneck)):
            self.taR_modules.append(TaR(self.ch, self.total_timesteps, self.K))
            self.taA_modules.append(TaA(self.ch, self.total_timesteps, self.K))
        
        # Keep track of total layers
        # Layers in encoder+decoder+latent: for indexing TaR/TaA

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: (B, 6, H, W) (condition + noise image concatenated)
        t: (B,) scalar timestep tensor
        """
        t_emb = self.timestep_enc(t)

        # Initial feature
        feats = []
        x = self.input_conv(x)  # shape: (B, C, H, W)

        # Encoder
        for lvl_idx, lvl in enumerate(self.encoder_levels):
            for res_idx, res_block in enumerate(lvl["res_blocks"]):
                # Apply TaR and TaA
                taR_idx = lvl_idx * self.res_blocks_per_level + res_idx
                taA_idx = taR_idx
                x = res_block(x, t_emb, self.taR_modules[taR_idx], self.taA_modules[taA_idx])
            feats.append(x)
            x = lvl["down"](x)  # downsample

        # Bottleneck
        for idx, res_block in enumerate(self.bottleneck):
            taR_idx = self.num_levels * self.res_blocks_per_level + idx
            taA_idx = taR_idx
            x = res_block(x, t_emb, self.taR_modules[taR_idx], self.taA_modules[taA_idx])

        # Decoder
        for lvl_idx in reversed(range(self.num_levels)):
            up = self.decoder_levels[lvl_idx]["up"]
            x = up(x)
            # Fuse with skip connection
            feat_enc = feats[lvl_idx]
            fused = self.cs_fusions[lvl_idx](feat_enc, x)
            # Residual blocks in decoder
            for res_idx, res_block in enumerate(self.decoder_levels[lvl_idx]["res_blocks"]):
                taR_idx = (self.num_levels + lvl_idx) * self.res_blocks_per_level + res_idx
                taA_idx = taR_idx
                x = res_block(x, t_emb, self.taR_modules[taR_idx], self.taA_modules[taA_idx])
            # Add fused features
            x = x + fused

        # Final output convolution
        out = self.output_conv(x)
        return out
```

## requirements.txt

## requirements.txt
```python
# Core deep learning framework
torch>=1.9.0
torchvision>=0.10.0

# Data handling and image processing
numpy>=1.19.0
scipy>=1.5.0
Pillow>=8.0.0

# Perceptual similarity metric
lpips

# Utility for progress tracking
tqdm
```

## sampling.py

```python
## sampling.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import argparse
from tqdm import tqdm
from model import UNet
from utils import (
    get_timestep_embedding,
    sign_bin,
    STESign,
)
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Diffusion parameters
T = config['diffusion'].get('total_timesteps', 2000)
inference_T = config['diffusion'].get('inference_timesteps', 50)  # number of sampling steps
eta = 0.0  # deterministic DDIM; set in code if needed

# Model path (should be provided or configured)
CHECKPOINT_PATH = 'checkpoints/model_final.pth'  # or replace with desired checkpoint

# Load trained model
model = UNet(config['model'])
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
model.to(device)
model.eval()

# Scheduler: betas, alphas, alpha_bars
def get_diffusion_schedule(T):
    """
    Generates betas, alphas, and cumulative alpha products with scheduled or cosine
    schedule (for simplicity, linear schedule here).
    """
    betas = np.linspace(0.0001, 0.02, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return torch.tensor(betas, dtype=torch.float32).to(device), torch.tensor(alpha_bars, dtype=torch.float32).to(device)

betas, alpha_bars = get_diffusion_schedule(T)

# Compute terms for DDIM
def get_ddim_parameters(alpha_bars, timesteps):
    """
    Compute DDIM parameters for a sequence of timesteps.
    """
    alphas_cumprod = alpha_bars
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    return {
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
    }

# Generate the sequence of timesteps for inference
timestep_seq = np.linspace(T - 1, 0, inference_T, dtype=int)

# Prepare schedule tensors
schedule = get_ddim_parameters(alpha_bars, torch.tensor(timestep_seq, dtype=torch.long).to(device))

# Function to extract schedule values at specific timesteps
def extract(schedule_tensor, t_indices):
    return schedule_tensor[t_indices]

# Main sampling function
def ddim_sample(condition_lr: torch.Tensor, condition_hr_size: Tuple[int, int],
                batch_size: int = 1, seed: int = None):
    """
    Perform DDIM sampling for image super-resolution conditioned on low-res image.
    
    Args:
        condition_lr: condition input images (LR images), shape: [B, 3, H, W], scaled [0,1]
        condition_hr_size: tuple (H, W), size of the final HR image 
        batch_size: number of samples to generate
        seed: random seed for reproducibility
    Returns:
        high_res: tensor [B, 3, H, W], in [-1,1]
    """

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    H, W = condition_hr_size
    # Initialize x_T with standard Gaussian noise
    x_t = torch.randn(batch_size, 3, H, W, device=device)

    # Expand condition LR to batch if needed
    condition_lr = condition_lr.to(device)

    # Loop over timesteps from high T to 0
    for idx in tqdm(range(inference_T):
        , desc='Sampling'):
        t_idx = timestep_seq[idx]
        t_b = torch.tensor([t_idx]*batch_size, device=device)
        
        # Timestep embedding
        t_emb = get_timestep_embedding(t_b, model.ch).to(device)  # shape: (B, ch)
        
        # Determine group index for biases/activation (if using TaR/TaA modules)
        # Here, as per the paper, group t into K groups
        K = config['model'].get('timestep_encoding_K',5)
        t_group_idx = min(int(K * t_idx / T), K -1)  # index in [0, K-1]
        # The bias and activation modules are inside model; here, for inference, we directly pass t and select bias/act

        # Prepare input: concatenate conditioned LR + current noisy HR
        # Input shape: [B, 6, H, W]
        model_input = torch.cat([condition_lr, x_t], dim=1)  # condition LR + noise image

        # Forward pass to predict noise epsilon
        with torch.no_grad():
            epsilon_pred = model(model_input, t_b.float())

        # Extract schedule values for current timestep
        alpha_cumprod_t = schedule['alphas_cumprod'][t_idx]
        alpha_cumprod_prev = schedule['alphas_cumprod'][t_idx -1] if t_idx >0 else torch.tensor(1.0, device=device)
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        sqrt_alpha_cumprod_prev = torch.sqrt(alpha_cumprod_prev)

        # Compute variance for stochastic or deterministic
        sigma_t = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev))
        # For deterministic DDIM, eta=0, so sigma_t=0

        # Compute the mean for x_{t-1}
        # According to DDIM equation (see e.g., https://arxiv.org/abs/2206.00364)
        pred_x0_coef = sqrt_alpha_cumprod_prev
        pred_eps_coef = torch.sqrt(1 - alpha_cumprod_prev)
        #
        # Predicted x0
        x0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * epsilon_pred) / sqrt_alpha_cumprod_t
        # Reconstructed mean
        mean_x_prev = pred_x0 * pred_x0_coef + torch.sqrt(1 - alpha_cumprod_prev) * epsilon_pred

        if eta > 0:
            # Add noise scaled by sigma_t for stochasticity
            noise = torch.randn_like(x_t)
            x_prev = mean_x_prev + sigma_t * noise
        else:
            x_prev = mean_x_prev  # deterministic

        # Clamp or normalize if needed
        x_t = x_prev

    # Final output
    return x_t

# Usage example
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import save_image

    # Load a sample LR condition image
    # For testing, replace with actual data loader or image
    from utils import image_to_tensor
    import cv2

    # Sample: prepare a LR image tensor [1,3,H,W]
    hr_size = (256, 256)  # sample size
    # Load low-res image
    lr_img_path = 'path_to_sample_lr_image.jpg'  # specify path here
    lr_img = cv2.imread(lr_img_path)
    lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
    lr_img_resized = cv2.resize(lr_img, (hr_size[1]//2, hr_size[0]//2), interpolation=cv2.INTER_CUBIC)  # scale=2

    lr_tensor = image_to_tensor(lr_img_resized).unsqueeze(0)  # shape [1,3,H,W]

    # Run sampling
    fake_hr = ddim_sample(lr_tensor, hr_size, batch_size=1, seed=42)

    # Convert to image
    gen_img = utils.tensor_to_image(fake_hr.squeeze(0))
    plt.imshow(gen_img)
    plt.axis('off')
    plt.show()

    # Save generated image
    save_image(torch.clamp(torch.tensor(gen_img).permute(2,0,1),0,1), 'generated_hr.png')
```

**Notes:**
- This script loads a trained UNet model, constructs the DDIM schedule, and performs inference by iteratively updating `x_t` from `t=T` down to `t=0`.
- It uses the `get_timestep_embedding` utility to provide timestep info, along with the `eta` parameter controlling stochasticity.
- The conditioning on the LR image is through concatenation; adapt as needed for your condition format.
- The bias/activation selection for TaR and TaA modules is simplified here by assuming the modules are integrated into the `model` and set to select the correct bias/activation based on `t_group_idx`. The actual selection occurs within model's forward pass based on `t` or preloaded biases.
- For actual training/inference, replace the dummy image loading with your dataset loader, and ensure the model checkpoint path is correct.

## trainer.py

```python
## trainer.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset_loader import DatasetLoader
from model import UNet
from utils import (
    get_timestep_embedding,
    binarize_weight,
    STESign,
    calculate_psnr,
    calculate_ssim,
    calculate_lpips
)

# Load configuration from yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Extract training params
lr = config['training'].get('learning_rate', 1e-4)
batch_size = config['training'].get('batch_size', 16)
total_iters = config['training'].get('total_iterations', 1_000_000)
iter_per_epoch = config['training'].get('iterations_per_epoch', 5000)
crop_size = config['training'].get('image_crop_size', 64)
aug_flip = config['training'].get('augmentation', {}).get('flip', True)
rotations = config['training'].get('augmentation', {}).get('rotations', [90,180,270])

# Diffusion parameters
T = config['diffusion'].get('total_timesteps', 2000)
inference_T = config['diffusion'].get('inference_timesteps', 50)

# Model params
channels = config['model'].get('channels', 64)
encoder_levels = config['model'].get('encoder_levels',4)
res_blocks_per_level = config['model'].get('res_blocks_per_level',2)
decoder_res_blocks = config['model'].get('decoder_res_blocks',3)
timesteps_K = config['model'].get('timestep_encoding_K',5)

# Binarization configs
bias_pairs_num = config['model'].get('binarization', {}).get('bias_pairs',5)
scale_weights = config['model'].get('binarization', {}).get('scale_weights', True)

# Initialize datasets and dataloaders
dataset_paths = {
    'DIV2K': 'path_to_DIV2K/',  # update path as needed
    'Flickr2K': 'path_to_Flickr2K/'
}
# Load data
data_loader_obj = DatasetLoader(dataset_paths, batch_size)
train_loader = data_loader_obj.get_train_loader()

# Placeholder for test loaders: load on-demand
test_loaders = {}
for test_name in ['Set5', 'B100', 'Urban100', 'Manga109']:
    test_loaders[test_name] = data_loader_obj.get_test_loader(test_name)

# Initialize model
model = UNet({
    'channels': channels,
    'encoder_levels': encoder_levels,
    'res_blocks_per_level': res_blocks_per_level,
    'decoder_res_blocks': decoder_res_blocks,
    'total_timesteps': T,
    'timestep_encoding_K': timesteps_K
}).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
# Use L1 loss for diffusion noise prediction
criterion = nn.L1Loss()

# Function for cosine schedule for beta (diffusion)
def get_diffusion_coeffs(T):
    # Use cosine schedule for alpha_bar for simplicity
    betas = np.linspace(0.0001, 0.02, T)  # or implement cosine schedule if desired
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    return torch.tensor(betas, dtype=torch.float32), torch.tensor(alpha_bars, dtype=torch.float32)

betas, alpha_bars = get_diffusion_coeffs(T)
betas = betas.to(device)
alpha_bars = alpha_bars.to(device)

# Training loop
global_iter = 0
pbar = tqdm(total=total_iters, desc='Training')
while global_iter < total_iters:
    for batch in train_loader:
        # Get high-res images (normalized to [0,1])
        hr_images: torch.Tensor = batch['HR'].to(device)  # shape: (B,C,H,W)
        lr_images: torch.Tensor = batch['LR'].to(device)
        B = hr_images.shape[0]

        # Sample random timestep t for each sample in batch
        t = torch.randint(1, T+1, (B,), device=device).long()
        t_emb = get_timestep_embedding(t, channels)  # shape: (B, channels)

        # Add noise according to diffusion schedule
        # Sample epsilon ~ N(0,1)
        epsilon = torch.randn_like(hr_images)
        # Gather alpha_bar for t
        alpha_bar_t = alpha_bars[t-1].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # shape: (B,1,1,1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
        x_t = sqrt_alpha_bar * hr_images + sqrt_one_minus_alpha_bar * epsilon

        # Prepare model input: concatenate LR condition + noisy HR
        # Concatenate along channel: [LR, x_t] with 3 channels each -> total 6 channels
        model_input = torch.cat([lr_images, x_t], dim=1)

        # Forward pass
        pred_noise = model(model_input, t.float())  # model expects float t (or int, but converted to float inside)
        # Compute loss
        loss = criterion(pred_noise, epsilon)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        pbar.update(1)
        if global_iter % 1000 == 0:
            # Optional: evaluate on a small batch or subset
            # For simplicity, skip validation during training here
            pass

        # Save checkpoints periodically
        if global_iter % 10000 == 0:
            save_path = f'checkpoints/model_{global_iter}.pth'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        global_iter +=1
        if global_iter >= total_iters:
            break
pbar.close()

# Save final model
torch.save(model.state_dict(), 'checkpoints/model_final.pth')

# --- Optional: Save activation distribution visualization, or validation style evaluations ---

print("Training complete. Model saved.")

# If desired, add inference functions or validation scripts below
# For example, run inferences on test datasets with DDIM sampling, compute PSNR, SSIM, LPIPS, etc.
# This would be implemented in separate scripts or a test/evaluate function.
```

## utils.py

```python
## utils.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
try:
    import lpips
except ImportError:
    lpips = None

from typing import Tuple, List
import math

# 1. Binarization Helpers

def sign_bin(x: torch.Tensor) -> torch.Tensor:
    """
    Binarize input tensor with sign function.
    Input:
        x (torch.Tensor): any real-valued tensor
    Output:
        torch.Tensor: +1 where x >= 0, -1 where x < 0
    """
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

class STESign(torch.autograd.Function):
    """
    Straight-Through Estimator for sign function.
    Forward pass: sign
    Backward pass: passes gradients unchanged (identity).
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return sign_bin(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        # Gradient is pass-through: just return grad_output
        return grad_output

def binarize_weight(weight: torch.Tensor, scale: bool = True) -> torch.Tensor:
    """
    Binarize weights with optional scaling.
    Args:
        weight (torch.Tensor): full-precision weight tensor
        scale (bool): whether to scale by mean absolute value
    Returns:
        torch.Tensor: binarized weight tensor
    """
    if scale:
        mean_abs = weight.abs().mean()
        w_binarized = sign_bin(weight) * mean_abs
    else:
        w_binarized = sign_bin(weight)
    return w_binarized

# 2. Activation Distribution Visualization

def plot_activation_distributions(activations: List[np.ndarray], timestep_indices: List[int], save_path: str = None):
    """
    Plot the activation distributions at various diffusion timesteps.
    Args:
        activations (List[np.ndarray]): list of activation arrays collected at different timesteps
        timestep_indices (List[int]): list of timestep indices corresponding to activations
        save_path (str): optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    for idx, act in zip(timestep_indices, activations):
        plt.hist(act.flatten(), bins=50, alpha=0.5, label=f'Timestep {idx}')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.title('Activation Distributions Across Timesteps')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# 3. Dataset Augmentation Methods

import torchvision.transforms as T

def random_crop(image: np.ndarray, crop_size: int = 64) -> np.ndarray:
    """
    Randomly crop a patch of size crop_size x crop_size from image.
    Args:
        image (np.ndarray): input image of shape (H, W, C)
        crop_size (int): size of the crop
    Returns:
        np.ndarray: cropped image patch
    """
    H, W, C = image.shape
    if H < crop_size or W < crop_size:
        raise ValueError("Image size smaller than crop size.")
    top = np.random.randint(0, H - crop_size + 1)
    left = np.random.randint(0, W - crop_size + 1)
    return image[top:top+crop_size, left:left+crop_size, :]

def random_flip(image: np.ndarray, flip_prob: float = 0.5) -> np.ndarray:
    """
    Random horizontal flip.
    Args:
        image (np.ndarray): input image
        flip_prob (float): probability to flip
    Returns:
        np.ndarray: flipped or original image
    """
    if np.random.rand() < flip_prob:
        return np.flip(image, axis=1)
    return image

def random_rotation(image: np.ndarray, rotations: List[int] = [90, 180, 270]) -> np.ndarray:
    """
    Randomly rotate image by one of the specified angles.
    Args:
        image (np.ndarray): input image
        rotations (List[int]): list of angles
    Returns:
        np.ndarray: rotated image
    """
    angle = np.random.choice(rotations)
    k = angle // 90
    return np.rot90(image, k)

def dataset_transform(image: np.ndarray, crop_size: int=64, flip: bool=True) -> np.ndarray:
    """
    Compose augmentation: random crop, flip, and rotation.
    Args:
        image (np.ndarray): input image
        crop_size (int): crop size
        flip (bool): whether to apply flip
    Returns:
        np.ndarray: augmented image
    """
    img = random_crop(image, crop_size)
    if flip:
        img = random_flip(img)
    img = random_rotation(img)
    return img

# 4. Metric Calculation Wrappers

def calculate_psnr(im_pred: np.ndarray, im_true: np.ndarray, border: int=0) -> float:
    """
    Compute PSNR between two images.
    Args:
        im_pred (np.ndarray): predicted image
        im_true (np.ndarray): ground truth image
        border (int): border width to exclude from computation
    Returns:
        float: PSNR value
    """
    if border > 0:
        im_pred = im_pred[border:-border, border:-border]
        im_true = im_true[border:-border, border:-border]
    mse = np.mean((im_pred - im_true) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / math.sqrt(mse))

def calculate_ssim(im_pred: np.ndarray, im_true: np.ndarray, border: int=0) -> float:
    """
    Compute SSIM between two images.
    Args:
        im_pred (np.ndarray): predicted image
        im_true (np.ndarray): ground truth image
        border (int): border width to exclude
    Returns:
        float: SSIM value
    """
    if border > 0:
        im_pred = im_pred[border:-border, border:-border]
        im_true = im_true[border:-border, border:-border]
    ssim_index = ssim(im_true, im_pred, data_range=1.0, gaussian=True)
    return ssim_index

def calculate_lpips(im_pred: np.ndarray, im_true: np.ndarray, device: str='cuda') -> float:
    """
    Calculate LPIPS perceptual metric.
    Args:
        im_pred (np.ndarray): predicted image in [0,1]
        im_true (np.ndarray): ground truth image in [0,1]
        device (str): 'cpu' or 'cuda'
    Returns:
        float: LPIPS score
    """
    if lpips is None:
        raise ImportError("LPIPS library is not installed.")
    # Convert to tensor [C, H, W], 3-channel normalized
    to_tensor = lambda img: torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device) * 2 - 1
    pred_tensor = to_tensor(im_pred)
    true_tensor = to_tensor(im_true)
    criterion = lpips.LPIPS(net='alex').to(device)
    with torch.no_grad():
        dist = criterion(pred_tensor, true_tensor)
    return dist.item()

# 5. Additional Utilities

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int=128) -> torch.Tensor:
    """
    Generate sinusoidal positional embeddings for diffusion timesteps.
    Args:
        timesteps (torch.Tensor): 1D tensor of shape (batch,)
        embedding_dim (int): dimension of the embedding
    Returns:
        torch.Tensor: tensor of shape (batch, embedding_dim)
    """
    device = timesteps.device
    half_dim = embedding_dim // 2
    exponent = torch.arange(half_dim, dtype=torch.float32, device=device) / float(half_dim)
    freqs = torch.exp(-np.log(10000) * exponent)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # shape (batch, half_dim)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if embedding_dim % 2 != 0:
        embedding = torch.cat([embedding, torch.zeros((embedding.shape[0],1), device=device)], dim=1)
    return embedding

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor [C, H, W] in [-1,1] to numpy image in [0,1].
    Args:
        tensor (torch.Tensor): image tensor
    Returns:
        np.ndarray: image in [0,1], shape (H, W, C)
    """
    tensor_clamped = torch.clamp(tensor, -1, 1)
    img = ((tensor_clamped + 1) / 2).cpu().numpy()
    return np.transpose(img, (1, 2, 0))

def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert numpy image [H, W, C] in [0,1] or [0,255] to tensor [-1,1].
    Args:
        image (np.ndarray): input image
    Returns:
        torch.Tensor: tensor in [-1,1], shape (C, H, W)
    """
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0)
    return tensor * 2 - 1

# 6. Gradient-compatible Binarization with STE (already defined as STESign class)

# 7. Error-Handling helper functions (added for robustness)

def check_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int]) -> None:
    """
    Utility to check if tensor shape matches expected.
    """
    if tensor.shape != expected_shape:
        raise ValueError(f"Tensor shape {tensor.shape} does not match expected {expected_shape}")

```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\BI-DiffSR\BI-DiffSR_repo`
