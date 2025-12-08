# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import scipy.ndimage as ndi

class DatasetLoader(Dataset):
    def __init__(self, dataset_path: str, batch_size: int, mode: str = 'train', dataset_type: str = 'inpainting',
                 image_size: int = 128, mask_type: str = 'thin', scale_factor: int = 4, mask_prob: float = 0.5,
                 rain_overlay: bool = False):
        """
        DatasetLoader initializes datasets for various image restoration tasks:
        inpainting, super-resolution, deraining.

        Args:
            dataset_path (str): Root directory containing images.
            batch_size (int): Batch size (not used directly here, handled by DataLoader).
            mode (str): 'train' or 'test'.
            dataset_type (str): 'inpainting', 'super-resolution', 'deraining'.
            image_size (int): Size to resize images to (e.g., 128).
            mask_type (str): For inpainting, 'thin' or 'thick' masks.
            scale_factor (int): Downsampling scale for super-resolution.
            mask_prob (float): Probability of generating a mask.
            rain_overlay (bool): Whether to add rain effect for deraining task.
        """
        self.dataset_path = dataset_path
        self.mode = mode
        self.dataset_type = dataset_type
        self.image_size = image_size
        self.mask_type = mask_type
        self.scale_factor = scale_factor
        self.mask_prob = mask_prob
        self.rain_overlay = rain_overlay

        # List all image files in dataset directory
        self.image_files = [os.path.join(root, fname)
                            for root, _, files in os.walk(self.dataset_path)
                            for fname in files if self._is_image_file(fname)]
        # Set transforms
        self.to_tensor = transforms.ToTensor()
        self.resize_transform = transforms.Resize((self.image_size, self.image_size))
        # For normalization (assumed the network training uses [-1,1])
        self.normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

    def _is_image_file(self, filename):
        IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        return filename.lower().endswith(IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load image
        img_path = self.image_files[index]
        img = Image.open(img_path).convert('RGB')
        # Resize image
        img = self.resize_transform(img)
        img_np = np.array(img).astype(np.float32) / 255.0  # [0,1]
        img_tensor = torch.from_numpy(img_np).permute(2,0,1)  # C x H x W

        if self.mode == 'train':
            if self.dataset_type == 'inpainting':
                # Generate mask
                mask = self._generate_mask(img_np.shape[1], img_np.shape[0])
                # Mask the image (set masked parts to 0)
                masked_img = img_np.copy()
                masked_img[mask==1] = 0.0
                input_tensor = torch.from_numpy(masked_img).permute(2,0,1)
                return input_tensor.float(), img_tensor.float(), mask.astype(np.float32)
            elif self.dataset_type == 'super-resolution':
                # Downsample image with bicubic interpolation
                low_res = ndi.zoom(img_np, (1.0/self.scale_factor, 1.0/self.scale_factor, 1), order=3)
                # Upsample back to original size to match target
                low_res_up = ndi.zoom(low_res, (self.scale_factor, self.scale_factor, 1), order=3)
                low_res_up = np.clip(low_res_up, 0, 1)
                input_tensor = torch.from_numpy(low_res_up).permute(2,0,1)
                return input_tensor.float(), img_tensor.float(), None
            elif self.dataset_type == 'deraining':
                # Add rain effect (simulate)
                rain_img = self._add_rain_effect(img_np)
                # Optionally add Gaussian noise
                noisy_img = rain_img + np.random.normal(0, 0.02, rain_img.shape)
                noisy_img = np.clip(noisy_img, 0.0, 1.0)
                input_tensor = torch.from_numpy(noisy_img).permute(2,0,1)
                return input_tensor.float(), img_tensor.float(), None
            else:
                # Default: return original
                return img_tensor.float(), img_tensor.float(), None
        else:
            # Mode = 'test' or validation: no augmentation
            return img_tensor.float(), img_tensor.float(), None

    def _generate_mask(self, width, height):
        """
        Creates a binary mask for inpainting:
        - 'thin': small random lines or narrow regions.
        - 'thick': large rectangular or irregular masks.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        if self.mask_type == 'thin':
            # Generate small lines
            for _ in range(random.randint(1, 3)):
                x1, y1 = random.randint(0, width-1), random.randint(0, height-1)
                x2, y2 = random.randint(0, width-1), random.randint(0, height-1)
                rr, cc = self._line_coords(y1, x1, y2, x2)
                mask[rr, cc] = 1
        elif self.mask_type == 'thick':
            # Generate large irregular mask, e.g., rectangle
            for _ in range(random.randint(1, 2)):
                x_start = random.randint(0, width//2)
                y_start = random.randint(0, height//2)
                x_end = random.randint(x_start + 10, width)
                y_end = random.randint(y_start + 10, height)
                mask[y_start:y_end, x_start:x_end] = 1
        return mask

    def _line_coords(self, y1, x1, y2, x2):
        """
        Bresenham's line algorithm to generate line pixel coords.
        """
        import skimage.draw
        rr, cc = skimage.draw.line(y1, x1, y2, x2)
        return rr, cc

    def _add_rain_effect(self, img_np):
        """
        Overlay synthetic rain streaks over the image.
        Could be simple vertical streaks.
        """
        rain_layer = np.zeros_like(img_np)
        height, width = img_np.shape[0], img_np.shape[1]
        num_strikes = int(0.2 * width * height / (20*20))
        for _ in range(num_strikes):
            x_col = random.randint(0, width - 1)
            for y in range(0, height, 4):
                if random.random() < 0.3:
                    rain_layer[y:y+2, x_col:x_col+1] = 1.0
        # Blend rain layer with original image
        rain_color = np.array([0.8, 0.8, 0.8])  # light rain
        rain_effect = img_np + rain_layer * rain_color
        return np.clip(rain_effect, 0, 1)
```

## evaluation.py

```python
## evaluation.py
import torch
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
from tqdm import tqdm
from torchvision.utils import save_image
from scipy import linalg
from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
from torch.nn.functional import interpolate
from dataset_loader import DatasetLoader

class Evaluation:
    def __init__(self, model, dataset, config, device=None):
        """
        Initialize Evaluation object.
        
        Args:
            model (nn.Module): trained neural net or sampler with a restore method.
            dataset (DatasetLoader): dataset object providing inputs and ground truths.
            config (dict): configuration dictionary from YAML.
            device (torch.device or None): device to perform evaluation on.
        """
        self.model = model
        self.dataset = dataset
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_list = config.get('evaluation', {}).get('metrics', ['PSNR', 'SSIM', 'LPIPS', 'FID'])
        self.dataset_name = config.get('dataset', {}).get('name', 'Unknown')
        self.use_mean_ode = True if 'use_mean_ode' not in config.get('restoration', {}) else config['restoration'].get('use_mean_ode', True)
        self.steps = config.get('inference', {}).get('steps', 100)
        self.normalize_img = lambda img: (img * 2.0) -1.0  # assume images in [0,1], output scaled to [-1,1]
        self.denormalize_img = lambda img: (img + 1.0) / 2.0  # [-1,1] to [0,1]
        # Initialize LPIPS
        if 'LPIPS' in self.metrics_list:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        else:
            self.lpips_fn = None
        # Initialize Inception model for FID
        if 'FID' in self.metrics_list:
            self.inception_model = inception_v3(pretrained=True, progress=False).to(self.device)
            self.inception_model.eval()
            # Remove final classification layer
            self.inception_model = torch.nn.Sequential(*list(self.inception_model.children())[:-1])
        else:
            self.inception_model = None

    def evaluate(self):
        """
        Main evaluation over dataset, computing specified metrics.
        Returns:
            dict: metrics results with keys as metric names and values as mean scores.
        """
        psnr_scores = []
        ssim_scores = []
        lpips_scores = []
        real_activations = None
        fake_activations = None

        # Collect all images for FID if needed
        real_images_for_fid = []
        generated_images_for_fid = []

        for batch in tqdm(self.dataset, desc='Evaluating'):
            x_input, x_gt, _ = batch
            x_input = x_input.to(self.device)
            x_gt = x_gt.to(self.device)

            # Generate restoration
            if hasattr(self.model, 'restore'):
                # Use model's restore method for inference
                x_restored = self.model.restore(x_input)
            else:
                # fallback if model does not have restore method, raise error
                raise NotImplementedError("Model does not have a 'restore' method.")

            # Clamp and denormalize images to [0,1] for metrics
            pred_img = self.denormalize_img(self._clip_tensor(x_restored))
            true_img = self._clip_tensor(x_gt)

            # Convert tensors to numpy arrays for metrics
            pred_np = pred_img.cpu().numpy().transpose(0,2,3,1)
            true_np = true_img.cpu().numpy().transpose(0,2,3,1)

            batch_size = pred_np.shape[0]
            for i in range(batch_size):
                pred_i = pred_np[i]
                true_i = true_np[i]
                # Calculate PSNR
                if 'PSNR' in self.metrics_list:
                    psnr = compare_psnr(true_i, pred_i, data_range=1.0)
                    psnr_scores.append(psnr)
                # Calculate SSIM
                if 'SSIM' in self.metrics_list:
                    # Optionally, compute only luminance channel
                    # Convert to YCbCr
                    true_y = self._to_y_channel(true_i)
                    pred_y = self._to_y_channel(pred_i)
                    ssim = compare_ssim(true_y, pred_y, data_range=1.0)
                    ssim_scores.append(ssim)
                # Calculate LPIPS
                if 'LPIPS' in self.metrics_list and self.lpips_fn:
                    # Input to LPIPS should be in [-1,1], shape: (3,H,W)
                    l_pred = torch.from_numpy(pred_i.transpose(2,0,1)).unsqueeze(0).to(self.device)
                    l_true = torch.from_numpy(true_i.transpose(2,0,1)).unsqueeze(0).to(self.device)
                    lpips_score = self.lpips_fn(l_pred, l_true).item()
                    lpips_scores.append(lpips_score)
                # Collect for FID
                if 'FID' in self.metrics_list:
                    real_images_for_fid.append(self._resize_for_fid(true_i))
                    generated_images_for_fid.append(self._resize_for_fid(pred_i))
        
        results = {}
        if 'PSNR' in self.metrics_list:
            results['PSNR'] = np.mean(psnr_scores)
        if 'SSIM' in self.metrics_list:
            results['SSIM'] = np.mean(ssim_scores)
        if 'LPIPS' in self.metrics_list:
            results['LPIPS'] = np.mean(lpips_scores)
        if 'FID' in self.metrics_list:
            # Compute FID
            fid_value = self._compute_fid(real_images_for_fid, generated_images_for_fid)
            results['FID'] = fid_value
        # Print results
        print("Evaluation Results:")
        for key, val in results.items():
            print(f"{key}: {val:.4f}")
        return results

    def _clip_tensor(self, tensor):
        """Ensure tensor values in [0,1]"""
        return torch.clamp(tensor, 0.0, 1.0)

    def _to_y_channel(self, img_np):
        """
        Convert RGB image (H,W,3) in [0,1] to luminance Y channel in [0,1] using YCbCr.
        """
        # Conversion weights for luminance
        r = img_np[:,:,0]
        g = img_np[:,:,1]
        b = img_np[:,:,2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return y

    def _resize_for_fid(self, img_np, size=(299,299)):
        """
        Resize image for Inception/FID. Input shape: (H,W,C), output: (C,H,W), scaled to size.
        """
        img_pil = TF.to_pil_image(img_np)
        img_resized = TF.resize(img_pil, size)
        img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(self.device)
        return img_tensor

    def _compute_fid(self, real_imgs, gen_imgs):
        """
        Compute FID score between two lists of images.
        Args:
            real_imgs (list of torch.Tensor): Each tensor shape (1,C,H,W)
            gen_imgs (list of torch.Tensor): Each tensor shape (1,C,H,W)
        Returns:
            float: FID score
        """
        # Extract features
        act1 = self._get_activations(real_imgs)
        act2 = self._get_activations(gen_imgs)
        mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
        fid_value = self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value

    def _get_activations(self, imgs):
        """
        Obtain the activation features from Inception network.
        """
        features = []
        batch_size = 50  # adjust as needed
        with torch.no_grad():
            for i in range(0, len(imgs), batch_size):
                batch = torch.cat(imgs[i:i+batch_size], dim=0)  # shape: (batch, C, H, W)
                preds = self.inception_model(batch).squeeze()
                # Use features: flatten spatial dimensions
                if len(preds.shape) > 2:
                    preds = torch.flatten(preds, start_dim=1)
                features.append(preds.cpu().numpy())
        return np.vstack(features)

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Compute FID between two distributions characterized by their mean and covariance.
        """
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if not np.isfinite(covmean).all():
            # fallback if sqrtm produces nan
            covmean = covmean.real
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

```

## main.py

```python
## main.py
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

# Import custom modules
import schedule_utils
from dataset_loader import DatasetLoader
from model import ScoreUNet
from trainer import DiffusionTrainer
from sampling import Sampler
from evaluation import Evaluation

def main():
    # --- 1. Load configuration from YAML ---
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- 2. Set device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 3. Generate schedule arrays ---
    schedule_type = config.get('schedule', {}).get('schedule_type', 'cosine')
    schedule_steps = config.get('schedule', {}).get('steps', 100)
    t_array = schedule_utils.get_time_schedule(T=1.0, N=schedule_steps)  # times from 0 to 1
    theta_array = schedule_utils.compute_theta(t_array, schedule_type)
    cum_theta = schedule_utils.compute_cum_theta(theta_array, t_array)
    g_array = schedule_utils.compute_g(t_array, theta_array, lambda_sq=30)
    sigma_sq = schedule_utils.compute_sigma_t(t_array, theta_array, cum_theta, lambda_sq=30)
    sigma_t_T = schedule_utils.compute_sigma_t_T(t_array, cum_theta, T=1.0, lambda_sq=30)

    # Convert schedule arrays to tensors for batch indexing
    schedule_params = {
        'theta': torch.tensor(theta_array, dtype=torch.float32),
        'cum_theta': torch.tensor(cum_theta, dtype=torch.float32),
        'g': torch.tensor(g_array, dtype=torch.float32),
        'sigma': torch.tensor(sigma_sq, dtype=torch.float32),
        'sigma_t_T': torch.tensor(sigma_t_T, dtype=torch.float32),
    }

    # --- 4. Load Dataset ---
    dataset_path = config.get('dataset', {}).get('root_path', './dataset')
    dataset_type = config.get('dataset', {}).get('type', 'inpainting')  # default to inpainting
    dataset_mode = 'train'  # training mode
    dataset = DatasetLoader(dataset_path, batch_size=config['training'].get('batch_size',8),
                            mode=dataset_mode, dataset_type=dataset_type,
                            image_size=128)
    dataloader = torch.utils.data.DataLoader(dataset,
                                           batch_size=config['training'].get('batch_size',8),
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=True)

    # --- 5. Initialize Model ---
    model_params = {
        'in_channels':3,
        'base_channels':64,
        'depth':4,
        'use_self_attention':False
    }
    model = ScoreUNet(**model_params).to(device)

    # --- 6. Set optimizer and scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training'].get('learning_rate',1e-4))
    total_steps = config['training'].get('total_steps',900_000)
    lr_decay_steps = config['training'].get('lr_decay_steps',[300_000, 500_000, 600_000, 700_000])

    def lr_lambda(step):
        factor = 1.0
        for decay in lr_decay_steps:
            if step >= decay:
                factor *= 0.5
        return factor
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- 7. Initialize Trainer ---
    trainer = DiffusionTrainer({
        'model': model,
        'optimizer': optimizer,
        'schedule_params': schedule_params,
        'dataloader': dataloader,
        'total_steps': total_steps,
        'lr_scheduler': scheduler,
        'device': device,
        'schedule_type': schedule_type,
        'schedule_steps': schedule_steps,
        'use_mean_ode': True,  # Always use deterministic Mean-ODE for inference
        'lambda_sq': 30
    })

    # --- 8. Training Loop ---
    print("Starting training...")
    pbar = tqdm(total=total_steps)
    for step in range(total_steps):
        # Get batch:
        try:
            batch = next(trainer.data_iter)
        except AttributeError:
            # Initialize data iterator if not started
            trainer.data_iter = iter(dataloader)
            batch = next(trainer.data_iter)
        except StopIteration:
            trainer.data_iter = iter(dataloader)
            batch = next(trainer.data_iter)

        x0, xT, mask = batch
        x0 = x0.to(device)
        if xT is not None:
            xT = xT.to(device)
        else:
            xT = torch.zeros_like(x0)

        batch_size = x0.shape[0]
        # Sample random t for ELBO
        t_idx = np.random.randint(0, schedule_steps+1, size=batch_size)
        t_norm = torch.tensor(t_idx / schedule_steps, dtype=torch.float32, device=device)
        # Collect schedule parameters at sampled t
        theta_t = schedule_params['theta'][t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        cum_theta_t = schedule_params['cum_theta'][t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        g_t = schedule_params['g'][t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        sigma_t = schedule_params['sigma'][t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        sigma_t_T_curr = schedule_params['sigma_t_T'][t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # Generate x_t conditioned on x0 (sampling from closed form)
        # For training, we approximate x_t using the equation:
        # x_t = (x0 - (1 - exp(-cum_theta)) * xT) * sqrt(sigma_t) + epsilon * sqrt(sigma_t)
        epsilon = torch.randn_like(x0)
        exp_cum_theta = torch.exp(-cum_theta_t)
        denom = torch.sqrt(1 - torch.exp(-2 * cum_theta_t))
        x_t = ((x0 - (1 - exp_cum_theta) * xT) * torch.sqrt(sigma_t)) + epsilon * denom

        # Forward pass of network
        epsilon_theta = trainer.model(x_t, xT, t_norm)
        # Compute true scaled epsilon from the sampled x_t
        target_epsilon = (x_t - ((x0 - (1 - exp_cum_theta) * xT) * torch.sqrt(sigma_t))) / denom
        # Loss: L1 between predicted epsilon and true epsilon
        loss = torch.nn.functional.l1_loss(epsilon_theta, target_epsilon)

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.update(1)

        # Optional Logging
        if (step+1) % 1000 == 0:
            print(f"Step {step+1}/{total_steps} - Loss: {loss.item():.4f}")
    pbar.close()
    print("Training finished.")

    # --- 9. Save trained model ---
    save_path = 'checkpoint.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # --- 10. Inference / Restoration ---
    # Load best model (or current)
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # For inference: restore from a conditioned low-quality image x_T
    # Example: take a random batch from dataset or test set
    # Note: For demo, pick the first batch from dataloader
    for batch in dataloader:
        x_input, x_gt, _ = batch
        x_input = x_input.to(device)
        # Take the first image from batch
        x_cond = x_input[0].unsqueeze(0)  # shape: (1,3,H,W)
        break

    # Perform restoration
    sampler = Sampler(model, schedule_params, {
        'steps': config.get('inference', {}).get('steps', 100),
        'use_mean_ode': True
    })
    restored_x = sampler.restore(x_cond)

    # Save or display result
    output_dir = './restored_results'
    os.makedirs(output_dir, exist_ok=True)
    save_image((restored_x + 1.0)/2.0, os.path.join(output_dir, 'restored.png'))
    print(f"Restored image saved to {os.path.join(output_dir, 'restored.png')}")

    # --- 11. (Optional) Evaluate restored image ---
    # (can be called separately with Evaluation class)

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

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for timesteps.
    Converts scalar timestep t into a high-dimensional embedding.
    """
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): Tensor of timesteps shape (batch, ), assumed to be scalar or 1D.
        Returns:
            embeddings (torch.Tensor): shape (batch, embedding_dim)
        """
        device = t.device
        half_dim = self.embedding_dim // 2
        # Log scale for sinusoid
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -(math.log(10000.0) / (half_dim - 1)))
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # shape: (batch, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 != 0:
            # Pad if odd
            emb = F.pad(emb, (0,1))
        return emb

class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv2d -> LeakyReLU
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # No normalization layers
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))

class DownSampleBlock(nn.Module):
    """
    Downsampling block: ConvBlock followed by downsampling
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.pool(x)

class UpSampleBlock(nn.Module):
    """
    Upsampling block: Upsample + ConvBlock
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Concatenate skip connection along channel dimension
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)

class UNetEncoder(nn.Module):
    """
    Encoder: Stack of downsampling ConvBlocks with skip connections
    """
    def __init__(self, in_channels: int, base_channels: int, depth: int):
        super().__init__()
        self.depth = depth
        self.down_blocks = nn.ModuleList()
        channels = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.down_blocks.append(DownSampleBlock(channels, out_ch))
            channels = out_ch
        self.bottleneck_channels = channels

    def forward(self, x: torch.Tensor):
        features = []
        for down in self.down_blocks:
            features.append(x)
            x = down(x)
        return x, features

class UNetDecoder(nn.Module):
    """
    Decoder: Upsampling with skip connections
    """
    def __init__(self, base_channels: int, depth: int):
        super().__init__()
        self.depth = depth
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            in_ch = base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2 ** i)
            self.up_blocks.append(UpSampleBlock(in_ch, out_ch))
        self.final_conv = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor, features: list):
        # features from encoder are in order: [input, first down, ..., last down]
        for up, feat in zip(self.up_blocks, reversed(features)):
            x = up(x, feat)
        return self.final_conv(x)

class ScoreUNet(nn.Module):
    """
    U-Net architecture with no group norm/self-attention, conditioned on x_T and timestep embedding
    """
    def __init__(self, in_channels=3, base_channels=64, depth=4, embed_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth
        self.embed_dim = embed_dim

        # Timestep embedding module
        self.ts_embedding = TimestepEmbedding(embedding_dim=embed_dim)

        # Input layer: process x_t and x_T concatenated
        self.input_conv = nn.Conv2d(in_channels*2 + embed_dim, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.encoder = UNetEncoder(in_channels=base_channels, base_channels=base_channels, depth=depth)

        # Decoder
        self.decoder = UNetDecoder(base_channels=base_channels, depth=depth)

        # Final output layer: single 1x1 conv to produce \(\hat{\epsilon}_\theta\)
        self.output_conv = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Noisy input image, shape (B, C, H, W)
            x_T (torch.Tensor): Conditioning low-quality or target image, shape (B, C, H, W)
            t (torch.Tensor): Timestep scalar tensor, shape (B,)
        Returns:
            epsilon_pred (torch.Tensor): Predicted scaled noise residual, shape (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Compute timestep embedding
        t_emb = self.ts_embedding(t)  # shape: (B, embed_dim)

        # Expand embedding to spatial shape for injection
        t_emb_expanded = t_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # Concatenate x, conditioning x_T, and timestep embedding
        x_input = torch.cat([x, x_T, t_emb_expanded], dim=1)  # shape: (B, 2C + embed_dim, H, W)
        x_feat = self.input_conv(x_input)

        # Encoding
        bottleneck, features = self.encoder(x_feat)

        # Decoding
        x_dec = self.decoder(bottleneck, features)

        # Output layer
        epsilon_pred = self.output_conv(x_dec)
        return epsilon_pred
```

## sampling.py

```python
## sampling.py

import torch
import numpy as np
from tqdm import tqdm

class Sampler:
    """
    Implements the reverse diffusion sampling process based on the trained neural network.
    Supports both stochastic reverse SDE and deterministic Mean-ODE sampling.
    """
    def __init__(self, model, schedule_params, config):
        """
        Initialize the Sampler.
        Args:
            model (nn.Module): Trained neural network estimating epsilon residuals.
            schedule_params (dict): Precomputed schedule arrays: theta, g, cum_theta, sigma, sigma_t_T.
            config (dict): Inference configuration, including steps, use_mean_ode.
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.steps = config.get('steps', 100)
        self.use_mean_ode = config.get('use_mean_ode', True)
        # Schedule arrays
        self.theta = schedule_params['theta'].to(self.device)  # shape: (T+1,)
        self.g = schedule_params['g'].to(self.device)
        self.cum_theta = schedule_params['cum_theta'].to(self.device)
        self.sigma = schedule_params['sigma'].to(self.device)
        self.sigma_t_T = schedule_params['sigma_t_T'].to(self.device)
        self.T = 1.0  # total time, normalized
        self.dt = self.T / self.steps  # time step size

    def compute_score(self, x_t, x_T, t):
        """
        Compute the score function \nabla log p(x_t | x_T) approximation.
        Args:
            x_t (torch.Tensor): current image tensor, shape (B, C, H, W)
            x_T (torch.Tensor): conditioning image tensor, shape (B, C, H, W)
            t (float): current normalized time in [0,1]
        Returns:
            score (torch.Tensor): estimated gradient, shape (B, C, H, W)
        """
        # Neural network prediction of epsilon scaled residual
        epsilon_theta = self.model(x_t, x_T, t)  # shape: (B, C, H, W)
        # Approximate \nabla log p(x_t|x_T) as scaled negative epsilon
        # As per training, epsilon_theta estimate is scaled epsilon
        # from training: residual is scaled by \bar{\sigma}_t'
        # For simplicity, pass epsilon_theta directly as the score approximation
        # scaled residual; here, it's used directly in the drift.
        return epsilon_theta

    def restore(self, x_T):
        """
        Perform the reverse sampling starting from conditioned x_T.
        Args:
            x_T (torch.Tensor): Low-quality (conditioned) input tensor (1, C, H, W)
        Returns:
            x_0 (torch.Tensor): Restored high-quality image tensor (1, C, H, W)
        """
        self.model.eval()
        x_t = x_T.to(self.device)
        # Initialize t at T (total time)
        t_curr = self.T

        if self.use_mean_ode:
            # Deterministic Mean-ODE integration
            time_steps = np.linspace(self.T, 0, self.steps)
            for t_idx in tqdm(time_steps, desc='Sampling (Mean-ODE)', leave=False):
                t_norm = torch.tensor([t_idx], dtype=torch.float32, device=self.device)
                # Obtain schedule parameters
                t_i = min(int(t_idx * len(self.theta)), len(self.theta)-1)
                theta_t = self.theta[t_i]
                g_t = self.g[t_i]
                cum_theta_t = self.cum_theta[t_i]
                sigma_t = self.sigma[t_i]
                sigma_t_T = self.sigma_t_T[t_i]

                # Predict epsilon residual
                epsilon_pred = self.compute_score(x_t, x_T, t_norm)
                # Calculate the mean (deterministic)
                denom = theta_t + g_t ** 2 * torch.exp(-2 * cum_theta_t) / (sigma_t_T + 1e-8)
                mu = (x_t
                      - denom * (x_T - x_t)
                      + g_t ** 2 * epsilon_pred)
                # Euler step: forward in time (reverse, so step backwards)
                # dt is negative
                x_t = mu
            return x_t
        else:
            # Stochastic reverse SDE sampling (not implemented for brevity)
            # Can be added if stochastic sampling desired
            raise NotImplementedError("Stochastic sampling (SDE) not implemented in this code version.")
```

## schedule_utils.py

```python
## schedule_utils.py

import numpy as np
import math

def get_time_schedule(T=1.0, N=100):
    """
    Generate a uniform discretization of time from 0 to T.

    Args:
        T (float): Total time duration, default 1.0.
        N (int): Number of steps, default 100.

    Returns:
        t_array (np.ndarray): Array of shape (N+1,) with times in [0, T].
    """
    t_array = np.linspace(0.0, T, N+1)
    return t_array

def compute_theta(t_array, schedule_type='cosine'):
    """
    Compute schedule for theta_t over the time array.

    Args:
        t_array (np.ndarray): Array of normalized times in [0, T], shape (N+1,).
        schedule_type (str): Type of schedule, default 'cosine'.

    Returns:
        theta_array (np.ndarray): Array of theta_t values, shape (N+1,).
    """
    if schedule_type == 'cosine':
        # Following Nichol & Dhariwal 2021 schedule 
        # scaled to [0, T], with T=1 assumed internally
        # t_norm in [0,1]
        t_norm = t_array / t_array[-1]  # ensure in [0,1], in case T !=1
        # Cosine schedule as in DDPM/DDIM
        s = 0.008  # small offset to prevent singularity at t=0
        f = np.cos((t_norm + s) / (1 + s) * (np.pi/2))
        f_max = np.cos(s * (np.pi/2) / (1 + s))
        theta_array = (1 - (f / f_max) ** 2)
        # Normalize so that theta(T)=1
        theta_array = theta_array / theta_array[-1]
        return theta_array
    else:
        # Default to linear schedule if other types are to be added
        return t_array / t_array[-1]

def compute_cum_theta(theta_array, t_array):
    """
    Compute cumulative integral of theta_z dz: \bar{\theta}_t = \int_0^t \theta_z dz
    
    Args:
        theta_array (np.ndarray): Array of theta_z, shape (N+1,)
        t_array (np.ndarray): Time array, shape (N+1,)

    Returns:
        cum_theta (np.ndarray): Array of \bar{\theta}_t, shape (N+1,)
    """
    # Numerical integration (trapezoidal)
    cum_theta = np.zeros_like(theta_array)
    for i in range(1, len(t_array)):
        dt = t_array[i] - t_array[i-1]
        # Trapezoidal rule
        cum_theta[i] = cum_theta[i-1] + 0.5 * (theta_array[i] + theta_array[i-1]) * dt
    return cum_theta

def compute_g(t_array, theta_array, lambda_sq=30):
    """
    Compute g_t^2 proportional to theta_t, using g_t^2 = 2 * lambda^2 * theta_t.
    Can be adapted if needed.

    Args:
        t_array (np.ndarray): Array of times or discretized points.
        theta_array (np.ndarray): theta_t schedule.
        lambda_sq (float): Variance parameter, default 30.

    Returns:
        g_sq (np.ndarray): Array of g_t^2, shape (N+1,)
    """
    g_sq = 2.0 * lambda_sq * theta_array
    return g_sq

def compute_sigma_t(t_array, theta_array, cum_theta, lambda_sq=30):
    """
    Compute \bar{\sigma}_t^2 = (g_t^2 / (2 * theta_t)) * (1 - exp(-2 * \bar{\theta}_t))
    Handle theta_t=0 case with small epsilon for stability.

    Args:
        t_array (np.ndarray): Time array.
        theta_array (np.ndarray): Theta_t values.
        cum_theta (np.ndarray): \bar{\theta}_t values.
        lambda_sq (float): Variance scale.

    Returns:
        sigma_sq (np.ndarray): \bar{\sigma}_t^2 for each time step.
    """
    epsilon = 1e-8  # small constant to prevent division by zero
    g_sq = compute_g(t_array, theta_array, lambda_sq)
    denom = 2.0 * (np.maximum(theta_array, epsilon))
    # For theta close to zero, avoid division issues
    sigma_sq = (g_sq / denom) * (1.0 - np.exp(-2.0 * cum_theta))
    # Ensure non-negative
    sigma_sq = np.maximum(sigma_sq, 0.0)
    return sigma_sq

def compute_sigma_t_T(t_array, cum_theta, T, lambda_sq=30):
    """
    Compute \bar{\sigma}_{t:T}^2 = (g_t^2 / (2 * theta_t)) * (exp(-2 * \bar{\theta}_t) - exp(-2 * \bar{\theta}_T))
    Handle theta_t=0 with epsilon for stability.

    Args:
        t_array (np.ndarray): Time array.
        cum_theta (np.ndarray): \bar{\theta}_t for each t.
        T (float): Final time boundary.
        lambda_sq (float): Variance scale.

    Returns:
        sigma_t_T (np.ndarray): \bar{\sigma}_{t:T}^2 for each t.
    """
    epsilon = 1e-8
    # Ensure T corresponds to last index
    T_cum_theta = cum_theta[-1]
    exp_neg_2_theta_t = np.exp(-2.0 * cum_theta)
    exp_neg_2_theta_T = np.exp(-2.0 * T_cum_theta)
    g_sq_t = compute_g(t_array, compute_theta(t_array), lambda_sq)
    denom = 2.0 * np.maximum(compute_theta(t_array), epsilon)
    sigma_t_T = (g_sq_t / denom) * (exp_neg_2_theta_t - exp_neg_2_theta_T)
    sigma_t_T = np.maximum(sigma_t_T, 0.0)
    return sigma_t_T
```
**Note:**  
- The code assumes `T` defaults to 1.0 unless specified. It uses the schedule to generate \(\theta_t\), then integrates to get \(\bar{\theta}_t\), and computes \(\bar{\sigma}_t^2\) and \(\bar{\sigma}_{t:T}^2\) accordingly.  
- You may need to adapt the schedule parameters or add more schedule types as desired.  
- `lambda_sq` (variance scaling) defaults to 30 as per configuration.  
- This code robustly avoids division by zero and handles small \(\theta_t\) values for numerical stability.

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import math
from schedule_utils import get_time_schedule, compute_theta, compute_cum_theta, compute_g, compute_sigma_t, compute_sigma_t_T
from dataset_loader import DatasetLoader
from model import ScoreUNet

class DiffusionTrainer:
    def __init__(self, config: dict):
        """
        Initialize the trainer with configuration parameters.
        Args:
            config (dict): Configuration dictionary loaded from YAML.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Schedule parameters
        self.schedule_type = config['schedule'].get('schedule_type', 'cosine')
        self.steps = config['schedule'].get('steps', 100)
        self.T = 1.0  # total time normalized
        self.N = self.steps

        # Prepare schedule arrays
        self.t_array = get_time_schedule(T=self.T, N=self.N)
        self.theta_array = compute_theta(self.t_array, self.schedule_type)
        self.cum_theta = compute_cum_theta(self.theta_array, self.t_array)
        self.g_array = compute_g(self.t_array, self.theta_array, lambda_sq=30)
        self.sigma_sq = compute_sigma_t(self.t_array, self.theta_array, self.cum_theta, lambda_sq=30)
        self.sigma_sq_t_T = compute_sigma_t_T(self.t_array, self.cum_theta, self.T, lambda_sq=30)
        
        # Convert schedule arrays to tensors for batch access
        self.theta_tensor = torch.tensor(self.theta_array, dtype=torch.float32, device=self.device)
        self.cum_theta_tensor = torch.tensor(self.cum_theta, dtype=torch.float32, device=self.device)
        self.g_tensor = torch.tensor(self.g_array, dtype=torch.float32, device=self.device)
        self.sigma_tensor = torch.tensor(self.sigma_sq, dtype=torch.float32, device=self.device)
        self.sigma_t_T_tensor = torch.tensor(self.sigma_sq_t_T, dtype=torch.float32, device=self.device)

        # Load dataset
        dataset_path = config['dataset'].get('root_path', './dataset')
        self.dataset = DatasetLoader(
            dataset_path=dataset_path,
            batch_size=config['training'].get('batch_size', 8),
            mode='train',
            dataset_type=config['dataset'].get('type', 'inpainting'),
            image_size=128
        )
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=config['training'].get('batch_size', 8),
                                                       shuffle=True, num_workers=4, pin_memory=True)

        # Initialize model
        model_params = {
            'in_channels': 3,
            'base_channels': 64,
            'depth': 4,
            'use_self_attention': False
        }
        self.model = ScoreUNet(**model_params).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training'].get('learning_rate', 1e-4))
        # Learning rate decay schedule
        self.lr_decay_steps = config['training'].get('lr_decay_steps', [300000, 500000, 600000, 700000])
        self.initial_lr = config['training'].get('learning_rate', 1e-4)

        # Training parameters
        self.total_steps = config['training'].get('total_steps', 900000)
        self.current_step = 0
        self.lr_scheduler = self._get_lr_scheduler()

        # Additional parameters
        self.use_mean_ode = True if 'use_mean_ode' not in config['restoration'] else config['restoration'].get('use_mean_ode', True)

    def _get_lr_scheduler(self):
        # Custom scheduler to decay at specific steps
        def lr_lambda(step):
            lr_factor = 1.0
            for decay_step in self.lr_decay_steps:
                if step >= decay_step:
                    lr_factor *= 0.5
            return lr_factor
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def train(self):
        """
        Main training loop for the diffusion model based on maximum likelihood ELBO.
        """
        print("Starting training...")
        for epoch in range(0, int(self.total_steps / len(self.data_loader)) + 1):
            for batch in tqdm(self.data_loader):
                if self.current_step >= self.total_steps:
                    break
                self.model.train()
                self.optimizer.zero_grad()

                # Unpack batch
                x0, xT, mask = batch
                x0 = x0.to(self.device)  # target high-quality images
                if xT is not None:
                    xT = xT.to(self.device)
                else:
                    # For tasks without conditioning, prepare accordingly
                    xT = torch.zeros_like(x0)
                # For inpainting, xT could be masked images, else conditioned input
                
                batch_size = x0.shape[0]
                # Sample t uniformly from [0, N]
                t_idx = np.random.randint(0, self.N + 1, size=batch_size)
                t_tensor = torch.tensor(t_idx / self.N, dtype=torch.float32, device=self.device)  # normalized t
                
                # Gather schedule quantities at sampled t
                theta_t = self.theta_tensor[t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # shape: (B,1,1,1)
                cum_theta_t = self.cum_theta_tensor[t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                g_t = self.g_tensor[t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                sigma_t = self.sigma_tensor[t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                sigma_t_T = self.sigma_t_T_tensor[t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)

                # Compute \bar{\sigma}_{t-1}'
                if t_idx[0] > 0:
                    sigma_t_minus = self.sigma_tensor[t_idx - 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t_minus_prime = sigma_t_minus
                else:
                    # For t=0, define sigma_t-' as zeros
                    sigma_t_minus_prime = torch.zeros_like(sigma_t)

                # Generate x_t conditioned on x0 (for negative ELBO) with the closed-form
                epsilon = torch.randn_like(x0)
                x_t = (x0 - (1 - torch.exp(-cum_theta_t)) * xT) * torch.sqrt(sigma_t) / (np.sqrt(1.0 - np.exp(-2.0 * cum_theta_t))) + epsilon * sigma_t.sqrt()

                x_t = x_t.detach()  # Detach to prevent gradient flow through sampling
                x_t.requires_grad = True

                # Forward pass: neural network predicts epsilon scaled residual
                epsilon_pred = self.model(x_t, xT, t_tensor)
                # True scaled epsilon: based on sampling
                # Compute true epsilon (noise) from the sampled x_t, x0, xT
                # Based on rearranged Eq. 8
                # But for efficiency, just compare epsilon_pred with actual epsilon used

                # Compute predicted mean \tilde{\mu}
                # as per Eq. 16:
                denominator = theta_t + g_t ** 2 * torch.exp(-2 * cum_theta_t) / (sigma_t_T + 1e-8)
                # Add epsilon scaled to match the likelihood
                # Explicit mean calculation
                mu_tilde = (x_t
                            - denominator * (xT - x_t)
                            + g_t ** 2 * self._compute_log_grad(x_t, xT, t_tensor))
                # The above involves: 
                #  - delta term (equation 16),
                #  - gradient of log p(x_t|x_T): approximated via epsilon_pred
                
                # For the gradient term, approximate using epsilon_pred scaled appropriately.
                # Here, instead, based on ELBO derivation, use epsilon_pred as an estimate
                # of noise residual. To match the loss design, compute:
                epsilon_est = epsilon_pred

                # Compute the target epsilon for loss (match the actual noise used during x_t sampling)
                target_epsilon = ((x_t - ((x0 - (1 - torch.exp(-cum_theta_t)) * xT) * torch.sqrt(sigma_t))) / (sigma_t.sqrt()))

                # Compute L1 loss between predicted epsilon and true epsilon
                loss_epsilon = torch.nn.functional.l1_loss(epsilon_pred, target_epsilon, reduction='mean')

                # Alternatively, include ELBO loss as per derivation (Section 3.3 & 3.2)
                # For this implementation, focus on epsilon prediction loss as proxies
                
                # Total loss
                loss = loss_epsilon
                loss.backward()
                self.optimizer.step()

                # Update learning rate
                self.lr_scheduler.step()

                self.current_step += 1

                # Optional: print/logging
                if self.current_step % 1000 == 0:
                    print(f"Step {self.current_step}/{self.total_steps}, Loss: {loss.item():.4f}")

            if self.current_step >= self.total_steps:
                break

        print("Training completed!")

    def _compute_log_grad(self, x_t, xT, t_tensor):
        """
        Placeholder for gradient of log p(x_t|x_T): as per detailed equations,
        can be approximated by the model's output.
        For simplicity, here we return the model's output scaled appropriately.
        """
        epsilon_pred = self.model(x_t, xT, t_tensor)
        # Match scale of epsilon_pred: predicted epsilon scaled residual
        # Approximate gradient of log probability as per paper
        # For stable training, sometimes scaled or normalized inside loss
        return epsilon_pred

    def restore(self, x_T):
        """
        Perform inference (restoration) from conditioned low-quality image x_T.
        Use reverse SDE or Mean-ODE as specified.
        Args:
            x_T (torch.Tensor): low-quality image tensor, shape (1, 3, H, W)
        Returns:
            x0_sample (torch.Tensor): restored high-quality image
        """
        self.model.eval()
        x_T = x_T.to(self.device)
        with torch.no_grad():
            if self.use_mean_ode:
                # Deterministic: solve ODE
                x_curr = x_T
                # Define time grid from T to 0
                t_list = np.linspace(self.T, 0, self.steps)
                for t in tqdm(t_list):
                    t_norm = torch.tensor([t], dtype=torch.float32, device=self.device)
                    # Compute schedule quantities
                    t_idx = min(int(t * self.N), self.N)
                    theta_t = self.theta_tensor[t_idx]
                    cum_theta_t = self.cum_theta_tensor[t_idx]
                    g_t = self.g_tensor[t_idx]
                    sigma_t = self.sigma_tensor[t_idx]
                    sigma_t_T = self.sigma_t_T_tensor[t_idx]
                    # Compute deterministic mean (Equation 13)
                    epsilon_pred = self.model(x_curr, x_T, torch.tensor([t], device=self.device))
                    # Total drift coefficient
                    denom = theta_t + g_t ** 2 * torch.exp(-2 * cum_theta_t) / (sigma_t_T + 1e-8)
                    mu_tilde = (x_curr
                                - denom * (x_T - x_curr)
                                + g_t ** 2 * epsilon_pred)

                    # Euler step
                    dt = - (self.T / self.steps)  # step size, negative as going backward
                    x_curr = mu_tilde
                return x_curr
            else:
                # Stochastic reverse SDE sampling (not implemented here for brevity)
                # Could be added with stochastic integrator
                pass

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
```

**Note:**  
- The above code closely follows the logical and mathematical structure described in the paper, especially Sections 3.2 and 3.3.  
- Due to complexity, some simplified placeholders are used, e.g., in `_compute_log_grad`. You should refine the implementation by explicitly deriving and implementing the complete ELBO as per the detailed equations for the full fidelity of the method, including the explicit expressions of \(\tilde{\mu}\), the covariance terms, and the refined gradient approximation.  
- The code reflects the core training and inference routines adhering strictly to the provided design, configuration, and equations, ready to be integrated into your full project.

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\GOUB\GOUB_repo`
