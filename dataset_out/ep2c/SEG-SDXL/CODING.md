# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## attention.py

```python
# attention.py
import torch
import torch.nn.functional as F
import math
from typing import Optional

class AttentionLayer:
    """
    Encapsulates self-attention with optional Gaussian blurring on query or key tensors,
    supporting varying sigma values for Smoothed Energy Guidance (SEG).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        apply_blur: bool = False,
        blur_on: str = "query",  # 'query' or 'key'
        default_sigma: float = 0.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the AttentionLayer.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            apply_blur (bool): Whether to enable Gaussian blurring.
            blur_on (str): Element to blur ('query' or 'key').
            default_sigma (float): Default sigma for Gaussian blur.
            device (torch.device): Computation device.
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.apply_blur = apply_blur
        self.blur_on = blur_on
        self.default_sigma = default_sigma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Linear projections for Q, K, V
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        sigma: float = None,
        attention_mask: Optional[torch.Tensor] = None,
        guidance_enabled: bool = False
    ) -> torch.Tensor:
        """
        Compute self-attention with optional Gaussian blurring on queries or keys.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, tokens, embed_dim].
            sigma (float): Sigma for Gaussian blur; if None, use default.
            attention_mask (torch.Tensor or None): Mask for attention.
            guidance_enabled (bool): If True, perform blurred attention (SEG).

        Returns:
            torch.Tensor: Attention output of shape [batch, tokens, embed_dim].
        """
        batch_size, seq_len, _ = x.shape
        sigma = sigma if sigma is not None else self.default_sigma

        # Project inputs
        Q = self.q_proj(x)  # shape: [batch, seq_len, embed_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head
        def reshape_for_heads(tensor):
            # [batch, seq_len, embed_dim] -> [batch, heads, seq_len, head_dim]
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        Q = reshape_for_heads(Q)
        K = reshape_for_heads(K)
        V = reshape_for_heads(V)

        # If guidance is enabled and sigma > 0, blur the queries
        if self.apply_blur and guidance_enabled and sigma > 1e-8:
            Q = self.apply_gaussian_blur(Q, sigma)

        # Compute scaled dot-product attention
        # Q, K: [batch, heads, seq_len, head_dim]
        # Attention scores: [batch, heads, seq_len, seq_len]
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)  # [batch, heads, seq_len, head_dim]

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return output

    def apply_gaussian_blur(self, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply 2D Gaussian blur on the spatial tokens of tensor.
        The tensor shape: [batch, heads, seq_len, head_dim]
        We reshape to treat seq_len as 2D spatial if possible.

        Since tokens are 1D sequences, we interpret sequence length as spatial dimension.
        For higher-dimensional tokens, reshape accordingly.

        Args:
            tensor (torch.Tensor): Input tensor to blur.
            sigma (float): Standard deviation of Gaussian kernel.

        Returns:
            torch.Tensor: Blurred tensor with same shape as input.
        """
        # For simplicity, we assume seq_len is a perfect square and treat it as 2D
        batch_size, heads, seq_len, head_dim = tensor.shape

        # Determine spatial size (assumes square for simplicity)
        spatial_size = int(math.sqrt(seq_len))
        if spatial_size * spatial_size != seq_len:
            # fallback: treat as 1D, just pad to next square or use 1D convolution
            # For generality, perform 1D convolution along sequence length
            return self._apply_gaussian_blur_1d(tensor, sigma)

        # Reshape to [batch, heads, H, W, head_dim], combine head_dim later
        tensor_2d = tensor.view(batch_size, heads, spatial_size, spatial_size, head_dim)

        # Generate Gaussian kernel
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
        gaussian_kernel = self._create_gaussian_kernel(kernel_size, sigma).to(self.device)

        # Blur along height
        tensor_blur_h = self._apply_conv2d_along_dim(tensor_2d, gaussian_kernel, dim=2)

        # Blur along width
        tensor_blur_w = self._apply_conv2d_along_dim(tensor_blur_h, gaussian_kernel, dim=3)

        # Flatten back
        tensor_blurred = tensor_blur_w.view(batch_size, heads, seq_len, head_dim)
        return tensor_blurred

    def _apply_gaussian_blur_1d(self, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply 1D Gaussian blur along the sequence dimension for tensors.
        Handles large sequences or when sequence length isn't perfect squares.

        Args:
            tensor (torch.Tensor): [batch, heads, seq_len, head_dim]
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            torch.Tensor: Blurred tensor.
        """
        batch_size, heads, seq_len, head_dim = tensor.shape
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
        gaussian_kernel_1d = self._create_gaussian_kernel(kernel_size, sigma).to(self.device)
        # Shape: [kernel_size]
        # Expand for convolution: [1, 1, kernel_size]
        kernel = gaussian_kernel_1d.view(1, 1, -1)

        # Permute for conv1d: [batch*heads*head_dim, seq_len]
        tensor_perm = tensor.permute(0, 1, 3, 2).contiguous()  # [batch, heads, head_dim, seq_len]
        tensor_reshaped = tensor_perm.view(-1, 1, seq_len)  # [batch*heads*head_dim, 1, seq_len]

        # Pad to maintain size
        pad = (kernel_size // 2, kernel_size // 2)
        blurred = F.conv1d(
            tensor_reshaped,
            weight=kernel,
            padding=pad
        )
        # Reshape back
        blurred = blurred.view(batch_size, heads, head_dim, seq_len)
        blurred = blurred.permute(0,1,3,2)  # [batch, heads, seq_len, head_dim]
        return blurred

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        Create a 1D Gaussian kernel normalized to sum to 1.

        Args:
            kernel_size (int): Size of the kernel.
            sigma (float): Standard deviation.

        Returns:
            torch.Tensor: 1D Gaussian kernel.
        """
        # Generate Gaussian
        center = kernel_size // 2
        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - center
        kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel

    def _apply_conv2d_along_dim(
        self,
        tensor: torch.Tensor,
        kernel: torch.Tensor,
        dim: int
    ) -> torch.Tensor:
        """
        Apply 2D convolution along a specified spatial dimension in tensor.

        Args:
            tensor (torch.Tensor): [batch, heads, H, W, head_dim]
            kernel (torch.Tensor): 2D Gaussian kernel
            dim (int): Dimension along which to convolve (2 for H, 3 for W)

        Returns:
            torch.Tensor: Blurred tensor.
        """
        # Permute tensor to [batch, heads, H or W, other spatial, head_dim]
        permute_dims = [0, 1, 2, 3, 4]
        if dim == 2:
            # Convolve along height
            tensor_perm = tensor.permute(0, 1, 2, 3, 4)
        elif dim == 3:
            # Convolve along width
            tensor_perm = tensor.permute(0, 1, 3, 2, 4)
        else:
            raise ValueError("dim must be 2 or 3")

        # Merge batch, heads, spatial, other dims for convolution
        shape = tensor_perm.shape
        # [batch, heads, spatial_dim, other_dim, head_dim]
        tensor_flat = tensor_perm.contiguous().view(-1, 1, shape[dim], shape[dim+1] if dim==2 else shape[dim-1])
        # Prepare kernel for conv2d: assuming kernel is square
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]

        # Pad to preserve size
        pad_size = kernel.shape[-1] // 2
        tensor_blurred = F.conv2d(
            tensor_flat,
            weight=kernel,
            padding=pad_size,
            groups=1
        )
        # Reshape back
        tensor_blurred = tensor_blurred.view(shape)
        # Permute back to original shape
        if dim == 2:
            tensor_blurred = tensor_blurred.permute(0, 1, 2, 3, 4)
        else:
            tensor_blurred = tensor_blurred.permute(0, 1, 3, 2, 4)

        return tensor_blurred

```

## dataset_loader.py

```python
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
```

## diffusion_sampler.py

```python
## diffusion_sampler.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict

class DiffusionSampler:
    """
    Implements the reverse diffusion sampling process with optional Smoothed Energy Guidance (SEG).
    Coordinates the iterative denoising, guiding the process using guidance scale, sigma schedule,
    and Gaussian-blurred attention provided by the diffusion model's attention modules.
    """
    def __init__(
        self,
        model,
        guidance_scale: float = 3.0,
        sigma_schedule: List[float] = None,
        steps: int = 1000,
        sampler_type: str = "ddim",
        guidance_type: str = "none"
    ):
        """
        Initializes the sampler with model, guidance parameters, and schedule.

        Args:
            model (DiffusionModel): The trained diffusion model with attention modules supporting Gaussian blur.
            guidance_scale (float): Guidance scale parameter.
            sigma_schedule (list): List of sigma values for each inference step.
            steps (int): Total number of denoising steps.
            sampler_type (str): 'ddim' or 'ddpm' (for this implementation, we'll focus on 'ddim').
            guidance_type (str): Guidance method, e.g., 'none', 'segmented_attention'.
        """
        self.model = model
        self.guidance_scale = guidance_scale
        self.sigma_schedule = sigma_schedule or [0.0]  # Default to no guidance if not specified
        self.total_steps = steps
        self.sampler_type = sampler_type.lower()
        self.guidance_type = guidance_type.lower()
        self.device = next(model.parameters()).device

        # Validate sigma_schedule length
        if len(self.sigma_schedule) != self.total_steps:
            # If provided schedule has different length, interpolate or repeat
            self.sigma_schedule = self._interpolate_sigma_schedule(self.sigma_schedule, self.total_steps)

    def _interpolate_sigma_schedule(self, schedule: List[float], steps: int) -> List[float]:
        """
        Optional: interpolate the given sigma schedule to match total steps.
        For simplicity, if the schedule length != steps, we can repeat or linear interpolate.
        Here, we will assume a linear schedule if length mismatches.
        """
        if len(schedule) == steps:
            return schedule
        # Linear interpolation
        start_sigma = schedule[0]
        end_sigma = schedule[-1]
        return list(torch.linspace(start_sigma, end_sigma, steps).numpy())

    def sample(self, conditioning: Optional[torch.Tensor] = None, initial_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate an image by reverse diffusion starting from pure noise.

        Args:
            conditioning (torch.Tensor, optional): Conditioning input for conditional generation.
            initial_noise (torch.Tensor, optional): Starting noisy tensor; if None, initialize as random noise.

        Returns:
            torch.Tensor: The final generated image tensor.
        """
        batch_size = 1  # For simplicity, generate one sample at a time
        image_size = self.model.model.get("unet").get("in_channels", 3)  # or define explicitly
        # Note: Replace above line with correct way to retrieve input channel or define fixed size
        # For demonstration, set image tensor shape to [batch, 3, H, W]
        # Use the model's expected input size, e.g., 512x512
        H, W = 512, 512
        if initial_noise is None:
            x = torch.randn((batch_size, 3, H, W), device=self.device)
        else:
            x = initial_noise.to(self.device)

        for step in range(self.total_steps):
            sigma = self.sigma_schedule[step]
            # Perform reverse diffusion step with guidance
            x = self.run_reverse_step(
                x, conditioning=conditioning, sigma=sigma, guidance_scale=self.guidance_scale
            )
        return x

    def run_reverse_step(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor],
        sigma: float,
        guidance_scale: float
    ) -> torch.Tensor:
        """
        Run one step of the reverse diffusion process, applying guidance and Gaussian-blurred attention.

        Args:
            x (torch.Tensor): Current noisy sample.
            conditioning (Optional[torch.Tensor]): Conditioning input if conditional.
            sigma (float): The current sigma for this timestep.
            guidance_scale (float): Guidance scale parameter.

        Returns:
            torch.Tensor: Updated denoised sample.
        """
        self.model.eval()  # Set model in eval mode

        # Indicate to the model that guidance type is 'segmented_attention'
        # and provide sigma as control parameter to attention modules
        # This assumes model's forward() accepts guidance_type and sigma
        pred_noise = self.model(
            x,
            guidance_scale=guidance_scale,
            sigma=sigma,
            conditioning=conditioning
        )

        # Typically, with DDIM or DDPM, estimate previous x
        # For simplicity, assuming direct Euler update (replace with actual scheduler if available)
        # A placeholder reverse step:
        # Compute the estimated mean, then sample from it
        # Here, we assume a simplified update: x_{t-1} = x - pred_noise
        # In practice, follow the diffusion schedule equations
        x_prev = x - pred_noise

        # Optional: add stochasticity for stochastic samplers
        if self.sampler_type == "ddpm":
            # Add scaling / noise as per DDPM
            # For demonstration, add small Gaussian noise
            noise = torch.randn_like(x_prev) * sigma
            x_prev = x_prev + noise
        elif self.sampler_type == "ddim":
            # Deterministic update for DDIM
            pass  # keep as is for simplicity

        return x_prev

    def generate(self, conditioning: Optional[torch.Tensor] = None, num_samples: int = 1) -> torch.Tensor:
        """
        Generate multiple images by repeating the sampling process.

        Args:
            conditioning (Optional[torch.Tensor]): Conditioning input.
            num_samples (int): Number of samples.

        Returns:
            torch.Tensor: Batch of generated images.
        """
        samples = []
        for _ in range(num_samples):
            x = torch.randn((1, 3, 512, 512), device=self.device)
            image = self.sample(conditioning=conditioning, initial_noise=x)
            samples.append(image)
        return torch.cat(samples, dim=0)

# Note: For proper integration, the model's `forward()` method must support passing 'sigma' and guidance parameters,
# and internally, the AttentionLayer's `compute_attention()` should use `apply_gaussian_blur()` with `sigma`.

# Additional helper functions or modifications in AttentionLayer or model are assumed to be implemented accordingly.
```

## evaluation.py

```python
## evaluation.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.inception import inception_v3
from PIL import Image
from scipy import linalg
from tqdm import tqdm
import clip

class Evaluation:
    """
    Evaluation class for computing FID, CLIP score, and LPIPS for generated images.
    Loads precomputed real dataset statistics for FID, and uses pretrained models for metrics.
    """
    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        fid_stats_path: str = "fid_stats.npy",
        clip_model_name: str = "ViT-B/32",
        lpips_model_type: str = "alex"  # options: "alex", "vgg", "squeeze"
    ):
        """
        Initialize the evaluation metrics models and load real dataset stats.
        """
        self.device = device

        # Load real dataset statistics for FID
        self.fid_mu, self.fid_sigma = self.load_fid_stats(fid_stats_path)

        # Initialize Inception v3 for feature extraction (FID)
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()
        # Replace final pooling and FC to get features
        self._modify_inception_for_fid()

        # Load CLIP model
        self.clip_model, self.clip_transform = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()

        # Initialize LPIPS model
        import lpips
        self.lpips_loss_fn = lpips.LPIPS(net=lpips_model_type).to(self.device)
        self.lpips_loss_fn.eval()

        # Minimum input size for CLIP (e.g., 224x224), ensure images are resized accordingly
        self.clip_image_size = 224
        self.clip_transform_clip = transforms.Compose([
            transforms.Resize((self.clip_image_size, self.clip_image_size)),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    def load_fid_stats(self, path: str):
        """
        Load precomputed real dataset mean and covariance for FID.
        Expected to be saved as a numpy npz or npy containing 'mu' and 'sigma'.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"FID statistics file not found: {path}")
        data = np.load(path)
        mu = data['mu']
        sigma = data['sigma']
        return mu, sigma

    def _modify_inception_for_fid(self):
        """
        Replace the final pooling/fc layers of InceptionV3 to output features for FID.
        """
        # Remove final linear layer and pooling
        self.inception_model.Mixed_7c.register_forward_hook(self._hook_features)
        self.fid_features = None

    def _hook_features(self, module, input, output):
        """
        Hook to extract features after the last pooled layer.
        """
        self.fid_features = output

    def _get_inception_feature(self, images: torch.Tensor) -> np.ndarray:
        """
        Compute features of images via InceptionV3 for FID.
        """
        with torch.no_grad():
            self.fid_features = None
            _ = self.inception_model(images)
            features = self.fid_features  # [batch, 2048, 1, 1]
            features = features.squeeze().cpu().numpy()
        return features

    def calculate_fid(self, images: torch.Tensor, real_data_path: str = "") -> float:
        """
        Compute FID score between generated images and real dataset stats.
        Args:
            images: tensor of shape [N, C, H, W], pixel values in [-1, 1]
            real_data_path: optional, path to real data (not used here, stats loaded in init)
        Returns:
            float: FID score
        """
        # Ensure images are in [0,1]
        images_input = (images.clamp(-1, 1) + 1) / 2
        # Resize if needed
        features_list = []
        batch_size = 32  # process in small batches
        for i in range(0, images_input.shape[0], batch_size):
            batch_imgs = images_input[i:i+batch_size]
            feats = self._get_inception_feature(batch_imgs)
            features_list.append(feats)
        mu_gen = np.mean(np.concatenate(features_list, axis=0), axis=0)
        sigma_gen = np.cov(np.concatenate(features_list, axis=0).T)
        # Compute FID
        fid_value = self._calculate_fid_score(self.fid_mu, self.fid_sigma, mu_gen, sigma_gen)
        return fid_value

    def _calculate_fid_score(self, mu1, sigma1, mu2, sigma2) -> float:
        """
        Calculate FID between two distributions (means and covariances).
        """
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        # Numerical issues can cause imaginary parts, take real part
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid_score = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid_score)

    def calculate_clip_score(self, images: torch.Tensor, prompts: list) -> float:
        """
        Compute average CLIP similarity score between images and prompts.
        Args:
            images: [N, C, H, W], pixel values in [-1, 1]
            prompts: list of strings, same length as images or batch
        Returns:
            float: average CLIP similarity
        """
        with torch.no_grad():
            # Process images: resize and normalize
            imgs_resized = []
            for img in images:
                img_pil = transforms.ToPILImage()( (img + 1) / 2 )
                img_clip = self.clip_transform_clip(img_pil)
                imgs_resized.append(img_clip)
            imgs_tensor = torch.stack(imgs_resized).to(self.device)
            # Get image features
            img_features = self.clip_model.encode_image(imgs_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            # Tokenize prompts
            text_tokens = clip.tokenize(prompts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            similarity = torch.sum(img_features * text_features, dim=-1)
            mean_score = similarity.mean().item()
        return mean_score

    def calculate_lpips(self, images: torch.Tensor, reference_images: Optional[torch.Tensor] = None) -> float:
        """
        Compute LPIPS perceptual distance. If reference_images provided, compute pairwise.
        Else, compute diversity across images.
        Args:
            images: Tensor of shape [N, C, H, W], pixel values in [-1, 1]
            reference_images: Optional tensor, same shape as images
        Returns:
            float: mean LPIPS distance
        """
        with torch.no_grad():
            # Convert images to [0, 1], since LPIPS expects [0,1]
            inputs = (images + 1) / 2
            if reference_images is not None:
                ref = (reference_images + 1) / 2
                dist = self.lpips_loss_fn(inputs, ref)
            else:
                # Compute pairwise LPIPS between all images
                num = inputs.shape[0]
                dist_sum = 0.0
                count = 0
                for i in range(num):
                    for j in range(i + 1, num):
                        dist = self.lpips_loss_fn(inputs[i:i+1], inputs[j:j+1])
                        dist_sum += dist
                        count += 1
                if count > 0:
                    return (dist_sum / count).item()
                else:
                    return 0.0
            return dist.mean().item()

    def evaluate(self, generated_images: torch.Tensor, prompts: list = None, reference_stats: dict = None) -> dict:
        """
        Run all metrics and return their scores.
        Args:
            generated_images: [N, C, H, W], pixel values in [-1,1]
            prompts: optional list of prompts for CLIP
            reference_stats: optional, dict with 'mu' and 'sigma' for real dataset FID
        Returns:
            dict: {'fid': float, 'clip': float, 'lpips': float}
        """
        metrics = {}
        # FID
        if reference_stats and 'mu' in reference_stats and 'sigma' in reference_stats:
            self.fid_mu = reference_stats['mu']
            self.fid_sigma = reference_stats['sigma']
        fid_score = self.calculate_fid(generated_images)
        metrics['fid'] = fid_score
        
        # CLIP score
        if prompts:
            clip_score = self.calculate_clip_score(generated_images, prompts)
        else:
            clip_score = None
        metrics['clip'] = clip_score

        # LPIPS
        lpips_score = self.calculate_lpips(generated_images)
        metrics['lpips'] = lpips_score

        return metrics
```

## main.py

```python
# main.py
import os
import yaml
import torch
from dataset_loader import DatasetLoader
from model import DiffusionModel
from diffusion_sampler import DiffusionSampler
from evaluation import Evaluation

def main():
    # Load configuration from 'config.yaml'
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Dataset Loader
    dataset_cfg = cfg.get('dataset', {})
    dataset_path = dataset_cfg.get('dataset_path', './dataset')
    image_size = tuple(dataset_cfg.get('image_size', [512, 512]))
    dataset_type = dataset_cfg.get('dataset_type', 'unconditional')
    dataset_name = dataset_cfg.get('dataset_name', 'laion')
    dataset_loader = DatasetLoader(
        dataset_path=dataset_path,
        image_size=image_size,
        dataset_type=dataset_type,
        dataset_name=dataset_name
    )

    # Optional: Load datasets for training/evaluation
    # Here, we prepare dataloader for training
    train_loader = torch.utils.data.DataLoader(
        dataset_loader,
        batch_size=cfg['training'].get('batch_size', 8),
        shuffle=True,
        drop_last=True
    )

    # Initialize Diffusion Model
    model_cfg = cfg.get('model', {})
    architecture = model_cfg.get('architecture', 'SDXL')
    pretrained_ckpt = model_cfg.get('pretrained_checkpoint', '')
    freeze_backbone = model_cfg.get('freeze_backbone', False)
    attention_blur = model_cfg.get('attention_blur', True)

    diffusion_model = DiffusionModel(architecture=architecture, pretrained_path=pretrained_ckpt)
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()
    if freeze_backbone:
        for param in diffusion_model.parameters():
            param.requires_grad = False
        # Optionally, freeze only backbone parts, depending on model structure

    # Initialize Sampler with guidance and sigma schedule
    sampling_cfg = cfg.get('sampling', {})
    guidance_cfg = cfg.get('guidance', {})
    guidance_scale = guidance_cfg.get('guidance_scale', 3.0)
    sigma_schedule = guidance_cfg.get('sigma_schedule', [0,1,2,5,10,20,50,100])
    steps = sampling_cfg.get('steps', 1000)
    sampler_type = sampling_cfg.get('sampler_type', 'ddim')
    guidance_variant = guidance_cfg.get('guidance_variant', 'segmented_attention')  # e.g., 'none', 'segmented_attention'

    sampler = DiffusionSampler(
        model=diffusion_model,
        guidance_scale=guidance_scale,
        sigma_schedule=sigma_schedule,
        steps=steps,
        sampler_type=sampler_type,
        guidance_type=guidance_variant
    )

    # Optional: Load checkpoint if provided, or train
    # Here, assuming no training loop; focus on inference
    # If training desired, implement a Trainer class or call existing train()

    # Run inference over different sigma values as per sigma_schedule
    for sigma in sigma_schedule:
        # Set the current sigma in sampler (if needed)
        # Here, sigma is passed during sampling; assumes sampler handles it internally
        print(f"Generating samples with sigma={sigma} and guidance_scale={guidance_scale}")
        samples = []
        num_samples = cfg.get('sampling', {}).get('num_samples', 1)
        for _ in range(num_samples):
            # Generate images
            sample = sampler.sample(conditioning=None)  # No conditioning for unconditional
            samples.append(sample)

        # Save generated images
        save_dir = os.path.join("outputs", f"sigma_{sigma}")
        os.makedirs(save_dir, exist_ok=True)
        for i, img in enumerate(samples):
            # Convert tensor to PIL image for saving
            img_cpu = img.squeeze(0).cpu()
            # Assume pixel range [-1,1], convert to [0,255]
            img_np = ((img_cpu + 1) / 2).clamp(0,1).permute(1,2,0).numpy() * 255
            img_pil = Image.fromarray(img_np.astype('uint8'))
            img_pil.save(os.path.join(save_dir, f"sample_{i+1}.png"))

    # Optional: Run evaluation with Evaluation class
    # Load generated images for metrics
    # For example, sample some images and compute FID, CLIP, LPIPS
    # Here, just as an example:
    # generated_images = torch.stack([...
    # evaluator = Evaluation()
    # metrics = evaluator.evaluate(generated_images, prompts=[])
    # print(metrics)

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
from attention import AttentionLayer
from typing import Optional

class DiffusionModel(nn.Module):
    """
    Encapsulates a pretrained diffusion backbone (e.g., SDXL),
    supporting attention modules with Gaussian blur (SEG) during inference.
    """
    def __init__(self, architecture: str, pretrained_path: str):
        """
        Initialize the diffusion model by loading the pretrained checkpoint and 
        locating attention layers.
        
        Args:
            architecture (str): Name/identifier of the model architecture.
            pretrained_path (str): Path to the pretrained checkpoint.
        """
        super().__init__()
        self.architecture = architecture
        self.pretrained_path = pretrained_path
        
        # Load the pretrained model (assuming a simple loading mechanism here)
        # For the example, using a generic U-Net-like backbone.
        # Replace with actual loading code for SDXL in practice.
        self.model = self._load_pretrained_model()
        self.attention_layers = self._extract_attention_layers(self.model)
        
        # Set model to eval and possibly FP16 if needed
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _load_pretrained_model(self):
        """
        Load the actual pretrained model.
        Note: Placeholder implementation; replace with actual model code.
        """
        # For demonstration, suppose model is a nn.Module with attention modules.
        # Load checkpoint
        loaded_model = torch.load(self.pretrained_path, map_location='cpu')
        model = loaded_model.get('model', loaded_model)  # Adapt as per checkpoint format
        # Ensure in eval mode
        model.eval()
        return model
    
    def _extract_attention_layers(self, model):
        """
        Traverse the model to find all AttentionLayer instances.
        """
        attention_layers = []

        def recurse_modules(module):
            for child in module.children():
                if isinstance(child, AttentionLayer):
                    attention_layers.append(child)
                recurse_modules(child)
        recurse_modules(model)
        return attention_layers

    def get_attention_layers(self):
        """
        Return list of attention layers for external control or debugging.
        """
        return self.attention_layers

    def forward(self, x: torch.Tensor, guidance_scale: float = 1.0, sigma: float = 0.0, conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the diffusion model.
        Integrates Gaussian blur into attention modules via guidance.
        
        Args:
            x (Tensor): Input tensor, noisy image or latent.
            guidance_scale (float): Guidance scale (ignored here, handled externally during sampling).
            sigma (float): Std dev for Gaussian blur in attention. Applies to all attention layers.
            conditioning (Tensor or None): Conditioning input (text, class, etc), optional.
        
        Returns:
            Tensor: Model prediction (e.g., noise prediction).
        """
        # During the forward, pass sigma to attention layers
        # For example, injecting sigma into attention modules
        # Here, assuming attention modules accept sigma as a parameter during call.
        # If not, we need to set a global parameter or hook.
        # The code below assumes each attention layer has a method 'set_sigma'
        # which sets the current sigma for that layer.
        
        # For simplicity, assign sigma to each attention layer
        for attn in self.attention_layers:
            if hasattr(attn, 'set_sigma'):
                attn.set_sigma(sigma)
        
        # Run the model forward
        # It is expected that AttentionLayer internally uses the set sigma
        # during attention computation.
        output = self.model(x, conditioning=conditioning)
        return output
```

## trainer.py

```python
## trainer.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List, Dict
import yaml
import numpy as np

from dataset_loader import DatasetLoader
from model import DiffusionModel
from attention import AttentionLayer

class Trainer:
    """
    Handles training, fine-tuning, and sampling of the diffusion model with SEG (Gaussian blurred attention).
    """
    def __init__(self, config: Dict, device: torch.device):
        # Load hyperparameters from config
        self.device = device
        # Model and dataset initialization
        self.dataset = DatasetLoader(
            dataset_path=config['dataset']['dataset_path'],
            image_size=tuple(config['dataset']['image_size']),
            dataset_type=config['dataset'].get('dataset_type', 'unconditional'),
            dataset_name=config['dataset'].get('dataset_name', 'laion')
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config['training'].get('batch_size', 8),
            shuffle=True,
            drop_last=True
        )

        # Load diffusion backbone
        self.model = DiffusionModel(
            architecture=config['model']['architecture'],
            pretrained_path=config['model']['pretrained_checkpoint']
        ).to(self.device)
        self.model.eval()  # Set to eval by default; fine-tuning optional

        # Optimize which parameters? For now, fine-tune entire model
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['training'].get('learning_rate', 3e-4),
            weight_decay=0.01
        )

        # Scheduler (optional)
        self.scheduler = None
        if 'scheduler' in config:
            # For simplicity, assume CosineAnnealingLR if specified
            # Replace or extend as needed
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100000)

        # Guidance parameters
        self.guidance_scale = config['guidance'].get('guidance_scale', 3.0)
        # Sigma schedule (for Gaussian blur)
        self.sigma_schedule = config['sampling'].get('sigma_schedule', [0, 1, 2, 5, 10, 20, 50, 100])
        self.current_sigma = self.sigma_schedule[0]

        self.guidance_variant = config['guidance'].get('guidance_variant', 'segmented_attention')
        self.total_steps = config['sampling'].get('steps', 1000)
        self.sampler_type = config['sampling'].get('sampler_type', 'ddim')
        self.guidance_gamma = 3.0  # fixed guidance scale for SEG as per paper

        # Save path
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save config for reproducibility
        self.config = config

    def get_batch(self):
        """Fetch a batch of data, move to device."""
        batch = next(iter(self.dataloader))
        if self.dataset.dataset_type == 'conditional':
            images, prompts_or_conditions = batch
            images = images.to(self.device)
            # prompts_or_conditions: process prompts if needed
            conditioning = prompts_or_conditions
        else:
            images = batch.to(self.device)
            conditioning = None
        return images, conditioning

    def train_step(self, images, conditioning):
        """
        Perform one training step: forward, loss, backward, optimize.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Prepare guidance input
        guidance_scale = self.guidance_scale

        # Forward pass with Gaussian blur at attention, passing current sigma
        # Model should accept sigma for attention blurring
        output = self.model(
            images,
            guidance_scale=guidance_scale,
            sigma=self.current_sigma,
            conditioning=conditioning
        )

        # Compute diffusion loss - assume model predicts noise for training
        # Placeholder: since actual diffusion training involves simulating noised images
        # and target is the true noise; here, we assume output is noise prediction
        # and inputs are the previous noised images and true noise; adapt as per actual training
        # For simplicity, assume a dummy loss (e.g., MSE with a target tensor)
        # In practice, replace with proper noise estimation
        target_noise = torch.randn_like(output)  # Placeholder
        loss = F.mse_loss(output, target_noise)

        loss.backward()
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        return loss.item()

    def save_checkpoint(self, step: int):
        """Save model and optimizer state."""
        save_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'step': step,
            'sigma': self.current_sigma,
            'guidance_scale': self.guidance_scale,
            'config': self.config
        }, save_path)

    def load_checkpoint(self, path: str):
        """Load model and optimizer."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_sigma = checkpoint.get('sigma', self.current_sigma)
        self.guidance_scale = checkpoint.get('guidance_scale', self.guidance_scale)

    def run_training(self, total_steps: int = 100000, save_interval: int = 5000):
        """Main training loop."""
        progress_bar = tqdm(range(total_steps))
        for step in progress_bar:
            images, conditioning = self.get_batch()
            loss = self.train_step(images, conditioning)
            # Log progress
            progress_bar.set_description(f"Loss: {loss:.4f}, sigma: {self.current_sigma}")
            # Update sigma schedule (if desired)
            # For example, schedule sigma change at certain steps
            # For simplicity, pick the nearest value in sigma_schedule
            # or implement a schedule (e.g., linear, cosine)
            # Here, keep sigma fixed at initial or last sigma
            # Optional: implement scheduling logic
            # Save checkpoints
            if (step + 1) % save_interval == 0:
                self.save_checkpoint(step + 1)

    def evaluate_metrics(self, sample_images: List[torch.Tensor], real_data_path: str, prompts: Optional[List[str]] = None):
        """
        Compute evaluation metrics like FID, CLIP, LPIPS for generated samples.
        Placeholder: assumes externally implemented functions.
        """
        from evaluation import Evaluation
        evaluator = Evaluation()
        fid_score = evaluator.calculate_fid(sample_images, real_data_path)
        clip_score = evaluator.calculate_clip_score(sample_images, prompts)
        # optional: LPIPS
        return {'fid': fid_score, 'clip': clip_score}

    def generate_samples(self, num_samples: int, conditioning: Optional[torch.Tensor] = None):
        """
        Generate images using the trained model with current sigma.
        """
        # Implement sampling with your diffusion scheduler and model
        # For simplicity, provide a stub
        # Use your diffusion sampling routine, passing model, guidance, sigma
        samples = []
        for _ in tqdm(range(num_samples)):
            # noise initialization
            x_T = torch.randn((1, 3, *self.dataset.image_size), device=self.device)
            sample = self.run_reverse_process(x_T, conditioning)
            samples.append(sample)
        return samples

    def run_reverse_process(self, x_T: torch.Tensor, conditioning: Optional[torch.Tensor] = None):
        """
        Run the diffusion reverse process for a predefined number of steps.
        Incorporate attention with Gaussian blur at each step.
        """
        # Placeholder: replace with actual reverse diffusion code with model
        # During each step, pass sigma for attention blurring
        x = x_T
        for step in reversed(range(self.total_steps)):
            # Optionally, adjust sigma over steps
            # Here, keep fixed sigma
            self.model.train()
            pred_noise = self.model(
                x,
                guidance_scale=self.guidance_scale,
                sigma=self.current_sigma,
                conditioning=conditioning
            )
            # Compute previous x (sample) from pred_noise
            # Placeholder: simple Euler step (replace with actual scheduler)
            # e.g., dt = ... 
            # For demonstration:
            x = x - pred_noise  # Not exact, replace with correct reverse step
        return x

# Usage example (for external script):
# if __name__ == "__main__":
#     with open("config.yaml", 'r') as f:
#         config = yaml.safe_load(f)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     trainer = Trainer(config, device)
#     trainer.run_training(total_steps=100000, save_interval=10000)
```


---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\SEG-SDXL\SEG-SDXL_repo`
