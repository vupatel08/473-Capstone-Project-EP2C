# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## coding.py

```python
## coding.py
import torch
import torch.nn.functional as F
import numpy as np

class BayesianCoder:
    """
    Implements Bayesian entropy coding for Gaussian variables, supporting
    encoding and decoding of weights (or parameters) using their Gaussian
    posteriors and Bernoulli (or continuous relaxation) approximations.
    Uses a simple bits-back A* or similar scheme implied by the paper.
    """
    def __init__(self):
        # Placeholder for actual entropy coding library or custom implementation.
        # Here, we mock basic interface for encode and decode.
        pass

    def encode_gaussian(self, data, mu, sigma, prior_mu, prior_sigma):
        """
        Encodes `data` assuming Gaussian likelihood and prior.
        Returns bits used (simulated), actual implementation depends on coder.
        """
        # For simplicity, compute negative log likelihood as bits estimate.
        # In practice, replace with actual entropy coder.
        # Avoid log(0): add epsilon
        epsilon = 1e-8
        nll = 0.5 * torch.log(2 * np.pi * prior_sigma**2 + epsilon) + \
              0.5 * ((data - mu)**2 / (prior_sigma**2 + epsilon))
        bits = torch.sum(nll) / torch.log(torch.tensor(2.0))
        return bits

    def decode_gaussian(self, bitstream, mu, sigma, prior_mu, prior_sigma):
        """
        Decodes data assuming Gaussian likelihood and prior.
        For the mock, simply returns the mean, as actual decoding requires
        the inverse process of encode.
        """
        # In an actual implementation, the decoding process retrieves values.
        # Here, we mock by returning previous data (which would not be available).
        # For illustration, return mu.
        return mu

    def encode_weights(self, weights, variational_params, prior_params, permutation=None):
        """
        Encodes a weight vector (flattened) with permutations.
        - weights: tensor of shape [total_params]
        - variational_params: dict with 'mu' and 'rho' tensors, each shape [total_params]
        - prior_params: dict with 'mu' and 'rho'
        - permutation: tensor of permutation indices (optional)
        Returns:
            bits: estimated bits (float)
            stored bits during process (placeholder)
        """
        # Apply permutation if provided
        if permutation is not None:
            weights_perm = weights[permutation]
            mu_perm = variational_params['mu'][permutation]
            rho_perm = variational_params['rho'][permutation]
            prior_mu_perm = prior_params['mu'][permutation]
            prior_rho_perm = prior_params['rho'][permutation]
        else:
            weights_perm = weights
            mu_perm = variational_params['mu']
            rho_perm = variational_params['rho']
            prior_mu_perm = prior_params['mu']
            prior_rho_perm = prior_params['rho']

        sigma_post = torch.exp(0.5 * rho_perm)
        sigma_prior = torch.exp(0.5 * prior_rho_perm)

        # Encode using Gaussian likelihood (approximate KL)
        bits = self.encode_gaussian(weights_perm, mu_perm, sigma_post, prior_mu_perm, sigma_prior)
        # In actual implementation, the bits above would be used to quantify bits in bitstream
        # For mock, we just return the estimate
        return bits

    def decode_weights(self, bitstream, variational_params, prior_params, permutation=None, shape=None):
        """
        Decodes weights given bitstream and prior/posterior parameters.
        Here, a mock returning the posterior mean.
        """
        # In a real implementation, decoding would retrieve values from bitstream
        # For simplicity, return the posterior mean
        if permutation is not None:
            mu_perm = variational_params['mu'][permutation]
        else:
            mu_perm = variational_params['mu']
        # Return reconstructed weights: just the mean
        reconstructed = mu_perm
        # Undo permutation if provided
        if permutation is not None:
            inv_perm = torch.argsort(permutation)
            reconstructed = reconstructed[inv_perm]
        return reconstructed

class Encoder:
    """
    Handles the encoding of weights (sampled from q(w)) using neural entropy coding
    with permutation strategies, block-wise processing, and prior models.
    """
    def __init__(self, prior_params, post_params, permutation_indices, block_size=1024):
        """
        Args:
            prior_params: dict with 'mu', 'rho' tensors of prior
            post_params: dict with 'mu', 'rho' tensors of q(w)
            permutation_indices: permutation tensor for the weights
            block_size: size of blocks to partition the weight vector
        """
        self.prior_params = prior_params
        self.post_params = post_params
        self.permutation = permutation_indices
        self.block_size = block_size
        self.coder = BayesianCoder()
        self.bitstream = []  # placeholder list for bits

    def encode(self, weights):
        """
        Encode the full weight vector with permutations and blocks.
        """
        # Apply permutation
        w_perm = weights[self.permutation]
        total_params = w_perm.shape[0]
        bits_total = 0.0
        # Divide into blocks
        for start in range(0, total_params, self.block_size):
            end = min(start + self.block_size, total_params)
            block_w = w_perm[start:end]
            block_mu = self.post_params['mu'][start:end]
            block_rho = self.post_params['rho'][start:end]
            prior_mu_block = self.prior_params['mu'][start:end]
            prior_rho_block = self.prior_params['rho'][start:end]
            # Encode block
            bits_block = self.coder.encode_gaussian(
                data=block_w,
                mu=block_mu,
                sigma=torch.exp(0.5 * block_rho),
                prior_mu=prior_mu_block,
                prior_sigma=torch.exp(0.5 * prior_rho_block)
            )
            bits_total += bits_block.item()
            # In practice, store bits to actual bitstream here
        self.bitstream.append(bits_total)
        return bits_total

    def get_bitstream(self):
        """
        Return the final bitstream (here, placeholder list).
        """
        return self.bitstream

class Decoder:
    """
    Handles Bayesian decoding of weights from bitstream, applying inverse permutations,
    reconstructing the weight vector for INR.
    """
    def __init__(self, prior_params, post_params, permutation_indices, shape):
        """
        Args:
            prior_params: dict with 'mu', 'rho'
            post_params: dict with 'mu', 'rho'
            permutation_indices: permutation tensor used during encoding
            shape: shape of the full weight tensor
        """
        self.prior_params = prior_params
        self.post_params = post_params
        self.permutation = permutation_indices
        self.shape = shape
        self.coder = BayesianCoder()

    def decode(self, bitstream):
        """
        Decode weights assuming stored bits (here, placeholder returns posterior mean).
        """
        total_params = np.prod(self.shape)
        # Reconstruct weights block-wise
        reconstructed_full = torch.zeros(total_params, device=self.post_params['mu'].device)
        for start in range(0, total_params, self.shape[0]):
            end = min(start + self.shape[0], total_params)
            # Decode block
            block_mu = self.post_params['mu'][start:end]
            block_rho = self.post_params['rho'][start:end]
            decoded_block = self.coder.decode_gaussian(
                bitstream=bitstream,
                mu=block_mu,
                sigma=torch.exp(0.5 * block_rho),
                prior_mu=self.prior_params['mu'][start:end],
                prior_sigma=torch.exp(0.5 * self.prior_params['rho'][start:end])
            )
            reconstructed_full[start:end] = decoded_block
        # Undo permutation
        inv_perm = torch.argsort(self.permutation)
        weights_reconstructed = reconstructed_full[inv_perm]
        return weights_reconstructed.view(self.shape)

def apply_permutation(vector: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """
    Applies a permutation to a vector.
    """
    return vector[permutation]

def inverse_permutation(vector: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """
    Applies the inverse permutation.
    """
    inv_perm = torch.argsort(permutation)
    return vector[inv_perm]
```

## dataset_loader.py

```python
## dataset_loader.py

import os
import glob
import numpy as np
import torch
from PIL import Image
import soundfile as sf
import cv2
from Bio.PDB import PDBParser

class Dataset:
    """Simple Dataset class holding dataset samples."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class DatasetLoader:
    def __init__(self, config):
        """
        Initialize DatasetLoader with configuration parameters.

        Args:
            config (dict): Dictionary with dataset configurations.
        """
        self.config = config
        # Parse dataset config
        dataset_cfg = self.config.get('datasets', {})
        self.data_dir = dataset_cfg.get('data_dir', '')
        self.dataset_type = dataset_cfg.get('type', 'image')  # 'image', 'audio', 'video', 'protein'
        self.patch_size = dataset_cfg.get('patch_size', None)
        self.patches_in_group = dataset_cfg.get('patches_in_group', None)
        self.structure_format = dataset_cfg.get('structure_format', 'voxel')  # for proteins
        self.voxel_resolution = dataset_cfg.get('voxel_resolution', 64)
        self.sample_rate = dataset_cfg.get('sample_rate', 16000)
        self.chunk_duration_sec = dataset_cfg.get('chunk_duration_sec', 1.0)  # for audio
        # Additional attributes
        self.samples = []

    def load_data(self):
        """
        Load dataset according to modality and preprocess.

        Returns:
            list of dicts: Each dict contains 'coordinates', 'values', 'metadata'.
        """
        if self.dataset_type == 'image':
            return self._load_image_dataset()
        elif self.dataset_type == 'audio':
            return self._load_audio_dataset()
        elif self.dataset_type == 'video':
            return self._load_video_dataset()
        elif self.dataset_type == 'protein':
            return self._load_protein_dataset()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _load_image_dataset(self):
        # Determine if CIFAR-10 or Kodak based on image_size
        # For simplicity, assume high-res: Kodak and CIFAR explicitly distinguished externally
        images = []
        # Load images: assume directory contains image files
        image_files = glob.glob(os.path.join(self.data_dir, '*.png')) + \
                      glob.glob(os.path.join(self.data_dir, '*.jpg')) + \
                      glob.glob(os.path.join(self.data_dir, '*.jpeg')) + \
                      glob.glob(os.path.join(self.data_dir, '*.bmp'))

        # For CIFAR-10: 32x32 images
        # For Kodak: 768x512 or 512x768 images
        # Here, just load all images and process
        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)  # (H,W,C)
            H, W, C = img_np.shape
            # Normalize pixel values to [0,1]
            pixel_vals = torch.from_numpy(img_np).float() / 255.0
            # Generate coordinate grid: shape (H*W, 2)
            coords = self._generate_grid(H, W)
            # reshape pixel_vals to (H*W, 3)
            vals = pixel_vals.view(-1, 3)
            # Store data sample
            sample = {
                'coordinates': coords,
                'values': vals,
                'metadata': {'image_path': img_path}
            }
            images.append(sample)
        return images

    def _load_audio_dataset(self):
        # Load raw audio clips from data_dir
        audio_files = glob.glob(os.path.join(self.data_dir, '*.flac')) + \
                      glob.glob(os.path.join(self.data_dir, '*.wav')) + \
                      glob.glob(os.path.join(self.data_dir, '*.mp3'))  # if mp3 is acceptable

        # For each audio file: load waveform, segment into patches
        samples = []
        for audio_path in audio_files:
            data, sr = sf.read(audio_path)
            # Resample if needed
            if sr != self.sample_rate:
                # For simplicity, assume same sr, or implement resampling here
                pass
            duration_samples = int(self.sample_rate * self.chunk_duration_sec)
            total_samples = len(data)
            # Trim or pad to 3 sec
            if total_samples < duration_samples:
                pad_width = duration_samples - total_samples
                data = np.pad(data, (0, pad_width), mode='constant')
            else:
                data = data[:duration_samples]
            # Segment into overlapping patches of size 800
            patch_size = 800
            num_patches = (duration_samples - patch_size) // patch_size + 1
            for p in range(num_patches):
                start_idx = p * patch_size
                end_idx = start_idx + patch_size
                patch_signal = data[start_idx:end_idx]
                # Generate normalized time indices in [0,1]
                t_coords = np.linspace(0,1,patch_size).astype(np.float32)
                coords = torch.from_numpy(t_coords).unsqueeze(1)  # (patch_size,1)
                vals = torch.from_numpy(patch_signal).unsqueeze(1)  # (patch_size,1)
                sample = {
                    'coordinates': coords,  # 1D time coordinate normalized
                    'values': vals,
                    'metadata': {'audio_path': audio_path, 'patch_idx': p}
                }
                samples.append(sample)
        return samples

    def _load_video_dataset(self):
        # Load videos, sample frames, crop/resize, patchify
        # For brevity, a placeholder: user should replace with actual video loading
        # Here, assume directory contains video files
        video_files = glob.glob(os.path.join(self.data_dir, '*.mp4')) + \
                      glob.glob(os.path.join(self.data_dir, '*.avi'))
        samples = []
        for v_path in video_files:
            cap = cv2.VideoCapture(v_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            # Convert frames to numpy array (T,H,W,3), resize to 128x128
            frames_np = np.stack(frames)
            resized_frames = [cv2.resize(f, (128, 128)) for f in frames_np]
            # Optionally crop or resize to 128x128
            # For each frame, convert to tensor
            for idx, frame in enumerate(resized_frames):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                val = torch.from_numpy(frame_rgb).float() / 255.0  # (H,W,3)
                H, W, C = val.shape
                # Generate 3D coordinate grid normalized [0,1]
                coords = self._generate_3d_grid(H, W)
                # For video, coordinate is (x,y,t)
                # Here, t is frame index normalized
                t_norm = np.array([[idx / len(resized_frames)]], dtype=np.float32)
                t_coord = torch.from_numpy(t_norm).repeat(H*W,1)  # (H*W,1)
                xy_coords = coords.view(-1, 2)
                all_coords = torch.cat([xy_coords, t_coord], dim=1)
                vals = val.view(-1, 3)
                sample = {
                    'coordinates': all_coords,
                    'values': vals,
                    'metadata': {'video_path': v_path, 'frame_idx': idx}
                }
                samples.append(sample)
        return samples

    def _load_protein_dataset(self):
        # Load protein structures from data_dir, parse with BioPython
        pdb_files = glob.glob(os.path.join(self.data_dir, '*.pdb'))
        parser = PDBParser(QUIET=True)
        samples = []
        for pdb_path in pdb_files:
            structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
            # For simplicity, process first chain and first 96 residues
            model = structure[0]
            chain_ids = list(model.get_chains())
            if len(chain_ids) == 0:
                continue
            chain = model[chain_ids[0]]
            coords_list = []
            for residue in chain:
                if residue.get_id()[0] != ' ':  # skip hetero residues
                    continue
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    coords_list.append(ca_atom.get_coord())
            coords_np = np.array(coords_list[:96])  # first 96 residues
            if coords_np.shape[0] < 96:
                # Pad with zeros if less than 96 residues
                pad = np.zeros((96 - coords_np.shape[0], 3))
                coords_np = np.vstack([coords_np, pad])
            # Normalize to [0,1] based on bounding box
            min_coords = coords_np.min(axis=0)
            max_coords = coords_np.max(axis=0)
            coords_normalized = (coords_np - min_coords) / (max_coords - min_coords + 1e-8)
            coords_tensor = torch.from_numpy(coords_normalized.astype(np.float32))
            # Values: 3D coordinates, or as per desired encoding
            values = coords_tensor.clone()  # store normalized coords as signal (for simplicity)
            sample = {
                'coordinates': coords_tensor,
                'values': values,
                'metadata': {'pdb_path': pdb_path}
            }
            samples.append(sample)
        return samples

    def _generate_grid(self, height, width):
        """
        Generate normalized 2D coordinate grid in [0,1]
        shape: (height*width, 2)
        """
        y_coords = np.linspace(0, 1, height, endpoint=False)
        x_coords = np.linspace(0, 1, width, endpoint=False)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        coords = np.stack([xx, yy], axis=-1).reshape(-1,2)
        return torch.from_numpy(coords).float()

    def _generate_3d_grid(self, height, width):
        """
        Generate normalized 2D coordinate grid for video frames
        shape: (height*width, 2)
        """
        y_coords = np.linspace(0, 1, height, endpoint=False)
        x_coords = np.linspace(0, 1, width, endpoint=False)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        coords = np.stack([xx, yy], axis=-1)
        return torch.from_numpy(coords).float()

```

## evaluation.py

```python
## evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class Evaluation:
    def __init__(self, test_dataset, reconstructed_samples, posterior_params,
                 prior_mu, prior_rho, modality='image', output_path='results'):
        """
        Initialize the evaluator.
        Args:
            test_dataset (list): List of original data samples. Each element is a dict containing:
                - 'coordinates': tensor (N_points, dim)
                - 'values': tensor (N_points, channels)
                - 'metadata': dict, includes info like 'image_path', 'audio_path', etc.
            reconstructed_samples (list): List of reconstructed data in same order as test_dataset.
            posterior_params (list): List of dicts with 'mu' and 'rho' tensors for each sample's q(w).
            prior_mu (torch.Tensor): Prior mean tensor.
            prior_rho (torch.Tensor): Prior log-variance tensor.
            modality (str): 'image', 'audio', 'video', 'protein'
            output_path (str): Directory to save plots
        """
        self.test_dataset = test_dataset
        self.reconstructed_samples = reconstructed_samples
        self.posterior_params = posterior_params
        self.prior_mu = prior_mu
        self.prior_rho = prior_rho
        self.modality = modality
        self.output_path = output_path
        import os
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def compute_rate(self, posterior):
        """
        Estimate rate in bits as KL divergence times log2(e).
        Args:
            posterior (dict): {'mu': tensor, 'rho': tensor}
        Returns:
            float: rate in bits
        """
        mu_post = posterior['mu']
        rho_post = posterior['rho']
        sigma_post = torch.exp(0.5 * rho_post)
        # KL divergence with prior (assumed Gaussian)
        kl = 0.5 * torch.sum(
            (sigma_post ** 2 + (mu_post - self.prior_mu) ** 2) / torch.exp(self.prior_rho)
            - 1 + self.prior_rho - rho_post
        )
        kl = kl.item()  # get scalar
        bits = kl / np.log(2)  # convert nats to bits
        return bits

    def compute_psnr(self, y_true, y_pred):
        """
        Compute PSNR for images or signals in [0,1].
        """
        mse = np.mean((y_true - y_pred) ** 2)
        max_val = 1.0  # assuming normalized data
        if mse < 1e-8:
            return 100  # very high PSNR for near perfect
        psnr = 10 * np.log10((max_val ** 2) / mse)
        return psnr

    def compute_rmsd(self, y_true, y_pred):
        """
        Compute RMSD for 3D point clouds (e.g., protein data).
        """
        diff = y_true - y_pred
        rmsd = np.sqrt(np.mean(np.linalg.norm(diff, axis=1) ** 2))
        return rmsd

    def evaluate(self):
        """
        Evaluate all samples, compute rate, distortion metrics, and produce RD curve.
        """
        rates = []
        distortions = []

        # Store modality-specific metric functions
        for idx, sample in enumerate(self.test_dataset):
            orig = sample['values'].cpu().numpy()
            recon = self.reconstructed_samples[idx]
            # In case reconstructed is tensor, convert to numpy
            if isinstance(recon, torch.Tensor):
                recon = recon.cpu().numpy()

            # Compute rate
            posterior = self.posterior_params[idx]
            rate = self.compute_rate(posterior)
            rates.append(rate)

            # Compute distortion
            if self.modality == 'image' or self.modality == 'video':
                # For images/video: expect shape (H,W,C)
                dist_metric = self.compute_psnr
            elif self.modality == 'audio':
                dist_metric = self.compute_psnr
            elif self.modality == 'protein':
                dist_metric = self.compute_rmsd
            else:
                # fallback: MSE
                def dist_metric(y_true, y_pred):
                    return -10 * np.log10(np.mean((y_true - y_pred) ** 2) + 1e-8)
            dist = dist_metric(orig, recon)
            distortions.append(dist)

        avg_rate = np.mean(rates)
        avg_distortion = np.mean(distortions)

        # Plot RD curve
        plt.figure()
        plt.scatter(rates, distortions, c='blue', label='Samples')
        plt.xlabel('Rate (bits per signal/atom)')
        if self.modality in ['image', 'video', 'audio']:
            plt.ylabel('PSNR (dB)')
        elif self.modality == 'protein':
            plt.ylabel('RMSD (Å)')
        plt.title(f'Rate-Distortion Curve: {self.modality}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_path}/rd_curve_{self.modality}.png')
        plt.close()

        # Return summary metrics
        return {
            'average_rate_bpp': avg_rate,
            'average_distortion': avg_distortion
        }

    def visual_comparison(self, sample_idx, max_display=3):
        """
        Save visual comparison of original and reconstructed data for sample_idx.
        Supports images, audio waveforms, point clouds.
        """
        sample = self.test_dataset[sample_idx]
        orig = sample['values'].cpu().numpy()
        recon = self.reconstructed_samples[sample_idx]
        if isinstance(recon, torch.Tensor):
            recon = recon.cpu().numpy()

        if self.modality in ['image', 'video']:
            # Create side-by-side image comparison
            import cv2
            # Denormalize if needed; here assume already in [0,1]
            orig_img = (orig * 255).astype(np.uint8).reshape(-1, orig.shape[0] if self.modality=='video' else -1, 3)
            recon_img = (recon * 255).astype(np.uint8).reshape(-1, orig.shape[0] if self.modality=='video' else -1, 3)
            # For simplicity, show first channel
            fig, axs = plt.subplots(1, 2, figsize=(8,4))
            axs[0].imshow(orig_img)
            axs[0].set_title('Original')
            axs[1].imshow(recon_img)
            axs[1].set_title('Reconstructed')
            plt.suptitle(f'Sample {sample_idx} {self.modality} comparison')
            plt.savefig(f'{self.output_path}/visual_{self.modality}_{sample_idx}.png')
            plt.close()
        elif self.modality == 'audio':
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(orig, label='Original')
            plt.plot(recon, label='Reconstructed')
            plt.legend()
            plt.title(f'Audio waveform sample {sample_idx}')
            plt.savefig(f'{self.output_path}/audio_compare_{sample_idx}.png')
            plt.close()
        elif self.modality == 'protein':
            # Visualize original and reconstructed point clouds
            fig = plt.figure()
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(orig[:,0], orig[:,1], orig[:,2], c='blue', label='Original')
            ax.set_title('Original')
            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(recon[:,0], recon[:,1], recon[:,2], c='red', label='Reconstructed')
            ax.set_title('Reconstructed')
            plt.suptitle(f'Protein structure {sample_idx}')
            plt.savefig(f'{self.output_path}/protein_{sample_idx}.png')
            plt.close()

# Usage example (not in code): Initialize this class with loaded dataset, reconstructed data, posterior params,
# prior mu/rho, and call evaluate() to produce RD curve and metrics.
```

## hierarchical_patch.py

```python
## hierarchical_patch.py
"""
Implements the HierarchicalPatchModel class for managing hierarchical Bayesian priors
over high-resolution data subdivided into patches, as described in Appendix B.2 and Figure 2.
It models, infers, and updates global, group, and patch-level weight representations,
supports permutation strategies, and maintains the dependencies and sharing necessary
for the hierarchical prior in the RECOMBINER framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import List, Optional, Tuple

class GaussianVariational:
    """
    Helper class to represent Gaussian variational distributions: q(μ,ρ)
    for hierarchical variables, with methods for sampling and KL.
    """
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor):
        """
        Initialize Gaussian variational distribution.
        Args:
            mu: mean tensor
            rho: log-variance tensor
        """
        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample from q(μ, ρ): h = μ + ε * σ
        Args:
            num_samples: number of samples
        Returns:
            samples: tensor of shape [num_samples, *mu.shape]
        """
        std = torch.exp(0.5 * self.rho)
        eps = torch.randn((num_samples,) + self.mu.shape, device=self.mu.device)
        return self.mu.unsqueeze(0) + eps * std.unsqueeze(0)

    def kl_divergence(self, prior_mu: torch.Tensor, prior_rho: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence KL(q || p) per Gaussian with diagonal covariance.
        Args:
            prior_mu: prior mean
            prior_rho: prior log-variance
        Returns:
            scalar tensor of KL
        """
        sigma_q = torch.exp(0.5 * self.rho)
        sigma_p = torch.exp(0.5 * prior_rho)
        term1 = (sigma_q ** 2) / (sigma_p ** 2)
        term2 = ((prior_mu - self.mu) ** 2) / (sigma_p ** 2)
        kl = 0.5 * torch.sum(term1 + term2 - 1 + torch.log((sigma_p ** 2) / (sigma_q ** 2) + 1e-8))
        return kl

class HierarchicalPatchModel:
    """
    Implements the hierarchical Bayesian prior for patch-based high-res signals,
    with global, group, and patch-level latent variables, permutation strategies,
    and variational inference.
    """
    def __init__(self, config, total_patches: int, group_size: int = 16, hierarchy_levels: int = 3,
                 seed: int = 42):
        """
        Initialize hierarchical prior and variational parameters.
        Args:
            config (dict): Configuration including prior variances, init params.
            total_patches (int): number of lowest-level patches.
            group_size (int): number of patches per group at second level.
            hierarchy_levels (int): total hierarchy levels (3 here).
            seed (int): random seed for permutations.
        """
        torch.manual_seed(seed)
        random.seed(seed)
        self.total_patches = total_patches
        self.group_size = group_size
        self.hierarchy_levels = hierarchy_levels

        # --- Priors over global h_w (top level) ---
        mu_global = torch.zeros_like(torch.tensor([]))  # will be set per dataset
        rho_global = torch.full_like(torch.tensor([]), fill_value=0.0)  # Variances initialized to 1

        # These will be set during training using dataset statistics; placeholders here
        self.prior_mu_global = torch.zeros(1)  # placeholder, updated during training
        self.prior_rho_global = torch.zeros(1)

        # --- Priors over group deviations at level 2 ---
        mu_group = torch.zeros(self.hierarchy_levels - 2, dtype=torch.float32)
        rho_group = torch.zeros(self.hierarchy_levels - 2, dtype=torch.float32)  # log-variance

        # --- Priors over patch deviations at level 1 ---
        mu_patch = torch.zeros(self.total_patches, dtype=torch.float32)
        rho_patch = torch.zeros(self.total_patches, dtype=torch.float32)

        # Variational parameters for global
        # Initialize with small variance (e.g., log-variance = -12)
        self.q_mu_global = nn.Parameter(torch.zeros_like(self.prior_mu_global))
        self.q_rho_global = nn.Parameter(torch.full_like(self.prior_rho_global, -12.0))
        self.q_global = GaussianVariational(self.q_mu_global, self.q_rho_global)

        # Variational for groups: shape (number of groups, mu, rho)
        self.num_groups = int(math.ceil(self.total_patches / self.group_size))
        self.q_mu_groups = nn.Parameter(torch.zeros(self.num_groups))
        self.q_rho_groups = nn.Parameter(torch.full((self.num_groups,), -12.0))
        self.q_groups = [GaussianVariational(self.q_mu_groups[i:i+1], self.q_rho_groups[i:i+1])
                         for i in range(self.num_groups)]

        # Variational for patches: shape (total_patches, mu, rho)
        self.q_mu_patches = nn.Parameter(torch.zeros(self.total_patches))
        self.q_rho_patches = nn.Parameter(torch.full((self.total_patches,), -12.0))
        self.q_patches = [GaussianVariational(self.q_mu_patches[i:i+1], self.q_rho_patches[i:i+1])
                          for i in range(self.total_patches)]

        # --- Permutation matrices/vectors ---
        self.perm_patch_current = torch.arange(self.total_patches)
        # Permutation for patches (shuffling across patches)
        self.permutation_patch = torch.randperm(self.total_patches)
        # Permutation within groups (across patches in each group)
        self.permutation_groups = [torch.randperm(self.group_size) for _ in range(self.num_groups)]

        # Save group assignments for each patch
        self.patch_to_group = [i // self.group_size for i in range(self.total_patches)]

        # --- Additional parameters for dependency modeling, if needed ---
        # For simplicity, assume no hyper-priors over covariances here.
        # Users can extend with hyper-priors if desired.

    def sample_global(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample global h_w from variational posterior.
        Returns:
            Tensor: shape (num_samples, global_dim)
        """
        return self.q_global.sample(num_samples)

    def sample_group(self, group_idx: int, num_samples: int=1) -> torch.Tensor:
        """
        Sample group deviation h_w^g.
        Args:
            group_idx: index of the group
        Returns:
            Tensor: shape (num_samples, group_dim)
        """
        return self.q_groups[group_idx].sample(num_samples)

    def sample_patch(self, patch_idx: int, num_samples: int=1) -> torch.Tensor:
        """
        Sample patch deviation h_w^π.
        Args:
            patch_idx: index of the patch
        Returns:
            Tensor: shape (num_samples, patch_dim)
        """
        return self.q_patches[patch_idx].sample(num_samples)

    def get_patch_weights(self, global_h: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate the per-patch weights from global and deviations.
        Args:
            global_h: sampled global h_w, shape (global_dim,)
        Returns:
            List of tensors: each shape (patch_dim,)
        """
        device = global_h.device
        patch_weights = []
        for pi in range(self.total_patches):
            group_idx = self.patch_to_group[pi]
            # Sample deviations
            h_g_samples = self.sample_group(group_idx)  # shape (samples, group_dim)
            h_p_samples = self.sample_patch(pi)        # shape (samples, patch_dim)

            # For deterministic extraction, take the mean of the variational posterior
            # Alternatively, sample once:
            # h_g_mean = h_g_samples.mean(0)
            # h_p_mean = h_p_samples.mean(0)
            # But typically, during inference, we take expectation (mean):
            h_g_mean = self.q_groups[group_idx].mu
            h_p_mean = self.q_patches[pi].mu
            # Combine: h_w^(\pi) = global + deviation (group and patch)
            h_w_pi = global_h + h_g_mean + h_p_mean
            patch_weights.append(h_w_pi)
        return patch_weights

    def sample_global_posterior(self) -> torch.Tensor:
        """
        Sample global h_w from the variational posterior
        """
        return self.q_global.sample()

    def sample_all_patch_weights(self, global_h: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate patch weights for all patches given global h_w.
        """
        return self.get_patch_weights(global_h)

    def update_variational_params(self, new_mu_global, new_rho_global,
                                    new_mu_groups, new_rho_groups,
                                    new_mu_patches, new_rho_patches):
        """
        Update all variational parameters with new values.
        """
        self.q_mu_global.data.copy_(new_mu_global)
        self.q_rho_global.data.copy_(new_rho_global)
        for i in range(self.num_groups):
            self.q_mu_groups.data[i:i+1] = new_mu_groups[i]
            self.q_rho_groups.data[i:i+1] = new_rho_groups[i]
        for pi in range(self.total_patches):
            self.q_mu_patches.data[pi:pi+1] = new_mu_patches[pi]
            self.q_rho_patches.data[pi:pi+1] = new_rho_patches[pi]

    def compute_kl(self,
                   prior_mu_global: torch.Tensor,
                   prior_rho_global: torch.Tensor,
                   prior_mu_group: torch.Tensor,
                   prior_rho_group: torch.Tensor,
                   prior_mu_patch: torch.Tensor,
                   prior_rho_patch: torch.Tensor) -> torch.Tensor:
        """
        Compute the total KL divergence upper bound (Equation 4)
        for the hierarchical model, summing over global, groups, and patches.
        """
        kl_global = self.q_global.kl_divergence(prior_mu_global, prior_rho_global)
        kl_groups = 0.0
        for i in range(self.num_groups):
            kl_groups += self.q_groups[i].kl_divergence(prior_mu_group, prior_rho_group)
        kl_patches = 0.0
        for pi in range(self.total_patches):
            kl_patches += self.q_patches[pi].kl_divergence(prior_mu_patch, prior_rho_patch)
        return kl_global + kl_groups + kl_patches

    def apply_permutation(self):
        """
        Permute the individual matrices/hierarchies as per permutation vectors.
        This applies to the concatenated representation matrix H(ℓ) at each level.
        For simplicity, user should perform permutations outside this class.
        """
        # Example: permute patch order
        self.perm_patch_current = self.permutation_patch
        # For groups, per-group permutation
        self.permutation_groups = [permutation for permutation in self.permutation_groups]
        # These permutation vectors can be used during representation stacking
  
    def get_permuted_indices(self, level: str) -> torch.Tensor:
        """
        Access current permutation indices for a given level.
        """
        if level == 'patch':
            return self.perm_patch_current
        elif level == 'group':
            # For group level, return list of permutations
            return self.permutation_groups
        else:
            return torch.arange(self.total_patches)

    # Additional utility methods for handling matrix stacking, slicing, etc.,
    # can be added as needed for the full encoding/decoding pipeline.

```

## main.py

```python
# main.py

import os
import yaml
import torch
import numpy as np
import random

from dataset_loader import DatasetLoader
from model import INRModel
from variational import VariationalDistribution
from hierarchical_patch import HierarchicalPatchModel
from trainer import Trainer
from coding import BayesianCoder
from evaluation import Evaluation

def main():
    # Load configuration from YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set random seeds for reproducibility
    seed = config.get('experiment', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets via DatasetLoader
    dataset_loader = DatasetLoader(config)
    dataset_samples = dataset_loader.load_data()

    # Partition dataset into training and test sets
    # For simplicity, assume dataset_samples is ordered; in practice, shuffle or separate datasets
    # For the primary experiments (e.g., CIFAR-10), assume all samples are test
    test_dataset = dataset_samples

    # Instantiate the INR model
    model_cfg = config.get('model', {})
    inr_model = INRModel(model_cfg).to(device)

    # Initialize learnable positional encodings h_z as a parameter if used
    # For simplicity, initialize h_z as a parameter; in practice, may be learned via a separate module
    positional_dim = model_cfg.get('positional_encoding_dim', 128)
    batch_size = 1  # for inference; adjust per dataset
    h_z = torch.randn(batch_size, positional_dim, device=device, requires_grad=True)

    # Initialize the upsampling CNN φ
    phi_cfg = {
        'input_dim': positional_dim,
        'output_dim': positional_dim
    }
    phi = Evaluation(None, None, None, None, None).__class__.evaluate  # placeholder to load architecture
    # As per instructions, define φ as in Appendix B.1: small CNN
    # For simplicity, here define a minimal CNN stub
    from model import PositionalEncodingCNN
    phi_net = PositionalEncodingCNN(input_dim=positional_dim, output_dim=positional_dim).to(device)

    # Initialize variational posteriors q(w) (for all weights)
    total_params = sum(p.numel() for p in inr_model.parameters())
    init_params = {
        'mu': torch.zeros(total_params, device=device),
        'rho': torch.full((total_params,), -12.0, device=device)  # std ~ 1e-6
    }
    variational_qw = VariationalDistribution(shape=(total_params,), init_params=init_params)

    # Initialize the linear reparameterization matrix A (block-diagonal structure)
    A_matrix = torch.eye(total_params, device=device)
    A_matrix = A_matrix.requires_grad_(True)

    # Initialize hierarchical patch model if using patches (from config)
    patch_model = None
    if 'hierarchical_levels' in model_cfg:
        # Determine number of patches based on dataset and patch size
        # For CIFAR-10: 50k train, but in test, assuming patches are given
        # For this example, create a dummy placeholder; replace with actual patch count
        total_patches = sum([s['coordinates'].shape[0] for s in test_dataset])  # placeholder
        total_patches = 100  # in real logic, set actual number
        patch_model = HierarchicalPatchModel(
            config=dict(),  # fill with appropriate hyperparameters if needed
            total_patches=total_patches,
            seed=seed
        )

    # Instantiate optimizer
    params_to_optimize = [p for p in inr_model.parameters()] + \
                         [variational_qw.mu, variational_qw.rho, A_matrix] + \
                         list(phi_net.parameters())

    optimizer = torch.optim.Adam(params_to_optimize, lr=config['training'].get('learning_rate', 0.0001),
                                 betas=(0.9, 0.999), weight_decay=1e-5)

    # Initialize beta scheduling
    beta = config['training'].get('beta_initial', 0.3)
    beta_min = config['training'].get('beta_min', 0.0)
    beta_max = config['training'].get('beta_max', 1.0)
    tau_C = config['training'].get('beta_adjust_step', 0.5)
    target_bpp = config['training'].get('beta_target_bpp', 0.3)  # bits per pixel/atom
    epsilon = 1e-4  # bet for adjustment

    # For training the prior hyperparameters, they might be updated per Algorithm 1
    # For simplicity, establish prior mu, rho as parameters, or keep fixed here
    prior_mu = torch.zeros(total_params, device=device)
    prior_rho = torch.full((total_params,), -12.0, device=device)

    # Setup a DataLoader for batching datasets if needed, or process all data at once
    # For simplicity, process entire dataset per epoch with custom batching
    # Masked here: assuming 'dataset_samples' is small enough for full-batch
    # Otherwise, implement DataLoader accordingly

    # Main training loop
    total_epochs = config['training'].get('epochs', 550)
    for epoch in range(total_epochs):
        optimizer.zero_grad()

        # --- Step 1: Infer q(w) via gradient steps ---
        # For this implementation, we perform a single gradient step per batch
        # For test purposes, perform a light inferring routine
        # For the actual implementation, run multiple gradient steps on each batch
        # To mimic this, perform a forward pass, compute loss, backprop, optimizer step
        total_loss = 0.0

        # Loop over all samples (or patches if subdivided)
        for sample in test_dataset:
            coords = sample['coordinates'].to(device)      # shape: [N_points, coord_dim]
            values = sample['values'].to(device)             # shape: [N_points, channels]
            # Generate positional encodings
            h_z_current = h_z
            pos_encodings = phi_net(h_z_current, coords.unsqueeze(0)).squeeze(0)  # shape: [N_points, dim]

            # Sample latents: for simplicity, one sample; in practice, multiple MC samples
            # Draw small epsilon for h_w
            epsilon_hw = torch.randn(total_params, device=device)
            sigma_hw = torch.exp(0.5 * variational_qw.rho)
            h_w_sample = variational_qw.mu + epsilon_hw * sigma_hw

            # Compute weights via linear reparameterization
            w_sample = torch.matmul(A_matrix, h_w_sample.unsqueeze(-1)).squeeze(-1)  # shape: [total_params]

            # Set model parameters accordingly
            # Map w_sample to model parameters
            def set_model_weights(model, weights_vector):
                offset = 0
                for param in model.parameters():
                    param_numel = param.numel()
                    param.data.copy_(weights_vector[offset:offset+param_numel].view_as(param))
                    offset += param_numel

            set_model_weights(inr_model, w_sample)

            # Forward pass
            preds = inr_model.forward(coords, h_z_current, data=None, params=None)
            # Compute distortion
            dist = torch.nn.functional.mse_loss(preds, values, reduction='mean')
            # Compute KL divergence for q(w) vs p(w)
            kl_qp = variational_qw.kl_divergence({'mu': prior_mu, 'rho': prior_rho})
            # Loss (negative ELBO scaled by beta)
            loss = beta * kl_qp + dist
            total_loss += loss

        # Backpropagate and update all parameters
        total_loss.backward()
        optimizer.step()

        # --- Step 2: Update prior parameters (Equation 7) ---
        # Calculate empirical mean and variance over the variational posteriors (here, approximate)
        with torch.no_grad():
            mu_post = variational_qw.mu.data
            rho_post = variational_qw.rho.data
            prior_mu_new = torch.mean(mu_post)
            prior_sigma_new = torch.mean((mu_post - prior_mu_new) ** 2 + torch.exp(rho_post))
            # Update prior mu and rho
            prior_mu.copy_(prior_mu_new)
            prior_rho.copy_(torch.log(prior_sigma_new.clamp(min=1e-8)))

        # --- Step 3: Calculate estimated bits and adjust beta ---
        with torch.no_grad():
            kl_estimate = variational_qw.kl_divergence({'mu': prior_mu, 'rho': prior_rho}).item()
        if kl_estimate > target_bpp + epsilon:
            beta = min(beta * (1 + tau_C), beta_max)
        elif kl_estimate < target_bpp - epsilon:
            beta = max(beta / (1 + tau_C), beta_min)

        # Optional: print training info
        print(f"Epoch {epoch+1}/{total_epochs} | Loss: {total_loss.item():.4f} | KL_est: {kl_estimate:.4f} | beta: {beta:.4f}")

    # After training:
    # Save final model, A, h_z, and variational params if needed
    torch.save({
        'inr_state_dict': inr_model.state_dict(),
        'A': A_matrix.detach().cpu(),
        'h_z': h_z.detach().cpu(),
        'mu': variational_qw.mu.detach().cpu(),
        'rho': variational_qw.rho.detach().cpu(),
        'prior_mu': prior_mu.detach().cpu(),
        'prior_rho': prior_rho.detach().cpu()
    }, 'trained_model.pth')

    # --- Inference and encoding on test data ---
    # For each test sample:
    posterior_samples = []
    reconstructed_data = []

    for sample in test_dataset:
        coords = sample['coordinates'].to(device)
        values = sample['values'].to(device)

        # Infer q(w) by same process as above but with multiple MC samples if desired
        # or with fixed point estimate for simplicity
        with torch.no_grad():
            # For consistency, do multiple MC samples (e.g., 5), but here just 1 for simplicity
            epsilon_hw = torch.randn(total_params, device=device)
            sigma_hw = torch.exp(0.5 * variational_qw.rho)
            h_w_sample = variational_qw.mu + epsilon_hw * sigma_hw

            # Compute weights
            weights = torch.matmul(A_matrix, h_w_sample.unsqueeze(-1)).squeeze(-1)

            # Store posterior parameters
            posterior_samples.append({
                'mu': variational_qw.mu.cpu(),
                'rho': variational_qw.rho.cpu()
            })

            # --- Bayesian coding: encode q(w) sample ---
            # Apply permutation strategy (random permutation)
            permutation = torch.randperm(total_params)
            # Use BayesianCoder to encode weights
            coder = BayesianCoder()
            # Encode weights
            bits_used = coder.encode_weights(weights, 
                                             {'mu': variational_qw.mu.cpu(), 'rho': variational_qw.rho.cpu()}, 
                                             {'mu': prior_mu.cpu(), 'rho': prior_rho.cpu()}, 
                                             permutation=permutation)
            # Save bits (or store in a list)
            # For this example, just store bits used
            # Save the permutation for decoding
            # Save the sample weight (or its bits) for later decoding
            # Store in a list for later use
            # For brevity, omit bits storage, assume access later

            # --- Decoding: retrieve weights from bits (simulate) ---
            decoded_weights = coder.decode_weights(None, 
                                                 {'mu': variational_qw.mu.cpu(), 'rho': variational_qw.rho.cpu()}, 
                                                 {'mu': prior_mu.cpu(), 'rho': prior_rho.cpu()}, 
                                                 permutation=permutation, shape=(total_params,))
            # Set model weights
            def set_model_weights(model, weights_vector):
                offset = 0
                for param in model.parameters():
                    param_numel = param.numel()
                    param.data.copy_(weights_vector[offset:offset+param_numel].view_as(param))
                    offset += param_numel
            set_model_weights(inr_model, decoded_weights)

        # Reconstruct data
        with torch.no_grad():
            preds = inr_model.forward(coords, h_z)
            reconstructed_data.append(preds.cpu())

    # --- Evaluation ---
    evaluator = Evaluation(test_dataset, reconstructed_data, posterior_samples,
                           prior_mu, prior_rho, modality='image', output_path='results')
    eval_metrics = evaluator.evaluate()
    print('RD Metrics:', eval_metrics)

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SineActivation(nn.Module):
    """
    Sine activation function for SIREN networks.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class FourierFeaturesEncoder:
    """
    Encodes input coordinates into Fourier features.
    """
    def __init__(self, max_freq, num_freqs, include_input=True):
        """
        Args:
            max_freq (float): maximum frequency (ω_max)
            num_freqs (int): number of frequency bands
            include_input (bool): whether to include raw input
        """
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self.include_input = include_input
        # Log-spaced frequencies
        self.freq_bands = torch.logspace(0, math.log2(self.max_freq), steps=self.num_freqs, base=2)

    def encode(self, x):
        """
        Generate Fourier features for input tensor x.
        Args:
            x: Tensor of shape [batch_size, coord_dim]

        Returns:
            features: Tensor of shape [batch_size, coord_dim * (include_input + 2 * num_freqs)]
        """
        features = []
        if self.include_input:
            features.append(x)
        # Expand x to [batch_size, coord_dim, 1]
        x_expand = x.unsqueeze(-1)  # [batch_size, coord_dim, 1]
        # Compute sine and cosine features for each frequency
        for freq in self.freq_bands:
            freq = freq.to(x.device)
            scaled = x_expand * freq * math.pi  # scale input
            features.append(torch.sin(scaled))
            features.append(torch.cos(scaled))
        return torch.cat(features, dim=-1)

class PositionalEncodingCNN(nn.Module):
    """
    Small CNN to produce positional encodings from learned h_z vector.
    """
    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): size of h_z vector
            output_dim (int): desired positional encoding per coordinate
        """
        super().__init__()
        # Example architecture: 3 conv layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=output_dim, kernel_size=3, padding=1)
        # Initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

    def forward(self, h_z, coords):
        """
        Generate positional encodings for each coordinate.
        Args:
            h_z: tensor of shape [batch_size, input_dim]
            coords: tensor of shape [batch_size, coord_dim], e.g., 2D coords

        Returns:
            pos_enc: tensor of shape [batch_size, coord_dim, output_dim]
        """
        batch_size, coord_dim = coords.shape
        # Expand h_z to [batch_size, 1, input_dim] for conv1d
        h_z_exp = h_z.unsqueeze(1)  # [batch_size,1,input_dim]
        # Pass through conv layers
        x = F.relu(self.conv1(h_z_exp))  # [batch_size,16,input_dim]
        x = F.relu(self.conv2(x))        # [batch_size,32,input_dim]
        x = self.conv3(x)                # [batch_size,output_dim,input_dim]
        # Pool (average across the input_dim dimension)
        x = torch.mean(x, dim=-1)        # [batch_size, output_dim]
        # Expand to per coordinate position: repeat for each coordinate
        pos_enc = x.unsqueeze(1).repeat(1, coord_dim, 1)  # [batch_size, coord_dim, output_dim]
        return pos_enc

class INRLayer(nn.Module):
    """
    Single sine-activated layer for SIREN.
    """
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        # Initialization for SIREN
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/ in_features, 1/ in_features]
                bound = 1 / self.in_features
            else:
                # Following layers: scaled Xavier
                bound = math.sqrt(6 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class INRModel(nn.Module):
    """
    INR with Fourier features, optional positional encodings, and sine activations (SIREN).
    """
    def __init__(self, config):
        """
        Args:
            config (dict): configuration parameters, see details below.
        """
        super().__init__()
        # Extract configuration with defaults
        self.num_layers = config.get('layers', 4)
        self.hidden_units = config.get('hidden_units', 32)
        self.activation = config.get('activation', 'sine')
        self.fourier_dim = config.get('fourier_features_dim', 16)
        self.positional_encoding_dim = config.get('positional_encoding_dim', 128)
        self.coord_dim = config.get('input_coord_dim', 2)  # e.g., 2 for images
        self.output_dim = config.get('output_dim', 3)      # e.g., RGB
        self.use_positional_encoding = config.get('use_pos_encoding', True)
        self.use_fourier = config.get('use_fourier', True)
        # Initialize Fourier feature encoder
        max_freq = config.get('fourier_max_freq', 30.0)
        self.fourier_encoder = FourierFeaturesEncoder(max_freq, self.fourier_dim)
        # Initialize positional encoding CNN
        self.pos_enc_cnn = PositionalEncodingCNN(self.positional_encoding_dim, self.positional_encoding_dim)
        # Build SIREN network layers
        layers = []
        input_size = None
        if self.use_fourier:
            input_size = self.fourier_dim * self.coord_dim
        else:
            input_size = self.coord_dim
        if self.use_positional_encoding:
            input_size += self.positional_encoding_dim * self.coord_dim
        # First layer
        layers.append(INRLayer(input_size, self.hidden_units, is_first=True))
        # Hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(INRLayer(self.hidden_units, self.hidden_units))
        # Output layer
        self.final_layer = nn.Linear(self.hidden_units, self.output_dim)
        # Register layers
        self.layers = nn.ModuleList(layers)
        # Activation function
        # Sine activation is used throughout,Initializer handled in INRLayer

    def forward(self, coords, h_z, data=None, params=None):
        """
        Compute forward pass.
        Args:
            coords: [batch_size, coord_dim] tensor with coordinate locations
            h_z: [batch_size, positional_encoding_dim] learnable positional encoding
            data: Optional additional data, unused here
            params: Optional dict of parameters for functional API
        Returns:
            output: [batch_size, output_dim]
        """
        device = coords.device
        # Generate Fourier features
        feat_list = []
        if self.use_fourier:
            gamma_x = self.fourier_encoder.encode(coords)  # [batch_size, feat_size]
            feat_list.append(gamma_x)
        # Generate positional encoding for each coordinate
        if self.use_positional_encoding:
            # h_z: [batch_size, positional_encoding_dim]
            z_enc = self.pos_enc_cnn(h_z, coords)  # [batch_size, coord_dim, pos_enc_dim]
            # Reshape to [batch_size, coord_dim * pos_enc_dim]
            # For simplicity, gather each coordinate's encoding
            z_enc_flat = z_enc.view(z_enc.shape[0], -1)
            # Repeat for each coordinate
            # But z_enc is per batch, we need coordinate-wise encoding
            # Since z_enc is [batch_size, coord_dim, pos_enc_dim], flatten per coordinate
            # So, per coordinate in batch
            for i in range(self.coord_dim):
                feat_list.append(z_enc[:, i, :])  # list of [batch_size, pos_enc_dim]
        # Concatenate all features
        feat = torch.cat(feat_list, dim=-1)  # [batch_size, input_size]
        x = feat
        # Pass through layers
        for layer in self.layers:
            x = layer(x)  # sine activation after each layer
        # Final layer (linear)
        out = self.final_layer(x)
        return out

```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional
from dataset_loader import Dataset, DatasetLoader
from model import INRModel
from variational import VariationalDistribution
from hierarchical_patch import HierarchicalPatchModel

class Trainer:
    """
    Implements the training routine for RECOMBINER as described:
    - Optimizes variational parameters, A matrices, and upsampling network φ
    - Updates the prior parameters
    - Adjusts β adaptively to meet target rate C
    - Supports hierarchical patch modeling
    """
    def __init__(self, 
                 model: INRModel,
                 variational: VariationalDistribution,
                 prior_mu: torch.Tensor,
                 prior_rho: torch.Tensor,
                 dataset: Dataset,
                 config: dict,
                 hierarchical_model: Optional[HierarchicalPatchModel] = None):
        """
        Initializes the Trainer.
        Args:
            model (INRModel): the INR network model.
            variational (VariationalDistribution): variational posterior q(w).
            prior_mu (torch.Tensor): prior mean parameters.
            prior_rho (torch.Tensor): prior log-variance parameters.
            dataset (Dataset): dataset object.
            config (dict): configuration with hyperparameters.
            hierarchical_model (Optional[HierarchicalPatchModel]): for patching.
        """
        self.model = model
        self.variational = variational
        self.prior_mu = prior_mu
        self.prior_rho = prior_rho
        self.dataset = dataset
        self.config = config
        self.hierarchical_model = hierarchical_model

        # Extract training hyperparameters
        self.learning_rate = self.config.get('training', {}).get('learning_rate', 1e-4)
        self.batch_size = self.config.get('training', {}).get('batch_size', 50)
        self.total_epochs = self.config.get('training', {}).get('epochs', 550)
        self.beta = self.config.get('training', {}).get('beta_initial', 0.3)
        self.beta_adjust_step = self.config.get('training', {}).get('beta_adjust_step', 0.5)
        self.beta_min = self.config.get('training', {}).get('beta_min', 0.0)
        self.beta_max = self.config.get('training', {}).get('beta_max', 1.0)
        self.C = self.config.get('training', {}).get('beta_target_bpp', 0.3)
        self.epsilon = 1e-4  # tolerance for rate control
        self.tau_C = self.beta_adjust_step
        self.seed = self.config.get('experiment', {}).get('seed', 42)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Optimizer over all parameters: model, variational params, A, phi (upsampling)
        params = list(self.model.parameters()) + \
                 list(self.variational.mu.parameters()) + \
                 list(self.variational.rho.parameters()) + \
                 [self.variational.A]
        if hasattr(self.model, 'pos_enc_cnn'):
            params += list(self.model.pos_enc_cnn.parameters())
        # Add A matrices if learnable
        self.optimizer = optim.Adam(params, lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)

        # Initialize prior parameters as tensors
        # These should be set appropriately from data statistics after initial training
        self.prior_mu = prior_mu
        self.prior_rho = prior_rho

    def train(self):
        """
        Run training over specified epochs with data loader.
        """
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.total_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                self.optimizer.zero_grad()
                # Compute β-ELBO loss: Equation (6)
                loss = self._compute_beta_elbo(batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

                # After each batch, update prior parameters
                self._update_prior()

                # Optionally, enable early stopping or logging here
            # Adjust beta at epoch level (or per batch if desired)
            self._adjust_beta()
            # Log epoch statistics
            print(f"Epoch {epoch+1}/{self.total_epochs}, Loss: {epoch_loss/len(dataloader):.4f}, "
                  f"Beta: {self.beta:.4f}")

    def _compute_beta_elbo(self, batch):
        """
        Compute the β-ELBO for a given batch, using Monte Carlo sampling.
        """
        # Sample w from q(w): shape [samples, *shape]
        num_samples = 1  # or 5 during inference for expectation estimation
        w_samples = self.variational.sample(num_samples)
        if torch.cuda.is_available():
            w_samples = w_samples.cuda()

        # For hierarchical models, generate patch weights if applicable
        if self.hierarchical_model:
            self.hierarchical_model.infer_patch_weights()  # update patch weights via variational approx

        # Compose w: w = h_w * A
        A = self.variational.get_A()
        # shape: [prod(shape)] for flattening
        w_flat_samples = []
        for i in range(num_samples):
            h_w = w_samples[i]  # shape: [*, ...]
            h_w_flat = h_w.view(-1)
            w_sample = torch.matmul(A, h_w_flat.unsqueeze(-1)).view(-1)
            w_flat_samples.append(w_sample)
        w_samples = torch.stack(w_flat_samples, dim=0)  # [samples, total_params]

        # Generate input coordinates for data points
        coords, pos_encs = self._prepare_input_for_batch(batch)

        # Forward pass through model with sampled weights
        recon_batch = []
        for i in range(num_samples):
            # Set model weights
            self._set_model_weights(w_samples[i])
            # Run model
            out = self.model.forward(coords, pos_encs)
            recon_batch.append(out)
        recon_mean = torch.mean(torch.stack(recon_batch), dim=0)

        # Calculate distortion (e.g., MSE or modality-specific)
        distortion_loss = self._compute_distortion(batch, recon_mean)

        # Compute KL divergence
        kl_div = self.variational.kl_divergence({'mu': self.prior_mu, 'rho': self.prior_rho})

        # Return scaled loss (Equation 6)
        loss = self.beta * kl_div + distortion_loss
        return loss

    def _prepare_input_for_batch(self, batch):
        """
        For each data point, generate coordinates and positional encodings.
        """
        # Assumes batch is a dict with 'coordinates' and 'values' tensors
        coords = batch['coordinates']  # shape: [N_points, coord_dim]
        # Generate positional encodings h_z for each coordinate
        if hasattr(self.model, 'pos_enc_cnn') and hasattr(self.model, 'h_z'):
            h_z = self.model.h_z  # shape: [batch_size, positional_dim]
            pos_encs = self.model.pos_enc_cnn(h_z, coords)
        else:
            pos_encs = None
        return coords, pos_encs

    def _set_model_weights(self, w_vector):
        """
        Map the sampled weight vector to the model parameters.
        """
        # This function sets model weights from the vector w
        # Assuming parameter ordering matches model architecture
        offset = 0
        for name, param in self.model.named_parameters():
            param_numel = param.numel()
            param_data = w_vector[offset:offset + param_numel].view_as(param)
            param.data.copy_(param_data)
            offset += param_numel

    def _compute_distortion(self, batch, recon_preds):
        """
        Compute the distortion metric; 
        e.g., MSE for pixels or other modality-specific metric.
        """
        # assuming batch['values'] is ground truth
        gt = batch['values']
        dist = torch.nn.functional.mse_loss(recon_preds, gt, reduction='mean')
        return dist

    def _update_prior(self):
        """
        Update the prior parameters based on the current q(w).
        """
        mu_post = self.variational.mu.data
        rho_post = self.variational.rho.data
        # Compute mean and variance
        prior_mu_new = torch.mean(mu_post, dim=0)
        prior_sigma_new = torch.mean((mu_post - prior_mu_new) ** 2 + torch.exp(rho_post), dim=0)
        # Update prior parameters (assuming Gaussian prior)
        self.prior_mu = prior_mu_new
        self.prior_rho = torch.log(prior_sigma_new.clip(min=1e-8))
        # Note: in practice, you might need to update these parameters stored elsewhere

    def _adjust_beta(self):
        """
        Adjust β dynamically based on estimated KL divergence (rate),
        aiming to match target bits per pixel/atom C.
        """
        # Estimate current KL divergence (rate estimate)
        with torch.no_grad():
            kl_estimate = self.variational.kl_divergence({'mu': self.prior_mu, 'rho': self.prior_rho}).item()

        if kl_estimate > self.C + self.epsilon:
            # Too high rate, increase β
            self.beta = min(self.beta * (1 + self.tau_C), self.beta_max)
        elif kl_estimate < self.C - self.epsilon:
            # Too low rate, decrease β
            self.beta = max(self.beta / (1 + self.tau_C), self.beta_min)

    # Optionally, implement save/load checkpoints, logging, and other utilities.

```

## variational.py

```python
## variational.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VariationalDistribution:
    """
    Variational Gaussian posterior q(w) over INR weights, supporting reparameterization
    via the learned linear transformation A and optional hierarchical Bayesian variables.
    """
    def __init__(self, shape, init_params=None):
        """
        Initialize variational parameters μ, ρ, and linear reparameterization matrix A.
        Optional hierarchical variables can be added if needed.
        
        Args:
            shape (list or tuple): shape of the weights tensor w (e.g., total number of parameters).
            init_params (dict): dictionary with optional initial 'mu' and 'rho' tensors.
        """
        # Set shape
        self.shape = shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize mu and rho (log-variance) for the mean-field Gaussian q(h_w)
        if init_params is not None:
            self.mu = nn.Parameter(init_params.get('mu', torch.zeros(shape, device=device)))
            self.rho = nn.Parameter(init_params.get('rho', torch.full(shape, -12.0, device=device)))  # variance ~ 10^{-6}
        else:
            self.mu = nn.Parameter(torch.zeros(shape, device=device))
            self.rho = nn.Parameter(torch.full(shape, -12.0, device=device))
        
        # Initialize A matrix for linear reparameterization (shape: [shape, shape])
        # For computational efficiency, store as a parameter with shape similar to the weight vector
        # and initialize as identity for simplicity
        self.A = nn.Parameter(torch.eye(shape[0], device=device))  # shape: [shape[0], shape[0]]
        
        # Variational parameters for hierarchical variables can be added here if needed
        # For simplicity, we consider only the local layer-wise q(w)

    def sample(self, num_samples=1):
        """
        Sample weights w from q(w) = N(mu, diag(sigma^2)), with w = h_w A.
        Samples h_w are drawn, then transformed via A.
        
        Args:
            num_samples (int): number of Monte Carlo samples.
        
        Returns:
            Tensor: sampled weights, shape [num_samples, *shape]
        """
        epsilon = torch.randn((num_samples,) + self.shape, device=self.mu.device)
        sigma = torch.exp(0.5 * self.rho)  # std: exp(0.5 * rho)
        h_w_samples = self.mu.unsqueeze(0) + epsilon * sigma.unsqueeze(0)
        # Apply linear reparameterization: w = h_w * A
        # Since shape of h_w: [num_samples, *shape], shape of A: [total_params, total_params]
        # Reshape h_w to 2D for matrix multiplication if necessary
        # Here, treat the last dimension as vector w
        # For each sample, perform: w = h_w * A
        # flatten last dims temporarily
        samples_list = []
        for i in range(num_samples):
            h_w_flat = h_w_samples[i].view(-1, 1)  # shape: [prod(shape), 1]
            w_sample = torch.matmul(self.A, h_w_flat).view(self.shape)
            samples_list.append(w_sample)
        return torch.stack(samples_list, dim=0)  # shape: [num_samples] + shape

    def kl_divergence(self, prior):
        """
        Compute KL divergence D_KL(q(w) || p(w)) between two Gaussians.
        Both are assumed diagonal covariances; the prior can be specified.
        For hierarchical, this can be extended.

        Args:
            prior (dict): a dict with keys 'mu' and 'rho' for the prior distribution.
                          they are tensors of shape matching self.mu and self.rho.

        Returns:
            float: KL divergence (scalar tensor)
        """
        # Variational q: mu_q, sigma_q
        mu_q = self.mu
        sigma_q = torch.exp(0.5 * self.rho)
        # Prior p: mu_p, sigma_p
        mu_p = prior.get('mu', torch.zeros_like(mu_q))
        sigma_p = torch.exp(0.5 * prior.get('rho', torch.zeros_like(mu_q)))

        # Compute KL element-wise for diagonal Gaussians
        # D_KL = 0.5 * [ (sigma_q^2 / sigma_p^2) + ((mu_p - mu_q)^2 / sigma_p^2) - 1 + log(sigma_p^2 / sigma_q^2) ]
        term1 = (sigma_q ** 2) / (sigma_p ** 2)
        term2 = ((mu_p - mu_q) ** 2) / (sigma_p ** 2)
        kl = 0.5 * torch.sum(term1 + term2 - 1 + torch.log((sigma_p ** 2).clamp_min(1e-8)/(sigma_q ** 2).clamp_min(1e-8)))
        return kl

    def update_params(self, new_params):
        """
        Update variational parameters μ and ρ.
        Args:
            new_params (dict): dictionary with 'mu' and 'rho' tensors
        """
        if 'mu' in new_params:
            self.mu.data = new_params['mu']
        if 'rho' in new_params:
            self.rho.data = new_params['rho']

    def get_weight(self):
        """
        Get the mean weight vector w = μ + noise via reparameterization.
        For usage in the functional API or explicit inference.
        """
        sigma = torch.exp(0.5 * self.rho)
        epsilon = torch.randn(self.shape, device=self.mu.device)
        h_w = self.mu + epsilon * sigma
        # Apply linear reparameterization
        w = torch.matmul(self.A, h_w.view(-1, 1)).view(self.shape)
        return w

    def set_A(self, A_new):
        """
        Set the learned linear transform A explicitly (fixed during inference).
        Args:
            A_new (Tensor): new A matrix.
        """
        self.A.data = A_new

    def get_A(self):
        """
        Get the current A matrix.
        """
        return self.A
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\RECOMBINER\RECOMBINER_repo`
