# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import glob
import random
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Define a class for the replay buffer to store past slot states
class SlotStateBuffer:
    def __init__(self, max_size: int):
        """
        Initialize the replay buffer.

        Args:
            max_size (int): Maximum number of states to store in the buffer (frames).
        """
        self.max_size = max_size
        self.buffer = []  # List to hold stored states
        self.pointer = 0

    def add(self, states: List[Dict[str, torch.Tensor]]):
        """
        Add new states to the buffer. Overwrites oldest if exceeding max size.

        Args:
            states (List[Dict]): List of state dicts, each containing slot features (r_k, etc.)
        """
        for state in states:
            if len(self.buffer) < self.max_size:
                self.buffer.append(state)
            else:
                self.buffer[self.pointer] = state
                self.pointer = (self.pointer + 1) % self.max_size

    def sample(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Randomly sample states from the buffer.

        Args:
            batch_size (int): Number of states to sample.

        Returns:
            List[Dict]: List of sampled states.
        """
        if len(self.buffer) == 0:
            return []

        sampled = random.choices(self.buffer, k=batch_size)
        return sampled

# Define the dataset class for MOVI sequences
class MoviSequenceDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 split_files: List[str],
                 sequence_length: int = 3,
                 training: bool = True,
                 transform: Optional[transforms.Compose] = None,
                 dataset_split: str = "official_split",
                 object_max_count: int = 10,
                 all_groundtruth_masks_dir: Optional[str] = None
                ):
        """
        Args:
            dataset_dir (str): Root directory containing the MOVI dataset videos.
            split_files (List[str]): List of file paths for videos in the selected split.
            sequence_length (int): Number of frames per sample sequence.
            training (bool): Whether dataset is used for training or evaluation.
            transform (callable, optional): Image transformations.
            dataset_split (str): Split type; uses for optional split logic.
            object_max_count (int): Max objects per video, used for filtering.
            all_groundtruth_masks_dir (str, optional): Directory with groundtruth masks for evaluation.
        """
        self.dataset_dir = dataset_dir
        self.sequence_length = sequence_length
        self.training = training
        self.transform = transform
        self.object_max_count = object_max_count
        self.all_groundtruth_masks_dir = all_groundtruth_masks_dir

        # Load list of video file paths
        self.video_paths = self._load_video_list(split_files)

        # Precompute indices for sampling
        self.video_indices = self._compute_video_indices()

    def _load_video_list(self, split_files: List[str]) -> List[str]:
        video_paths = []
        for split_file in split_files:
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Each line is a relative path to a video folder or file
                    full_path = os.path.join(self.dataset_dir, line)
                    if os.path.isdir(full_path):
                        video_paths.append(full_path)
                    elif os.path.isfile(full_path):
                        video_paths.append(full_path)
        # Optionally filter videos based on object count or other criteria if labels are available
        return sorted(video_paths)

    def _compute_video_indices(self):
        # For each video, store (video_path, total_frames)
        indices = []
        for vpath in self.video_paths:
            frame_files = sorted(glob.glob(os.path.join(vpath, 'frames', '*.png')))
            total_frames = len(frame_files)
            if total_frames >= self.sequence_length:
                indices.append({'video_path': vpath, 'total_frames': total_frames, 'frame_files': frame_files})
        return indices

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Returns a sequence of frames (and masks if available) sampled from a video.

        Output:
            images: Tensor of shape (sequence_length, 3, 128, 128)
            masks: Optional tensor of shape (sequence_length, H, W), if groundtruth masks are provided
            meta: dict with info, e.g., video path, index
        """
        vinfo = self.video_indices[index]
        frame_files = vinfo['frame_files']
        total_frames = vinfo['total_frames']

        # Sample a start index
        if self.training:
            start_idx = random.randint(0, total_frames - self.sequence_length)
        else:
            start_idx = 0  # For validation/test, can fix or use entire video

        # Sample sequence of frame indices
        seq_indices = list(range(start_idx, start_idx + self.sequence_length))
        # Load frames
        frames = []
        for fi in seq_indices:
            img_path = frame_files[fi]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                # Default transform: resize and toTensor
                img = transforms.ToTensor()(img)
            frames.append(img)

        images = torch.stack(frames, dim=0)  # shape: (sequence_length, 3, 128, 128)

        # Optional: load groundtruth masks if available
        masks = None
        if self.all_groundtruth_masks_dir is not None:
            masks_list = []
            for fi in seq_indices:
                # Assume masks are stored as images in a directory parallel to frames
                mask_path = os.path.join(self.all_groundtruth_masks_dir, os.path.basename(vinfo['video_path']), 'masks', f'frame_{fi:04d}.png')
                if os.path.exists(mask_path):
                    mask_img = Image.open(mask_path)
                    mask_tensor = transforms.ToTensor()(mask_img)  # shape: (1, H, W)
                    masks_list.append(mask_tensor.squeeze(0))
                else:
                    # If mask not available, fill with zeros
                    masks_list.append(torch.zeros((128, 128)))
            masks = torch.stack(masks_list, dim=0)  # shape: (sequence_length, 128, 128)

        meta = {
            'video_path': vinfo['video_path'],
            'start_frame': start_idx,
            'indices': seq_indices,
        }

        return images, masks, meta

# Optional: collate_fn for DataLoader to handle variable batch sizes, if needed
def collate_fn(batch: List[Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]]):
    """
    Collates a batch of samples.
    """
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    metas = [item[2] for item in batch]
    batch_images = torch.stack(images, dim=0)  # (batch, seq_len, 3, 128, 128)
    if masks[0] is not None:
        batch_masks = torch.stack(masks, dim=0)  # (batch, seq_len, 128, 128)
    else:
        batch_masks = None
    return batch_images, batch_masks, metas
```

## evaluation.py

```python
## evaluation.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from utils import mask_logits_to_mask, normalize_image, denormalize_image

# Assuming the model, dataset, and data loader are properly initialized and passed
class Evaluator:
    def __init__(self, model, dataset, device=None, fg_ari=True, mip_iu=True,
                 mask_threshold=0.3, visualization=True, save_dir='./results'):
        """
        Initializes the evaluator.

        Args:
            model (nn.Module): Trained VONet model set in eval mode.
            dataset (Dataset): Dataset object for inference (validation/test set).
            device (torch.device): Device for computation.
            fg_ari (bool): Whether to compute FG-ARI.
            mip_iu (bool): Whether to compute mIoU.
            mask_threshold (float): Threshold for masks.
            visualization (bool): Whether to generate visualization images.
            save_dir (str): Path to save results.
        """
        self.model = model
        self.dataset = dataset
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.fg_ari = fg_ari
        self.mIoU = mip_iu
        self.mask_threshold = mask_threshold
        self.visualization = visualization
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def generate_masks(self, x):
        """
        Generate segmentation masks for input batch of frames.
        Args:
            x (Tensor): [B, 3, H, W]
        Returns:
            masks: [B, H, W] integer tensor with mask labels (0: null/background)
            mask_probs: [B, K, H, W] probabilities for each slot (before argmax)
        """
        with torch.no_grad():
            features = self.model.extract_features(x)
            # Generate masks with model's attention U-Net (assuming c_prev is zeros here)
            batch_size = x.shape[0]
            K = self.model.num_slots
            # Initialize context vectors as zeros for inference
            c_prev = torch.zeros(batch_size, K, self.model.slot_dim, device=self.device)
            masks_prob = self.model.generate_attention(features, c_prev)
            # masks_prob: [batch, K+1, H, W]
            # Exclude background for masks
            fg_masks_prob = masks_prob[:,1:,:,:]  # [B, K, H, W]
            # Compute maximum mask probability per pixel for thresholding
            max_prob, _ = torch.max(fg_masks_prob, dim=1, keepdim=True)  # [B,1,H,W]
            # Assign null/background label where max prob < threshold
            pred_mask = torch.argmax(fg_masks_prob, dim=1)  # [B, H, W]
            pred_mask[max_prob.squeeze(1) < self.mask_threshold] = 0  # null label = 0
            return pred_mask, fg_masks_prob

    def reconstruct_scene(self, z_slots):
        """
        Reconstruct scene given slot latent vectors.
        Args:
            z_slots (Tensor): [B, K, D]
        Returns:
            recon (Tensor): [B, 3, H, W]
        """
        with torch.no_grad():
            recon = self.model.decode_scene(z_slots)
        return recon

    def compute_metrics_seq(self, pred_masks_seq, gt_masks_seq):
        """
        Compute FG-ARI and mIoU metrics over entire sequence.
        Args:
            pred_masks_seq (list of [H,W]): Predicted masks per frame.
            gt_masks_seq (list of [H,W]): Ground truth masks per frame.
        Returns:
            dict: {'FG-ARI': value, 'mIoU': value}
        """
        T = len(gt_masks_seq)
        fg_ari_scores = []
        iou_scores = []

        for t in range(T):
            pred_mask = pred_masks_seq[t]
            gt_mask = gt_masks_seq[t]
            # Convert to numpy
            pred_mask_np = pred_mask.cpu().numpy()
            gt_mask_np = gt_mask.cpu().numpy()

            # ====== Compute FG-ARI ======
            if self.fg_ari:
                # FG masks: only foreground pixels in gt
                fg_idx = gt_mask_np > 0
                if np.sum(fg_idx) == 0:
                    continue
                ari = adjusted_rand_score(
                    gt_mask_np[fg_idx].flatten(),
                    pred_mask_np[fg_idx].flatten()
                )
                fg_ari_scores.append(ari)

            # ====== Compute mIoU ======
            if self.mIoU:
                # Hungarian matching between predicted masks and ground truth masks
                pred_labels = np.unique(pred_mask_np)
                gt_labels = np.unique(gt_mask_np)
                pred_labels = pred_labels[pred_labels != 0]
                gt_labels = gt_labels[gt_labels != 0]
                if len(pred_labels)==0 or len(gt_labels)==0:
                    continue
                cost_mat = np.zeros((len(pred_labels), len(gt_labels)))
                for i, pl in enumerate(pred_labels):
                    pred_bin = pred_mask_np == pl
                    for j, gl in enumerate(gt_labels):
                        gt_bin = gt_mask_np == gl
                        intersection = np.logical_and(pred_bin, gt_bin).sum()
                        union = pred_bin.sum() + gt_bin.sum() - intersection
                        iou = intersection / union if union > 0 else 0
                        cost_mat[i,j] = 1 - iou  # Hungarian minimizes total cost
                row_ind, col_ind = linear_sum_assignment(cost_mat)
                match_ious = []
                for r_idx, c_idx in zip(row_ind, col_ind):
                    pl = pred_labels[r_idx]
                    gl = gt_labels[c_idx]
                    pred_bin = pred_mask_np == pl
                    gt_bin = gt_mask_np == gl
                    intersection = np.logical_and(pred_bin, gt_bin).sum()
                    union = pred_bin.sum() + gt_bin.sum() - intersection
                    iou = intersection / union if union >0 else 0
                    match_ious.append(iou)
                if len(match_ious) > 0:
                    iou_scores.append(np.mean(match_ious))

        metrics = {}
        if self.fg_ari:
            metrics['FG-ARI'] = np.mean(fg_ari_scores) if len(fg_ari_scores) >0 else 0.0
        if self.mIoU:
            metrics['mIoU'] = np.mean(iou_scores) if len(iou_scores) >0 else 0.0
        return metrics

    def overlay_masks_on_frame(self, frame, masks, save_path, title=None):
        """
        Overlay colored masks with the frame and save.
        Args:
            frame: [H,W,3], uint8 or float
            masks: [K, H, W], probabilities
            save_path: str
            title: str optional
        """
        import matplotlib.pyplot as plt
        H,W,_=frame.shape
        if frame.dtype != np.uint8:
            frame = denormalize_image(frame).astype(np.uint8)
        overlay = frame.copy()
        K = masks.shape[0]
        # Generate random colors per slot
        np.random.seed(0)
        colors = np.random.rand(K,3)
        for k in range(K):
            mask_prob = masks[k].cpu().numpy()
            color_mask = (mask_prob[..., None] * colors[k] * 255).astype(np.uint8)
            mask_thresholded = mask_prob >= 0.5
            overlay = np.where(mask_thresholded[..., None], color_mask, overlay)
        plt.figure(figsize=(W/100, H/100), dpi=100)
        plt.imshow(overlay)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.savefig(save_path)
        plt.close()

    def evaluate_video(self, video_frames, gt_masks=None, max_frames=24, save_visualization=True):
        """
        Evaluate a single video sequence with optional ground truth masks.
        Args:
            video_frames (list of Tensor): List of length T, each [3,H,W]
            gt_masks (list of Tensor): List of length T, [H,W], optional
            max_frames (int): Limit number of frames for eval
            save_visualization (bool): Save masks overlays.
        Returns:
            dict: metrics across sequence
        """
        pred_masks_seq = []
        gt_masks_seq = [] if gt_masks is not None else None
        scene_recons = []

        # Initialize context/c_prev as zeros
        batch_size = 1
        K = self.model.num_slots

        r_prev = torch.randn(batch_size, K, self.model.slot_dim, device=self.device)
        c_prev = torch.zeros(batch_size, K, self.model.slot_dim, device=self.device)

        for t_idx, frame in enumerate(video_frames):
            if t_idx >= max_frames:
                break
            x = frame.unsqueeze(0).to(self.device)  # [1,3,H,W]
            # Generate masks
            pred_mask, mask_probs = self.generate_masks(x)
            pred_masks_seq.append(pred_mask.squeeze(0))
            # Save overlay if visualization
            if self.visualization:
                overlay_path = os.path.join(self.save_dir, f'video_frame_{t_idx}.png')
                self.overlay_masks_on_frame(
                    normalize_image(frame.permute(1,2,0)).cpu().numpy(),
                    mask_probs.squeeze(0),
                    overlay_path,
                    title=f'Frame {t_idx}'
                )
            # For metrics: if groundtruth exists, store groundtruth mask
            if gt_masks is not None:
                gt_mask = gt_masks[t_idx]
                gt_masks_seq.append(gt_mask)

            # Encode slot features
            features = self.model.extract_features(x)
            masks_exp = F.softmax(mask_probs, dim=1)
            fg_masks = masks_exp[:,1:,:,:]
            slot_feats = self.model.encode_slots(features, fg_masks)
            # Update slot states
            r_t = self.model.update_slot_states(slot_feats, r_prev)
            r_prev = r_t.detach()
            c_prev = r_t.detach()  # for simplicity, in actual code consider differently

            # Variational z
            mu_z, logvar_z = self.model.posterior_z(r_t)
            z_slots = self.model.posterior_z.sample(mu_z, logvar_z)
            scene_recon = self.model.decode_scene(z_slots)
            scene_recons.append(scene_recon)

        # Compute metrics if gt available
        metrics = {}
        if gt_masks is not None and len(gt_masks_seq) > 0:
            result_metrics = self.compute_metrics_seq(pred_masks_seq, gt_masks_seq)
            metrics.update(result_metrics)
        return metrics

# Example usage:
# Assuming loaded model and dataset
if __name__ == "__main__":
    # Load model
    # model = ...
    # dataset = ...
    # For a video of frames: list of tensors
    # evaluator = Evaluator(model, dataset)
    # video_frames, gt_masks = ... load as list of tensors and masks
    # metrics = evaluator.evaluate_video(video_frames, gt_masks)
    pass
```

## main.py

```python
# main.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_loader import MoviSequenceDataset
from model import build_vonet_from_config
from utils import (
    kl_annealing,
    set_seed,
    load_checkpoint,
    save_checkpoint,
    adjust_learning_rate
)
from trainer import Trainer
from evaluation import Evaluator

def main():
    # 1. Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['misc'].get('device', 'cuda'))
    seed = config['misc'].get('seed', 42)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 2. Dataset and DataLoader
    dataset_dir = './dataset'  # update if different
    split_files = ['./splits/train_split.txt']
    # For validation/testing
    val_split_files = ['./splits/validation_split.txt']

    dataset = MoviSequenceDataset(
        dataset_dir=dataset_dir,
        split_files=split_files,
        sequence_length=config['training'].get('segment_length', 3),
        training=True,
        transform=None,
        dataset_split='official_split',
        object_max_count=10  # for MOVI-A/B/C, change to 16 for D/E if needed
    )

    val_dataset = MoviSequenceDataset(
        dataset_dir=dataset_dir,
        split_files=val_split_files,
        sequence_length=config['training'].get('segment_length', 3),
        training=False,
        transform=None,
        dataset_split='official_split',
        object_max_count=10
    )

    batch_size = config['training'].get('batch_size', 32)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=None, pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=None, pin_memory=True
    )

    # 3. Instantiate model
    model = build_vonet_from_config(config)
    model.to(device)

    # 4. Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    schedule_params = config['training'].get('learning_rate_schedule', {})

    # Implement simple LambdaLR as placeholder for schedule
    def lr_lambda(current_step):
        return adjust_learning_rate(optimizer, current_step, schedule_params)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        dataset_loader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )

    # 6. Start training
    trainer.train()

    # 7. Optional: run final evaluation on validation set
    evaluator = Evaluator(
        model=model,
        dataset=val_dataset,
        device=device,
        fg_ari=True,
        mip_iu=True,
        mask_threshold=0.3,
        visualization=True,
        save_dir=os.path.join(trainer.result_save_path, 'final_eval')
    )

    os.makedirs(evaluator.save_dir, exist_ok=True)

    print("Running final evaluation on validation set...")
    total_metrics = []
    for idx in tqdm(range(len(val_dataset)), desc='Evaluating'):
        # Load a single video sequence (simulate)
        # Can load full video or sample one sequence:
        # Here, for illustration, pick one sequence (e.g., index=0)
        # For full validation, iterate through val_dataloader
        # Alternatively, process all validation videos individually
        # Here, process the first validation sequence
        # You can extend to run over the set
        video_frames, gt_masks, meta = val_dataset[idx]
        # Convert list of frames to list of tensors
        # Each frame is tensor [3, H, W]
        if isinstance(video_frames, list):
            frames_list = video_frames
        else:
            frames_list = [video_frames[t] for t in range(len(video_frames))]
        metrics = evaluator.evaluate_video(
            video_frames=frames_list,
            gt_masks=gt_masks if gt_masks is not None else None,
            max_frames=24,
            save_visualization=True
        )
        total_metrics.append(metrics)
    # Aggregate metrics
    # Here, simply compute mean over the batch of videos
    fg_ari_vals = [m.get('FG-ARI', 0) for m in total_metrics]
    mIoU_vals = [m.get('mIoU', 0) for m in total_metrics]
    print(f"Final FG-ARI: {np.mean(fg_ari_vals):.4f} ± {np.std(fg_ari_vals):.4f}")
    print(f"Final mIoU: {np.mean(mIoU_vals):.4f} ± {np.std(mIoU_vals):.4f}")

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Utility functions
def init_weights(module, init_fn=nn.init.kaiming_normal_):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init_fn(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        return self.relu(out)

# CNN Backbone: Simple ResNet-like feature extractor
class CNNBackbone(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128, base_channels=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = ResidualBlock(base_channels)
        self.layer2 = ResidualBlock(base_channels)
        # Final conv to get desired feature_dim
        self.final_conv = nn.Conv2d(base_channels, feature_dim, kernel_size=1)
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, x):
        x = self.initial(x)  # [B, base_channels, H/4, W/4]
        x = self.layer1(x)
        x = self.layer2(x)
        features = self.final_conv(x)  # [B, feature_dim, H/4, W/4]
        return features

# U-Net with transformer bottleneck
class UNet(nn.Module):
    def __init__(self, in_channels: int=128+128, base_channels: int=64, depth: int=5, num_slots: int=11):
        super().__init__()
        self.depth = depth
        self.num_slots = num_slots
        # Downsampling path
        self.downs = nn.ModuleList()
        channels = base_channels
        for i in range(depth):
            block = nn.Sequential(
                nn.Conv2d(in_channels if i==0 else channels, channels*2, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(channels*2, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            self.downs.append(block)
            channels *= 2
        # Bottleneck transformer
        self.bottleneck_channels = channels
        self.transformer_decoder = MaskTransformerDecoder(num_slots=num_slots, feature_dim=channels, num_layers=3, n_heads=3)
        # Upsampling path
        self.ups = nn.ModuleList()
        for i in range(depth):
            up_block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(channels, channels//2, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(channels//2, affine=True),
                nn.ReLU(inplace=True)
            )
            self.ups.append(up_block)
            channels //= 2
        # Final conv to produce mask logits
        self.final_conv = nn.Conv2d(channels, num_slots+1, kernel_size=1)
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, x, slot_contexts):
        # x: backbone features [B, C, H, W]
        # slot_contexts: list of context vectors for each slot: shape [B, K, context_dim]
        skip_connections = []
        out = x
        for down in self.downs:
            out = down(out)
            skip_connections.append(out)
        # Prepare shared features for transformer
        feat_shape = out.shape  # [B, C, H', W']
        B, C, Hp, Wp = feat_shape
        # Expand slot contexts for communication (B,K,C)
        # Concatenate slot contexts along batch dimension for transformer
        # Generate initial slot mask estimates (delta logs) in a way that can be refined by the U-Net
        # For simplicity, assume the delta mask is output of U-Net from initial input (see authors' proposal)
        # but here we process using a shared transformer bottleneck
        # We flatten spatial dimensions
        feat_seq = rearrange(out, 'b c h w -> b (h w) c')
        # Feed into transformer decoder to get slot embeddings
        slot_embeddings = self.transformer_decoder(feat_seq)
        # Reshape back to spatial map per slot
        slot_embeddings = rearrange(slot_embeddings, 'b (h w) k d -> b k h w d', h=Hp, w=Wp)
        # Generate mask logits for each slot
        mask_logits = self.final_conv(slot_embeddings)  # [B, K+1, H, W]
        # Compute softmax across slot dimension (excluding background? or include background as extra slot)
        mask_prob = F.softmax(mask_logits, dim=1)
        return mask_prob

# Transformer decoder for masks' interaction
class MaskTransformerDecoder(nn.Module):
    def __init__(self, num_slots: int=11, feature_dim: int=64, num_layers: int=3, n_heads: int=3):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_heads, norm_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Initialize learnable query tokens for slots if needed, but in this case, input is features
        # We assume input features are already prepared with spatial flattened sequence
        # For simplicity, no positional encoding is used
    def forward(self, feat_sequence):
        # feat_sequence: [B, H*W, C]
        # For this module, use a placeholder learnable token: here, treat feat_sequence as the keys & queries is feat_sequence
        # In design, avoid learnable tokens, just pass the features directly
        # For matching the description, suppose we process the features with a set of slot tokens:
        # Noticing that authors communicate among slots at bottleneck, so the input is features per spatial location
        # For implementation, we just pass features through transformer decoder
        # For more fidelity, implement a set of queries per slot
        # For simplicity, assume each slot corresponds to a query: use slot embeddings as queries
        # Since we're not storing them globally here, we'll implement a set of learnable slot queries
        # but as per design, the query vectors are learned or fixed; assuming fixed:
        # Let's implement slot queries as a parameter
        self.slot_queries = getattr(self, 'slot_queries', None)
        if self.slot_queries is None:
            self.slot_queries = nn.Parameter(torch.randn(1, self.transformer.num_layers, feat_sequence.shape[-1]))
        B = feat_sequence.shape[0]
        query = self.slot_queries.expand(B, -1, -1)  # [B, K, C]
        # expand to (K, B, C) for transformer
        query = rearrange(query, 'b k c -> k b c')
        memory = rearrange(feat_sequence, 'b s c -> s b c')  # transpose for src memory
        # Use transformer decoder
        out = self.transformer(tgt=query, memory=memory)
        # out: [K, B, C]
        out = rearrange(out, 'k b c -> b k c')
        return out

# Slot Encoder: extract per-slot features from masked features
class SlotEncoder(nn.Module):
    def __init__(self, feature_dim: int=128, slot_feature_dim: int=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, slot_feature_dim)
        )
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, features, masks):
        # features: [B, C, H, W]
        # masks: [B, K, H, W]
        B, C, H, W = features.shape
        K = masks.shape[1]
        # Expand features for masking
        features_exp = features.unsqueeze(1).expand(-1, K, -1, -1, -1)  # [B, K, C, H, W]
        masks_exp = masks.unsqueeze(2)  # [B, K, 1, H, W]
        masked_feats = features_exp * masks_exp  # [B, K, C, H, W]
        # Average pooling over spatial dims
        pooled_feats = masked_feats.view(B, K, C, -1).mean(-1)  # [B, K, C]
        # Pass through MLP
        slot_feats = self.mlp(pooled_feats)  # [B, K, slot_feature_dim]
        return slot_feats

# Slot Trajectory RNN - GRU with LayerNorm
class SlotTrajectoryRNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, slot_feats, r_prev):
        # slot_feats: [B, K, input_dim]
        # r_prev: [B, K, hidden_dim]
        # Process each slot independently
        B, K, D = slot_feats.shape
        r_prev_flat = r_prev.reshape(B*K, -1)  # [B*K, D]
        slot_feats_flat = slot_feats.reshape(B*K, -1)  # [B*K, D]
        # Concatenate slot feature + previous state
        input_seq = slot_feats_flat.unsqueeze(1)  # [B*K, 1, D]
        r_prev_seq = r_prev_flat.unsqueeze(1)  # same shape
        # Run through GRU
        # We can model each slot as a sequence length 1 for simplicity
        r_output, _ = self.gru(input_seq, r_prev_seq.unsqueeze(0))
        r_new = r_output.squeeze(1)  # [B*K, hidden_dim]
        r_new = self.layer_norm(r_new + self.mlp(r_new))
        # Reshape back
        r_new = r_new.view(B, K, -1)
        return r_new

# Variational posterior encoder for z_{t,k}
class VariationalPosterior(nn.Module):
    def __init__(self, slot_feature_dim=128, latent_dim=128):
        super().__init__()
        self.mu_layer = nn.Linear(slot_feature_dim, latent_dim)
        self.logvar_layer = nn.Linear(slot_feature_dim, latent_dim)
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, slot_state):
        # slot_state: [B, K, slot_feature_dim]
        mu = self.mu_layer(slot_state)  # [B, K, latent_dim]
        logvar = self.logvar_layer(slot_state)
        return mu, logvar

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z

# Prior transformer to predict r'_{t,k}
class PriorTransformer(nn.Module):
    def __init__(self, num_slots: int=11, slot_dim: int=128, num_layers: int=2, n_heads: int=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=slot_dim, nhead=n_heads, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_mu = nn.Linear(slot_dim, slot_dim)
        self.mlp_logvar = nn.Linear(slot_dim, slot_dim)
        self.num_slots = num_slots
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, prev_r):
        # prev_r: [B, K, D]
        B, K, D = prev_r.shape
        # No positional encoding, raw input
        r_seq = prev_r  # [B, K, D]
        r_seq = r_seq.permute(1, 0, 2)  # [K, B, D]
        encoded = self.transformer(r_seq)  # [K, B, D]
        encoded = encoded.permute(1, 0, 2)  # [B, K, D]
        mu_prior = self.mlp_mu(encoded)  # [B, K, D]
        logvar_prior = self.mlp_logvar(encoded)
        return mu_prior, logvar_prior

# Transformer-based Scene Decoder (autoregressive)
class SceneDecoder(nn.Module):
    def __init__(self, slot_num, slot_dim=128, decoder_layers=3, decoder_heads=3, image_size=128):
        super().__init__()
        self.slot_num = slot_num
        self.slot_dim = slot_dim
        self.image_size = image_size
        # We flatten the image into patches for autoregressive decoding
        self.patch_size = 8  # e.g., 8x8 patches -> (128/8)=16 patches
        self.num_patches = (image_size // self.patch_size) ** 2
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        # Input: concatenate slot embeddings to scene tokens
        self.scene_tokens = nn.Parameter(torch.randn(1, self.num_patches, slot_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=slot_dim, nhead=decoder_heads, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        # Output projection to RGB (or features)
        self.output_projection = nn.Linear(slot_dim, 3 * self.patch_size * self.patch_size)
        self.apply_weights()

    def apply_weights(self):
        init_weights(self)

    def forward(self, z_slots, x_prev=None):
        # z_slots: [B, K, D]
        B, K, D = z_slots.shape
        # Expand slot embeddings for each scene patch
        # For simplicity, expand z_slots over patches
        z_expanded = z_slots.unsqueeze(2).expand(-1, -1, self.num_patches, -1)  # [B, K, num_patches, D]
        # Flatten across slots and patches
        scene_queries = z_expanded.view(B, self.num_patches * K, D).permute(1, 0, 2)  # [skip: can process individually]
        # The decoder attends over all slot embeddings for each patch
        # Generate initial scene tokens, possibly learned or fixed
        scene_tokens = self.scene_tokens.expand(B, -1, -1)  # [B, num_patches, D]
        scene_tokens = scene_tokens.permute(1, 0, 2)  # [num_patches, B, D]
        # Decode scene patches autoregressively
        decoded = self.transformer_decoder(tgt=scene_tokens, memory=scene_queries)
        # decoded: [num_patches, B, D]
        decoded = decoded.permute(1, 0, 2)  # [B, num_patches, D]
        # Map to pixel patches
        patches = self.output_projection(decoded)  # [B, num_patches, 3*patch_size*patch_size]
        # Reshape to image
        batch_img = self._assemble_image_from_patches(patches, B)
        return batch_img

    def _assemble_image_from_patches(self, patches, B):
        # Convert patches to [B, 3, H, W]
        patches = patches.view(B, self.num_patches, 3, self.patch_size, self.patch_size)
        # Reassemble patches into full image
        h_blocks = w_blocks = int(self.image_size / self.patch_size)
        img = torch.zeros(B, 3, self.image_size, self.image_size, device=patches.device)
        idx = 0
        for i in range(h_blocks):
            for j in range(w_blocks):
                img[:, :, i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size] = patches[:, idx]
                idx += 1
        return img

# Main VONet Model class
class VONet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # Config parsing
        self.num_slots = config.get('model', {}).get('slot_number', 11)
        self.slot_dim = config.get('model', {}).get('slot_embedding_dim', 128)
        self.feature_dim = 128
        self.backbone_channels = 64
        self.attention_unet_depth = config.get('model', {}).get('attention_unet_depth', 5)
        self.attention_unet_channels = config.get('model', {}).get('attention_unet_channels', 64)
        self.transformer_mask_layers = config.get('model', {}).get('transformer_mask_layers', 3)
        self.transformer_mask_heads = config.get('model', {}).get('transformer_mask_heads', 3)
        self.transformer_prior_layers = config.get('model', {}).get('transformer_prior_layers', 2)
        self.transformer_prior_heads = config.get('model', {}).get('transformer_prior_heads', 3)
        self.decoder_layers = config.get('model', {}).get('decoder_layers', 3)
        self.decoder_heads = config.get('model', {}).get('decoder_heads', 3)
        self.image_size = 128

        # Shared feature extractor
        self.backbone = CNNBackbone(in_channels=3, feature_dim=self.feature_dim, base_channels=self.backbone_channels)

        # Attention module
        self.attention_unet = UNet(in_channels=self.feature_dim + self.slot_dim,
                                     base_channels=self.attention_unet_channels,
                                     depth=self.attention_unet_depth,
                                     num_slots=self.num_slots)
        # Slot encoder
        self.slot_encoder = SlotEncoder(feature_dim=self.feature_dim, slot_feature_dim=self.slot_dim)

        # Slot trajectory RNN (GRU + MLP + LayerNorm)
        self.slot_rnn = SlotTrajectoryRNN(input_dim=self.slot_dim, hidden_dim=self.slot_dim)

        # Variational encoder for z_{t,k}
        self.posterior_z = VariationalPosterior(slot_feature_dim=self.slot_dim, latent_dim=self.slot_dim)

        # Prior transformer for r'_{t,k}
        self.prior_transformer = PriorTransformer(num_slots=self.num_slots,
                                                  slot_dim=self.slot_dim,
                                                  num_layers=self.transformer_prior_layers,
                                                  n_heads=self.transformer_prior_heads)
        # Scene decoder
        self.scene_decoder = SceneDecoder(self.num_slots, slot_dim=self.slot_dim,
                                          decoder_layers=self.decoder_layers,
                                          decoder_heads=self.decoder_heads,
                                          image_size=self.image_size)

        # Initialization
        self.apply_weights()

        # Placeholder for previous slot states (r_{t-1,k})
        self.register_buffer('initial_slot_states', torch.randn(1, self.num_slots, self.slot_dim))

    def apply_weights(self):
        init_weights(self)

    def extract_features(self, x):
        """Extract backbone features from input images."""
        return self.backbone(x)

    def generate_attention(self, features, c_prev):
        """Generate masks for all slots using parallel attention module."""
        masks = self.attention_unet(features, c_prev)
        # masks: shape [B, K+1, H, W]
        return masks

    def encode_slots(self, features, masks):
        """Extract per-slot features from the features weighted by attention masks."""
        slot_feats = self.slot_encoder(features, masks)
        return slot_feats

    def update_slot_states(self, slot_feats, r_prev):
        """Update slot states using RNN/GRU and residual connection."""
        r_new = self.slot_rnn(slot_feats, r_prev)
        return r_new

    def compute_z_posterior(self, r_tk):
        """Compute the variational posterior (q) parameters."""
        mu, logvar = self.posterior_z(r_tk)
        z = self.posterior_z.sample(mu, logvar)
        return z, mu, logvar

    def predict_slot_prior(self, r_prev):
        """Predict the future slot states using prior transformer."""
        mu_prior, logvar_prior = self.prior_transformer(r_prev)
        # Sample from prior
        std = torch.exp(0.5 * logvar_prior)
        epsilon = torch.randn_like(std)
        r_prior = mu_prior + std * epsilon
        return r_prior, mu_prior, logvar_prior

    def decode_scene(self, z_slots):
        """Decode scene from slot embeddings."""
        scene_rec = self.scene_decoder(z_slots)
        return scene_rec

    def forward(self, x, r_prev, c_prev):
        """
        x: input image tensor [B, 3, H, W]
        r_prev: previous slot states [B, K, D]
        c_prev: context vectors for each slot [B, K, D]
        """
        features = self.extract_features(x)  # shape [B, C, H', W']
        masks = self.generate_attention(features, c_prev)  # [B, K+1, H, W]
        # Extract foreground masks (excluding null/background at index 0)
        masks_probs = masks[:, 1:, :, :]  # [B, K, H, W]
        # Normalize masks to sum to 1 + background, but authors use softmax separately
        # For inference, can use these masks directly
        slot_features = self.encode_slots(features, masks[:,1:,:,:])  # [B,K,128]
        r_t = self.update_slot_states(slot_features, r_prev)  # [B,K,128]
        # Variational encoding
        z_t, mu_z, logvar_z = self.compute_z_posterior(r_t)  # [B,K,128]
        # Prior prediction
        r_prior, mu_prior, logvar_prior = self.predict_slot_prior(r_prev)
        # Decode scene
        recon_scene = self.decode_scene(z_t)
        return recon_scene, masks, r_t, r_prior, mu_z, logvar_z, mu_prior, logvar_prior

# Initialize the network with configuration
def build_vonet_from_config(config: dict) -> VONet:
    return VONet(config)
```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import kl_annealing, match_masks_hungarian, set_seed, save_checkpoint, load_checkpoint, adjust_learning_rate
from dataset_loader import SlotStateBuffer
import os
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, dataset_loader, optimizer, scheduler, config, device=None):
        """
        Args:
            model (nn.Module): VONet model.
            dataset_loader (DataLoader): DataLoader for training sequences.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler): LR scheduler.
            config (dict): Configuration dict from YAML.
            device (torch.device): Compute device.
        """
        self.model = model
        self.dataset_loader = dataset_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_steps = config['training'].get('total_steps', 150000)
        self.batch_size = config['training'].get('batch_size', 32)
        self.segment_length = config['training'].get('segment_length', 3)
        self.use_replay = config['misc'].get('use_replay_buffer', True)
        self.replay_buffer_size = config['misc'].get('replay_buffer_size', 10000)
        self.zeros_clip_norm = config['optimization'].get('max_gradient_norm', 0.1)
        self.gradient_clipping = config['optimization'].get('gradient_clipping', True)
        self.model_save_path = config['misc'].get('model_save_path', './checkpoints/')
        self.result_save_path = config['misc'].get('result_save_path', './results/')
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.result_save_path, exist_ok=True)

        # Initialize replay buffer if used
        if self.use_replay:
            self.replay_buffer = SlotStateBuffer(self.replay_buffer_size)
        else:
            self.replay_buffer = None

        # State variables
        self.global_step = 0
        # For KL schedule
        self.kl_start_step = self.config['training'].get('kl_anneal_start_step', 0)
        self.kl_end_step = self.config['training'].get('kl_anneal_end_step', 50000)
        self.beta_final = self.config['training'].get('kl_final_weight', 0.7)

        # Visualization interval
        self.vis_interval = self.config['evaluation'].get('evaluation_interval', 10000)

        # Loss tracking
        self.losses = []

        # Set seed
        seed = self.config['misc'].get('seed', 42)
        set_seed(seed)

    def train(self):
        device = self.device
        model = self.model.to(device)
        model.train()

        pbar = tqdm(total=self.total_steps, desc='Training')
        # Initialize previous slot states (r_{0,k}) for each slot, shape [batch, K, D]
        # Start from Gaussian noise
        r_prev = torch.randn(1, self.model.num_slots, self.model.slot_dim, device=device)
        r_prev = r_prev.repeat(self.batch_size, 1, 1)  # batch dimension

        # Initialize previous context (c_{t-1,k}) as zeros, optional
        c_prev = torch.zeros(self.batch_size, self.model.num_slots, self.model.slot_dim, device=device)

        # Prepare optimizer and scheduler
        # Scheduler is assumed to handle LR updates
        # No explicit independent, so we step scheduler based on step count

        for step in range(self.global_step, self.total_steps):
            # Dynamic learning rate adjustment if needed
            adjust_learning_rate(self.optimizer, step, self.config['training'].get('learning_rate_schedule', {}))
            self.optimizer.zero_grad()

            try:
                # Load batch: shape [B, L, 3, 128, 128]
                batch = next(self._data_iterator)
            except StopIteration:
                self._data_iterator = iter(self.dataset_loader)
                batch = next(self._data_iterator)

            x_seq, gt_masks_seq, meta_seq = batch
            # x_seq: [B, L, 3, 128, 128], move to device
            x_seq = x_seq.to(device)
            B, L, C, H, W = x_seq.shape

            # For current step, consider only current frames
            # Assume we process the last frame in sequence for training
            # Alternatively, process all frames with separate losses (authors' approach is to process sequences)
            # Here, following the authors, process full sequence
            total_loss = 0.0

            # For handling the residual states, initialize from replay buffer if used
            if self.use_replay:
                # Sample states for batch segments
                buffer_samples = self.replay_buffer.sample(B)
                # buffer_samples is list of dicts with keys: 'r', etc.
                r_prev_batch = []
                for sample in buffer_samples:
                    r_prev_batch.append(sample['r'])  # shape [K, D]
                r_prev_batch = torch.stack(r_prev_batch, dim=0).to(device)  # [B, K, D]
            else:
                r_prev_batch = torch.randn(B, self.model.num_slots, self.model.slot_dim, device=device)

            # Initialize context vectors c_prev as zeros or from buffer
            # For simplicity, initialize as zeros
            c_prev_batch = torch.zeros(B, self.model.num_slots, self.model.slot_dim, device=device)

            # Save per-batch metrics
            batch_recon_loss = 0.0
            batch_kl_loss = 0.0
            batch_total_loss = 0.0

            # Process sequence: for each frame
            for t in range(self.segment_length):
                x_t = x_seq[:, t, :, :, :]  # [B, 3, 128,128]
                # Forward pass
                outputs = self._forward_single_frame(x_t, r_prev_batch, c_prev_batch, step)
                recon_scene = outputs['recon']
                masks = outputs['masks']
                r_t = outputs['r']
                r_prior = outputs['r_prior']
                mu_z = outputs['mu_z']
                logvar_z = outputs['logvar_z']
                mu_prior = outputs['mu_prior']
                logvar_prior = outputs['logvar_prior']

                # Compute reconstruction loss
                # Assuming Gaussian decoder, negative log likelihood
                recon_loss = self._compute_reconstruction_loss(x_t, recon_scene)
                # Compute KLD
                kld_loss = self._compute_kld(mu_z, logvar_z, mu_prior, logvar_prior)
                # Calculate current beta
                beta = kl_annealing(step, self.kl_start_step, self.kl_end_step, self.beta_final)

                loss = recon_loss + beta * kld_loss

                total_loss += loss

                # Prepare for next timestep
                # Update r_prev_batch
                r_prev_batch = r_t.detach()
                # Update c_prev_batch, here we assume c_prev is same as r_t for simplicity
                c_prev_batch = r_t.detach()

            # Backpropagation
            total_loss.backward()

            # Gradient clipping
            if self.gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), self.max_gradient_norm)

            # Optimizer step
            self.optimizer.step()

            # Save to metrics/log
            self.losses.append(total_loss.item() / self.segment_length)

            # Save slot states to replay buffer
            if self.use_replay:
                # Save current slot states (r_t) for each timestep (here last timestep)
                # As per design, buffer operates on states per frame, so we could store r_t for the last frame in sequence
                for t in range(self.segment_length):
                    # For simplicity, store only last step's r_t
                    state = {'r': r_prev_batch.clone().detach().cpu()}
                    self.replay_buffer.add([state])  # add individually

            # Step learning rate scheduler
            self.scheduler.step()

            # Periodic visualization and validation
            if (step+1) % self.vis_interval == 0 or step == self.total_steps -1:
                self._save_training_metrics(step)
                self._visualize_masks_and_recon(x_seq, masks, step)

            # Save checkpoint
            if (step+1) % 50000 == 0 or step == self.total_steps -1:
                save_checkpoint(model, self.optimizer, step+1, os.path.join(self.model_save_path, 'model_step_{}.pt'.format(step+1)))

            pbar.update(1)

        pbar.close()

    def _forward_single_frame(self, x_t, r_prev, c_prev, step):
        """
        Run model forward for a single frame, returning outputs dict.
        """
        # Extract backbone features
        features = self.model.extract_features(x_t)
        # Generate attention masks
        masks = self.model.generate_attention(features, c_prev)
        # masks shape: [B, K+1, H, W]
        # Foreground masks: exclude background (index 0)
        masks_fg = masks[:, 1:, :, :]  # [B, K, H, W]

        # Encode slot features
        slot_feats = self.model.encode_slots(features, masks_fg)
        # Update slot states
        r_t = self.model.update_slot_states(slot_feats, r_prev)  # [B, K, D]
        # Variational posterior for z_{t,k}
        mu_z, logvar_z = self.model.posterior_z(r_t)
        z = self.model.posterior_z.sample(mu_z, logvar_z)
        # Prior prediction r'
        r_prior, mu_prior, logvar_prior = self.model.predict_slot_prior(r_prev)
        # Scene reconstruction
        recon_scene = self.model.decode_scene(z)

        return {
            'recon': recon_scene,
            'masks': masks,
            'r': r_t,
            'r_prior': r_prior,
            'mu_z': mu_z,
            'logvar_z': logvar_z,
            'mu_prior': mu_prior,
            'logvar_prior': logvar_prior
        }

    def _compute_reconstruction_loss(self, x, recon):
        """
        Compute per-pixel negative log likelihood assuming Gaussian with fixed variance.
        """
        # Assuming standard Gaussian with unit variance
        # As per authors, often uses gaussian NLL
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.shape[0]
        return recon_loss

    def _compute_kld(self, mu_q, logvar_q, mu_p, logvar_p):
        """
        Compute KL divergence between q(z|r) and p(z|r')
        """
        # KL divergence for diagonal Gaussians
        kld = 0.5 * (logvar_p - logvar_q + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p) -1 )
        return kld.sum() / mu_q.shape[0]

    def _save_training_metrics(self, step):
        """
        Save metrics, losses, and plots
        """
        # Save losses list
        plt.figure()
        plt.plot(self.losses)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss over Steps')
        plt.savefig(os.path.join(self.result_save_path, 'loss_curve_step_{}.png'.format(step)))
        plt.close()

    def _visualize_masks_and_recon(self, x_seq, masks, step):
        """
        Create visualizations for attention masks and scene reconstructions.
        """
        B, L, C, H, W = x_seq.shape
        for t in range(min(L, self.config['evaluation'].get('visualization_frames', 3))):
            x_frame = x_seq[0, t]
            mask_probs = masks[0, 1:, :, :]  # assume all K masks
            thresh_mask = (mask_probs >= 0.3).float()
            # Save overlays
            save_mask_path = os.path.join(self.result_save_path, f'seq_{step}_frame_{t}_mask.png')
            # Assume utils.py has visualization functions
            from utils import visualize_attention_masks, visualize_reconstruction
            # For visualization, get input image as numpy
            visualize_attention_masks(x_frame.permute(1,2,0).cpu().numpy(), mask_probs.cpu(), save_mask_path, frame_idx=t)
            # Save reconstructed image
            recon_img = self.model.decode_scene(z).detach().cpu()
            save_recon_path = os.path.join(self.result_save_path, f'seq_{step}_frame_{t}_recon.png')
            visualize_reconstruction(x_frame.cpu(), recon_img[0], save_recon_path, frame_idx=t)
            

# Usage example (assuming proper config, and dataset loader implemented)
if __name__ == '__main__':
    import yaml
    from dataset_loader import MoviSequenceDataset
    from model import build_vonet_from_config
    import torch.optim as optim
    import torch.nn as nn
    import torch
    import os

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(config['misc'].get('device', 'cuda'))

    # Prepare dataset
    dataset = MoviSequenceDataset(
        dataset_dir='./dataset',  # set accordingly
        split_files=['./splits/train_split.txt'],
        sequence_length=config['training'].get('segment_length',3),
        training=True,
        transform=None,
        dataset_split='official_split',
        object_max_count=10 # or 16 for D/E
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training'].get('batch_size', 32),
        shuffle=True,
        num_workers=4,
        collate_fn=None  # optional, default collate
    )

    # Build model
    model = build_vonet_from_config(config)
    model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Define LR scheduler
    def lr_lambda(current_step):
        # Implement the schedule: warmup, plateau, decay
        schedule_params = config['training'].get('learning_rate_schedule', {})
        return ... # define as needed
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Instantiate trainer
    trainer = Trainer(model, data_loader, optimizer, scheduler, config)

    # Run training
    trainer.train()
```

## utils.py

```python
## utils.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import random


def kl_annealing(current_step: int, anneal_start: int = 0, anneal_end: int = 50000, target_beta: float = 0.7) -> float:
    """
    Compute the KL annealing coefficient (beta) based on current training step.

    Args:
        current_step (int): Current training step.
        anneal_start (int, optional): Step at which annealing starts. Defaults to 0.
        anneal_end (int, optional): Step at which beta reaches target. Defaults to 50_000.
        target_beta (float, optional): Target value of beta. Defaults to 0.7.

    Returns:
        float: The KL weight (beta) for current step.
    """
    if current_step < anneal_start:
        return 0.0
    elif current_step >= anneal_end:
        return target_beta
    else:
        progress = (current_step - anneal_start) / (anneal_end - anneal_start)
        return min(max(progress * target_beta, 0.0), target_beta)


def flatten_batch_slots(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reshape tensor with shape [batch, K, ...] into [batch * K, ...].

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, K, ...].

    Returns:
        torch.Tensor: Reshaped tensor.
    """
    B, K = tensor.shape[:2]
    return tensor.reshape(B * K, *tensor.shape[2:])


def unflatten_batch_slots(tensor: torch.Tensor, batch_size: int, K: int) -> torch.Tensor:
    """
    Reshape tensor from [batch * K, ...] to [batch, K, ...].

    Args:
        tensor (torch.Tensor): Input tensor of shape [B*K, ...].
        batch_size (int): Batch size.
        K (int): Number of slots.

    Returns:
        torch.Tensor: Reshaped tensor.
    """
    return tensor.reshape(batch_size, K, *tensor.shape[1:])


def expand_to_slots(tensor: torch.Tensor, K: int) -> torch.Tensor:
    """
    Expand tensor of shape [batch, ...] to [batch, K, ...].

    Args:
        tensor (torch.Tensor): Input tensor [B, ...].
        K (int): Number of slots.

    Returns:
        torch.Tensor: Expanded tensor [B, K, ...].
    """
    B = tensor.shape[0]
    return tensor.unsqueeze(1).expand(-1, K, *tensor.shape[1:])


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize image pixel values from [0, 255] or [0,1] to [0, 1].

    Args:
        image (torch.Tensor): Image tensor.

    Returns:
        torch.Tensor: Normalized image tensor.
    """
    if image.max() > 1.0:
        return image / 255.0
    else:
        return image


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image from [0,1] to [0,255], clip, and cast to uint8.

    Args:
        image (torch.Tensor): Image tensor with [0,1].

    Returns:
        torch.Tensor: Denormalized image with [0,255], dtype uint8.
    """
    img = image * 255.0
    return img.clamp(0, 255).byte()


def mask_logits_to_mask(logits: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """
    Convert raw logits of shape [B, K+1, H, W] to integer masks [B, H, W].

    Args:
        logits (torch.Tensor): Mask logits.
        threshold (float): Confidence threshold to assign null/background.

    Returns:
        torch.Tensor: Integer mask of shape [B, H, W], values in [0..K], where 0 is null/background.
    """
    probs = F.softmax(logits, dim=1)  # [B, K+1, H, W]
    max_probs, idxs = torch.max(probs, dim=1)  # [B, H, W]
    # Initialize mask with null label 0
    mask = idxs.clone()
    # Assign null labels where max probability is below threshold
    mask[max_probs < threshold] = 0
    return mask  # int tensor


def visualize_attention_masks(input_img: torch.Tensor, masks: torch.Tensor, save_path: str, frame_idx: int = None):
    """
    Overlay attention masks on the input image and save.

    Args:
        input_img (torch.Tensor): [H, W, 3], uint8 or float with [0,255] or [0,1].
        masks (torch.Tensor): [K, H, W], with mask values [0..K].
        save_path (str): Path to save the visualization image.
        frame_idx (int, optional): Frame index for labeling. Defaults to None.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    H, W = input_img.shape[:2]
    if input_img.dtype != torch.uint8:
        display_img = denormalize_image(input_img).cpu().numpy()
    else:
        display_img = input_img.cpu().numpy()

    overlay = (display_img * 0.8).astype(np.uint8).copy()

    # Generate random colors for each slot
    K = masks.shape[0]
    colors = np.random.rand(K, 3)
    for k in range(K):
        mask_np = masks[k].cpu().numpy()
        color_mask = np.zeros_like(overlay, dtype=np.uint8)
        for c in range(3):
            color_mask[:, :, c] = (mask_np * colors[k, c] * 255).astype(np.uint8)
        overlay = np.where(mask_np[:, :, None] > 0.5, color_mask, overlay)

    plt.figure(figsize=(W/100, H/100), dpi=100)
    plt.imshow(overlay)
    plt.axis('off')
    if frame_idx is not None:
        plt.title(f'Frame {frame_idx}')
    plt.savefig(save_path)
    plt.close()


def visualize_reconstruction(input_img: torch.Tensor, reconstructed_img: torch.Tensor, save_path: str, frame_idx: int = None):
    """
    Save side-by-side input and reconstructed images.

    Args:
        input_img (torch.Tensor): [H, W, 3], [0,255] or [0,1].
        reconstructed_img (torch.Tensor): same shape as input.
        save_path (str): Path to save visualization.
        frame_idx (int, optional): Index label.
    """
    import matplotlib.pyplot as plt

    if input_img.dtype != torch.uint8:
        input_np = denormalize_image(input_img).cpu().numpy()
        recon_np = denormalize_image(reconstructed_img).cpu().numpy()
    else:
        input_np = input_img.cpu().numpy()
        recon_np = reconstructed_img.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.transpose(input_np, (1, 2, 0)))
    axes[0].set_title("Input")
    axes[0].axis('off')
    axes[1].imshow(np.transpose(recon_np, (1, 2, 0)))
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')
    if frame_idx is not None:
        plt.suptitle(f'Frame {frame_idx}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_kld_over_time(kld_list: list, save_path: str):
    """
    Plot the KLD evolution over frames/timesteps.

    Args:
        kld_list (list): Sequence of KLD values.
        save_path (str): Path to save plot.
    """
    plt.figure()
    plt.plot(kld_list, marker='o')
    plt.xlabel("Frame / Time")
    plt.ylabel("KLD / Slot")
    plt.title("KLD over Time per Slot")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_fg_ari(pred_masks: np.ndarray, gt_masks: np.ndarray) -> float:
    """
    Compute FG-ARI between predicted and ground truth masks.

    Args:
        pred_masks (np.ndarray): [T, H, W], integer labels.
        gt_masks (np.ndarray): [T, H, W], integer labels.

    Returns:
        float: Averaged FG-ARI over the sequence.
    """
    from sklearn.metrics import adjusted_rand_score

    total_score = 0.0
    T = pred_masks.shape[0]
    for t in range(T):
        pred_flat = pred_masks[t].flatten()
        gt_flat = gt_masks[t].flatten()
        # Foreground only (mask > 0)
        fg_idx = gt_flat > 0
        if np.sum(fg_idx) == 0:
            continue
        score = adjusted_rand_score(gt_flat[fg_idx], pred_flat[fg_idx])
        total_score += score
    return total_score / max(T, 1)


def compute_mIoU(pred_masks: np.ndarray, gt_masks: np.ndarray) -> float:
    """
    Compute mean IoU over a sequence, matching predicted masks to gt masks using Hungarian algorithm.

    Args:
        pred_masks (np.ndarray): [T, H, W], with labels including null/background.
        gt_masks (np.ndarray): [T, H, W], with groundtruth labels.

    Returns:
        float: mean IoU over sequence.
    """
    T = pred_masks.shape[0]
    ious = []
    for t in range(T):
        pred_mask = pred_masks[t]
        gt_mask = gt_masks[t]
        # Get unique labels excluding background (0)
        pred_labels = np.unique(pred_mask)
        gt_labels = np.unique(gt_mask)
        pred_labels = pred_labels[pred_labels != 0]
        gt_labels = gt_labels[gt_labels != 0]
        if len(pred_labels) == 0 or len(gt_labels) == 0:
            continue
        # Build IoU matrix
        cost_matrix = np.zeros((len(pred_labels), len(gt_labels)))
        for i, pl in enumerate(pred_labels):
            pred_bin = pred_mask == pl
            for j, gl in enumerate(gt_labels):
                gt_bin = gt_mask == gl
                intersection = (pred_bin & gt_bin).sum()
                union = pred_bin.sum() + gt_bin.sum() - intersection
                iou = intersection / union if union > 0 else 0
                cost_matrix[i, j] = 1 - iou  # For Hungarian, minimize cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        match_ious = []
        for r, c in zip(row_ind, col_ind):
            pl = pred_labels[r]
            gl = gt_labels[c]
            pred_bin = pred_mask == pl
            gt_bin = gt_mask == gl
            intersection = (pred_bin & gt_bin).sum()
            union = pred_bin.sum() + gt_bin.sum() - intersection
            iou = intersection / union if union > 0 else 0
            match_ious.append(iou)
        if len(match_ious) > 0:
            ious.append(np.mean(match_ious))
    return np.mean(ious) if len(ious) > 0 else 0.0


def match_masks_hungarian(pred_masks: np.ndarray, gt_masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match predicted mask labels to ground truth labels using Hungarian matching based on IoU.

    Args:
        pred_masks (np.ndarray): [H, W], predicted labels.
        gt_masks (np.ndarray): [H, W], groundtruth labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: remapped predicted labels, matched gt labels
    """
    pred_labels = np.unique(pred_masks)
    gt_labels = np.unique(gt_masks)
    pred_labels = pred_labels[pred_labels != 0]
    gt_labels = gt_labels[gt_labels != 0]

    cost_matrix = np.zeros((len(pred_labels), len(gt_labels)))
    for i, pl in enumerate(pred_labels):
        pred_bin = pred_masks == pl
        for j, gl in enumerate(gt_labels):
            gt_bin = gt_masks == gl
            intersection = np.logical_and(pred_bin, gt_bin).sum()
            union = pred_bin.sum() + gt_bin.sum() - intersection
            iou = intersection / union if union > 0 else 0
            cost_matrix[i, j] = 1 - iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Establish remapping based on matching
    pred_remap = np.zeros_like(pred_masks)
    for i, j in zip(row_ind, col_ind):
        pred_remap[pred_masks == pred_labels[i]] = gt_labels[j]
    # Assign null to unmatched pixels
    # For unmatched predicted labels, assign 0 (background)
    return pred_remap, gt_labels[col_ind]


def set_seed(seed: int):
    """
    Set seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Get the device object.

    Returns:
        torch.device: CUDA if available else CPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, path: str):
    """
    Save model state and optimizer.

    Args:
        model (torch.nn.Module): Network.
        optimizer (torch.optim.Optimizer): Optimizer.
        step (int): Current step.
        path (str): Save path.
    """
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """
    Load checkpoint into model (and optimizer).

    Args:
        path (str): Path to checkpoint.
        model (torch.nn.Module): Model to load into.
        optimizer (torch.optim.Optimizer, optional): If provided, load optimizer state.

    Returns:
        int: Last training step.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('step', 0)


def adjust_learning_rate(optimizer, current_step: int, schedule_params: dict):
    """
    Adjust learning rate based on schedule.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        current_step (int): Current training step.
        schedule_params (dict): Schedule parameters with keys 'warmup_steps', 'max_lr', 'plateau_steps', 'decay_steps'.
    """
    warmup = schedule_params.get('warmup_steps', 0)
    max_lr = schedule_params.get('max_lr', 1e-4)
    plateau = schedule_params.get('plateau_steps', 100000)
    decay = schedule_params.get('decay_steps', 50000)

    if current_step < warmup:
        lr = schedule_params.get('warmup_start_lr', 1e-6) + (max_lr - schedule_params.get('warmup_start_lr', 1e-6)) * (current_step / warmup)
    elif current_step <= warmup + plateau:
        lr = max_lr
    elif current_step <= warmup + plateau + decay:
        progress = (current_step - warmup - plateau) / decay
        lr = max_lr * (1 - progress)
    else:
        lr = schedule_params.get('final_lr', 1e-5)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\vonet\vonet_repo`
