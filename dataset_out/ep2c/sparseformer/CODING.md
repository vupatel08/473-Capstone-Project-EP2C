# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset.py

```python
## dataset.py
import os
from typing import Optional, Callable, List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageNetDataset(Dataset):
    """
    Dataset class for ImageNet-1K, supporting train and validation splits,
    with configurable augmentation and preprocessing pipelines.
    """
    def __init__(self, config: Dict, split: str = 'train'):
        """
        Args:
            config (Dict): Configuration dictionary parsed from config.yaml.
            split (str): 'train' or 'val'. Determines dataset split.
        """
        self.data_dir = config['dataset']['data_dir']
        self.image_size = config['dataset'].get('image_size', 224)
        self.num_workers = config['dataset'].get('num_workers', 8)
        self.augmentation = config['dataset'].get('augmentation', [])
        self.split = split.lower()

        # Determine root directory based on split
        if self.split == 'train':
            root_dir = os.path.join(self.data_dir, config['dataset']['train_split'])
            is_training = True
        elif self.split == 'val' or self.split == 'test':
            root_dir = os.path.join(self.data_dir, config['dataset']['val_split'])
            is_training = False
        else:
            raise ValueError(f"Unknown dataset split: {split}")

        # Setup transforms based on augmentation config
        self.transform = self._build_transform(is_training)

        # Initialize underlying dataset
        self.dataset = torchvision.datasets.ImageFolder(root=root_dir, transform=self.transform)

    def _build_transform(self, is_training: bool) -> Callable:
        """
        Build torchvision transforms pipeline based on augmentation configuration.
        """
        transform_list: List[Callable] = []

        # Parse augmentation settings
        augmentation_cfg = self.augmentation

        # For training, apply augmentation: random resized crop, flip, normalization
        if is_training:
            # Check if 'random_resized_crop' specified
            if any('random_resized_crop' in str(step) for step in augmentation_cfg):
                transform_list.append(transforms.RandomResizedCrop(self.image_size))
            else:
                transform_list.append(transforms.Resize(256))
                transform_list.append(transforms.RandomCrop(self.image_size))
            # Horizontal flip
            if any('horizontal_flip' in str(step) for step in augmentation_cfg):
                transform_list.append(transforms.RandomHorizontalFlip())
        else:
            # For validation/test
            transform_list.append(transforms.Resize(256))
            transform_list.append(transforms.CenterCrop(self.image_size))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalize
        norm_cfg = {}
        for step in augmentation_cfg:
            if isinstance(step, dict) and 'normalization' in step:
                norm_cfg = step['normalization']
                break
        if not norm_cfg:
            # Default normalization if not specified
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = norm_cfg.get('mean', [0.485, 0.456, 0.406])
            std = norm_cfg.get('std', [0.229, 0.224, 0.225])

        transform_list.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Return:
            image (Tensor): Transformed image tensor, shape [3, H, W]
            label (int): Class index label
        """
        image, label = self.dataset[index]
        return image, label
```

## evaluation.py

```python
## evaluation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from torchvision.utils import draw_bounding_boxes
from dataset import ImageNetDataset  # Assuming dataset.py provides ImageNetDataset
from model import SparseFormer          # Assuming model.py provides SparseFormer
from utils import bilinear_sample, setup_logging  # Utility functions
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None  # Require user to install 'ptflops' if needed

# Load the config for evaluation parameters
import yaml
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_top1_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device, max_samples: int = 1000) -> float:
    """
    Evaluate top-1 accuracy over the validation set.
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            if total_samples >= max_samples:
                break

    accuracy = 100.0 * total_correct / total_samples
    model.train()
    return accuracy

def measure_model_flops(model: nn.Module, input_res: Tuple[int, int] = (224, 224)) -> float:
    """
    Calculate the FLOPs of the model on a dummy input.
    Requires 'ptflops' package.
    """
    if get_model_complexity_info is None:
        print("ptflops is not installed. Please install it for FLOPs measurement.")
        return -1.0
    model.eval()
    dummy_input = torch.randn(1, 3, input_res[0], input_res[1]).to(DEVICE)
    flops, params = get_model_complexity_info(model, input_res, as_strings=False,
                                              print_per_layer=False)
    model.train()
    return flops / 1e9  # Convert to GFLOPs

def measure_throughput(model: nn.Module, dataloader: DataLoader, device: torch.device, 
                       num_iters: int = 50, batch_size: int = 128) -> float:
    """
    Measure inference throughput (images/sec) over a fixed number of iterations.
    """
    model.eval()
    images_iter = iter(dataloader)
    # Prepare a batch of dummy images
    dummy_batch = next(images_iter)[0][:batch_size]
    dummy_batch = dummy_batch.to(device, non_blocking=True)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_batch)

    # Timer start
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iters):
            try:
                images = next(images_iter)[0][:batch_size]
            except StopIteration:
                images_iter = iter(dataloader)
                images = next(images_iter)[0][:batch_size]
            images = images.to(device, non_blocking=True)
            _ = model(images)
    torch.cuda.synchronize()
    end_time = time.time()

    total_images = batch_size * num_iters
    throughput = total_images / (end_time - start_time)
    model.train()
    return throughput

def visualize_token_rois(model: nn.Module, image: torch.Tensor, save_dir: str, idx: int):
    """
    Visualize RoIs of tokens on the input image.
    """
    import cv2
    import numpy as np

    model.eval()
    image = image.unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

    # Hook or modify model to output token RoIs during forward
    # Here, we assume model provides an attribute/function to get RoIs
    # For demonstration, we rerun the forward with hook to capture RoIs.
    token_rois = None

    def hook_fn(module, input, output):
        nonlocal token_rois
        # Assuming the model stores the latest RoIs in an accessible attribute
        # Or modify model code to output RoIs explicitly
        if hasattr(model, 'last_token_rois'):
            token_rois = model.last_token_rois.detach().cpu()

    # Register hook on the focusing transformer stage or relevant module
    # For simplicity, assuming model has method 'forward_with_rois'
    # and returns RoIs along with logits, otherwise you'll need to modify model to output RoIs.
    try:
        # Run inference with hook
        _ = model(image)
        # After inference, token_rois should be set in model
        rois_np = model.last_token_rois.cpu().numpy()  # shape: [N, 4], normalized [0,1]
    except AttributeError:
        print("Model does not provide token RoIs access. Implement model hook to get RoIs.")
        return

    # Load original image
    img_path = None
    # In actual implementation, you should keep track of the original image path
    # or pass the original image tensor for visualization.
    # Here, we assume the original image tensor was saved separately or passed as input.
    # For demonstration, create dummy image
    image_np = image.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    # Convert normalized RoIs to pixel coordinates
    H, W = image_np.shape[:2]
    boxes = []
    for (x, y, w, h) in rois_np:
        x1 = int((x - 0.5 * w) * W)
        y1 = int((y - 0.5 * h) * H)
        x2 = int((x + 0.5 * w) * W)
        y2 = int((y + 0.5 * h) * H)
        boxes.append([x1, y1, x2, y2])

    # Draw bounding boxes
    box_tensor = torch.tensor(boxes)
    img_bboxes = draw_bounding_boxes(
        torch.tensor(image_np).permute(2,0,1),
        boxes=box_tensor,
        width=2,
        colors="red",
        labels=None
    )

    # Save or show image
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"token_rois_{idx}.png")
    img_bboxes = img_bboxes.permute(1,2,0).numpy()
    plt.imshow(img_bboxes)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_sampling_density(model: nn.Module, dataset: Dataset, idx: int, save_dir: str):
    """
    Visualize sampling point density maps for a specific image.
    """
    import seaborn as sns
    import cv2

    # Run inference on selected image
    image, label = dataset[idx]
    image_tensor = image.unsqueeze(0).to(DEVICE)

    # Hook or modify model to save sampling points at each stage
    sampling_points_list: List[np.ndarray] = []

    # For demonstration, assuming model or its focusing layers have a method or attribute to get sampling points
    # If not, modify model to store sampling points during forward pass.
    try:
        _ = model(image_tensor)
        # Assuming model has attribute 'last_sampling_points' as list of P x 2 arrays per stage
        sampling_points_list = model.last_sampling_points  # List of NxP x 2 arrays
    except AttributeError:
        print("Model does not provide sampling points. Please modify model to output them.")
        return

    # Prepare original image
    image_np = image.permute(1,2,0).cpu().numpy()

    for stage_idx, points in enumerate(sampling_points_list):
        # points: [N, P, 2], normalized coords
        # For visualization, project to pixel location
        H, W = image.shape[1], image.shape[2]
        points = np.clip(points, 0, 1)
        abs_points = points.copy()
        abs_points[:,:,0] = points[:,:,0] * W
        abs_points[:,:,1] = points[:,:,1] * H

        # For density map, flatten all points
        all_points = abs_points.reshape(-1, 2)
        x_coords, y_coords = all_points[:,0], all_points[:,1]
        # Generate density map
        density_map, xedges, yedges = np.histogram2d(
            y_coords, x_coords, bins=(H//4, W//4),
            range=[[0, H], [0, W]]
        )

        # Smooth density with kernel
        from scipy.ndimage import uniform_filter
        density_map_smooth = uniform_filter(density_map, size=7)

        # Plot heatmap
        plt.figure(figsize=(6,6))
        plt.imshow(image_np)
        plt.imshow(density_map_smooth, alpha=0.6, cmap='hot', extent=[0, W, H, 0])
        plt.axis('off')
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"sampling_density_stage_{stage_idx+1}_{idx}.png")
        plt.savefig(save_path)
        plt.close()

def main():
    # Initialize dataset and model for evaluation
    dataset_val = ImageNetDataset(CONFIG, split='val')
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=CONFIG['dataset'].get('num_workers', 8))
    model = SparseFormer(CONFIG)
    model.to(DEVICE)
    model.eval()

    # Load trained model checkpoint
    checkpoint_path = os.path.join(CONFIG['saving'].get('checkpoint_dir', './checkpoints/'), 'final.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print("Model checkpoint not found. Run training first.")
        return

    # 1. Compute top-1 accuracy
    print("Calculating top-1 accuracy on validation set...")
    val_acc1 = compute_top1_accuracy(model, dataloader_val, DEVICE)
    print(f"Validation Top-1 Accuracy: {val_acc1:.2f}%")

    # 2. Measure FLOPs
    print("Measuring model FLOPs...")
    flops_g = measure_model_flops(model, input_res=(224,224))
    print(f"Model GFLOPs: {flops_g:.2f} GF")

    # 3. Measure throughput
    print("Measuring throughput (images/sec)...")
    throughput = measure_throughput(model, dataloader_val, DEVICE)
    print(f"Approximate throughput: {throughput:.2f} images/sec")

    # 4. Visualization: RoIs
    if CONFIG['evaluation'].get('visualize_token_rois', False):
        print("Visualizing token RoIs for selected images...")
        for idx in range(min(10, len(dataset_val))):  # sample 10 images
            image, _ = dataset_val[idx]
            visualize_token_rois(model, image, save_dir='visualizations/rois', idx=idx)

    # 5. Visualization: Sampling Density Maps
    if CONFIG['evaluation'].get('visualize_sampling_density', False):
        print("Visualizing sampling point density maps...")
        for idx in range(min(10, len(dataset_val))):
            image, _ = dataset_val[idx]
            visualize_sampling_density(model, dataset_val, idx, save_dir='visualizations/sampling_density')

if __name__ == "__main__":
    main()
```

## main.py

```python
## main.py
import os
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
import torchvision
from torch.utils.data import DataLoader
from dataset import ImageNetDataset
from model import SparseFormer
from utils import get_lr_scheduler, setup_logging, save_checkpoint, initialize_weights
import math

def main():
    # Load configuration from YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional: deterministic cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup logging (e.g., TensorBoard)
    writer = setup_logging()

    # Prepare datasets and dataloaders
    print("Preparing datasets and dataloaders...")
    train_dataset = ImageNetDataset(config, split='train')
    val_dataset = ImageNetDataset(config, split='val')

    batch_size = config['training'].get('batch_size', 128)
    num_workers = config['dataset'].get('num_workers', 8)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Initialize model
    print("Initializing model...")
    model = SparseFormer(config)
    model.to(device)

    # Initialize weights
    model.apply(initialize_weights)

    # Load from checkpoint if resume
    checkpoint_dir = config['saving'].get('checkpoint_dir', './checkpoints/')
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = 1
    best_acc1 = 0.0
    checkpoint_path = os.path.join(checkpoint_dir, 'latest.pth')
    if config['training'].get('resume_from_checkpoint', False) and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer_state = checkpoint.get('optimizer')
        if optimizer_state:
            # will initialize optimizer later
            pass
        start_epoch = checkpoint.get('epoch', 1)
        best_acc1 = checkpoint.get('best_acc1', 0.0)
    else:
        optimizer_state = None

    # Build optimizer
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['training'].get('lr', 0.001),
                            weight_decay=config['training'].get('weight_decay', 0.05))
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    # Build scheduler
    total_steps = int(len(train_loader) * config['training'].get('epochs', 50))
    scheduler = get_lr_scheduler(optimizer, config, total_steps)

    # AMP scaler for mixed precision
    scaler = amp.GradScaler(enabled=config['training'].get('mixed_precision', False))

    # Training loop
    max_epochs = config['training'].get('epochs', 50)
    gradient_clip_norm = config['training'].get('gradient_clip_norm', 0.0)
    save_every = config['saving'].get('save_every_epochs', 10)

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        epoch_start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with amp.autocast(enabled=config['training'].get('mixed_precision', False)):
                outputs = model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)

            scaler.scale(loss).backward()

            # Gradient clipping if set
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            # Step learning rate scheduler
            scheduler.step()

            # Compute accuracy
            _, preds = torch.max(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            total_loss += loss.item() * images.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch}/{max_epochs}] Step [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
                # Log to TensorBoard
                global_step = (epoch - 1) * len(train_loader) + batch_idx + 1
                writer.add_scalar('train/loss', loss.item(), global_step)
                train_acc = 100.0 * total_correct / total_samples
                writer.add_scalar('train/accuracy', train_acc, global_step)

        epoch_loss = total_loss / len(train_dataset)
        epoch_acc = 100.0 * total_correct / total_samples
        print(f"Epoch [{epoch}] Training Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
        writer.add_scalar('train/epoch_accuracy', epoch_acc, epoch)

        # Save checkpoint
        if epoch % save_every == 0:
            save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc1': best_acc1
            }, save_path)
            print(f"Checkpoint saved at {save_path}")

        # Validation
        print(f"Running validation...")
        model.eval()
        val_total_correct = 0
        val_total_samples = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with amp.autocast(enabled=config['training'].get('mixed_precision', False)):
                    outputs = model(images)
                    loss_val = nn.CrossEntropyLoss()(outputs, labels)
                val_loss_sum += loss_val.item() * images.size(0)
                _, preds = torch.max(outputs, dim=1)
                val_total_correct += torch.sum(preds == labels).item()
                val_total_samples += labels.size(0)

        val_loss_epoch = val_loss_sum / len(val_dataset)
        val_acc1 = 100.0 * val_total_correct / val_total_samples
        print(f"Validation Loss: {val_loss_epoch:.4f} | Top-1 Accuracy: {val_acc1:.2f}%")
        writer.add_scalar('val/loss', val_loss_epoch, epoch)
        writer.add_scalar('val/accuracy', val_acc1, epoch)

        # Save best model
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved at epoch {epoch} with accuracy {best_acc1:.2f}%")

        print(f"Epoch {epoch} completed in {time.time() - epoch_start_time:.2f} seconds\n")

    print("Training finished.")
    print(f"Best val accuracy: {best_acc1:.2f}%")
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'final.pth')
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_acc1': best_acc1
    }, final_path)
    print(f"Final model saved at {final_path}")

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
from typing import Tuple

class SharedBackbone(nn.Module):
    """
    Lightweight convolutional backbone as described:
    7x7 stride-2 conv -> ReLU -> 3x3 stride-2 max pool.
    Output feature map has shape [batch, C, H/4, W/4]
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x  # shape: [batch, C, H/4, W/4]

class MLP(nn.Module):
    """
    Simple MLP with one hidden layer, activation GELU.
    Used for adaptive feature decoding.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TokenEmbeddings(nn.Module):
    """
    Manage learnable token embeddings and associated RoIs.
    """
    def __init__(self, num_tokens: int, token_dim: int, init_rois: torch.Tensor):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(num_tokens, token_dim))
        self.rois = nn.Parameter(init_rois.clone())  # shape: [N, 4], normalized [0,1]
    def get_embeddings(self):
        return self.embeddings
    def get_rois(self):
        return self.rois
    def refine_rois(self, delta: torch.Tensor):
        """
        delta shape: [N, 4], corresponding to (x,y,w,h) adjustments
        RoI update equations:
        x' = x + Δt_x * w
        y' = y + Δt_y * h
        w' = w * exp(Δt_w)
        h' = h * exp(Δt_h)
        """
        x, y, w, h = self.rois[:,0], self.rois[:,1], self.rois[:,2], self.rois[:,3]
        Δx, Δy, Δw, Δh = delta[:,0], delta[:,1], delta[:,2], delta[:,3]
        x_new = x + Δx * w
        y_new = y + Δy * h
        w_new = w * torch.exp(Δw)
        h_new = h * torch.exp(Δh)
        self.rois.data = torch.stack([x_new, y_new, w_new, h_new], dim=1).clamp(0,1)

class BilinearSampler:
    """
    Utility class for bilinear sampling from feature maps
    using sampling locations in normalized coordinates.
    """
    @staticmethod
    def sample(feature_map: torch.Tensor, sampling_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W]
            sampling_points: [N, P, 2], normalized to [0, 1]
        Returns:
            sampled_features: [N, P, C]
        """
        B, C, H, W = feature_map.shape
        N, P, _ = sampling_points.shape
        # Convert normalized coords to absolute xy in feature map
        x = sampling_points[:,:,0] * (W - 1)
        y = sampling_points[:,:,1] * (H - 1)
        grid = torch.stack([x, y], dim=3)  # [N, P, 2]
        # For batch processing, replicate feature_map
        # Note: In this context, sampling is per token, so batch size=1
        grid = grid.unsqueeze(0)  # [1, N, P, 2]
        # Reshape for grid_sample
        # But since we process per token, map batch independently
        # We assume batch size of 1 for simplicity
        grid = grid.squeeze(0).permute(2,0,1).unsqueeze(0)  # [1, 2, N, P]
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode='bilinear',
            align_corners=True
        )  # shape: [B, C, 1, N]
        sampled = sampled.squeeze(2).permute(2,0,1)  # [N, P, C]
        return sampled

class FocusLinearGenerator(nn.Module):
    """
    Generate P sampling offsets conditioned on token embedding t.
    Outputs offsets: [N, P, 2]
    """
    def __init__(self, token_dim: int, P: int):
        super().__init__()
        self.linear = nn.Linear(token_dim, 2 * P)
        self.P = P
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [N, d_c]
        Returns:
            offsets: [N, P, 2]
        """
        offset = self.linear(t)  # [N, 2*P]
        offset = offset.view(-1, self.P, 2)  # [N, P, 2]
        return offset

class RoIAdjuster(nn.Module):
    """
    Generate RoI deltas for refinement from token embedding t.
    Outputs: delta_x, delta_y, delta_w, delta_h each [N, 1]
    """
    def __init__(self, token_dim: int):
        super().__init__()
        self.linear = nn.Linear(token_dim, 4)
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [N, d_c]
        Returns:
            delta: [N, 4]
        """
        delta = self.linear(t)
        return delta

class SparseFocusingTransformer(nn.Module):
    """
    One stage of the focusing transformer:
    - Generate sampling points
    - Sample features
    - Decode features to update token embeddings
    - Refine RoIs
    """
    def __init__(self, token_dim: int, num_points: int, image_size: int, feature_map_size: Tuple[int,int]):
        super().__init__()
        self.P = num_points
        self.image_size = image_size  # e.g., 224
        self.feature_map_size = feature_map_size  # (H_feat, W_feat)
        self.offset_generator = FocusLinearGenerator(token_dim, self.P)
        self.roi_delta_generator = RoIAdjuster(token_dim)
        # Adaptive decoder
        self.decoder = MLP(in_dim=self.P * feature_map_size[0]*feature_map_size[1], hidden_dim=token_dim//4, out_dim=token_dim)
        # Map for converting offsets
        self.norm_std = 3.0  # standard deviations for normalization
    def forward(self, t: torch.Tensor, rois: torch.Tensor, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t: [N, d_c]
            rois: [N, 4], normalized [0,1]
            feature_map: [B, C, Hf, Wf], shared feature map
        Returns:
            new_t: [N, d_c]
            new_rois: [N, 4]
        """
        N, d_c = t.shape
        # Generate offsets conditioned on t
        offsets = self.offset_generator(t)  # [N, P, 2]
        # Normalize offsets with std
        offsets = offsets / self.norm_std
        # Convert relative offsets to absolute sampling locations
        x0, y0, w, h = rois[:,0], rois[:,1], rois[:,2], rois[:,3]
        # [N,1] for broadcasting
        xc = x0.unsqueeze(1)  # [N,1]
        yc = y0.unsqueeze(1)
        W = w.unsqueeze(1)
        H = h.unsqueeze(1)
        # absolute sampling locations
        x_samples = xc + 0.5 * offsets[:,:,0] * W  # [N, P]
        y_samples = yc + 0.5 * offsets[:,:,1] * H
        sampling_points = torch.stack([x_samples, y_samples], dim=2)  # [N, P, 2]
        # Clamp to [0, 1]
        sampling_points = sampling_points.clamp(0,1)
        # Sample features
        sampled_feats = BilinearSampler.sample(feature_map, sampling_points)  # [N, P, C]
        # Decode features with adaptive decoding
        feat_flat = sampled_feats.view(N, -1)  # [N, P*C]
        decoded = self.decoder(feat_flat)  # [N, d_c]
        # Residual update of token
        new_t = t + decoded
        # Generate RoI deltas
        delta = self.roi_delta_generator(t)  # [N,4]
        # Update RoIs
        new_x = rois[:,0] + delta[:,0] * rois[:,2]
        new_y = rois[:,1] + delta[:,1] * rois[:,3]
        new_w = rois[:,2] * torch.exp(delta[:,2])
        new_h = rois[:,3] * torch.exp(delta[:,3])
        new_rois = torch.stack([new_x, new_y, new_w, new_h], dim=1).clamp(0,1)
        return new_t, new_rois

class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer Encoder Layer
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float=4.0, dropout: float=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, embed_dim]
        """
        # MultiheadAttention expects [N, embed_dim], batch_first
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm.unsqueeze(0), x_norm.unsqueeze(0), x_norm.unsqueeze(0))
        attn_output = attn_output.squeeze(0)
        x = residual + self.dropout(attn_output)
        residual = x
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = residual + self.dropout(x_mlp)
        return x

class CortexTransformerEncoder(nn.Module):
    """
    Multiple layers of transformer encoder over token set.
    """
    def __init__(self, embed_dim: int, num_layers: int, num_heads: int=8, mlp_ratio: float=4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class SparseFormer(nn.Module):
    """
    Main class implementing the SparseFormer architecture.
    Combines backbone, token set, focusing transformer, cortex transformer, and classifier.
    """
    def __init__(self, config: dict):
        super().__init__()
        # Extract configuration
        self.num_tokens = config['model'].get('num_tokens', 81)
        self.token_dim = config['model'].get('token_dim', 768)
        self.focusing_layers = config['model'].get('focusing_layers', 1)
        self.cortex_layers = config['model'].get('cortex_layers', 12)
        self.sampling_points = config['model'].get('sampling_points', 36)
        self.stage_repeats = config['model'].get('stage_repeats', 1)
        self.image_size = 224  # as per training config
        # Backbone
        self.backbone = SharedBackbone(in_channels=3, out_channels=96)
        # Initialize token embeddings and RoIs
        init_rois = self._initialize_rois()
        self.token_set = TokenEmbeddings(self.num_tokens, self.token_dim, init_rois)
        # Focusing transformer stage
        self.focusing_transformer = nn.ModuleList([
            SparseFocusingTransformer(
                token_dim=self.token_dim,
                num_points=self.sampling_points,
                image_size=self.image_size,
                feature_map_size=(self.image_size//4, self.image_size//4)
            ) for _ in range(self.focusing_layers)
        ])
        # Cortex transformer
        self.cortex_transformer = CortexTransformerEncoder(
            embed_dim=self.token_dim,
            num_layers=self.cortex_layers
        )
        # Classification head
        self.head = nn.Linear(self.token_dim, 1000)
    def _initialize_rois(self):
        """
        Initialize RoIs to cover the image on a grid.
        """
        # For simplicity, we initialize grid centered at uniform points
        n_grid = int(math.sqrt(self.num_tokens))
        coords = torch.linspace(0.1, 0.9, n_grid)
        centers_x, centers_y = torch.meshgrid(coords, coords)
        centers_x = centers_x.contiguous().view(-1)
        centers_y = centers_y.contiguous().view(-1)
        widths = torch.full_like(centers_x, 0.5)
        heights = torch.full_like(centers_y, 0.5)
        rois = torch.stack([centers_x, centers_y, widths, heights], dim=1)  # [N,4]
        return rois
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W]
        Returns:
            logits: [B, 1000]
        """
        B = images.shape[0]
        # Extract shared feature map
        feature_map = self.backbone(images)  # [B, C, H/4, W/4]
        # Prepare tokens and RoIs
        t = self.token_set.get_embeddings()  # [N, d_c]
        rois = self.token_set.get_rois()     # [N, 4]
        # Repeat for batch
        token_embeddings = t.unsqueeze(0).expand(B, -1, -1)  # [B, N, d_c]
        rois_batch = rois.unsqueeze(0).expand(B, -1, -1)  # [B, N, 4]

        # Initialize tokens for this batch
        tokens = token_embeddings  # [B, N, d_c]
        rois = rois_batch  # [B, N, 4]

        # Focus stages
        for stage_idx in range(self.focusing_layers):
            stage_fn = self.focusing_transformer[stage_idx]
            new_tokens_list = []
            new_rois_list = []
            for b in range(B):
                # For each batch element
                tokens_b = tokens[b]  # [N, d_c]
                rois_b = rois[b]      # [N, 4]
                # Apply focusing transformer stage
                new_tokens, new_rois = stage_fn(tokens_b, rois_b, feature_map[b:b+1])
                new_tokens_list.append(new_tokens)
                new_rois_list.append(new_rois)
            tokens = torch.stack(new_tokens_list, dim=0)  # [B, N, d_c]
            rois = torch.stack(new_rois_list, dim=0)      # [B, N, 4]

        # Prepare tokens for cortex transformer
        tokens = tokens  # [B, N, d_c]
        # Reshape for transformer (batch, seq, embed)
        tokens = tokens
        # Process with cortex transformer
        tokens = self.cortex_transformer(tokens)  # [B, N, d_c]
        # Readout: average over tokens
        pooled = tokens.mean(dim=1)  # [B, d_c]
        logits = self.head(pooled)   # [B, 1000]
        return logits
```

## train.py

```python
## train.py
import os
import yaml
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from dataset import ImageNetDataset
from model import SparseFormer
from utils import get_lr_scheduler, setup_logging, save_checkpoint, initialize_weights
import time
from typing import Dict

def main():
    # -----------------------------------------------------
    # 1. Configuration Parsing
    # Load config.yaml
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config: Dict = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Fix seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize logging
    from torch.utils.tensorboard import SummaryWriter
    writer = setup_logging()

    # -----------------------------------------------------
    # 2. Dataset and DataLoader Preparation
    print("Preparing datasets and dataloaders...")
    train_dataset = ImageNetDataset(config, split='train')
    val_dataset = ImageNetDataset(config, split='val')
    batch_size = config['training'].get('batch_size', 128)
    num_workers = config['dataset'].get('num_workers', 8)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # -----------------------------------------------------
    # 3. Model Initialization
    print("Initializing model...")
    model = SparseFormer(config)
    model.to(device)

    # Initialize weights (if training from scratch)
    model.apply(initialize_weights)

    # Load checkpoint if resuming or pretraining
    resume = config['saving'].get('resume_from_checkpoint', False)
    checkpoint_path = os.path.join(config['saving'].get('checkpoint_dir', './checkpoints/'), 'latest.pth')

    start_epoch = 1
    best_acc1 = 0.0

    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else None
        start_epoch = checkpoint.get('epoch', 1)
        best_acc1 = checkpoint.get('best_acc1', 0.0)
        print(f"Resumed at epoch {start_epoch}")

    # -----------------------------------------------------
    # 4. Loss function, optimizer, LR scheduler
    print("Setting up loss, optimizer, scheduler...")
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(),
                            lr=config['training']['optimizer'].get('lr', 1e-3),
                            weight_decay=config['training'].get('weight_decay', 0.05))
    total_steps = int(len(train_loader) * config['training']['epochs'])
    scheduler = get_lr_scheduler(optimizer, config, total_steps)

    # Use GradScaler for AMP
    scaler = amp.GradScaler(enabled=config['training'].get('mixed_precision', False))

    # Optional gradient clipping
    clip_norm = config['training'].get('gradient_clip_norm', 0.0)

    # -----------------------------------------------------
    # 5. Training Loop
    print("Starting training...")
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with amp.autocast(enabled=config['training'].get('mixed_precision', False)):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Optional gradient clipping
            if clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Compute metrics
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
            running_loss += loss.item() * images.size(0)

            # Logging every few iterations
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch}/{config['training']['epochs']}], Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
                # Log to TensorBoard
                global_step = (epoch - 1) * len(train_loader) + batch_idx + 1
                writer.add_scalar('train/loss', loss.item(), global_step)
                acc = 100.0 * correct / total
                writer.add_scalar('train/accuracy', acc, global_step)

        train_loss = running_loss / len(train_dataset)
        train_acc = 100.0 * correct / total
        print(f"Epoch [{epoch}] training loss: {train_loss:.4f}, accuracy: {train_acc:.2f}%")
        writer.add_scalar('train/epoch_loss', train_loss, epoch)
        writer.add_scalar('train/epoch_accuracy', train_acc, epoch)
        # Save checkpoint periodically
        if epoch % config['saving'].get('save_every_epochs', 10) == 0:
            save_path = os.path.join(config['saving'].get('checkpoint_dir', './checkpoints/'), f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc1': best_acc1
            }, save_path)
            print(f"Saved checkpoint: {save_path}")

        # -------------------------------------------------
        # 6. Validation and Evaluation
        print(f"Running validation for epoch {epoch}...")
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True)
                with amp.autocast(enabled=config['training'].get('mixed_precision', False)):
                    val_outputs = model(val_images)
                    loss_val = criterion(val_outputs, val_labels)
                val_loss += loss_val.item() * val_images.size(0)
                # Accuracy
                _, val_preds = torch.max(val_outputs, 1)
                val_correct += torch.sum(val_preds == val_labels).item()
                val_total += val_labels.size(0)

            val_epoch_loss = val_loss / len(val_dataset)
            val_acc1 = 100.0 * val_correct / val_total
            print(f"Validation Epoch [{epoch}] Loss: {val_epoch_loss:.4f}, Accuracy: {val_acc1:.2f}%")
            writer.add_scalar('val/loss', val_epoch_loss, epoch)
            writer.add_scalar('val/accuracy', val_acc1, epoch)

        # Save best model
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            best_model_path = os.path.join(config['saving'].get('checkpoint_dir', './checkpoints/'), 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch} with accuracy {best_acc1:.2f}%.")

        print(f"Epoch {epoch} completed in {(time.time() - epoch_start_time):.2f} seconds.\n")

    # -----------------------------------------------------
    # 7. Finish
    print("Training completed.")
    print(f"Best validation accuracy: {best_acc1:.2f}%")
    # Save final model
    final_checkpoint_path = os.path.join(config['saving'].get('checkpoint_dir', './checkpoints/'), 'final.pth')
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'best_acc1': best_acc1
    }, final_checkpoint_path)
    print(f"Final model saved at {final_checkpoint_path}")

if __name__ == "__main__":
    main()
```

## utils.py

```python
## utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def bilinear_sample(feature_map: torch.Tensor, sampling_points: torch.Tensor) -> torch.Tensor:
    """
    Sample features from a feature map at given normalized sampling points using bilinear interpolation.
    
    Args:
        feature_map: Tensor of shape [B, C, H, W]
        sampling_points: Tensor of shape [N, P, 2], with values in [0,1], representing normalized (x, y) locations.
    
    Returns:
        sampled_features: Tensor of shape [N, P, C]
    """
    B, C, H, W = feature_map.shape
    N, P, _ = sampling_points.shape
    
    # Convert normalized coords to absolute pixel coordinates
    x = sampling_points[:,:,0] * (W - 1)
    y = sampling_points[:,:,1] * (H - 1)
    
    # Compute coordinates for the four neighbors
    x0 = x.floor().clamp(0, W - 1)
    y0 = y.floor().clamp(0, H - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    y1 = (y0 + 1).clamp(0, H - 1)
    
    # Gather pixel values at four corners
    # Expand dims for broadcasting
    B_idx = torch.arange(B, device=feature_map.device).view(-1, 1, 1)
    # For batch index, sample the same batch for all points
    # Reshape for gather
    def gather_pixel(x_idx, y_idx):
        """
        Gather pixel values at (x_idx, y_idx) locations for each batch.
        """
        # shape: [N, P]
        grid = torch.stack([y_idx, x_idx], dim=2)  # [N, P, 2]
        # Normalize to [-1,1] for grid_sample: not directly used here, so do direct indexing
        # Instead, do pixel gather:
        # flatten index
        flatten_idx = (grid[:,:,0] * W + grid[:,:,1]).long()  # [N, P]
        # Expand for batch dims
        pixel_vals = []
        for b in range(B):
            fmap = feature_map[b]
            fmap_flat = fmap.view(C, -1)  # [C, H*W]
            vals = fmap_flat[:, flatten_idx.view(-1)]  # [C, N*P]
            vals = vals.view(C, N, P).permute(1, 2, 0)  # [N, P, C]
            pixel_vals.append(vals)
        pixel_vals = torch.stack(pixel_vals, dim=0)  # [B, N, P, C]
        return pixel_vals
    
    # Gather pixel values for each neighbor
    Ia = gather_pixel(x0, y0)
    Ib = gather_pixel(x1, y0)
    Ic = gather_pixel(x0, y1)
    Id = gather_pixel(x1, y1)
    
    # Compute interpolation weights
    wx = (x - x0)
    wy = (y - y0)
    
    wx = wx.unsqueeze(-1)  # [N, P, 1]
    wy = wy.unsqueeze(-1)
    
    # Interpolate
    # shape: [B, N, P, C]
    sampled = (Ia * (1 - wx) * (1 - wy) +
               Ib * wx * (1 - wy) +
               Ic * (1 - wx) * wy +
               Id * wx * wy)
    return sampled

def generate_sampling_offsets(token_embedding: torch.Tensor, P: int, device: torch.device) -> torch.Tensor:
    """
    Generate relative sampling offsets conditioned on token embedding.
    
    Args:
        token_embedding: [N, D]
        P: int, number of sampling points
        device: torch device
    
    Returns:
        offsets: [N, P, 2], relative offsets
    """
    linear_layer = nn.Linear(token_embedding.shape[1], 2 * P).to(device)
    offsets = linear_layer(token_embedding)  # [N, 2*P]
    offsets = offsets.view(-1, P, 2)  # [N, P, 2]
    # Normalize offsets to roughly 3 std deviations as in paper
    # Here, just return raw; normalization can be applied externally
    return offsets

def convert_offsets_to_points(rois: torch.Tensor, rel_offsets: torch.Tensor) -> torch.Tensor:
    """
    Convert relative offsets conditioned on RoIs to absolute sampling points in normalized [0,1] coords.
    
    Args:
        rois: [N, 4], (x, y, w, h), normalized
        rel_offsets: [N, P, 2], (delta_x, delta_y), possibly normalized
    
    Returns:
        sampling_points: [N, P, 2], absolute locations in [0,1]
    """
    x, y, w, h = rois[:,0], rois[:,1], rois[:,2], rois[:,3]
    # Expand to shape [N, P]
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    w = w.unsqueeze(1)
    h = h.unsqueeze(1)
    delta_x = rel_offsets[:,:,0]
    delta_y = rel_offsets[:,:,1]
    # Compute absolute locations
    abs_x = x + 0.5 * delta_x * w
    abs_y = y + 0.5 * delta_y * h
    # Clamp to [0,1]
    abs_x = abs_x.clamp(0,1)
    abs_y = abs_y.clamp(0,1)
    sampling_points = torch.stack([abs_x, abs_y], dim=2)  # [N, P, 2]
    return sampling_points

def generate_roi_deltas(token_embedding: torch.Tensor) -> torch.Tensor:
    """
    Generate RoI adjustment deltas from token embedding.
    Following equation:
    (Δx, Δy, Δw, Δh) = Linear(t)
    """
    linear_layer = nn.Linear(token_embedding.shape[1], 4).to(token_embedding.device)
    delta = linear_layer(token_embedding)  # [N, 4]
    return delta

def update_rois(rois: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Update RoIs based on deltas using:
    x' = x + Δt_x * w
    y' = y + Δt_y * h
    w' = w * exp(Δt_w)
    h' = h * exp(Δt_h)
    
    Args:
        rois: [N, 4]
        delta: [N,4]
    Returns:
        new_rois: [N,4], clamped to [0,1]
    """
    x, y, w, h = rois[:,0], rois[:,1], rois[:,2], rois[:,3]
    Δx, Δy, Δw, Δh = delta[:,0], delta[:,1], delta[:,2], delta[:,3]
    x_new = x + Δx * w
    y_new = y + Δy * h
    w_new = w * torch.exp(Δw)
    h_new = h * torch.exp(Δh)
    # Clamp to [0,1]
    x_new = x_new.clamp(0,1)
    y_new = y_new.clamp(0,1)
    w_new = w_new.clamp(0,1)
    h_new = h_new.clamp(0,1)
    new_rois = torch.stack([x_new, y_new, w_new, h_new], dim=1)
    return new_rois

def initialize_weights(module: nn.Module, std: float = 0.02) -> None:
    """
    Initialize weights of linear, conv, and other layers with Xavier or normal.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'weight') and hasattr(module, 'bias'):
        if module.weight is not None:
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)

def get_lr_scheduler(optimizer: torch.optim.Optimizer, config: dict, total_steps: int) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build cosine warmup scheduler as per config.
    """
    warmup_epochs = config['training'].get('warmup_epochs', 5)
    def lr_lambda(current_step):
        warmup_steps = warmup_epochs * total_steps // config['training'].get('epochs', 1)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

def setup_logging():
    """
    Setup logging for training, including TensorBoard.
    """
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    return writer

def save_checkpoint(state: dict, filename: str):
    """
    Save model checkpoint.
    """
    torch.save(state, filename)

def load_checkpoint(filename: str) -> dict:
    """
    Load model checkpoint.
    """
    return torch.load(filename, map_location='cpu')
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\sparseformer\sparseformer_repo`
