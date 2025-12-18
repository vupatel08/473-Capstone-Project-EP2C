# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset.py

```python
## dataset.py

import os
import glob
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A

# Disable warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

class PascalVOCScribbleDataset(Dataset):
    """
    Pascal VOC 2012 Dataset with Scribble Annotations for Weakly Supervised Semantic Segmentation.
    Loads images, scribble masks, and (optionally) ground truth labels.
    Supports data augmentation suitable for training, and can serve validation data without augmentation.
    """

    def __init__(
        self,
        data_dir: str = "path/to/VOC2012",
        scribble_dir: str = "path/to/ScribbleAnnotations",
        split: str = "train",
        image_size: int = 512,
        transforms: Optional[Callable] = None,
        is_train: bool = True,
        return_mask: bool = False,
        ignore_index: int = 255,
    ):
        """
        Args:
            data_dir (str): Root directory of VOC dataset.
            scribble_dir (str): Directory containing scribble annotation masks.
            split (str): 'train', 'val', or 'test'.
            image_size (int): Resize/crop size for images.
            transforms (callable, optional): Albumentations augmentation pipeline.
            is_train (bool): Flag indicating train or eval mode.
            return_mask (bool): During validation, whether to return ground truth labels.
            ignore_index (int): The label value to ignore during loss computation.
        """
        super().__init__()
        self.data_dir = data_dir
        self.scribble_dir = scribble_dir
        self.split = split
        self.image_size = image_size
        self.transforms = transforms
        self.is_train = is_train
        self.return_mask = return_mask
        self.ignore_index = ignore_index

        # Initialize list of image IDs based on Pascal VOC structure
        self.image_ids = self._init_image_ids()

        # Define image and annotation paths
        self.image_paths = self._get_image_paths()
        self.scribble_paths = self._get_scribble_paths()
        self.gt_paths = self._get_gt_paths() if return_mask else None

        assert len(self.image_paths) == len(self.scribble_paths), \
            "Mismatch between images and scribble annotations"

        if self.return_mask:
            assert len(self.image_paths) == len(self.gt_paths), \
                "Mismatch between images and ground truth labels"

        # Define albumentations pipeline for training data
        if self.is_train:
            self.augmentation = A.Compose([
                A.Resize(height=self.image_size, width=self.image_size),
                A.RandomScale(scale_limit=(0.5, 2.0), p=1.0),
                A.Rotate(limit=10, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.RandomCrop(height=self.image_size, width=self.image_size),
            ])
        else:
            # For validation or inference, only resize for consistency
            self.augmentation = A.Compose([
                A.Resize(height=self.image_size, width=self.image_size),
            ])

    def _init_image_ids(self) -> List[str]:
        """
        Initialize list of image IDs based on the dataset split.
        Assumes standard Pascal VOC directory structure.
        """
        split_file = os.path.join(self.data_dir, "ImageSets", "Segmentation", f"{self.split}.txt")
        with open(split_file, 'r') as f:
            ids = [line.strip() for line in f.readlines()]
        return ids

    def _get_image_paths(self) -> List[str]:
        """
        Get list of image file paths for the dataset split.
        Assumes images stored under 'JPEGImages' folder.
        """
        image_paths = []
        for img_id in self.image_ids:
            img_path = os.path.join(self.data_dir, "JPEGImages", f"{img_id}.jpg")
            if os.path.exists(img_path):
                image_paths.append(img_path)
            else:
                # Alternatively, handle other image extensions if needed
                img_path = os.path.join(self.data_dir, "JPEGImages", f"{img_id}.png")
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                else:
                    raise FileNotFoundError(f"Image file not found for ID {img_id}")
        return image_paths

    def _get_scribble_paths(self) -> List[str]:
        """
        Get list of scribble mask paths.
        Assumes scribble masks stored under scribble_dir with same filenames as images.
        """
        scribble_paths = []
        for img_path in self.image_paths:
            filename = os.path.basename(img_path).replace('.jpg', '').replace('.png', '')
            scribble_path = os.path.join(self.scribble_dir, self.split, f"{filename}.png")
            if os.path.exists(scribble_path):
                scribble_paths.append(scribble_path)
            else:
                raise FileNotFoundError(f"Scribble annotation not found for {filename}")
        return scribble_paths

    def _get_gt_paths(self) -> List[str]:
        """
        Get ground truth label paths for validation.
        Assumes stored under 'SegmentationClass' folder.
        """
        gt_paths = []
        for img_path in self.image_paths:
            filename = os.path.basename(img_path).replace('.jpg', '').replace('.png', '')
            gt_path = os.path.join(self.data_dir, 'SegmentationClass', f"{filename}.png")
            if os.path.exists(gt_path):
                gt_paths.append(gt_path)
            else:
                raise FileNotFoundError(f"Ground truth not found for {filename}")
        return gt_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # Load image
        img_path = self.image_paths[index]
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load scribble mask
        scribble_path = self.scribble_paths[index]
        scribble_mask_img = np.array(Image.open(scribble_path))
        # Convert scribble mask image to class labels
        # Assuming scribble masks are stored as PNG with class IDs per pixel
        # For unlabeled pixels, assume pixel value == 255 (common practice)
        label_mask = np.full_like(scribble_mask_img, self.ignore_index, dtype=np.int64)
        # Mask out ignored pixels
        label_mask[scribble_mask_img != 255] = scribble_mask_img[scribble_mask_img != 255]

        # For validation, load full ground truth labels if needed
        gt_label = None
        if self.return_mask:
            gt_path = self.gt_paths[index]
            gt_mask_np = np.array(Image.open(gt_path))
            gt_label = gt_mask_np.astype(np.int64)

        # Convert to dict for augmentation
        augmented_data = self.augmentation(image=image, mask=label_mask)

        img_tensor = augmented_data['image'].astype(np.float32) / 255.0  # normalize to [0,1]
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).contiguous()  # (C,H,W)

        label_tensor = torch.from_numpy(augmented_data['mask']).long()

        sample = {
            'image': img_tensor,
            'label': label_tensor,
            'image_path': img_path,
        }

        if self.return_mask:
            sample['ground_truth'] = torch.from_numpy(gt_label)

        return sample

# Note:
# - The class expects that your dataset directory has the standard Pascal VOC structure with
#   'JPEGImages/', 'ImageSets/Segmentation/', 'SegmentationClass/', and separate scribble annotation directories.
# - The file paths and labels should be consistent with the experiment setup.
# - For actual use, consider adding __repr__() and optional caching for efficiency.

```

## evaluation.py

```python
## evaluation.py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, List
import time
import json
import os

def evaluate(
    model,
    val_loader,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    config: Dict = None,
    progress_display: bool = True
) -> Dict:
    """
    Evaluate the trained model on the validation set, compute per-class and mean IoU.
    
    Args:
        model (nn.Module): Trained segmentation model.
        val_loader (DataLoader): DataLoader for validation dataset.
        device (torch.device): computation device.
        config (dict): Configuration dictionary loaded from 'config.yaml'.
        progress_display (bool): Whether to print progress info.

    Returns:
        Dict: contains per_class_iou (dict) and mean_iou (float).
    """

    # Set model to eval
    model.eval()

    # Load relevant config parameters with defaults
    use_prototypes: bool = True
    class_guidance: bool = False
    image_size: int = 512
    num_classes: int = 21

    if config is not None:
        use_prototypes = config.get('inference', {}).get('use_prototypes', True)
        class_guidance = config.get('inference', {}).get('class_guidance', False)

    # Prepare accumulators
    total_inter = np.zeros((num_classes,), dtype=np.float64)
    total_union = np.zeros((num_classes,), dtype=np.float64)

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch['image'].to(device)   # shape: [B, 3, H, W]
            labels = batch['ground_truth'].cpu().numpy()  # shape: [B, H, W]

            # === Inference ===
            if use_prototypes:
                # During inference with prototypes, can incorporate prototype guidance
                # For simplicity, in this implementation, we assume no special prototype-guided steps
                # For task-specific application, you can extend this part to retrieve prototypes 
                # and perform feature augmentation as in training.
                # Here, we just perform standard forward.
                preds_logits, extra_outputs = model(images)
            else:
                preds_logits, extra_outputs = model(images)

            # Convert logits to predicted labels
            preds_label = torch.argmax(preds_logits, dim=1)  # shape: [B, H, W]
            preds_label_np = preds_label.cpu().numpy()

            # Compute per-image metrics
            for i in range(images.size(0)):
                pred_mask = preds_label_np[i]
                gt_mask = labels[i]
                # Compute for each class
                for c in range(num_classes):
                    pred_c = (pred_mask == c)
                    gt_c = (gt_mask == c)

                    # Ignore unlabeled (ignore_index=255), which we skip in IoU
                    # But in ground truth, labels with 255 are ignored.
                    # For IoU, only compute on pixels with valid GT labels (i.e., not 255).
                    valid_mask = (gt_mask != 255)
                    pred_c_valid = pred_c[valid_mask]
                    gt_c_valid = gt_c[valid_mask]

                    intersection = np.logical_and(pred_c_valid, gt_c_valid).sum()
                    union = np.logical_or(pred_c_valid, gt_c_valid).sum()

                    total_inter[c] += intersection
                    total_union[c] += union

    # Compute per-class IoU
    ious = total_inter / (total_union + 1e-6)
    per_class_iou = {}
    for c in range(num_classes):
        per_class_iou[str(c)] = float(ious[c])
    mean_iou = float(np.nanmean(ious))

    end_time = time.time()
    total_time = end_time - start_time

    if progress_display:
        print(f"Evaluation completed in {total_time:.2f} seconds")
        print("Per-class IoU:")
        for c in range(num_classes):
            print(f"  Class {c:2d}: {per_class_iou[str(c)]]:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}%")

    # Optionally, save detailed report (e.g., JSON)
    results = {
        "per_class_iou": per_class_iou,
        "mean_iou": mean_iou,
        "total_time_sec": total_time
    }
    return results
```

## main.py

```python
## main.py
import os
import torch
import yaml
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from evaluation import evaluate
from dataset import PascalVOCScribbleDataset
from model import ScribbleSegModel

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Extract training configs with defaults
    training_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    prototypes_cfg = config.get('prototypes', {})
    inference_cfg = config.get('inference', {})

    # Set hyperparameters and paths
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset setup
    dataset_path = data_cfg.get('dataset_path', 'path/to/VOC2012')
    scribble_path = data_cfg.get('scribble_annotations_path', 'path/to/ScribbleAnnotations')
    image_size = data_cfg.get('image_size', 512)

    # Instantiate datasets
    train_dataset = PascalVOCScribbleDataset(
        data_dir=dataset_path,
        scribble_dir=scribble_path,
        split='train',
        image_size=image_size,
        transforms=None,  # can define augmentations here if needed
        is_train=True,
        return_mask=False
    )
    val_dataset = PascalVOCScribbleDataset(
        data_dir=dataset_path,
        scribble_dir=scribble_path,
        split='val',
        image_size=image_size,
        transforms=None,
        is_train=False,
        return_mask=True
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get('batch_size', 16),
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Instantiate model
    model = ScribbleSegModel(config)
    model.to(device)
    model.train()

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=training_cfg.get('learning_rate', 3e-5))
    lr_decay_epoch = training_cfg.get('lr_decay_epoch', 80)
    lr_decay_factor = training_cfg.get('lr_decay_factor', 0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_decay_epoch], gamma=lr_decay_factor)

    # Instantiate prototype memory bank and tracking
    proto_num_per_class = model_cfg.get('proto_num_per_class', 5)
    proto_momentum = model_cfg.get('proto_momentum', 0.99)
    prototype_bank = model.PrototypeMemoryBank(
        num_classes=21,
        prototypes_per_class=proto_num_per_class,
        feature_dim=256,
        momentum=proto_momentum
    )

    # Training setup
    warmup_epochs = prototypes_cfg.get('warmup_epochs', 10)
    total_epochs = training_cfg.get('epochs', 100)
    use_prototypes_guidance = inference_cfg.get('use_prototypes', True)

    # Training phase flags
    global_prototypes_full = False
    current_phase = 'warmup'  # will change: 'warmup', 'local_proto', 'full_proto'

    # For logging
    best_mIoU = 0.0
    save_path = 'checkpoints'
    os.makedirs(save_path, exist_ok=True)

    print("Starting training...")

    for epoch in range(1, total_epochs + 1):
        # Manage training phase transition
        if epoch <= warmup_epochs:
            current_phase = 'warmup'
        else:
            # Decide if global prototypes are full
            if not global_prototypes_full:
                global_prototypes_full = all(proto_bank.full_mask.tolist())
                if global_prototypes_full:
                    current_phase = 'full_proto'
                else:
                    current_phase = 'local_proto'
            else:
                current_phase = 'full_proto'

        # Adjust loss weights
        if current_phase == 'warmup':
            loss_weights = {'pce': 1.0, 'con_l': 0.0, 'con_g': 0.0}
        elif current_phase == 'local_proto':
            loss_weights = {'pce': 1.0, 'con_l': 0.02, 'con_g': 0.0}
        else:
            loss_weights = {'pce': 1.0, 'con_l': 0.02, 'con_g': 0.05}

        print(f"\nEpoch {epoch}/{total_epochs} - Phase: {current_phase}")

        epoch_loss_pce = 0.0
        epoch_loss_con_l = 0.0
        epoch_loss_con_g = 0.0
        epoch_start_time = time.time()

        model.train()

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)  # [B,3,H,W]
            labels = batch['label'].to(device)  # [B,H,W], ignore=255

            # Forward pass
            preds, extra_outputs = model(images, labels=labels, training_phase=current_phase)
            # Compute partial cross-entropy
            loss_pce = torch.nn.functional.cross_entropy(
                preds.permute(0,2,3,1)[labels != 255],
                labels[labels != 255],
                reduction='mean'
            ) if (labels != 255).sum() > 0 else torch.tensor(0.0, device=device)
            total_loss = loss_weights['pce'] * loss_pce

            # Extract features for prototypes
            feats = model.encoder.extract_features(images)
            feats_proj = [layer(feats[i]) for i, layer in enumerate(model.proj_layers)]
            last_feat = feats_proj[-1]
            pred_probs = torch.nn.functional.softmax(preds, dim=1)

            # Prototype extraction & update (except during warmup)
            if current_phase != 'warmup':
                # Determine class set present
                class_mask = labels.unique()
                class_mask = class_mask[class_mask != 255]
                class_list = class_mask.tolist()

                # Extract local prototypes
                local_proto_dict = model.prototype_extractor.compute_prototypes(
                    last_feat, pred_probs, labels, class_list)
                # Store local prototypes in model (simulate)
                model.local_prototypes = local_proto_dict

                # Update global prototypes if in 'full_proto'
                if current_phase == 'full_proto':
                    class_idxs = []
                    proto_list = []
                    for c in class_list:
                        if c in local_proto_dict:
                            class_idxs.append(c)
                            proto_list.append(local_proto_dict[c])
                    if class_idxs:
                        # Update memory bank and get global prototypes
                        model.proto_bank.update(class_idxs, torch.stack(proto_list))
                        model.global_prototypes = model.proto_bank.get()
                        # Check if fill complete
                        global_prototypes_full = all(model.proto_bank.full_mask.tolist())

            # Augment features from prototypes if enabled
            feats_for_aug = feats_proj.copy()
            if use_prototypes_guidance and (current_phase != 'warmup'):
                # Local prototypes augmentation
                for c, proto in model.local_prototypes.items():
                    class_mask_map = (labels == c).to(torch.long).squeeze(1)  # [B,H,W]
                    for lvl in range(len(feats_for_aug)):
                        feats_for_aug[lvl] = model.local_augmenter.augment_with_prototypes(
                            feats_for_aug[lvl], proto.unsqueeze(0), class_mask_map)
                # Global prototypes augmentation
                if global_prototypes_full:
                    for c in range(21):
                        proto_set = model.global_prototypes[c]  # [K,D]
                        class_mask_map = (labels == c).to(torch.long).squeeze(1)
                        for lvl in range(len(feats_for_aug)):
                            feats_for_aug[lvl] = model.global_augmenter.augment_with_prototypes(
                                feats_for_aug[lvl], proto_set, class_mask_map)

            # Get augmented predictions
            preds_aug, _ = model.decoder(feats_for_aug), None

            # Compute consistency loss if not warmup
            if current_phase != 'warmup':
                probs_initial = torch.nn.functional.softmax(preds.detach(), dim=1)
                probs_aug = torch.nn.functional.softmax(preds_aug, dim=1)
                loss_con_l = torch.nn.functional.mse_loss(probs_initial, probs_aug)
                total_loss += loss_weights['con_l'] * loss_con_l
                loss_con_g = 0.0
                if global_prototypes_full:
                    # Optional global constraint
                    probs_initial_g = probs_initial
                    probs_aug_g = probs_aug
                    loss_con_g = torch.nn.functional.mse_loss(probs_initial_g, probs_aug_g)
                    total_loss += loss_weights['con_g'] * loss_con_g
            else:
                loss_con_l = torch.tensor(0.0, device=device)
                loss_con_g = torch.tensor(0.0, device=device)

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            epoch_loss_pce += loss_pce.item()
            epoch_loss_con_l += loss_con_l.item() if hasattr(loss_con_l, 'item') else loss_con_l
            epoch_loss_con_g += loss_con_g.item() if hasattr(loss_con_g, 'item') else loss_con_g

        # LR schedule
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        # Validation
        if epoch % 5 == 0 or epoch == total_epochs:
            miou = evaluate(model, val_loader, device=device, config=config)
            print(f"Validation mIoU at epoch {epoch}: {miou:.2f}%")
            # Save best checkpoint
            if miou > best_mIoU:
                best_mIoU = miou
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                print(f"New best model saved with mIoU: {best_mIoU:.2f}%")

    print("Training completed.")

    # Final evaluation with best model
    print("Loading best model for final evaluation.")
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))
    final_mIoU = evaluate(model, val_loader, device=device, config=config)
    print(f"Final validation mIoU: {final_mIoU:.2f}%")
    # Additional visualization can be added here if necessary

if __name__ == '__main__':
    main()
```

## model.py

```python
# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Dict, Optional, Tuple
import timm

class SegformerEncoder(nn.Module):
    """
    Segformer (MiT-B1) encoder based on timm's implementation.
    Extracts multi-scale features from the backbone.
    """
    def __init__(self, backbone_name: str = "mit_b1", pretrained: bool = True):
        super().__init__()
        # Load a pre-trained Segformer model from timm
        # Using timm, which supports 'mit_b1' as pre-trained model
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        # The backbone outputs features from 4 stages with different resolutions
        # For modularity, define feature extraction layers
        # We assume the backbone provides 'forward_features' method or similar
        # ttm's models may need custom extraction; here, rely on its standard interface
        # For simplicity, assume 'forward_features' returns list of features
        # If not, need to modify accordingly
        self.out_channels = [64, 128, 320, 512]  # for MiT-B1
    
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Use the backbone's forward_features method
        features = self.backbone.forward_features(x)
        # features is a list of tensors: [F1, F2, F3, F4]
        # Ensure they are in order from high resolution to low resolution
        return features


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder for semantic segmentation.
    Fuses multi-scale features and refines into logits.
    """
    def __init__(self, num_classes: int = 21, feature_channels: List[int] = [64, 128, 320, 512]):
        super().__init__()
        # We will fuse multi-level features
        # Upsample lower resolution features and concatenate
        # For simplicity, use a series of convolutions and transposed convs
        self.conv1 = nn.Conv2d(feature_channels[-1], 256, kernel_size=1)
        self.conv2 = nn.Conv2d(feature_channels[-2], 256, kernel_size=1)
        self.conv3 = nn.Conv2d(feature_channels[-3], 256, kernel_size=1)
        self.conv4 = nn.Conv2d(feature_channels[-4], 256, kernel_size=1)
        
        # Decoder layers (could be transformer blocks, here simple convs)
        self.decode_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def fuse_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        '''
        Fuse multi-scale features into a single high-resolution feature map.
        '''
        # Expect features: [F1, F2, F3, F4]
        # Upsample lower resolution features to F1 size
        size = features[0].size()[2:]  # H, W
        upsampled = []
        for i, feat in enumerate(features):
            if feat.size()[2:] != size:
                feat = F.interpolate(feat, size=size, mode='bilinear', align_corners=False)
            upsampled.append(feat)
        # Concatenate all features along channel dimension
        fused = torch.cat(upsampled, dim=1)  # shape: (B, sum of channels, H, W)
        return fused

    def forward(self, features: List[torch.Tensor], prototypes: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        '''
        Forward pass, with optional augmentation via prototypes.
        '''
        fused = self.fuse_features(features)
        # Optionally, apply prototype-guided augmentation (implemented externally)
        # For core decoder, simply pass through
        logits = self.decode_head(fused)
        return logits


class PrototypeExtractor(nn.Module):
    """
    Extracts high-confidence feature prototypes for each class.
    """
    def __init__(self, topk_percentage: float = 0.5):
        super().__init__()
        self.topk_percentage = topk_percentage  # e.g., 0.5 for top 50%
        
    def compute_prototypes(self, features: torch.Tensor, predictions: torch.Tensor,
                           labels: torch.Tensor, class_ids: List[int]) -> Dict[int, torch.Tensor]:
        """
        Compute class-wise prototypes from high-confidence pixels.
        Args:
            features: [B, D, H, W]
            predictions: [B, C, H, W]
            labels: [B, H, W], sparse labels with ignore_idx for unlabeled
            class_ids: list of classes present in current batch
        Returns:
            prototypes: dict {class_idx: [K, D]} with K=number of prototypes per class
        """
        B, D, H, W = features.shape
        # Flatten spatial dimensions
        feats_flat = features.view(B, D, -1)  # [B, D, HW]
        pred_flat = predictions.view(B, predictions.shape[1], -1)  # [B, C, HW]
        label_flat = labels.view(B, -1)  # [B, HW]
        
        prototypes = {}
        for c in class_ids:
            class_prototypes = []
            for b in range(B):
                # Create mask for class c, excluding ignore_idx
                mask = (label_flat[b] == c)
                if mask.sum() == 0:
                    continue
                # Select confidence scores for class c
                class_pred_confidence = pred_flat[b, c, :]
                # Get top-k pixels
                k = max(1, int(self.topk_percentage * mask.sum().item()))
                topk_confidence, topk_indices = torch.topk(class_pred_confidence[mask], k=k)
                # Gather features for top pixels
                selected_indices = torch.where(mask)[0][topk_indices]
                sel_feats = feats_flat[b, :, selected_indices]  # [D, K]
                # Compute weighted average
                # Use confidence scores if desired for weighting
                # Here, simple average
                proto = sel_feats.mean(dim=1)  # [D]
                # Normalize
                proto = F.normalize(proto, p=2, dim=0)
                class_prototypes.append(proto)
            if len(class_prototypes) > 0:
                # Average over batch instances
                class_proto = torch.stack(class_prototypes, dim=0).mean(dim=0)  # [D]
                prototypes[c] = class_proto
        return prototypes


class FeatureAugmenter(nn.Module):
    """
    Augments features using prototypes via attention.
    Can be used with local or global prototypes.
    """
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(2 * feature_dim, feature_dim)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Linear(feature_dim, feature_dim)
        
    def augment_with_prototypes(self, features: torch.Tensor, prototypes: torch.Tensor, class_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform feature augmentation with prototypes.
        Args:
            features: [B, D, H, W]
            prototypes: [C, K, D] or [C, D]
            class_mask: [B, H, W], indicating class at each pixel
        Returns:
            augmented_features: [B, D, H, W]
        """
        B, D, H, W = features.shape
        device = features.device

        # Reshape features to [B, D, H*W]
        feat_flat = features.view(B, D, -1)  # [B, D, HW]
        feat_flat_t = feat_flat.permute(0, 2, 1)  # [B, HW, D]

        # For simplicity, assume prototypes are [C, D]
        # For each pixel, get the prototypes of its class using class_mask
        # We'll process cross class, for high efficiency, process per class
        # Create a new tensor for augmented features
        augmented_feat = feat_flat.clone()

        for c in range(prototypes.shape[0]):  # For each class
            class_indices = (class_mask == c).nonzero(as_tuple=False)
            if class_indices.shape[0] == 0:
                continue
            class_proto = prototypes[c]  # [D]
            # Expand to match features of class pixels
            proto_expanded = class_proto.unsqueeze(0).unsqueeze(0)  # [1,1,D]
            class_feat_indices = class_indices[:, 1:]  # [N, 3], spatial coords
            for idx in class_feat_indices:
                b_idx, y, x = idx
                feat_vector = features[b_idx, :, y, x]  # [D]
                # Compute attention weight via dot product
                attn = torch.matmul(feat_vector, class_proto)  # scalar
                # Attention weight via softmax over prototypes of this class
                # For simplicity, use sigmoid or softmax over 1 scalar
                # Here, since only one prototype, set weight
                weight = torch.sigmoid(attn)
                # Interpolated prototype
                proto_weighted = weight * class_proto
                # Linear transform
                feat_aug = self.linear(torch.cat([feat_vector, proto_weighted], dim=0))
                feat_aug = self.relu(feat_aug)
                # Residual connection
                features[b_idx, :, y, x] += self.proj(feat_aug)
        return features


class PrototypeMemoryBank:
    """
    Manages global prototypes for each class, updated with cosine similarity.
    """
    def __init__(self, num_classes: int = 21, prototypes_per_class: int = 5, feature_dim: int = 256,
                 momentum: float = 0.99):
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.feature_dim = feature_dim
        self.momentum = momentum
        # Initialize memory bank: tensor [C, K, D]
        self.prototypes = torch.zeros((self.num_classes, self.prototypes_per_class, self.feature_dim))
        self.full_mask = torch.zeros((self.num_classes,), dtype=torch.bool)  # track if full
        # For simplicity, fill with zeros initially
        self.initialized_counts = torch.zeros((self.num_classes,), dtype=torch.long)
        
    def update(self, class_indices: List[int], new_prototypes: torch.Tensor):
        """
        Update prototypes for given class indices.
        Args:
            class_indices: list of class indices [batch_size]
            new_prototypes: [len(class_indices), D]
        """
        for idx, c in enumerate(class_indices):
            proto = new_prototypes[idx]  # [D]
            if not self.full_mask[c]:
                # Fill remaining slots
                count = self.initialized_counts[c]
                remaining = self.prototypes_per_class - count
                if remaining >= 1:
                    self.prototypes[c, count:count+1, :] = proto.unsqueeze(0)
                    self.initialized_counts[c] += 1
                if self.initialized_counts[c] >= self.prototypes_per_class:
                    self.full_mask[c] = True
            else:
                # Replace the most similar prototype
                bank_protos = self.prototypes[c]  # [K, D]
                # Compute cosine similarity
                sim = F.cosine_similarity(bank_protos, proto.unsqueeze(0), dim=1)  # [K]
                min_sim_idx = torch.argmin(sim)
                # Update with momentum
                self.prototypes[c, min_sim_idx, :] = self.momentum * bank_protos[min_sim_idx] + (1 - self.momentum) * proto
                self.prototypes[c, min_sim_idx, :] = F.normalize(self.prototypes[c, min_sim_idx, :], p=2, dim=0)
    
    def get(self) -> torch.Tensor:
        """
        Return prototypes: shape [C, K, D]
        """
        return self.prototypes


class ScribbleSegModel(nn.Module):
    """
    Complete segmentation model with backbone, decoder, and prototype handling.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.num_classes = config['model'].get('num_classes', 21)
        self.backbone_name = config['model'].get('backbone', 'mit_b1')
        self.prototype_num_per_class = config['model'].get('proto_num_per_class', 5)
        self.prototype_momentum = config['model'].get('proto_momentum', 0.99)
        self.prototype_extraction_topk = config['model'].get('prototype_extraction_topk', 0.5)
        
        # Initialize backbone encoder
        self.encoder = SegformerEncoder(self.backbone_name)
        feat_channels = self.encoder.out_channels  # [64, 128, 320, 512]
        # For this model, previous code assumes features are 256 channels, so unify
        # Apply a projection for each feature to D
        self.feat_dim = 256
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(c, self.feat_dim, kernel_size=1) for c in feat_channels
        ])
        # Initialize decoder
        self.decoder = TransformerDecoder(num_classes=self.num_classes)
        # Initialize prototype extractor
        self.prototype_extractor = PrototypeExtractor(topk_percentage=self.prototype_extraction_topk)
        # Initialize feature augmenters
        self.local_augmenter = FeatureAugmenter(feature_dim=self.feat_dim)
        self.global_augmenter = FeatureAugmenter(feature_dim=self.feat_dim)
        # Initialize global prototype memory bank
        self.global_proto_bank = PrototypeMemoryBank(
            num_classes=self.num_classes,
            prototypes_per_class=self.prototype_num_per_class,
            feature_dim=self.feat_dim,
            momentum=self.prototype_momentum
        )
        # Flags for training phases
        self.global_prototypes_full = False
        self.use_prototypes = True  # can be toggled during training schedule
        # Placeholder for prototypes during training
        self.local_prototypes: Dict[int, torch.Tensor] = {}
        self.global_prototypes: torch.Tensor = torch.zeros(
            (self.num_classes, self.prototype_num_per_class, self.feat_dim),
            device='cpu'
        )
        
    def forward(self, images: torch.Tensor, predictions: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, training_phase: str = 'warmup') -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.
        Args:
            images: [B, 3, H, W]
            predictions: optional initial predictions [B, C, H, W]
            labels: optional labels for prototype extraction
            training_phase: 'warmup' / 'local_proto' / 'full_proto'
        Returns:
            logits: segmentation logits [B, C, H, W]
            extra_outputs: dict with prototypes etc.
        """
        B = images.size(0)
        # Encode
        feats = self.encoder.extract_features(images)  # List of 4 features
        # Project features to fix dimension
        feats_proj = [layer(feats[i]) for i, layer in enumerate(self.proj_layers)]
        # Compose feature tensor storing concatenated features
        # For augmentation, keep separate
        feats_for_aug = [feat for feat in feats_proj]
        # Predicted logits before augmentation
        preds = self.decoder(feats_for_aug)

        # Prepare for prototype extraction
        proto_info = {}
        if self.use_prototypes and (training_phase != 'warmup'):
            with torch.no_grad():
                # Use predictions and features to extract prototypes
                # Only proceed if labels provided
                if labels is not None:
                    # Compute class set present in batch
                    class_mask = labels.unique()
                    class_mask = class_mask[class_mask != 255].tolist()
                else:
                    class_mask = list(range(self.num_classes))
                # Compute class-wise prototypes from high-confidence pixels
                # Use features from the last layer for prototype extraction
                last_feat = feats_for_aug[-1]  # [B, D, H, W]
                pred_probs = F.softmax(preds, dim=1)
                # Extract local prototypes
                local_proto_dict = self.prototype_extractor.compute_prototypes(
                    last_feat, pred_probs, labels if labels is not None else torch.full_like(labels, 255), class_mask
                )
                # Store local prototypes
                self.local_prototypes = local_proto_dict

                # Update global prototypes if in full phase
                if self.global_prototypes_full:
                    class_idxs_empty = []
                    class_protos_tensor = []
                    for c in class_mask:
                        if c in local_proto_dict:
                            class_idxs_empty.append(c)
                            class_protos_tensor.append(local_proto_dict[c])
                    if len(class_idxs_empty) > 0:
                        # update memory bank
                        self.global_proto_bank.update(class_idxs_empty, torch.stack(class_protos_tensor))
                        # Update global prototypes tensor
                        self.global_prototypes = self.global_proto_bank.get()
        # Apply feature augmentation
        augmented_feats = feats_for_aug.copy()

        if self.use_prototypes:
            # For local prototypes
            for c, proto in self.local_prototypes.items():
                # Create class mask map
                class_mask_map = (labels == c).to(torch.long) if labels is not None else None
                if class_mask_map is not None:
                    class_mask_map = class_mask_map.squeeze(1)  # [B, H, W]
                else:
                    # fallback: assign zeros
                    class_mask_map = torch.zeros_like(labels)
                # Augment features for each level
                for lvl in range(len(augmented_feats)):
                    augmented_feats[lvl] = self.local_augmenter.augment_with_prototypes(
                        augmented_feats[lvl], proto.unsqueeze(0), class_mask_map
                    )
            # For global prototypes
            if self.global_prototypes_full:
                for c in range(self.num_classes):
                    proto = self.global_prototypes[c, :, :]  # [K, D]
                    if self.global_proto_bank.full_mask[c]:
                        class_mask_map = (labels == c).to(torch.long) if labels is not None else None
                        if class_mask_map is not None:
                            class_mask_map = class_mask_map.squeeze(1)
                        else:
                            class_mask_map = torch.zeros_like(labels)
                        for lvl in range(len(augmented_feats)):
                            augmented_feats[lvl] = self.global_augmenter.augment_with_prototypes(
                                augmented_feats[lvl], proto, class_mask_map
                            )

        # Recompute logits with augmented features
        final_logits = self.decoder(augmented_feats)

        return final_logits, {
            'local_prototypes': self.local_prototypes,
            'global_prototypes': self.global_prototypes
        }
```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math
import time
import os
import numpy as np

from dataset import PascalVOCScribbleDataset
from model import ScribbleSegModel

class Trainer:
    """
    Manages the training loop for scribble-supervised semantic segmentation with prototype-based feature augmentation.
    Handles prototype extraction, global prototype updates, loss scheduling, and model optimization.
    """
    def __init__(self, 
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 config: Dict[str, Any],
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Load configuration params
        training_cfg = config.get('training', {})
        model_cfg = config.get('model', {})
        loss_cfg = config.get('loss', {})
        proto_cfg = config.get('prototypes', {})
        inference_cfg = config.get('inference', {})

        # Hyperparameters
        self.learning_rate = training_cfg.get('learning_rate', 3e-5)
        self.batch_size = training_cfg.get('batch_size', 16)
        self.epochs = training_cfg.get('epochs', 100)
        self.lr_decay_epoch = training_cfg.get('lr_decay_epoch', 80)
        self.lr_decay_factor = training_cfg.get('lr_decay_factor', 0.01)

        self.num_classes = model_cfg.get('num_classes', 21)
        self.prototypes_per_class = model_cfg.get('proto_num_per_class', 5)
        self.proto_momentum = model_cfg.get('proto_momentum', 0.99)
        self.prototype_topk = model_cfg.get('prototype_extraction_topk', 0.5)

        self.partial_ce_scale = loss_cfg.get('partial_ce_scale', 1.0)
        self.lambda_local = loss_cfg.get('lambda_local', 0.02)
        self.lambda_global = loss_cfg.get('lambda_global', 0.05)

        self.warmup_epochs = proto_cfg.get('warmup_epochs', 10)
        self.interaction_topk = proto_cfg.get('interaction_topk', 0.5)

        self.use_prototypes_in_infer = inference_cfg.get('use_prototypes', True)
        self.class_guidance_in_infer = inference_cfg.get('class_guidance', False)

        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.lr_decay_epoch], gamma=self.lr_decay_factor)

        # Initialize prototype memory bank
        self.proto_bank = PrototypeMemoryBank(
            num_classes=self.num_classes,
            prototypes_per_class=self.prototypes_per_class,
            feature_dim=256,
            momentum=self.proto_momentum
        )

        # Track training phase
        self.current_epoch = 0
        self.global_prototypes_full = False
        self.train_phase = 'warmup'  # can be 'warmup', 'local_proto', 'full_proto'

        # Prototype placeholders
        self.local_prototypes: Dict[int, torch.Tensor] = {}  # {class_idx: tensor [D]}
        self.global_prototypes: torch.Tensor = torch.zeros(
            (self.num_classes, self.prototypes_per_class, 256), device=self.device)

    def _update_training_phase(self):
        """
        Manages transition of phases based on epoch count and prototype bank fill status.
        """
        if self.current_epoch < self.warmup_epochs:
            self.train_phase = 'warmup'
        else:
            # After warm-up, depending on whether global prototypes are full
            if not self.global_prototypes_full:
                # Check if all classes' prototypes in memory are full
                self.global_prototypes_full = self._check_global_prototypes_full()
                if self.global_prototypes_full:
                    self.train_phase = 'full_proto'
                else:
                    self.train_phase = 'local_proto'
            else:
                self.train_phase = 'full_proto'

    def _check_global_prototypes_full(self) -> bool:
        """
        Checks if all class prototypes in memory bank are fully filled.
        """
        return all(self.proto_bank.full_mask.tolist())

    def _adjust_loss_weights(self):
        """
        Returns the current weights of the loss components based on phase.
        """
        if self.train_phase == 'warmup':
            return {'pce': self.partial_ce_scale, 'con_l': 0.0, 'con_g': 0.0}
        elif self.train_phase == 'local_proto':
            return {'pce': self.partial_ce_scale, 'con_l': self.lambda_local, 'con_g': 0.0}
        else:  # 'full_proto'
            return {'pce': self.partial_ce_scale, 'con_l': self.lambda_local, 'con_g': self.lambda_global}

    def train(self):
        """
        Main training loop over epochs.
        """
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            self._update_training_phase()
            loss_weights = self._adjust_loss_weights()

            print(f"Epoch {epoch}/{self.epochs} - Phase: {self.train_phase}")
            epoch_loss_pce = 0.0
            epoch_loss_con_l = 0.0
            epoch_loss_con_g = 0.0
            epoch_miou = 0.0

            self.model.train()
            start_time = time.time()

            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                preds, extra_outputs = self.model(images, labels=labels, training_phase=self.train_phase)

                # Compute partial cross-entropy loss
                loss_pce = self.compute_partial_ce_loss(preds, labels)
                total_loss = loss_weights['pce'] * loss_pce

                # Extract features for prototypes
                feats = self.model.encoder.extract_features(images)
                feats_proj = [layer(feats[i]) for i, layer in enumerate(self.model.proj_layers)]
                last_feat = feats_proj[-1]
                pred_probs = F.softmax(preds, dim=1)

                # Prototype extraction and update
                if self.train_phase != 'warmup':
                    class_mask = labels.unique()
                    class_mask = class_mask[class_mask != 255].tolist()
                    # Extract local prototypes
                    local_proto_dict = self.model.prototype_extractor.compute_prototypes(
                        last_feat, pred_probs, labels, class_mask)
                    self.local_prototypes = local_proto_dict

                    # Update global prototypes if full
                    if self.train_phase == 'full_proto':
                        class_idxs, proto_list = [], []
                        for c in class_mask:
                            if c in local_proto_dict:
                                class_idxs.append(c)
                                proto_list.append(local_proto_dict[c])
                        if len(class_idxs) > 0:
                            self.proto_bank.update(class_idxs, torch.stack(proto_list))
                            self.global_prototypes = self.proto_bank.get()
                            # If all classes are full, set flag
                            if self._check_global_prototypes_full():
                                self.global_prototypes_full = True

                # Prototype-based augmentation if applicable
                feats_for_aug = feats_proj.copy()
                if self.use_prototypes_in_infer and self.train_phase != 'warmup':
                    # Augment with local prototypes
                    for c, proto in self.local_prototypes.items():
                        class_mask_map = (labels == c).to(torch.long).squeeze(1)  # B x H x W
                        for lvl in range(len(feats_for_aug)):
                            feats_for_aug[lvl] = self.model.local_augmenter.augment_with_prototypes(
                                feats_for_aug[lvl], proto.unsqueeze(0), class_mask_map)
                    # Augment with global prototypes if full
                    if self.global_prototypes_full:
                        for c in range(self.num_classes):
                            proto_set = self.global_prototypes[c]  # [K, D]
                            class_mask_map = (labels == c).to(torch.long).squeeze(1)
                            for lvl in range(len(feats_for_aug)):
                                feats_for_aug[lvl] = self.model.global_augmenter.augment_with_prototypes(
                                    feats_for_aug[lvl], proto_set, class_mask_map)

                # Generate augmented predictions
                preds_aug, _ = self.model.decoder(feats_for_aug), None
                # Compute consistency or auxiliary losses
                if self.train_phase != 'warmup':
                    # Obtain probability maps of initial predictions
                    probs_initial = F.softmax(preds.detach(), dim=1)
                    probs_aug = F.softmax(preds_aug, dim=1)
                    loss_con_l = self.compute_consistency_loss(probs_initial, probs_aug)
                    total_loss += loss_weights['con_l'] * loss_con_l
                    loss_con_g = 0.0
                    if self.global_prototypes_full:
                        # Additionally apply global prototype augmentation
                        probs_aug_g, _ = self.model.decoder(feats_for_aug), None
                        probs_initial_g = F.softmax(preds.detach(), dim=1)
                        loss_con_g = self.compute_consistency_loss(probs_initial_g, probs_aug_g)
                        total_loss += loss_weights['con_g'] * loss_con_g
                else:
                    loss_con_l = 0.0
                    loss_con_g = 0.0

                # Backprop and optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_loss_pce += loss_pce.item()
                epoch_loss_con_l += loss_con_l if isinstance(loss_con_l, float) else loss_con_l.item()
                epoch_loss_con_g += loss_con_g if isinstance(loss_con_g, float) else loss_con_g.item()

            # Step LR scheduler
            self.scheduler.step()
            epoch_time = time.time() - start_time
            print(f" Epoch {epoch} completed in {epoch_time:.2f}s")
            # Periodically validate
            if epoch % 5 == 0 or epoch == self.epochs:
                miou = self.evaluate()
                print(f"Validation mIoU at epoch {epoch}: {miou:.2f}%")
            # Save checkpoints if desired
            # (not shown here for brevity)

    def compute_partial_ce_loss(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute partial cross-entropy loss on labeled pixels only.
        """
        # Mask labeled pixels
        mask = (labels != 255)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        preds_masked = preds.permute(0,2,3,1)[mask]  # BxHxWxC -> NxC
        labels_masked = labels[mask]
        loss = F.cross_entropy(preds_masked, labels_masked, reduction='mean')
        return loss

    def compute_consistency_loss(self, prob1: torch.Tensor, prob2: torch.Tensor) -> torch.Tensor:
        """
        Compute mean squared error (MSE) between probability maps.
        """
        return F.mse_loss(prob1, prob2)

    def evaluate(self) -> float:
        """
        Run inference on validation set and compute mean IoU.
        """
        self.model.eval()
        total_inter = np.zeros(self.num_classes)
        total_union = np.zeros(self.num_classes)
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['ground_truth'].to(self.device)
                preds, _ = self.model(images)
                preds_label = torch.argmax(preds, dim=1)
                for i in range(images.size(0)):
                    pred_np = preds_label[i].cpu().numpy()
                    label_np = labels[i].cpu().numpy()
                    # Compute per class intersection and union
                    for c in range(self.num_classes):
                        pred_mask = (pred_np == c)
                        label_mask = (label_np == c)
                        inter = np.logical_and(pred_mask, label_mask).sum()
                        union = np.logical_or(pred_mask, label_mask).sum()
                        total_inter[c] += inter
                        total_union[c] += union
            # Compute per class IoU
            ious = total_inter / (total_union + 1e-6)
            mean_iou = np.nanmean(ious) * 100
        return mean_iou

    # Additional functions and cleanup can be added as needed; including save/load checkpoints.
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\PFA\PFA_repo`
