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
