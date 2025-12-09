## evaluation.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple

# Import dataset loaders, models, prompt, label mapping, and configuration
from dataset_loader import DatasetLoader
from model import PretrainedModel
from prompt_module import PromptGenerator
from label_mapping import LabelMapper

# Optional: For segmentation IoU computation
def compute_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor, num_classes: int) -> float:
    """
    Compute average IoU for batch predictions.
    Args:
        pred_mask: (N, H, W) long tensor with predicted classes
        target_mask: (N, H, W) long tensor with ground truth classes
        num_classes: total number of classes
    Returns:
        average IoU score over batch
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        target_cls = (target_mask == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            # No ground truth and no prediction for this class
            continue
        ious.append(intersection / union)
    if len(ious) == 0:
        return 1.0  # If no classes present, assume perfect
    return sum(ious) / len(ious)

class Evaluation:
    def __init__(self,
                 model: PretrainedModel,
                 prompts: PromptGenerator,
                 dataset_loader: DatasetLoader,
                 label_mapper: LabelMapper,
                 config: Dict[str, Any]):
        """
        Initialize evaluation with model, prompts, dataset loader, label mapper, and config.
        """
        self.model = model
        self.prompts = prompts
        self.dataset_loader = dataset_loader
        self.label_mapper = label_mapper
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        # Dataset info
        self.dataset_name = self.config['dataset']['name']
        self.num_classes = self.config['dataset'].get('num_classes', 10)
        # Visualization directory
        self.vis_dir = self.config.get('logging', {}).get('log_dir', './logs')
        os.makedirs(self.vis_dir, exist_ok=True)

    def evaluate(self):
        """
        Run inference on test dataset, compute metrics, generate visualizations.
        Returns:
            dict: metrics results and optional visualizations info
        """
        # Prepare data loader
        test_loader = self.dataset_loader['test']
        # Initialize metrics accumulators
        total_samples = 0
        correct_preds = 0
        total_iou = 0.0
        total_batches = 0
        total_robust_acc = 0.0  # if robustness test included
        total_confidence = 0.0  # average confidence
        total_confidence_samples = 0

        # For segmentation, collect predicted and target masks
        pred_masks = []
        target_masks = []

        # Set model in eval mode
        self.model.model.eval()
        self.prompts.prompt_tensor.eval()
        if hasattr(self.prompts, 'real_coeffs'):
            self.prompts.real_coeffs.eval()
        if hasattr(self.prompts, 'imag_coeffs'):
            self.prompts.imag_coeffs.eval()

        with torch.no_grad():
            for batch in tqdm(test_loader):
                # Handle different dataset types
                imgs = batch.get('image', None)
                labels = batch.get('label', None)
                masks = batch.get('mask', None)  # For segmentation
                targets = batch.get('target', None)  # For detection
                # Move to device
                if imgs is not None:
                    imgs = imgs.to(self.device)
                # For segmentation/detection, may need special handling
                # For classification:
                if imgs is None:
                    continue  # skip if no images
                
                # Resize images as per config
                imgs_resized = self._resize_images(imgs)

                # Get prompt
                prompt = self.prompts.get_prompt()
                # Apply prompt and resize
                prompted_imgs = self._apply_prompt(imgs_resized, prompt)

                # Forward pass
                preds = self.model.forward(prompted_imgs)
                # Get predictions (logits or features)
                if self.model.model_name == 'clip':
                    # CLIP: compute text similarity
                    # preds: image features (normalized)
                    # Compute cosine similarity with class text embeddings
                    class_embeddings = self._get_class_text_embeddings()
                    # Normalize image features
                    img_embs = self.model.extract_features(prompted_imgs)
                    # Compute cosine similarity (N, T)
                    sims = torch.matmul(img_embs, class_embeddings.T)
                    pred_logit = sims
                else:
                    # Vision model: logits or features
                    pred_logit = preds

                # Map to target classes
                mapped_preds = self.label_mapper.map(pred_logit)

                # Compute classification accuracy
                if labels is not None:
                    pred_labels = torch.argmax(mapped_preds, dim=1)
                    correct_preds += (pred_labels == labels).sum().item()
                    total_samples += labels.shape[0]

                # For segmentation: compute IoU
                if masks is not None:
                    pred_mask = torch.argmax(mapped_preds, dim=1)  # shape (N, H, W)
                    total_iou += compute_iou(pred_mask, masks, self.num_classes)
                    pred_masks.append(pred_mask.cpu())
                    target_masks.append(masks.cpu())

                # For detection: could extend with mAP calculation (not shown here)

                # Optional: robustness evaluation
                if 'corrupted' in batch:
                    # Evaluate on corrupted images if provided
                    corrupted_imgs = batch['corrupted'].to(self.device)
                    with torch.no_grad():
                        prompted_corr = self._apply_prompt(self._resize_images(corrupted_imgs), prompt)
                        preds_corr = self.model.forward(prompted_corr)
                        if self.model.model_name == 'clip':
                            sims_corr = torch.matmul(self.model.extract_features(prompted_corr), class_embeddings.T)
                            preds_corr_logits = sims_corr
                        else:
                            preds_corr_logits = preds_corr
                        mapped_preds_corr = self.label_mapper.map(preds_corr_logits)
                        pred_labels_corr = torch.argmax(mapped_preds_corr, dim=1)
                        correct_corr = (pred_labels_corr == batch['label'].to(self.device)).sum().item()
                        total_robust_acc += correct_corr
                        total_confidence += torch.max(F.softmax(mapped_preds_corr, dim=1), dim=1).sum().item()
                        total_confidence_samples += batch['label'].size(0)

        # Final metrics
        accuracy = 100.0 * correct_preds / total_samples if total_samples > 0 else 0.0
        avg_iou = total_iou / max(1, len(pred_masks)) if pred_masks else 0.0
        robustness_acc = (total_robust_acc / total_confidence_samples * 100.0) if total_confidence_samples > 0 else None
        avg_confidence = (total_confidence / total_confidence_samples) if total_confidence_samples > 0 else None

        results = {
            'accuracy': accuracy,
            'iou': avg_iou,
            'robust_accuracy': robustness_acc,
            'average_confidence': avg_confidence,
        }

        # Generate visualizations
        self._visualize_prompts()
        self._visualize_label_mapping()

        return results

    def _resize_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Resize images internally if needed per current config.
        Currently assuming fixed scale; extend with differentiable resize if needed.
        """
        # Placeholder: do nothing, return images directly
        return imgs

    def _apply_prompt(self, imgs: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        Apply pixel prompts or insert prompts into images.
        For pixel prompts: overlay or concatenate.
        """
        # For simplicity, assuming prompts are padding: pad images
        p = prompt.shape[1]
        # Padding with zeros (or prompts), adjust as needed
        batch_size, C, H, W = imgs.shape
        padded_imgs = F.pad(imgs, pad=(p, p, p, p), mode='constant', value=0)
        return padded_imgs

    def _get_class_text_embeddings(self):
        """
        Return class text embeddings for semantic similarity.
        Only relevant for CLIP.
        """
        # Assumes class names are available globally or in self
        # For simplicity, assume self._class_text_embeddings exists
        if hasattr(self, '_class_text_embeddings'):
            return self._class_text_embeddings
        else:
            # Needs to be initialized before
            # Placeholder: random embeddings
            return torch.randn((self.num_classes, 512), device=self.device)

    def _visualize_prompts(self):
        """
        Visualize current prompts (pixel or frequency) for inspection.
        """
        if hasattr(self.prompts, 'visualize'):
            try:
                prompt_img = self.prompts.visualize()
                plt.figure(figsize=(4,4))
                plt.imshow(prompt_img)
                plt.axis('off')
                save_path = os.path.join(self.vis_dir, 'prompt_visualization.png')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
            except Exception:
                pass

    def _visualize_label_mapping(self):
        """
        Visualize label mapping matrices or semantic similarities.
        """
        if hasattr(self.label_mapper, 'visualize_mapping'):
            try:
                self.label_mapper.visualize_mapping()
            except Exception:
                pass
