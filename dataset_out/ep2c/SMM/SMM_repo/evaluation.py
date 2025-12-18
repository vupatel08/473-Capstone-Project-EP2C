## evaluation.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

class Evaluation:
    """
    Evaluation class for testing the fixed pre-trained classifier on reprogrammed images.
    Computes metrics such as accuracy and optionally visualizes reprogrammed images and masks.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        mask_generator=None,
        pattern=None,
        config: dict = None,
        device: torch.device = None
    ):
        """
        Initialize Evaluation with the classifier, optional reprogramming components, configuration.
        Args:
            model (torch.nn.Module): Fixed pre-trained classifier (f_P).
            mask_generator (object): Optional, for visualizing masks if available.
            pattern (torch.nn.Parameter): Optional, for visualizing pattern delta.
            config (dict): Configuration dict (from YAML).
            device (torch.device): Computation device.
        """
        self.model = model
        self.mask_generator = mask_generator
        self.pattern = pattern
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        self.visualize = self.config.get('evaluation', {}).get('visualize', False)
        self.num_visualize = 8  # Number of images to visualize if needed
        self.device = self.device

        # Set model to eval mode
        self.model.eval()
        self.model.to(self.device)

    def evaluate(self, data_loader, mask_generator=None, pattern=None):
        """
        Run inference on the dataset, compute accuracy, optionally visualize.
        Args:
            data_loader (torch.utils.data.DataLoader): Loader for test/validation dataset.
            mask_generator (object): Optional, for visualization.
            pattern (torch.nn.Parameter): Optional, for visualization.
        Returns:
            metrics (dict): Contains 'accuracy' (float).
        """
        total_samples = 0
        correct_predictions = 0

        # Accumulators for visualization if enabled
        vis_images, vis_reprogrammed, vis_masks = [], [], []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Run reprogramming step: resize, generate masks, add pattern
                # The evaluation can re-use the pipeline externally if needed,
                # here we assume no additional reprogramming, or define an internal method
                
                # For visualization, generate reprogrammed images
                reprogrammed_images = None
                masks_batch = None
                if self.mask_generator or mask_generator:
                    # Use provided mask_generator or class attribute
                    mg = mask_generator or self.mask_generator
                    delta = pattern or self.pattern
                    images_resized = self._resize(images, self.config['sampling'].get('image_resize', [32, 32]))
                    masks_batch = self._generate_masks(images_resized, mg)
                    delta_exp = delta.unsqueeze(0).to(self.device)
                    pattern_masked = delta_exp * masks_batch
                    reprogrammed_images = images_resized + pattern_masked
                else:
                    # No reprogramming; just use images
                    reprogrammed_images = images

                # Obtain logits
                logits = self.model(reprogrammed_images)
                _, predicted = torch.max(logits, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                # For visualization, store first batch images
                if self.config.get('evaluation', {}).get('visualize', False) and batch_idx == 0:
                    # Save original and reprogrammed images for visualization
                    # Denormalize images if normalization applied
                    if hasattr(reprogrammed_images, 'cpu'):
                        reprogrammed_images_vis = reprogrammed_images.cpu()
                        images_vis = images.cpu()
                        if masks_batch is not None:
                            masks_vis = masks_batch.cpu()
                        else:
                            masks_vis = None
                        vis_images.extend(images_vis[:self.num_visualize])
                        vis_reprogrammed.extend(reprogrammed_images_vis[:self.num_visualize])
                        if masks_vis is not None:
                            vis_masks.extend(masks_vis[:self.num_visualize])

        accuracy = correct_predictions / total_samples
        metrics = {'accuracy': accuracy}

        # Visualize if requested
        if self.config.get('evaluation', {}).get('visualize', False):
            self._visualize_results(vis_images, vis_reprogrammed, vis_masks)

        return metrics

    def _resize(self, images, size: list):
        """
        Resize images to target size using bilinear interpolation.
        """
        size_tuple = tuple(size)
        return torch.nn.functional.interpolate(images, size=size_tuple, mode='bilinear', align_corners=False)

    def _generate_masks(self, images_resized, mask_generator):
        """
        Generate sample-specific masks batch for input images.
        """
        batch_size = images_resized.size(0)
        H, W = images_resized.shape[2], images_resized.shape[3]
        masks_list = []

        # Generate singleton masks one by one due to patch-wise upsampling constraints
        for i in range(batch_size):
            img = images_resized[i].unsqueeze(0)  # shape [1, 3, H, W]
            mask_low_res = mask_generator.generate_mask(img)  # shape [1, 3, H', W']
            mask_upsampled = self._patch_upsample(mask_low_res, (H, W))
            masks_list.append(mask_upsampled)
        masks_batch = torch.cat(masks_list, dim=0)
        return masks_batch

    def _patch_upsample(self, mask, size: Tuple[int, int]):
        """
        Upsample mask via patch-wise (pixel) repetition to match original size.
        """
        H_in, W_in = size
        _, C, H', W' = mask.shape
        # Calculate patch size (tiles per pixel)
        patch_size_h = max(1, H_in // H')
        patch_size_w = max(1, W_in // W')
        # Repeat each pixel patch-wise
        upsampled = mask.repeat_interleave(patch_size_h, dim=2)
        upsampled = upsampled.repeat_interleave(patch_size_w, dim=3)
        # Crop to exact size
        upsampled_cropped = upsampled[:, :, :H_in, :W_in]
        return upsampled_cropped

    def _visualize_results(self, original_images, reprogrammed_images, masks):
        """
        Generate visualizations for the first few images: original, reprogrammed, masks, overlays.
        """
        num_images = min(self.num_visualize, len(original_images))
        plt.figure(figsize=(15, 5))
        for i in range(num_images):
            # Original image
            plt.subplot(3, num_images, i+1)
            self._imshow_tensor(original_images[i])
            plt.title('Original')
            plt.axis('off')

            # Reprogrammed image
            plt.subplot(3, num_images, i+1+num_images)
            self._imshow_tensor(reprogrammed_images[i])
            plt.title('Reprogrammed')
            plt.axis('off')

            # Mask overlay
            plt.subplot(3, num_images, i+1+2*num_images)
            mask_vis = masks[i] if masks is not None else None
            if mask_vis is not None:
                overlay = self._overlay_mask(reprogrammed_images[i], mask_vis)
                self._imshow_tensor(overlay)
            else:
                self._imshow_tensor(reprogrammed_images[i])
            plt.title('Mask Overlay')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _imshow_tensor(self, tensor_img):
        """
        Helper to show tensor image after denormalization.
        """
        img = tensor_img.clone()
        # If normalization was applied, unnormalize
        # Assuming normalization mean,std from dataset
        # For simplicity, assume images are in [0,1]
        np_img = img.permute(1, 2, 0).numpy()
        np_img = np.clip(np_img, 0, 1)
        plt.imshow(np_img)

    def _overlay_mask(self, image_tensor, mask_tensor, alpha: float=0.3):
        """
        Overlay mask on image for visualization.
        """
        image = image_tensor.permute(1, 2, 0).cpu().numpy()
        mask = mask_tensor.permute(1, 2, 0).cpu().numpy()
        mask_normalized = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        overlay = image * (1 - alpha) + mask_normalized * alpha
        overlay = np.clip(overlay, 0, 1)
        return overlay
