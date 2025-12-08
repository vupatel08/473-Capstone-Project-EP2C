# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## attack.py

```python
## attack.py
import torch
import torch.nn.functional as F
import numpy as np
from utils import clip_tensor, compute_cosine_similarity

class Attack:
    """
    The Attack class encapsulates the CosPGD white-box adversarial attack for pixel-wise prediction tasks.
    It supports untargeted and targeted settings, various norms (primarily l_infinity), and pixel-wise scaled loss
    based on cosine similarity for balanced perturbation across all spatial locations.
    """

    def __init__(self,
                 model,
                 epsilon=8/255,
                 step_size=2/255,
                 max_iters=10,
                 task='classification',
                 targeted=False,
                 target=None,
                 device=torch.device('cpu')):
        """
        Initialize the attack parameters and the model.

        Args:
            model (object): An instance of Model class with predict() method.
            epsilon (float): Maximum perturbation (for l_infinity norm).
            step_size (float): Step size for each iteration.
            max_iters (int): Number of attack iterations.
            task (str): 'classification' for semantic segmentation, 'regression' for optical flow/image restoration.
            targeted (bool): Whether attack is targeted.
            target (torch.Tensor or None): Target labels or images (if targeted). Shape depends on task.
            device (torch.device): Device to run computations on.
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = step_size
        self.max_iters = max_iters
        self.task = task
        self.targeted = targeted
        self.target = target
        self.device = device

    def initialize(self, x_clean):
        """
        Initialize the adversarial example by adding small uniform noise within epsilon bounds.

        Args:
            x_clean (torch.Tensor): Original clean input tensor.

        Returns:
            torch.Tensor: Initialized adversarial input.
        """
        # Uniform random noise within [-epsilon, epsilon]
        delta = torch.rand_like(x_clean, device=self.device) * 2 * self.epsilon - self.epsilon
        x_adv = x_clean + delta
        # Clip to [0,1]
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv

    def compute_scaled_loss(self, pred, y, targeted=False):
        """
        Compute scale-aware pixel-wise loss scaled by cosine similarity / dissimilarity.

        Args:
            pred (torch.Tensor): Model predictions (logits or outputs), shape [B, C, H, W].
            y (torch.Tensor): Targets labels/images, shape [B, H, W] for segmentation, [B, 2, H, W] for flow, etc.
            targeted (bool): True if the attack is targeted, False otherwise.

        Returns:
            torch.Tensor: Scalar loss tensor, scaled pixel-wise.
        """
        # Apply softmax to model predictions for classification tasks
        # For regression, identity may be used; here we assume classification task
        pred_probs = F.softmax(pred, dim=1)  # shape: [B, C, H, W]

        # For each pixel, get the probability vector across classes
        # For 'Y', if labels are class indices, convert to one-hot
        # Assuming y is class idx tensor; for regression, use y as is
        if y.dim() == pred_probs.dim() - 1:
            # Convert y (class indices) to one-hot
            num_classes = pred_probs.shape[1]
            y_one_hot = F.one_hot(y, num_classes=num_classes).permute(0,3,1,2).float()
        else:
            # y is already one-hot or continuous
            y_one_hot = y

        # Compute pixel-wise cosine similarity
        # pred_probs and y_one_hot shape: [B,C,H,W]
        cosine_score = compute_cosine_similarity(pred_probs, y_one_hot)

        # For classification goals, shape: [B, H, W]
        # For regression, you may need a different similarity measure
        # Here we implement for classification as default

        # Compute pixel-wise loss (e.g., cross-entropy)
        # Using negative log likelihood for numerical stability
        # Alternatively, use torch.nn.functional.cross_entropy directly
        ce_loss = F.cross_entropy(pred, y, reduction='none')  # shape: [B, H, W]

        # Scale loss: for untargeted attack, scale by cosine similarity
        # For targeted attack, scale by (1 - cosine similarity)
        if self.targeted:
            scale_factor = 1.0 - cosine_score
        else:
            scale_factor = cosine_score

        # Expand scale_factor to match shape of ce_loss
        # shape: [B, H, W]
        scaled_loss = scale_factor * ce_loss

        # Return mean over batch and spatial dimensions
        return scaled_loss.mean()

    def update_input(self, x_adv, grad):
        """
        Update the adversarial input tensor based on gradient, step size, and clipping.

        Args:
            x_adv (torch.Tensor): Current adversarial example.
            grad (torch.Tensor): Gradient of loss w.r.t. x_adv.

        Returns:
            torch.Tensor: Updated adversarial example.
        """
        # Sign of the gradient as in FGSM
        grad_sign = grad.sign()
        # Update step
        x_adv = x_adv + self.alpha * grad_sign
        # Clip to epsilon-ball around original x_clean
        delta = x_adv - self.x_clean
        delta = clip_tensor(delta, -self.epsilon, self.epsilon)
        # update adversarial example
        x_adv = self.x_clean + delta
        # Clip pixel values to [0,1]
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv

    def clip(self, x, x_orig):
        """
        Ensure x remains within allowed bounds ([0,1]) and within epsilon constraint from x_orig.

        Args:
            x (torch.Tensor): Input tensor after update.
            x_orig (torch.Tensor): Original clean input tensor.

        Returns:
            torch.Tensor: Clipped tensor respecting constraints.
        """
        delta = x - x_orig
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        x_clipped = x_orig + delta
        x_clipped = torch.clamp(x_clipped, 0.0, 1.0)
        return x_clipped

    def attack(self, x_clean, y=None, targeted=False, target=None):
        """
        Run the iterative CosPGD attack.

        Args:
            x_clean (torch.Tensor): Original clean input.
            y (torch.Tensor): Ground-truth labels or images.
            targeted (bool): Whether attack is targeted.
            target (torch.Tensor or None): Target labels or images if targeted.

        Returns:
            torch.Tensor: The adversarial example after attack.
        """
        # Save the original clean input for clipping
        self.x_clean = x_clean.detach().clone()
        # Initialize x_adv with small random noise within epsilon
        x_adv = self.initialize(x_clean).detach()
        x_adv.requires_grad = True

        for iter_idx in range(self.max_iters):
            # Enable gradient
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            else:
                x_adv.requires_grad = True

            # Forward pass
            pred = self.model.predict(x_adv)
            # Ensure pred is of shape: [B, C, H, W]

            # Compute target tensor
            if y is None and self.target is not None:
                # For targeted attack with specific target images or labels
                y_input = self.target
            elif y is not None:
                y_input = y
            else:
                y_input = None

            # Calculate cosine similarity
            # Helper function in utils: compute_cosine_similarity
            pred_probs = F.softmax(pred, dim=1)
            # Prepare y for similarity: for classification, ensure one-hot
            if y_input is not None and y_input.dim() == pred_probs.dim() - 1:
                num_classes = pred_probs.shape[1]
                y_one_hot = F.one_hot(y_input, num_classes=num_classes).permute(0,3,1,2).float()
            elif y_input is not None:
                y_one_hot = y_input
            else:
                # For regression tasks, possible to set psi as identity
                # Here, fall back to identity (not using cosine scaling)
                y_one_hot = None

            # Compute cosine similarity per pixel
            if y_one_hot is not None:
                cosine_score = compute_cosine_similarity(pred_probs, y_one_hot)
            else:
                # For regression or other, set cosine_score to 1
                # or skip scaling
                cosine_score = torch.ones_like(pred_probs[:,0,...], device=self.device)

            # Compute scaled pixel loss
            scaled_loss = self.compute_scaled_loss(pred, y_input if y_input is not None else pred, targeted)

            # Backpropagate
            scaled_loss.backward()
            grad = x_adv.grad.detach()

            # Update input
            x_adv = self.update_input(x_adv.detach(), grad)

            # Detach for next iteration
            x_adv = x_adv.detach()
            x_adv.requires_grad = True

        return x_adv
```

## dataset_loader.py

```python
## dataset_loader.py
import os
import glob
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import cv2

class DatasetLoader(Dataset):
    """
    DatasetLoader class for loading datasets for pixel-wise prediction tasks:
    semantic segmentation (Pascal VOC 2012),
    optical flow (KITTI 2015),
    image restoration (GoPro),
    and image denoising (SSID).

    Args:
        dataset_name (str): Name of the dataset (e.g., 'PascalVOC2012', 'KITTI2015', 'GoPro', 'SSID')
        root_dir (str): Root directory of dataset.
        task (str): One of 'semantic_segmentation', 'optical_flow', 'image_restoration', 'image_denoising'.
        split (str): Dataset split, e.g., 'train', 'validation', 'test'.
        augment (bool): Whether to apply data augmentation (primarily for training).
        input_size (tuple): Desired input size (height, width) for resizing images.
    """
    def __init__(self,
                 dataset_name: str,
                 root_dir: str,
                 task: str,
                 split: str = 'train',
                 augment: bool = False,
                 input_size: Optional[Tuple[int, int]] = None):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.task = task
        self.split = split
        self.augment = augment
        self.input_size = input_size  # e.g., (512, 512)

        # Initialize file lists based on dataset and split
        if self.dataset_name.lower() == 'pascalvoc2012':
            self._load_pascalvoc2()
        elif self.dataset_name.lower() == 'kitti2015':
            self._load_kitti()
        elif self.dataset_name.lower() == 'gopro':
            self._load_gopro()
        elif self.dataset_name.lower() == 'ssid':
            self._load_ssid()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")

        # Set transforms for images and labels
        self.transform_img = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor()
        ])
        # Normalization to [0,1] is handled in ToTensor

        # Additional augmentation transforms
        if self.augment:
            self.augment_transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomResizedCrop(self.input_size, scale=(0.8, 1.0))
            ])
        else:
            self.augment_transforms = None

    def _load_pascalvoc2(self):
        """
        Load Pascal VOC dataset file paths.
        Expected folder structure:
        root_dir/
            JPEGImages/
            SegmentationClass/
            ImageSets/
        """
        img_dir = os.path.join(self.root_dir, 'JPEGImages')
        label_dir = os.path.join(self.root_dir, 'SegmentationClass')
        split_file = os.path.join(self.root_dir, 'ImageSets', 'Segmentation', f'{self.split}.txt')

        with open(split_file, 'r') as f:
            file_list = [line.strip() for line in f.readlines()]

        self.image_paths = [os.path.join(img_dir, f'{fname}.jpg') for fname in file_list]
        self.label_paths = [os.path.join(label_dir, f'{fname}.png') for fname in file_list]
        # Store filenames for reference
        self.filenames = file_list

    def _load_kitti(self):
        """
        Load KITTI 2015 optical flow dataset.
        Assumes:
        root_dir/
            image_2/
            flow_occ/
        """
        img_dir = os.path.join(self.root_dir, 'image_2')
        flow_dir = os.path.join(self.root_dir, 'flow_occ')

        # For split, assume separate lists are provided or infer from directory
        # Here, assume 'validation' split
        image_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        flow_files = sorted(glob.glob(os.path.join(flow_dir, '*.png')))

        self.image_paths = image_files
        self.flow_paths = flow_files

    def _load_gopro(self):
        """
        Load GoPro dataset images.
        Assumes:
        root_dir/
            images/
            ground_truth/
        """
        img_dir = os.path.join(self.root_dir, 'images')
        label_dir = os.path.join(self.root_dir, 'ground_truth')

        image_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        label_files = sorted(glob.glob(os.path.join(label_dir, '*.png')))

        self.image_paths = image_files
        self.label_paths = label_files

    def _load_ssid(self):
        """
        Load SSID dataset images and ground truths.
        Assumes:
        root_dir/
            noisy/
            clean/
        """
        noisy_dir = os.path.join(self.root_dir, 'noisy')
        clean_dir = os.path.join(self.root_dir, 'clean')

        noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.png')))
        clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.png')))

        self.image_paths = noisy_files
        self.label_paths = clean_files

    def __len__(self):
        # Return length based on which dataset is loaded
        if hasattr(self, 'image_paths') and hasattr(self, 'label_paths'):
            return len(self.image_paths)
        elif hasattr(self, 'image_paths'):
            return len(self.image_paths)
        else:
            return 0

    def __getitem__(self, idx):
        """
        Load and process data sample with optional augmentation.
        Returns:
            dict: {'image': tensor, 'label' or 'flow' or 'target': tensor}
        """
        if hasattr(self, 'image_paths'):
            img_path = self.image_paths[idx]
        else:
            raise IndexError("Image path not set for dataset.")

        # Load image
        img = Image.open(img_path).convert('RGB')

        # For segmentation task
        if self.task == 'semantic_segmentation':
            label_path = self.label_paths[idx]
            label_img = Image.open(label_path)
            label = np.array(label_img).astype(np.int64)  # class indices
            # Resize label similarly
            label = cv2.resize(label, self.input_size, interpolation=cv2.INTER_NEAREST)
            label_tensor = torch.from_numpy(label)  # shape: HxW
        elif self.task == 'optical_flow':
            flow_path = self.flow_paths[idx]
            flow = self._load_flow(flow_path)
        elif self.task in ['image_restoration', 'image_denoising']:
            label_path = self.label_paths[idx]
            label_img = Image.open(label_path)
            label = np.array(label_img).astype(np.float32) / 255.0
            label_tensor = torch.from_numpy(label).permute(2,0,1)  # CHW
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        # Apply augmentation if specified
        if self.augment and self.augment_transforms:
            seed = np.random.randint(0, 2**32)
            torch.manual_seed(seed)
            img = self.augment_transforms(img)
            if self.task == 'semantic_segmentation':
                label_img = self.augment_transforms(label_img)
            elif self.task == 'optical_flow':
                # For flow, skip color jitter, only flip or crop
                pass  # or create separate transforms
            elif self.task in ['image_restoration', 'image_denoising']:
                label_img = self.augment_transforms(label_img)

        # Resize and convert to tensor
        img = self.transform_img(img)  # normalized [0,1]
        sample = {'image': img}

        if self.task == 'semantic_segmentation':
            label_tensor = torch.from_numpy(label).long()
            sample['label'] = label_tensor
        elif self.task == 'optical_flow':
            flow_tensor = self._load_flow(flow_path)  # normalized
            sample['flow'] = flow_tensor
        elif self.task in ['image_restoration', 'image_denoising']:
            label_img = Image.fromarray((label * 255).astype(np.uint8))
            label_img = self.transform_img(label_img)
            sample['target'] = label_img

        return sample

    def _load_flow(self, flow_path):
        """
        Load flow stored as PNG or .flo, convert to tensor.
        """
        if flow_path.endswith('.png'):
            flow_np = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            # Assuming flow stored with 2 channels in 2D
            flow = flow_np[:,:,0:2]
            flow = cv2.resize(flow, self.input_size, interpolation=cv2.INTER_LINEAR)
            # Normalize flow if needed
            flow_tensor = torch.from_numpy(flow).permute(2,0,1)  # shape: 2 x H x W
            return flow_tensor
        elif flow_path.endswith('.flo'):
            # Implement .flo reading if necessary
            # For now, placeholder
            return torch.zeros(2, self.input_size[1], self.input_size[0])
        else:
            raise ValueError(f"Unknown flow format: {flow_path}")
```

## evaluation.py

```python
## evaluation.py
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

class Evaluation:
    """
    Evaluation class for computing and saving performance metrics for pixel-wise prediction tasks:
    semantic segmentation (IoU, pixel accuracy),
    optical flow (EPE), and image restoration (PSNR, SSIM).
    
    Supports evaluation over datasets, multiple attack iterations, saving metrics, and plotting trends.
    """

    def __init__(self,
                 model,
                 dataset_loader,
                 task='classification',
                 device=torch.device('cpu'),
                 save_dir='./evaluation_results',
                 verbose=True):
        """
        Initialize evaluation parameters.

        Args:
            model (object): Instance of Model class with predict() method.
            dataset_loader (DatasetLoader): Dataset loader object.
            task (str): 'semantic_segmentation', 'optical_flow', 'image_restoration', 'image_denoising'.
            device (torch.device): Device for inference.
            save_dir (str): Directory to save results and plots.
            verbose (bool): Whether to print progress.
        """
        self.model = model
        self.dataset_loader = dataset_loader
        self.task = task
        self.device = device
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.verbose = verbose

        # Metrics storage
        self.scores = {
            'per_sample': [],
            'average': {}
        }
        self.results_summary = {}
        # Determine metrics to compute based on task
        if self.task == 'semantic_segmentation':
            self.metrics = ['IoU', 'pixel_accuracy']
        elif self.task == 'optical_flow':
            self.metrics = ['EPE', 'EPE_f1_all']
        elif self.task in ['image_restoration', 'image_denoising']:
            self.metrics = ['PSNR', 'SSIM']
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def compute_iou(self, pred_mask, true_mask, num_classes):
        """
        Compute mean Intersection over Union (IoU) over classes.

        Args:
            pred_mask (np.ndarray): Predicted class labels, shape (H,W)
            true_mask (np.ndarray): Ground truth class labels, shape (H,W)
            num_classes (int): Number of classes

        Returns:
            float: mean IoU over classes
        """
        iou_per_class = []
        for cls in range(num_classes):
            pred_inds = (pred_mask == cls)
            true_inds = (true_mask == cls)
            intersection = np.logical_and(pred_inds, true_inds).sum()
            union = np.logical_or(pred_inds, true_inds).sum()
            if union == 0:
                # If no pixels for this class in gt and pred, ignore
                continue
            iou_per_class.append(intersection / union)
        if len(iou_per_class) == 0:
            return 0.0
        return np.mean(iou_per_class)

    def compute_pixel_accuracy(self, pred_mask, true_mask):
        """
        Compute pixel accuracy.

        Args:
            pred_mask (np.ndarray): Predicted labels, shape (H,W)
            true_mask (np.ndarray): Ground truth labels, shape (H,W)

        Returns:
            float: pixel accuracy
        """
        valid_mask = (true_mask >= 0)  # ignore index if needed
        correct = (pred_mask == true_mask) & valid_mask
        total = valid_mask.sum()
        if total == 0:
            return 0.0
        return correct.sum() / total

    def compute_epe(self, pred_flow, true_flow):
        """
        Compute End-Point Error (EPE) per pixel.

        Args:
            pred_flow (np.ndarray): Shape (H,W,2)
            true_flow (np.ndarray): Shape (H,W,2)

        Returns:
            float: mean EPE over pixels
        """
        diff = pred_flow - true_flow
        epe_map = np.linalg.norm(diff, axis=2)
        return np.mean(epe_map)

    def compute_epe_f1_all(self, pred_flow, true_flow):
        """
        Compute EPE-f1-all metric as specified.

        Args:
            pred_flow (np.ndarray): Shape (H,W,2)
            true_flow (np.ndarray): Shape (H,W,2)

        Returns:
            float: proportion of pixels with EPE > 3.0 or normalized EPE > 0.05
        """
        diff = pred_flow - true_flow
        epe_map = np.linalg.norm(diff, axis=2)
        threshold_mask = (epe_map > 3.0) | ((epe_map / (np.linalg.norm(true_flow, axis=2) + 1e-6)) > 0.05)
        return np.mean(threshold_mask)

    def compute_psnr(self, pred_img, true_img):
        """
        Compute PSNR between two images.

        Args:
            pred_img (np.ndarray): Shape (H,W,3)
            true_img (np.ndarray): Shape (H,W,3)

        Returns:
            float: PSNR value
        """
        mse = np.mean((pred_img - true_img) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0  # images normalized to [0,1]
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def compute_ssim(self, pred_img, true_img):
        """
        Compute SSIM between two images.

        Args:
            pred_img (np.ndarray): Shape (H,W,3)
            true_img (np.ndarray): Shape (H,W,3)

        Returns:
            float: SSIM value
        """
        pred_np = (pred_img * 255).astype(np.uint8)
        true_np = (true_img * 255).astype(np.uint8)
        ssim_value = compare_ssim(pred_np, true_np, multichannel=True, data_range=255)
        return ssim_value

    def run_evaluation(self, attack_iterations_list=None, save_visuals=False):
        """
        Run the evaluation over the dataset, optionally over multiple attack iterations.

        Args:
            attack_iterations_list (list): List of attack iteration counts to evaluate, e.g., [3,5,10,...].
            save_visuals (bool): Whether to save visualizations of predictions, overlays, adversarial examples.
        """
        dataset = self.dataset_loader
        model = self.model
        device = self.device
        results_per_iter = {str(iter): {metric: [] for metric in self.metrics} for iter in attack_iterations_list} if attack_iterations_list else None

        all_pred_masks = []
        all_true_masks = []
        all_pred_flows = []
        all_true_flows = []
        all_pred_images = []
        all_true_images = []

        if self.verbose:
            print("Starting evaluation...")

        for idx in range(len(dataset)):
            sample = dataset[idx]
            # Load input based on task
            input_img = sample['image'].unsqueeze(0).to(device)  # add batch dim
            with torch.no_grad():
                pred = model.predict(input_img)

            # For segmentation
            if self.task == 'semantic_segmentation':
                true_mask = sample['label'].cpu().numpy()  # H x W
                pred_logits = pred.cpu()
                pred_probs = F.softmax(pred_logits, dim=1)
                pred_mask = torch.argmax(pred_probs, dim=1).squeeze(0).cpu().numpy()  # H x W
                num_classes = pred_probs.shape[1]
                # Store for overall metrics
                all_true_masks.append(true_mask)
                all_pred_masks.append(pred_mask)

                # Compute metrics
                iou = self.compute_iou(pred_mask, true_mask, num_classes)
                pix_acc = self.compute_pixel_accuracy(pred_mask, true_mask)
                if attack_iterations_list:
                    for iter_idx, T in enumerate(attack_iterations_list):
                        results_per_iter[str(T)]['IoU'].append(iou)
                        results_per_iter[str(T)]['pixel_accuracy'].append(pix_acc)

            elif self.task == 'optical_flow':
                true_flow = sample['flow'].permute(1,2,0).cpu().numpy()  # H x W x 2
                pred_flow = pred.squeeze(0).permute(1,2,0).cpu().numpy()  # H x W x 2
                epe = self.compute_epe(pred_flow, true_flow)
                epe_f1 = self.compute_epe_f1_all(pred_flow, true_flow)
                if attack_iterations_list:
                    for T in attack_iterations_list:
                        self._append_metric('EPE', epe)
                        self._append_metric('EPE_f1_all', epe_f1)

                # Store for potential further analysis
                all_true_flows.append(true_flow)
                all_pred_flows.append(pred_flow)

            elif self.task in ['image_restoration', 'image_denoising']:
                true_img = sample['target'].permute(1,2,0).cpu().numpy()  # H x W x 3
                pred_img = pred.squeeze(0).permute(1,2,0).cpu().numpy()  # H x W x 3
                psnr_val = self.compute_psnr(pred_img, true_img)
                ssim_val = self.compute_ssim(pred_img, true_img)
                if attack_iterations_list:
                    for T in attack_iterations_list:
                        self._append_metric('PSNR', psnr_val)
                        self._append_metric('SSIM', ssim_val)

                # Save for potential visual export
                if save_visuals:
                    self._save_image(pred_img, idx, T=None, suffix='restoration_pred')
                    self._save_image(true_img, idx, T=None, suffix='ground_truth')
                all_true_images.append(true_img)
                all_pred_images.append(pred_img)

        # Aggregate results
        self._aggregate_and_save_results(attack_iterations_list)
        # Plot the metrics over iterations if applicable
        if attack_iterations_list:
            self._plot_metrics_curves(attack_iterations_list)

    def _append_metric(self, metric_name, value):
        """Append a single metric value to a temporary list."""
        if metric_name not in self.scores['per_sample']:
            self.scores['per_sample'].append({metric_name: value})
        else:
            self.scores['per_sample'][-1][metric_name] = value

    def _aggregate_and_save_results(self, attack_iterations_list):
        """Compute mean metrics over dataset and save to JSON."""
        # For simplicity, compute overall mean per metric
        summary = {}
        for metric in self.metrics:
            values = []
            for sample_metrics in self.scores.get('per_sample', []):
                if metric in sample_metrics:
                    values.append(sample_metrics[metric])
            if len(values) > 0:
                summary[metric] = {'mean': np.mean(values), 'std': np.std(values)}
        self.results_summary = summary
        # Save to JSON
        results_path = os.path.join(self.save_dir, 'evaluation_summary.json')
        with open(results_path, 'w') as f:
            json.dump(self.results_summary, f, indent=4)
        if self.verbose:
            print(f"Saved evaluation summary to {results_path}")

    def _plot_metrics_curves(self, attack_iterations_list):
        """Plot metrics trends over attack iterations."""
        for metric in self.metrics:
            values = []
            for T in attack_iterations_list:
                if metric in self.results_summary:
                    # For demonstration, assume values stored during run; detailed implementation may differ
                    val = self.results_summary.get(metric, {}).get('mean', None)
                    values.append(val)
                elif metric in self.scores:
                    # fallback or placeholder
                    pass
            plt.figure()
            plt.plot(attack_iterations_list, values, marker='o')
            plt.xlabel('Attack iterations')
            plt.ylabel(f'{metric}')
            plt.title(f'{metric} over attack iterations')
            plt.grid()
            plt_path = os.path.join(self.save_dir, f'{metric}_curve.png')
            plt.savefig(plt_path)
            plt.close()

    def _save_image(self, img_np, idx, T=None, suffix=''):
        """Save image array to disk."""
        filename = f"sample_{idx}"
        if T:
            filename += f"_iter_{T}"
        filename += f"_{suffix}.png"
        save_path = os.path.join(self.save_dir, filename)
        import imageio
        imageio.imwrite(save_path, (img_np * 255).astype(np.uint8))
        if self.verbose:
            print(f"Saved image: {save_path}")

    def get_results(self):
        """Return the overall evaluation results."""
        return self.results_summary
```

## main.py

```python
## main.py
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Import custom modules
from dataset_loader import DatasetLoader
from model import Model
from attack import Attack
from evaluation import Evaluation
from utils import set_seed, plot_metrics, save_image_batch

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set device and seed for reproducibility
    device_str = config.get('hardware', {}).get('device', 'cuda:0')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    seed = config.get('hardware', {}).get('seed', 42)
    set_seed(seed)

    # Log setup
    print(f"Using device: {device}")
    print(f"Random seed: {seed}")

    # Load datasets
    datasets_cfg = config['datasets']
    data_loaders = {}
    for key, cfg in datasets_cfg.items():
        name = cfg['name']
        root_dir = cfg['root_dir']
        split = cfg['split']
        input_size = tuple(map(int, cfg.get('input_size', (512, 512)).split('x'))) if 'x' in str(cfg.get('input_size', '512')) else cfg.get('input_size', (512, 512))
        augment = cfg.get('augment', False)

        print(f"Loading dataset: {name}, split: {split}")
        dataset_loader = DatasetLoader(
            dataset_name=name,
            root_dir=root_dir,
            task=key,  # use key as task identifier
            split=split,
            augment=augment,
            input_size=input_size
        )
        data_loader = torch.utils.data.DataLoader(dataset_loader, batch_size=config.get('training', {}).get('batch_size', 16), shuffle=False)
        data_loaders[key] = data_loader

    # Load models
    models_cfg = config['models']
    models_obj = {}
    for key, m_cfg in models_cfg.items():
        model_name = key
        checkpoint_path = m_cfg['checkpoint_path']
        print(f"Loading model: {model_name} from {checkpoint_path}")
        model = Model(model_name, checkpoint_path, device).model
        model.eval()
        models_obj[key] = model

    # Prepare Attack parameters
    attack_params = config['attack_parameters']
    epsilon = attack_params.get('epsilon', 8/255)
    step_size = attack_params.get('step_size', 2/255)
    attack_iters_list = attack_params.get('attack_iters', [3,5,10,20,40,100])
    targeted = attack_params.get('targeted', False)
    target_label = attack_params.get('target_label', None)
    # Note: target_label can be dataset-specific; here kept generic

    # Prepare output directory
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)

    # For each dataset (e.g., semantic segmentation, optical flow, restoration)
    for task_name, dataloader in data_loaders.items():
        print(f"\n=== Starting evaluation on task: {task_name} ===")
        # Select dataset-specific model(s). For simplicity, pick one model per task.
        # In practice, you may want to attack multiple models.
        model_key = list(models_obj.keys())[0]
        model = models_obj[model_key]
        model_name = list(models_cfg.keys())[0]
        # Initialize Evaluation object
        eval_obj = Evaluation(
            model=Model(model_name, '', device),  # We can pass dummy if not used
            dataset_loader=None,  # Will set later
            task=task_name,
            device=device,
            save_dir=os.path.join(output_dir, task_name),
            verbose=True
        )

        # To run evaluation, we need to iterate over dataset
        # For each attack iteration count, perform attack
        all_metrics = {str(t): {'IoU': [], 'pixel_accuracy': [], 'EPE': [], 'EPE_f1_all': [], 'PSNR': [], 'SSIM': []}
                       for t in attack_iters_list}

        for batch in tqdm(dataloader, desc=f"Attacking dataset: {task_name}"):
            # Extract inputs based on task
            images = batch['image'].to(device)
            y = None
            # For semantic segmentation
            if task_name == 'semantic_segmentation':
                y = batch['label'].to(device)
            elif task_name == 'optical_flow':
                y = batch['flow'].to(device)
            elif task_name in ('image_restoration', 'image_denoising'):
                y = batch['target'].to(device)

            # Run attacks for each attack iteration count
            for T in attack_iters_list:
                attack_instance = Attack(
                    model=Model(model_name, '', device),
                    epsilon=epsilon,
                    step_size=step_size,
                    max_iters=T,
                    task=task_name,
                    targeted=targeted,
                    target=target_label,
                    device=device
                )

                # Generate adversarial example
                x_adv = attack_instance.attack(x_clean=images, y=y, targeted=targeted, target=target_label)

                # Get model prediction
                pred = model.predict(x_adv)

                # Evaluate metrics
                if task_name == 'semantic_segmentation':
                    true_mask = y.cpu().numpy()
                    pred_logits = pred.cpu()
                    pred_probs = torch.nn.functional.softmax(pred_logits, dim=1)
                    pred_mask = torch.argmax(pred_probs, dim=1).squeeze(0).cpu().numpy()
                    num_classes = pred_probs.shape[1]
                    # Evaluate IoU and pixel accuracy
                    iou = eval_obj.compute_iou(pred_mask, true_mask.squeeze(0), num_classes)
                    pix_acc = eval_obj.compute_pixel_accuracy(pred_mask, true_mask.squeeze(0))
                    all_metrics[str(T)]['IoU'].append(iou)
                    all_metrics[str(T)]['pixel_accuracy'].append(pix_acc)
                elif task_name == 'optical_flow':
                    true_flow = y.squeeze(0).permute(1,2,0).cpu().numpy()
                    pred_flow = pred.squeeze(0).permute(1,2,0).cpu().numpy()
                    epe = eval_obj.compute_epe(pred_flow, true_flow)
                    epe_f1 = eval_obj.compute_epe_f1_all(pred_flow, true_flow)
                    all_metrics[str(T)]['EPE'].append(epe)
                    all_metrics[str(T)]['EPE_f1_all'].append(epe_f1)
                elif task_name in ('image_restoration', 'image_denoising'):
                    true_img = y.squeeze(0).permute(1,2,0).cpu().numpy()
                    pred_img = pred.squeeze(0).permute(1,2,0).cpu().numpy()
                    psnr = eval_obj.compute_psnr(pred_img, true_img)
                    ssim = eval_obj.compute_ssim(pred_img, true_img)
                    all_metrics[str(T)]['PSNR'].append(psnr)
                    all_metrics[str(T)]['SSIM'].append(ssim)
                else:
                    continue

                # Optionally save adversarial images or predictions
                # save_image_batch(x_adv, os.path.join(output_dir, task_name, f"adv_{T}_{np.random.randint(0,10000)}.png"))

        # After processing dataset, compute mean metrics and save
        # Save per-iteration and overall results
        summary = {}
        for T_str, metrics_dict in all_metrics.items():
            summary[T_str] = {}
            for metric_name, values in metrics_dict.items():
                if len(values) > 0:
                    summary[T_str][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
        # Save summary JSON
        summary_path = os.path.join(output_dir, task_name, 'evaluation_summary.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        # Plot metrics trends over attack steps
        for metric in ['IoU', 'pixel_accuracy', 'EPE', 'EPE_f1_all', 'PSNR', 'SSIM']:
            x_vals = list(all_metrics.keys())
            y_vals = [np.mean(all_metrics[t][metric]) if len(all_metrics[t][metric])>0 else 0 for t in x_vals]
            plt.figure()
            plt.plot(x_vals, y_vals, marker='o')
            plt.xlabel('Attack iterations')
            plt.ylabel(metric)
            plt.title(f"{task_name} - {metric} over attack steps")
            plt.grid()
            plt.savefig(os.path.join(output_dir, task_name, f"{metric}_trend.png"))
            plt.close()

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torchvision.models as models
import torch.nn as nn

# Assume custom UNet and NAFNet implementations are available and imported.
# For demonstration, placeholder classes are provided.
# In actual use, replace with the actual implementation or import accordingly.
class UNet(nn.Module):
    def __init__(self, encoder='ConvNeXt-tiny', num_classes=21):
        super().__init__()
        # Initialize UNet with ConvNeXt-tiny encoder
        # Placeholder for actual UNet implementation
        self.model = nn.Identity()  # Replace with real UNet

    def forward(self, x):
        return self.model(x)

class NAFNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholder for actual NAFNet implementation
        self.model = nn.Identity()

    def forward(self, x):
        return self.model(x)

class Model:
    def __init__(self, model_name: str, checkpoint_path: str, device: torch.device):
        """
        Initialize and load the specified model architecture with weights.
        Supports 'deeplabv3', 'psnet', 'unet', 'nafnette'.

        Args:
            model_name (str): 'deeplabv3', 'psnet', 'unet', 'nafnette'
            checkpoint_path (str): Path to checkpoint weights
            device (torch.device): Device to load the model onto
        """
        self.model_name = model_name.lower()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        if self.model_name == 'deeplabv3':
            # Load DeepLabV3 with ResNet50 backbone
            model = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=21)
            # Load weights from checkpoint if available
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            return model

        elif self.model_name == 'psnet':
            # Assume a custom PSPNet implementation; placeholder here
            # You need to replace with actual PSPNet implementation
            # For illustration, using DeepLabV3 as placeholder
            model = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=21)
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            return model

        elif self.model_name == 'unet':
            # Instantiate custom UNet with ConvNeXt tiny encoder
            model = UNet(encoder='ConvNeXt-tiny', num_classes=21)
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            return model

        elif self.model_name == 'nafnette':
            # Instantiate NAFNet (assumed custom or external)
            model = NAFNet()
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            return model

        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference on input tensor x.

        Args:
            x (torch.Tensor): Input tensor, shape [B, C, H, W], normalized to [0,1].

        Returns:
            torch.Tensor: Model prediction (logits or outputs), shape depends on task:
                - segmentation: [B, num_classes, H, W]
                - optical flow: [B, 2, H, W]
                - restoration: [B, 3, H, W]
        """
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            # Handle output shape
            if self.model_name in ['deeplabv3', 'psnet', 'unet']:
                # Usually segmentation logits
                # output can be a dict with 'out', handle accordingly
                if isinstance(output, dict):
                    # For torchvision models, the output is like {'out': tensor}
                    return output['out']
                else:
                    return output
            elif self.model_name == 'nafnette':
                # Assume output is directly logits or images
                return output
            else:
                # Default fallback
                return output
```


## utils.py

```python
## utils.py
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility across random, numpy and torch.
    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clip_tensor(x: torch.Tensor, epsilon: float, x_clean: torch.Tensor, norm_type: str='l_infinity') -> torch.Tensor:
    """
    Clip tensor x to the epsilon ball around x_clean under specified norm.
    Args:
        x (torch.Tensor): Tensor to clip.
        epsilon (float): Max perturbation.
        x_clean (torch.Tensor): Original clean tensor.
        norm_type (str): 'l_infinity' or 'l_2'.
    Returns:
        torch.Tensor: Clipped tensor.
    """
    if norm_type == 'l_infinity':
        delta = x - x_clean
        delta = torch.clamp(delta, -epsilon, epsilon)
        x_clipped = x_clean + delta
        return torch.clamp(x_clipped, 0.0, 1.0)
    elif norm_type == 'l_2':
        delta = x - x_clean
        batch_size = delta.shape[0]
        delta_flat = delta.view(batch_size, -1)
        norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
        norm = torch.max(norm, torch.tensor([1e-12], device=delta.device))
        factor = torch.min(torch.ones_like(norm), epsilon / norm)
        delta = delta_flat * factor.view(-1,1)
        delta = delta.view_as(delta)
        x_clipped = x_clean + delta
        return torch.clamp(x_clipped, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

def normalize_tensor(x: torch.Tensor, method: str='softmax') -> torch.Tensor:
    """
    Normalize tensor x.
    Args:
        x (torch.Tensor): Input logits/tensor.
        method (str): 'softmax' or 'identity'.
    Returns:
        torch.Tensor: Normalized tensor.
    """
    if method == 'softmax':
        return torch.nn.functional.softmax(x, dim=1)
    elif method == 'identity':
        return x
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def compute_cosine_similarity(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    """
    Compute per-pixel cosine similarity between pred and target tensors.
    Args:
        pred (torch.Tensor): Shape [B, C, H, W], normalized (probabilities or features).
        target (torch.Tensor): Shape [B, C, H, W].
        eps (float): Small epsilon to prevent division by zero.
    Returns:
        torch.Tensor: [B, H, W], cosine similarity values in [-1,1].
    """
    dot = torch.sum(pred * target, dim=1)  # [B, H, W]
    pred_norm = torch.norm(pred, p=2, dim=1)  # [B, H, W]
    target_norm = torch.norm(target, p=2, dim=1)  # [B, H, W]
    denom = pred_norm * target_norm + eps
    cosine = dot / denom
    # Clamp to [-1,1] for numerical stability
    cosine = torch.clamp(cosine, -1.0, 1.0)
    return cosine

def pixelwise_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str='cross_entropy') -> torch.Tensor:
    """
    Compute pixel-wise loss.
    Args:
        pred (torch.Tensor): [B, C, H, W], logits or predicted output.
        target (torch.Tensor): [B, H, W] (class indices) or [B, 1, H, W] (regression).
        loss_type (str): 'cross_entropy' or 'mse'
    Returns:
        torch.Tensor: [B, H, W], per-pixel loss.
    """
    if loss_type == 'cross_entropy':
        # For cross_entropy, pred is logits, target is class indices
        loss = torch.nn.functional.cross_entropy(pred, target, reduction='none')
        # loss: [B, H, W]
        return loss
    elif loss_type == 'mse':
        # For regression tasks
        loss = torch.nn.functional.mse_loss(pred, target, reduction='none')  # shape [B, C, H, W]
        # For MSE, optionally reduce per pixel
        if loss.shape[1] == 1:
            loss = loss.squeeze(1)  # shape [B, H, W]
        else:
            loss = torch.mean(loss, dim=1)  # average over channels
        return loss
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

def compute_scaled_loss(pred: torch.Tensor, target: torch.Tensor, targeted: bool=False, loss_type: str='cross_entropy') -> torch.Tensor:
    """
    Compute pixel-wise loss scaled by cosine similarity or dissimilarity.
    Args:
        pred (torch.Tensor): Model outputs [B, C, H, W].
        target (tensor): For classification: class labels [B, H, W]; For regression: [B, C, H, W].
        targeted (bool): Targeted attack indicator.
        loss_type (str): 'cross_entropy' or 'mse'.
    Returns:
        torch.Tensor: Scalar loss for backprop.
    """
    pred_probs = normalize_tensor(pred, method='softmax')  # normalize for similarity
    # For labels as class indices (segmentation)
    if isinstance(target, torch.Tensor) and target.dtype==torch.long:
        # Convert class indices to one-hot
        num_classes = pred_probs.shape[1]
        y_one_hot = torch.nn.functional.one_hot(target, num_classes).permute(0,3,1,2).float()
    else:
        # assume target is already one-hot or continuous
        y_one_hot = target

    cosine = compute_cosine_similarity(pred_probs, y_one_hot)
    # Compute pixel-wise loss
    pixel_loss = pixelwise_loss(pred, target, loss_type=loss_type)  # shape [B, H, W]
    # Scale factor: (1 - cosine) if targeted else cosine
    if targeted:
        scale = 1.0 - cosine
    else:
        scale = cosine
    # Apply scale
    scaled_loss = scale * pixel_loss
    # Return mean for gradient stability
    return scaled_loss.mean()

def plot_metrics(metrics: dict, title: str='Metrics over Attack Iterations') -> None:
    """
    Plot metrics over attack iterations.
    Args:
        metrics (dict): {metric_name: list of values}
        title (str): Plot title
    """
    plt.figure()
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)
    plt.xlabel('Attack Iterations')
    plt.ylabel('Metric Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def save_image_batch(images: torch.Tensor, filename: str) -> None:
    """
    Save a batch of images to disk.
    Args:
        images (torch.Tensor): [B, C, H, W], assume in [0,1].
        filename (str): Save path ending with .png or .jpg.
    """
    import os
    import torchvision.transforms as T
    # Take first image or combine as grid
    # For demonstration, save first image
    img = images[0].cpu().permute(1,2,0).numpy()
    img = (np.clip(img, 0, 1)*255).astype(np.uint8)
    from PIL import Image
    pil_img = Image.fromarray(img)
    pil_img.save(filename)

def compute_pixelwise_angle(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    """
    Compute the angle between prediction and target vectors in radians.
    Useful for debugging.
    Args:
        pred (torch.Tensor): [..., C]
        target (torch.Tensor): [..., C]
        eps (float): small epsilon
    Returns:
        torch.Tensor: [...], angles in radians
    """
    cosine = compute_cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0), eps)
    angles = torch.acos(torch.clamp(cosine, -1+eps, 1-eps))
    return angles

def log_progress(current: int, total: int, logs: dict) -> None:
    """
    Log progress during attack or evaluation.
    Args:
        current (int): current step.
        total (int): total steps.
        logs (dict): Dictionary of metrics.
    """
    percent = (current+1)/total*100
    metrics_str = ', '.join([f"{k}: {v:.4f}" for k,v in logs.items()])
    print(f"[{current+1}/{total}] ({percent:.1f}%) | {metrics_str}")
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\cospgd\cospgd_repo`
