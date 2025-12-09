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
