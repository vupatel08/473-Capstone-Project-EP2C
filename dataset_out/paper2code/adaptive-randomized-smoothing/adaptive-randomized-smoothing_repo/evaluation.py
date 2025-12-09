## evaluation.py
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
from math import sqrt
from utils import add_gaussian_noise, compute_coefficients, combine_predictions, compute_cert_radius

class Evaluator:
    def __init__(self, mask_unet, classifier, sigma1, sigma2, device, background_scale=1):
        """
        Initialize the evaluator with trained models and parameters.
        Args:
            mask_unet (nn.Module): Mask model w (predicts mask w(m1))
            classifier (nn.Module): Base classifier g
            sigma1 (float): Noise scale for first step (mask generation)
            sigma2 (float): Noise scale for second step (masked image)
            device (torch.device): Computation device
            background_scale (int): Scale factor for background in certification formulas
        """
        self.mask_unet = mask_unet
        self.classifier = classifier
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.device = device
        self.background_scale = background_scale

    def predict_mask(self, noisy_input):
        """
        Generate mask w(m1) given noisy input.
        Args:
            noisy_input (torch.Tensor): Input tensor, shape (C,H,W).
        Returns:
            torch.Tensor: Mask w in [0,1], shape (1,H,W).
        """
        with torch.no_grad():
            mask_raw = self.mask_unet(noisy_input.unsqueeze(0).to(self.device))
            mask = torch.sigmoid(mask_raw)
        return mask.squeeze(0)  # shape: (1,H,W)

    def generate_masked_noisy_image(self, w, X):
        """
        Generate m2 = masked X + noise, with adaptive noise scale.
        Args:
            w (torch.Tensor): mask, shape (1,H,W).
            X (torch.Tensor): original input, shape (C,H,W).
        Returns:
            m2 (torch.Tensor): noisy masked image, shape (C,H,W).
            sigma_w (float): noise scale for m2
        """
        # Compute norm_2 of w for noise scale in Eq. 2.5
        w_flat = w.view(-1)
        norm_w2 = torch.norm(w_flat, p=2).item()
        d = w.numel()  # total number of pixels in mask
        # variance of noise in m2
        noise_std = (norm_w2 / np.sqrt(d)) * self.sigma2
        # Add Gaussian noise
        z2 = torch.randn_like(X) * noise_std
        masked_X = w * X
        m2 = masked_X + z2
        return m2, noise_std

    def predict(self, X, n_samples=1000, class_prior=None):
        """
        Perform Monte Carlo prediction and classify.
        Args:
            X (torch.Tensor): input image, shape (C,H,W)
            n_samples (int): number of MC samples
            class_prior (torch.Tensor): optional prior probabilities for classes, shape (num_classes,)
        Returns:
            dict: containing class probabilities, predicted class, lower bounds, radius
        """
        self.classifier.eval()
        # Store class counts for class probability estimation
        class_counts = None
        # Store predictions
        probs_list = []

        for _ in range(n_samples):
            # Sample z1 noise
            z1 = add_gaussian_noise(torch.zeros_like(X), self.sigma1).to(self.device)
            noisy_input = X + z1

            # 1. Generate mask w(m1)
            w = self.predict_mask(noisy_input)  # shape: (1,H,W)
            # 2. Generate m2 = masked X + noise
            m2, sigma_w = self.generate_masked_noisy_image(w, X)

            # Compute coefficients c1, c2 for unbiased combination
            c1, c2 = compute_coefficients(w, self.sigma1, sigma_w)

            # 3. Predict for m1 (original noisy input for m1 for consistency)
            # Alternatively, to follow the paper's estimator, combine the images:
            # Instead of logits, it's better to combine images via the estimated unbiased estimate
            # Here, for practical intuition, pass m2 through classifier
            # For the training purpose, we perform one forward per sample
            
            # Forward classifier
            logits = self.classifier(m2.unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs.squeeze(0))
        
        # Aggregate class probabilities over samples
        probs_stack = torch.stack(probs_list, dim=0)  # shape: (n_samples, num_classes)
        class_probs_mean = probs_stack.mean(dim=0)  # mean probability per class
        # Find predicted class
        pred_class = torch.argmax(class_probs_mean).item()
        p_plus = class_probs_mean[pred_class].item()
        # For bound estimation, we need class counts
        pred_counts = (probs_stack.argmax(dim=1) == pred_class).sum().item()

        # Count for other classes
        class_counts = torch.zeros(probs_stack.shape[1])
        class_counts = torch.zeros_like(class_probs_mean)
        for c_idx in range(probs_stack.shape[1]):
            class_counts[c_idx] = (probs_stack.argmax(dim=1) == c_idx).sum()

        # Compute confidence bounds
        p_plus_lower, p_minus_upper = self.compute_bounds(class_counts, pred_class,
                                                          conf_level=0.99, error_tol=0.01)

        # Compute the certified radius
        radius = self.compute_radius(p_plus_lower, p_minus_upper, self.sigma1, 
                                      dim=w.numel(), norm_type='L_infinity', bg_scale=self.background_scale)

        # Prepare output
        result = {
            'predicted_class': pred_class,
            'confidence_lower_bound': p_plus_lower,
            'confidence_upper_bound': p_minus_upper,
            'radius': radius,
            'class_probabilities': class_probs_mean.detach().cpu().numpy()
        }
        return result

    def compute_bounds(self, class_counts, pred_class_idx, conf_level=0.99, error_tol=0.01):
        """
        Using binomial confidence intervals (Clopper-Pearson, normal approximation),
        compute lower/upper bounds for class probabilities.
        Args:
            class_counts (torch.Tensor): counts of each class from MC, shape: (num_classes,)
            pred_class_idx (int): index of predicted class
            conf_level (float): e.g., 0.99
            error_tol (float): confidence interval tolerance for bounds
        Returns:
            p_plus_lower (float): lower bound for top class
            p_minus_upper (float): upper bound for max of other classes
        """
        total = class_counts.sum().item()
        p_c = class_counts[pred_class_idx].item() / total
        # Confidence interval for p_plus
        lower_p_plus = self.estimate_binom_bound(count=class_counts[pred_class_idx].item(),
                                                  n=total, conf_level=conf_level)
        # For top class's lower bound
        p_plus_lower = lower_p_plus

        # For other classes: find maximum over classes != pred_class
        max_other = max([class_counts[i].item() for i in range(len(class_counts)) if i != pred_class_idx])
        p_minus_upper = self.estimate_binom_bound(count=max_other,
                                                  n=total, conf_level=conf_level, upper=True)
        return p_plus_lower, p_minus_upper

    def estimate_binom_bound(self, count, n, conf_level=0.99, upper=False):
        """
        Use normal approximation or Clopper-Pearson for binomial CI.
        Args:
            count (int): number of successes
            n (int): total trials
            conf_level (float): confidence level
            upper (bool): if True, compute upper bound, else lower.
        Returns:
            float: probability bound
        """
        p_hat = count / n
        alpha = 1 - conf_level
        z = norm.ppf(1 - alpha/2)
        # standard error
        se = sqrt(p_hat * (1 - p_hat) / n)
        if upper:
            p_bound = p_hat + z * se
        else:
            p_bound = p_hat - z * se
        return min(max(p_bound, 0), 1)

    def compute_radius(self, p_plus, p_minus, sigma, dim, norm_type='L_infinity', bg_scale=1):
        """
        Calculate the certified radius using Eq. 2.2 or 2.4.
        Args:
            p_plus (float): lower bound on correct class probability
            p_minus (float): upper bound on maximum other class probability
            sigma (float): noise scale
            dim (int): dimension of input (e.g., number of pixels in mask image)
            norm_type (str): 'L_infinity' or default (for this code, 'L_infinity')
            bg_scale (int): scale factor of background for $L_\infty$ bounds
        Returns:
            float: certified radius
        """
        # For L-infinity, formula includes division by sqrt(d)
        if norm_type == 'L_infinity':
            radius = (sigma / (2 * bg_scale)) * (norm.ppf(p_plus) - norm.ppf(p_minus))
        else:
            # default: standard
            radius = (sigma / 2) * (norm.ppf(p_plus) - norm.ppf(p_minus))
        # Ensure radius is non-negative
        return max(radius, 0.0)

    def get_certified_radius(self, X, n_samples=1000, conf_level=0.99, error_tol=0.01):
        """
        Main interface: Given input X, return class, radius, and probability bounds.
        Args:
            X (torch.Tensor): input image, shape (C,H,W)
            n_samples (int): Monte Carlo sample count
            conf_level (float): confidence level
            error_tol (float): epsilon for bounds
        Returns:
            dict: class prediction, certified radius, and bounds.
        """
        result = self.predict(X, n_samples, conf_level)
        return result
