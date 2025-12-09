## utils.py
import torch
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np

def add_gaussian_noise(input: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Adds Gaussian noise to the input tensor.
    
    Args:
        input (torch.Tensor): Input tensor of shape (B, C, H, W) or similar.
        sigma (float): Standard deviation of the Gaussian noise.
    
    Returns:
        torch.Tensor: Noisy tensor of the same shape as input.
    """
    noise = torch.randn_like(input) * sigma
    return input + noise

def compute_coefficients(w: torch.Tensor, sigma1: float, sigma2: float, epsilon: float = 1e-8):
    """
    Computes pixel-wise coefficients c1 and c2 for unbiased combination of m1 and m2 predictions,
    minimizing variance under the constraint c1 + w * c2 = 1.
    
    Args:
        w (torch.Tensor): mask weights, shape (B, 1, H, W), values in [0,1].
        sigma1 (float): Standard deviation used in m1.
        sigma2 (float): Standard deviation used in m2.
        epsilon (float): Small constant for numerical stability.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: c1 and c2 tensors, shape matching w, suitable for combining m1 and m2.
    """
    # Compute norm_2 of w (per sample)
    w_flat = w.view(w.shape[0], -1)
    norm_w2 = torch.norm(w_flat, p=2, dim=1, keepdim=True)  # shape (B,1)
    
    # Expand for pixel-wise coefficients
    w_sq = w ** 2
    # Compute denominator avoiding division by zero
    denom = (w_sq.sum(dim=(2,3), keepdim=True) * sigma2 ** 2 + norm_w2 ** 2 * sigma1 ** 2) + epsilon
    
    # Coefficient c1: for m1
    c1 = (w_sq * sigma2 ** 2) / denom
    c1 = c1 * sigma1 ** 2  # shape (B,1,H,W)
    
    # Coefficient c2: for m2
    denom_c2 = (w_sq * sigma1 ** 2 + norm_w2 ** 2 * sigma2 ** 2) + epsilon
    c2 = (sigma1 ** 2 * w) / denom_c2  # same shape as w
    
    return c1, c2

def combine_predictions(m1: torch.Tensor, m2: torch.Tensor, w: torch.Tensor, sigma1: float, sigma2: float) -> torch.Tensor:
    """
    Combine two noisy predictions m1 and m2 into an unbiased estimate with minimized variance.
    
    Args:
        m1 (torch.Tensor): First noisy prediction, shape (B, C, H, W).
        m2 (torch.Tensor): Second noisy prediction, shape same as m1.
        w (torch.Tensor): Mask weights, shape (B, 1, H, W), values in [0,1].
        sigma1 (float): Noise level in m1.
        sigma2 (float): Noise level in m2.
    
    Returns:
        torch.Tensor: Combined estimate $\hat{x}$, shape (B, C, H, W).
    """
    c1, c2 = compute_coefficients(w, sigma1, sigma2)

    hat_x = c1 * m1 + c2 * w * m2
    return hat_x

def compute_cert_radius(p_plus: float, p_minus: float, sigma: float) -> float:
    """
    Compute certified radius (e.g., Eq. 2.2) for classification based on class probabilities.
    
    Args:
        p_plus (float): Estimated probability of the top class.
        p_minus (float): Estimated probability of the top competing class.
        sigma (float): Noise scale used in the Gaussian mechanism.
    
    Returns:
        float: Certified robustness radius.
    """
    # Clip probabilities to avoid invalid values
    p_plus = min(max(p_plus, 1e-10), 1 - 1e-10)
    p_minus = min(max(p_minus, 1e-10), 1 - 1e-10)

    # Compute inverse CDF
    inv_p_plus = norm.ppf(p_plus)
    inv_p_minus = norm.ppf(p_minus)

    radius = (sigma / 2.0) * (inv_p_plus - inv_p_minus)
    return radius

def estimate_class_probabilities(input: torch.Tensor, classifier: callable, num_samples: int, sigma: float, device: torch.device = torch.device('cpu')) -> dict:
    """
    Estimates class probabilities via Monte Carlo sampling with noisy inputs.
    
    Args:
        input (torch.Tensor): Input tensor, shape (C, H, W), original input.
        classifier (callable): Classifier function accepting tensor batch, returning class probabilities.
        num_samples (int): Number of Monte Carlo samples.
        sigma (float): Noise level used in addition to the input.
        device (torch.device): Device for computation.
    
    Returns:
        dict: Mapping class label -> estimated probability.
    """
    input = input.to(device)
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            noisy_input = add_gaussian_noise(input.unsqueeze(0), sigma).squeeze(0)
            probs = classifier(noisy_input.unsqueeze(0))
            pred_class = probs.argmax(dim=1).item()
            predictions.append(pred_class)

    # Count predictions
    counts = {}
    for c in predictions:
        counts[c] = counts.get(c, 0) + 1

    class_probs = {}
    total = float(len(predictions))
    for c, count in counts.items():
        class_probs[c] = count / total

    return class_probs

def search_sigma(validation_function: callable, sigma_values: list, *args, **kwargs):
    """
    Performs grid search over sigma values for the best certified accuracy.
    
    Args:
        validation_function (callable): Function to evaluate model performance at given sigma.
        sigma_values (list): List of sigma values to evaluate.
        *args, **kwargs: Additional arguments for validation_function.
    
    Returns:
        float: Best sigma value.
        dict: Performance metrics for each sigma.
    """
    best_sigma = None
    best_score = -np.inf
    results = {}
    for sigma in sigma_values:
        score = validation_function(sigma, *args, **kwargs)
        results[sigma] = score
        if score > best_score:
            best_score = score
            best_sigma = sigma
    return best_sigma, results

def search_beta(validation_function: callable, beta_values: list, *args, **kwargs):
    """
    Performs grid search over beta (distribution tail parameter) for tuning.
    
    Args:
        validation_function (callable): Function to evaluate model at given beta.
        beta_values (list): List of beta values.
        *args, **kwargs: Additional args.
    
    Returns:
        float: Best beta.
        dict: Performance metrics per beta.
    """
    best_beta = None
    best_score = -np.inf
    results = {}
    for beta in beta_values:
        score = validation_function(beta, *args, **kwargs)
        results[beta] = score
        if score > best_score:
            best_score = score
            best_beta = beta
    return best_beta, results

def load_background_images(background_paths: list, background_scale: int = 640) -> list:
    """
    Load and resize background images from file paths.
    
    Args:
        background_paths (list): List of file paths to images.
        background_scale (int): Target size (width and height).
    
    Returns:
        List[Image.Image]: Resized background images.
    """
    from PIL import Image
    bg_images = []
    for path in background_paths:
        img = Image.open(path).convert('RGB')
        img_resized = img.resize((background_scale, background_scale), Image.BILINEAR)
        bg_images.append(img_resized)
    return bg_images

def resize_backgrounds(bg_images: list, background_scale: int) -> list:
    """
    Resize a list of background images to a specific size.
    
    Args:
        bg_images (list): List of PIL Images.
        background_scale (int): Target size.
    
    Returns:
        list: Resized PIL Images.
    """
    resized_bgs = []
    for img in bg_images:
        resized_bgs.append(img.resize((background_scale, background_scale), Image.BILINEAR))
    return resized_bgs
