## evaluation.py
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.linalg import sqrtm
from typing import List
import torch.nn.functional as F

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained InceptionV3 model for feature extraction
# We will extract features from the pool3 layer for FID
class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        # We want activations from the pool3 layer
        # In torchvision's inception, pool3 is the last pooling layer
        # We can access it via hook or by setting output
        # To keep it simple, use hook to grab features
        
        self.feature = None
        def hook_fn(module, input, output):
            self.feature = output.detach()

        # Register hook on 'avgpool'
        self.model.Mixed_7a.register_forward_hook(hook_fn)

    def forward(self, x):
        # x: tensors RGB images [B,3,H,W], normalized if needed
        # Resize to 299x299 as expected
        # Expect input in [0,1], normalize to ImageNet mean/std
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        x = (x - mean) / std
        # Forward
        self.feature = None
        _ = self.model(x)
        # feature stored in self.feature
        return self.feature.cpu().numpy()

# Instantiate feature extractor globally
feature_extractor = InceptionFeatureExtractor()

def extract_features(images: List, batch_size: int = 50):
    """
    Extract features (activations from pool3 layer) for a list of images.
    Args:
        images (List or np.ndarray): list of PIL Images or numpy arrays in [0,1]
        batch_size (int): batch size for processing
    Returns:
        numpy.ndarray: shape (N, feature_dim)
    """
    features = []
    n = len(images)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            imgs_batch = images[i:i+batch_size]
            if isinstance(imgs_batch[0], np.ndarray):
                imgs_tensor = torch.stack([torch.from_numpy(np.transpose(img, (2,0,1))) for img in imgs_batch])  # [B,3,H,W]
            else:  # assume PIL Image
                imgs_tensor = torch.stack([T.ToTensor()(img) for img in imgs_batch])
            feats = feature_extractor(imgs_tensor.to(device))
            features.append(feats)
    return np.concatenate(features, axis=0)

def compute_mean_covariance(features: np.ndarray):
    """
    Compute mean and covariance of features.
    Args:
        features: numpy array (N, D)
    Returns:
        mean: (D,)
        cov: (D, D)
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    # Add epsilon to diagonal for numerical stability
    eps = 1e-6
    sigma += np.eye(sigma.shape[0]) * eps
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2):
    """
    Calculate Frechet Inception Distance between two distributions.
    """
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    # Numerical stability: if covmean yields imaginary part, take real
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid

def compute_fid(generated_images: List, reference_images: List, batch_size: int = 50):
    """
    Compute FID score between generated images and reference images.
    Args:
        generated_images: list of PIL images or numpy arrays in [0,1]
        reference_images: list of images in the same format
        batch_size: batch size for feature extraction
    Returns:
        float: FID score
    """
    gen_feats = extract_features(generated_images, batch_size)
    ref_feats = extract_features(reference_images, batch_size)
    mu_gen, sigma_gen = compute_mean_covariance(gen_feats)
    mu_ref, sigma_ref = compute_mean_covariance(ref_feats)
    fid_score = calculate_fid(mu_ref, sigma_ref, mu_gen, sigma_gen)
    return fid_score

def compute_inception_score(generated_images: List, batch_size: int = 50, splits: int = 10):
    """
    Compute the Inception Score (IS) for generated images.
    Args:
        generated_images: list of PIL images or np arrays
        batch_size: batch size
        splits: number of splits for estimation
    Returns:
        float: average IS
    """
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(generated_images), batch_size):
            imgs_batch = generated_images[i:i+batch_size]
            if isinstance(imgs_batch[0], np.ndarray):
                imgs_tensor = torch.stack([torch.from_numpy(np.transpose(img, (2,0,1))) for img in imgs_batch])
            else:
                imgs_tensor = torch.stack([T.ToTensor()(img) for img in imgs_batch])
            imgs_tensor = imgs_tensor.to(device)
            # Resize and normalize as in feature extractor
            _ = feature_extractor.model.transforms
            # For inference, get softmax outputs
            preds = feature_extractor.model(imgs_tensor)
            probs = F.softmax(torch.from_numpy(preds), dim=1)  # shape: [B,1000]
            all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    # Compute splits for stable estimation
    split_scores = []
    n_images = all_probs.shape[0]
    split_size = n_images // splits
    for k in range(splits):
        part = all_probs[k*split_size:(k+1)*split_size]
        p_y = np.mean(part, axis=0)
        kl_divs = part * (np.log(part + 1e-6) - np.log(p_y + 1e-6))
        kl_divs = np.sum(kl_divs, axis=1)
        split_score = np.exp(np.mean(kl_divs))
        split_scores.append(split_score)
    return np.mean(split_scores)

def compute_precision_recall(generated_features: np.ndarray,
                             real_features: np.ndarray,
                             k_neighbors: int = 3):
    """
    Compute Precision and Recall between generated and real features.
    Implementation as in Kynkäänniemi et al., 2019
    Args:
        generated_features: numpy array (N, D)
        real_features: numpy array (M, D)
        k_neighbors: number of neighbors for k-NN coverage
    Returns:
        tuple: (precision, recall)
    """
    # Use nearest neighbors in feature space
    from sklearn.neighbors import NearestNeighbors

    # Precision: fraction of generated features within real data manifold
    nbrs_real = NearestNeighbors(n_neighbors=k_neighbors).fit(real_features)
    distances, _ = nbrs_real.kneighbors(generated_features)
    # For each generated feature, check if within threshold (distance to kth neighbor)
    # For a simplified approach, threshold can be the max distance to kth neighbor
    radius_real = np.max(distances, axis=1)
    # Count how many generated features are within radius of real features
    nbrs_gen = NearestNeighbors(n_neighbors=k_neighbors).fit(generated_features)
    distances_gen, _ = nbrs_gen.kneighbors(real_features)
    radius_gen = np.max(distances_gen, axis=1)
    # Precision calculation
    radius_threshold = np.median(radius_real)
    precision = np.sum(distances[:, -1] < radius_threshold) / len(distances)
    # Recall calculation
    recall = np.sum(distances_gen[:, -1] < radius_threshold) / len(real_features)
    return float(precision), float(recall)

# Example utility function to compute features and metrics for a set of images
def evaluate_metrics(generated_images: List, real_images: List,
                     batch_size: int = 50,
                     splits: int = 10):
    """
    Compute all metrics: FID, IS, Precision, Recall.
    """
    # Extract features
    gen_feats = extract_features(generated_images, batch_size)
    real_feats = extract_features(real_images, batch_size)

    # FID
    mu_gen, sigma_gen = compute_mean_covariance(gen_feats)
    mu_real, sigma_real = compute_mean_covariance(real_feats)
    fid_value = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)

    # IS
    # Prepare images in PIL or np format for IS calculation, see above
    # For simplicity, assume images are in [0,1] np arrays
    is_value = compute_inception_score(generated_images, batch_size, splits)

    # Precision & Recall
    prec, rec = compute_precision_recall(gen_feats, real_feats)

    return {
        'FID': fid_value,
        'IS': is_value,
        'Precision': prec,
        'Recall': rec
    }
