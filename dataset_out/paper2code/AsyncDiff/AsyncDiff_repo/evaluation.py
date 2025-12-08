## evaluation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from skimage import measure
from scipy.linalg import sqrtm
import os

# Load pretrained models for metrics
# For FID, use torchvision's InceptionV3
from torchvision.models.inception import inception_v3
# For CLIP
import clip
# For DISTS, assume we have a local implementation
# For NIQE, use skimage.measure
# For MUSIQ, assume we have a pretrained model (mocked here)

# Assume external pretrained DISTS implementation
try:
    from dists import DISTS  # You should have a DISTS implementation accessible
except ImportError:
    DISTS = None  # Placeholder

# Assume MUSIQ is available via some library (mocked as function)
def compute_musiq_score(images: np.ndarray) -> np.ndarray:
    # Placeholder: in real code, load pretrained MUSIQ model and compute scores
    # Here, simply return dummy scores
    return np.random.uniform(70, 80, size=(images.shape[0],))

class Evaluation:
    def __init__(self, config: dict):
        """
        Initialize models and datasets needed for metrics.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load CLIP model
        self.clip_model, self.clip_transform = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

        # Load Inception model for FID feature extraction
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()
        # Remove final classification layer
        self.inception_features = nn.Sequential(*list(self.inception_model.children())[:-1])

        # For FID: precompute real dataset features (could be cached outside)
        self.real_features_cache = None
        self._real_features_dataset = None

        # Load DISTS model if available
        if DISTS is not None:
            self.dists_model = DISTS().to(self.device).eval()
        else:
            self.dists_model = None

        # Other configurations
        self.image_size = config.get('image_size', 512)

    def compute_fid(self, generated: torch.Tensor, real: torch.Tensor) -> float:
        """
        Compute FID between generated and real images.
        Inputs:
            generated: torch.Tensor B x C x H x W, values in [0,1]
            real: torch.Tensor B x C x H x W, values in [0,1]
        Returns:
            float: FID score
        """
        # Extract features
        gen_feat = self._get_inception_features(generated)
        real_feat = self._get_inception_features(real)

        mu_gen = gen_feat.mean(dim=0).cpu().numpy()
        sigma_gen = np.cov(gen_feat.cpu().numpy(), rowvar=False)

        mu_real = real_feat.mean(dim=0).cpu().numpy()
        sigma_real = np.cov(real_feat.cpu().numpy(), rowvar=False)

        diff = mu_gen - mu_real
        covmean, _ = sqrtm(sigma_gen @ sigma_real, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid_score = diff @ diff + np.trace(sigma_gen + sigma_real - 2 * covmean)
        return float(fid_score)

    def _get_inception_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get features from inception model.
        Args:
            images: tensor in [0,1], shape B x C x H x W
        """
        with torch.no_grad():
            # Resize images to inception input size (typically 299)
            images_resized = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
            # Normalize as per Inception requirement
            # Assuming images are in [0,1], apply normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1,3,1,1)
            images_norm = (images_resized - mean) / std
            features = self.inception_features(images_norm)
            features = features.squeeze()
        return features

    def compute_clip(self, images: torch.Tensor, texts: List[str]) -> float:
        """
        Compute CLIP similarity between images and texts.
        Args:
            images: tensor B x C x H x W in [0,1]
            texts: list of strings, length B
        """
        with torch.no_grad():
            # Encode images
            image_embeddings = self.clip_model.encode_image(self.clip_transform(images))
            image_embeddings = F.normalize(image_embeddings, dim=-1)

            # Encode texts
            text_tokens = clip.tokenize(texts).to(self.device)
            text_embeddings = self.clip_model.encode_text(text_tokens)
            text_embeddings = F.normalize(text_embeddings, dim=-1)

            # Compute cosine similarity
            similarity = (image_embeddings * text_embeddings).sum(dim=-1)
            mean_similarity = similarity.mean().item()
        return mean_similarity

    def compute_niqe(self, images: np.ndarray) -> float:
        """
        Compute NIQE score for each image, then average.
        Inputs:
            images: numpy array shape B x H x W x C, values in [0,255]
        """
        scores = []
        for img in images:
            # NIQE expects [0,255], grayscale or color
            score = measure.niqe(img)
            scores.append(score)
        return float(np.mean(scores))

    def compute_musiq(self, images: np.ndarray) -> np.ndarray:
        """
        Compute MUSIQ scores (assuming function provided elsewhere)
        Input:
            images: numpy array B x H x W x C, in [0,255]
        """
        scores = compute_musiq_score(images)
        return scores

    def compute_dists(self, generated: torch.Tensor, reference: torch.Tensor) -> float:
        """
        Compute DISTS between generated and reference images.
        Inputs:
            generated, reference: B x C x H x W in [0,1]
        """
        if self.dists_model is None:
            # If DISTS is not available, return NaN or dummy
            return float('nan')
        with torch.no_grad():
            gen = generated.to(self.device)
            ref = reference.to(self.device)
            score = self.dists_model(gen, ref)
        return float(score)

    def evaluate(
        self,
        generated_outputs: List[torch.Tensor],
        ground_truth: Optional[List[torch.Tensor]] = None,
        prompts: Optional[List[str]] = None,
        real_images: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Compute all specified metrics given generated and ground-truth data.

        Args:
            generated_outputs: list of tensors (B x C x H x W), in [0,1]
            ground_truth: ground truth images (for FID, DISTS), same format
            prompts: list of prompts for CLIP
            real_images: list of real images for FID, etc.

        Returns:
            dict: scores for each metric
        """
        results = {}

        # Convert list of tensors into batch ND arrays in [0,1]
        gen_batch = torch.stack(generated_outputs, dim=0).clamp(0,1)

        # FID
        if ground_truth is not None:
            real_batch = torch.stack(ground_truth, dim=0).clamp(0,1)
            fid_score = self.compute_fid(gen_batch, real_batch)
            results['FID'] = fid_score

        # CLIP
        if prompts is not None:
            # For CLIP, consider prompt for each sample
            clip_score = self.compute_clip(gen_batch, prompts)
            results['CLIP'] = clip_score

        # NIQE
        niqe_scores = []
        gen_np = gen_batch.permute(0,2,3,1).cpu().numpy() * 255  # Convert to H,W,C and [0,255]
        for img in gen_np:
            niqe_score = measure.niqe(img)
            niqe_scores.append(niqe_score)
        results['NIQE'] = float(np.mean(niqe_scores))

        # MUSIQ
        musiq_scores = self.compute_musiq(gen_np)
        results['MUSIQ'] = float(np.mean(musiq_scores))

        # DISTS
        if ground_truth is not None and self.dists_model is not None:
            ref_batch = torch.stack(ground_truth, dim=0).clamp(0,1)
            dists_score = self.compute_dists(gen_batch, ref_batch)
            results['DISTS'] = dists_score

        return results
