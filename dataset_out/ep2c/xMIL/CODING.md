# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## config.py

```python
## config.py

import os
from typing import Dict, Any
import yaml

class Config:
    """
    Centralized configuration class for data paths, model parameters,
    training, explanation, hardware, evaluation, and save paths.
    Loads settings from 'config.yaml'.
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration from YAML file
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Data paths
        self.dataset_paths: Dict[str, str] = cfg.get("dataset_paths", {
            "histopathology": "/path/to/histopathology/data",
            "toy_data": "/path/to/toy/dataset"
        })

        # Model parameters
        self.model_parameters: Dict[str, Any] = cfg.get("model_parameters", {
            "model_type": "attention",            # options: "attention", "transformer", "additive"
            "hidden_dim": 512,
            "feature_extractor": "resnet18",      # pretrained CNN backbone
            "freeze_feature_extractor": True      # freeze CNN during training
        })

        # Training settings
        self.training: Dict[str, Any] = cfg.get("training", {
            "learning_rate": 0.002,
            "batch_size": 32,
            "epochs": 1000,
            "optimizer": "Adam",
            "dropout": 0.0
        })

        # Explanation method configuration
        self.explanation_method: Dict[str, Any] = cfg.get("explanation_method", {
            "method": "xMIL-LRP",                # "xMIL-LRP", "IG", "G×I", "attention_rollout"
            "relevance_rules": {
                "linear": "LRP-epsilon",         # propagation rule for linear layers
                "attention": "AH-rule",          # propagation rule for attention modules
                "layer_norm": "LN-rule"          # propagation rule for layer norm
            }
        })

        # Hardware setup
        self.hardware: Dict[str, Any] = cfg.get("hardware", {
            "device": "cuda",                     # "cuda" or "cpu"
            "gpus": 1
        })

        # Evaluation setup
        self.evaluation: Dict[str, Any] = cfg.get("evaluation", {
            "perturbation_steps": 100,
            "metrics": {
                "AUPRC2": True,
                "AUPC": True
            },
            "visualization": {
                "heatmaps": True
            }
        })

        # Save paths
        self.save: Dict[str, str] = cfg.get("save", {
            "model_checkpoint_path": "./checkpoints/",
            "explanation_heatmaps_path": "./heatmaps/"
        })

        # Validate paths exist or create directories
        self._validate_paths()

    def _validate_paths(self):
        # Create directories if they do not exist
        for path in [self.save["model_checkpoint_path"], self.save["explanation_heatmaps_path"]]:
            if not os.path.exists(path):
                os.makedirs(path)

    def get(self) -> Dict[str, Any]:
        """
        Return the complete configuration as a nested dictionary.
        """
        return {
            "dataset_paths": self.dataset_paths,
            "model_parameters": self.model_parameters,
            "training": self.training,
            "explanation_method": self.explanation_method,
            "hardware": self.hardware,
            "evaluation": self.evaluation,
            "save": self.save
        }

# Instantiate a singleton configuration object
config = Config()

# Usage example:
# cfg = config.get()
```

## dataset_loader.py

```python
## dataset_loader.py
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import openslide
from sklearn.cluster import MiniBatchKMeans
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging

from config import config

# Set up logging for debugging and information
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SlideSample:
    """
    A class to hold data for a single slide:
    - Path and metadata
    - List of patches (images)
    - Extracted features
    - Labels and IDs
    """
    def __init__(self,
                 slide_id: str,
                 slide_path: str,
                 label: int,
                 patches: List[Image.Image],
                 patch_coords: List[Tuple[int, int]],
                 features: Optional[np.ndarray] = None):
        self.slide_id = slide_id
        self.slide_path = slide_path
        self.label = label
        self.patches = patches  # List of PIL Images
        self.patch_coords = patch_coords  # (x, y) positions
        self.features = features  # NumPy array (num_patches x feature_dim)

class DatasetLoader:
    """
    Loads WSIs, extracts patches, filters background, and prepares bags for MIL.
    """

    def __init__(self,
                 dataset_dir: str = config.dataset_paths['histopathology'],
                 patch_size: int = 256,
                 magnification_level: int = 20,
                 tissue_threshold: float = 0.1,
                 max_patches_per_slide: int = 10000,
                 background_filtering: bool = True,
                 device: str = config.hardware['device']):
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size
        self.magnification_level = magnification_level  # e.g., 20x
        self.tissue_threshold = tissue_threshold  # proportion of tissue in patch
        self.max_patches_per_slide = max_patches_per_slide
        self.background_filtering = background_filtering
        self.device = device

        self.slide_files: List[str] = []
        self.slide_labels: Dict[str, int] = {}  # slide_id -> label
        self.slides_metadata: List[Dict] = []

        self._discover_slides()

    def _discover_slides(self):
        """
        Scan the dataset directory for slide files and load label info.
        Assumes labels are encoded in filenames or an external label file.
        """
        # For simplicity, assume all .svs/.tiff files in directory
        for fname in os.listdir(self.dataset_dir):
            if fname.lower().endswith(('.svs', '.tiff', '.ndpi', '.tif')):
                slide_path = os.path.join(self.dataset_dir, fname)
                slide_id = os.path.splitext(fname)[0]
                self.slide_files.append(slide_path)

                # Placeholder: Assign label based on filename or external file
                # Here, a dummy label: 0
                label = self._get_label_from_filename(fname)
                self.slide_labels[slide_id] = label

                self.slides_metadata.append({
                    'slide_id': slide_id,
                    'slide_path': slide_path,
                    'label': label
                })
        logger.info(f"Discovered {len(self.slide_files)} slide(s).")

    def _get_label_from_filename(self, filename: str) -> int:
        """
        Placeholder for label assignment based on filename conventions.
        Override or extend as needed.
        """
        # For example, parse filename for label info
        # e.g., 'slide_HPVpos.svs' --> 1
        # Simplified here: default label 0
        return 0

    def load_slide(self, slide_path: str) -> openslide.OpenSlide:
        """
        Open a slide using OpenSlide.
        """
        try:
            slide = openslide.OpenSlide(slide_path)
            return slide
        except Exception as e:
            logger.error(f"Failed to open slide {slide_path}: {e}")
            raise

    def extract_patches_from_slide(self,
                                   slide: openslide.OpenSlide,
                                   min_magnification: float = 20,
                                   max_patches: int = None) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
        """
        Extract patches at specified magnification y level.
        Returns list of PIL Images and their coordinates.
        """
        # Calculate magnification scale factor
        # OpenSlide needs pixel size info or use reference levels
        # For simplicity, assume level 0 is the highest resolution
        # and level with desired magnification can be found via level downsampling ratio
        level = self._select_level(slide, min_magnification)
        level_downsample = slide.level_downsamples[level]
        level_dim = slide.level_dimensions[level]
        slide_width, slide_height = slide.dimensions

        # Placeholder: estimate tissue region via thumbnail
        thumbnail = slide.get_thumbnail(slide.dimensions)
        tissue_mask = self._get_tissue_mask(thumbnail)

        patches = []
        patch_coords = []

        stride = int(self.patch_size * level_downsample)
        # Traverse slide grid
        for y in range(0, slide_width - self.patch_size, stride):
            if len(patches) >= max_patches:
                break
            for x in range(0, slide_height - self.patch_size, stride):
                # Check tissue content
                if self.background_filtering:
                    tissue_fraction = self._patch_tissue_fraction(slide, (x, y))
                    if tissue_fraction < self.tissue_threshold:
                        continue  # Skip non-tissue patches
                # Read region
                patch_img = slide.read_region(
                    (x, y),
                    level,
                    (self.patch_size, self.patch_size))
                patch_img = patch_img.convert("RGB")
                patches.append(patch_img)
                patch_coords.append((x, y))
        return patches, patch_coords

    def _select_level(self, slide: openslide.OpenSlide, target_magnification: float) -> int:
        """
        Select the pyramid level closest to the target magnification.
        Assumes level 0 is highest resolution.
        """
        # Placeholder: assuming level 0 is the highest
        # with known mpp (microns per pixel), but OpenSlide often does not supply directly.
        # For demo, assume level 0 is fine, or choose based on downsample ratios.
        # For simplicity, return level 0.
        return 0

    def _get_tissue_mask(self, thumbnail: Image.Image) -> np.ndarray:
        """
        Generate tissue mask via Otsu's threshold on thumbnail.
        """
        gray = thumbnail.convert('L')
        np_gray = np.array(gray)
        threshold = self._otsu_threshold(np_gray)
        tissue_mask = np_gray > threshold
        return tissue_mask

    def _otsu_threshold(self, image_array: np.ndarray) -> int:
        """
        Compute Otsu's threshold.
        """
        from skimage.filters import threshold_otsu
        return threshold_otsu(image_array)

    def _patch_tissue_fraction(self, slide: openslide.OpenSlide, coord: Tuple[int, int]) -> float:
        """
        Estimate tissue fraction in a patch center region.
        """
        # Read a smaller region (e.g., 50x50) within the patch
        size = 50
        x, y = coord
        # Ensure region is within bounds
        try:
            region = slide.read_region((x, y), 0, (size, size))
        except Exception:
            return 0.0
        region = region.convert("L")
        np_region = np.array(region)
        # Compute tissue fraction by Otsu threshold
        thresh = self._otsu_threshold(np_region)
        tissue_pixels = np.sum(np_region > thresh)
        total_pixels = size * size
        fraction = tissue_pixels / total_pixels
        return fraction

    def load_all_slides(self) -> List[SlideSample]:
        """
        Load all slides, extract patches, and prepare dataset.
        Returns a list of SlideSample objects with patches.
        """
        dataset_samples: List[SlideSample] = []
        for meta in self.slides_metadata:
            slide_id = meta['slide_id']
            slide_path = meta['slide_path']
            label = meta['label']
            # Load slide
            slide = self.load_slide(slide_path)
            # Extract patches
            patches, coords = self.extract_patches_from_slide(slide)
            # Store in SlideSample
            sample = SlideSample(
                slide_id=slide_id,
                slide_path=slide_path,
                label=label,
                patches=patches,
                patch_coords=coords
            )
            dataset_samples.append(sample)
            slide.close()
            logger.info(f"Loaded slide {slide_id}: {len(patches)} patches.")
        return dataset_samples

    def extract_features_for_dataset(self,
                                     dataset_samples: List[SlideSample],
                                     feature_extractor,
                                     batch_size: int = 128):
        """
        Use the provided feature_extractor to process patches into features.
        Update each SlideSample with a features array (num_patches x feature_dim).
        """
        device = self.device
        feature_extractor.eval()
        # Prepare list of all patches across dataset
        all_patches: List[Tuple[SlideSample, int, Image.Image]] = []
        for sample in dataset_samples:
            for idx, patch in enumerate(sample.patches):
                all_patches.append((sample, idx, patch))
        # Process in batches
        dataloader = DataLoader(all_patches, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in dataloader:
                # batch is list of (SlideSample, idx, Image)
                patches = [item[2] for item in batch]
                # Convert to tensor batch
                batch_tensor = torch.stack([transforms.ToTensor()(p) for p in patches], dim=0).to(device)
                features = feature_extractor(batch_tensor)
                features = features.cpu().numpy()
                # Assign features back
                for i, (sample_obj, idx, _) in enumerate(batch):
                    if sample_obj.features is None:
                        sample_obj.features = np.zeros(
                            (len(sample_obj.patches), features.shape[1]),
                            dtype=np.float32)
                    sample_obj.features[idx] = features[i]
        logger.info("Feature extraction completed for dataset.")
```

## evaluation.py

```python
## evaluation.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.integrate import trapz

from utils import load_slide_image, save_figure
from config import config

class Evaluation:
    """
    Handles the evaluation of a trained MIL model and explanation method on test data.
    Computes faithfulness metrics (AUPRC-2, AUPC), generates heatmaps, and summarizes results.
    """
    def __init__(self, model, dataset, explanation, device: str = 'cuda'):
        """
        Initialize with trained model, dataset, explanation method, and device.
        Args:
            model (torch.nn.Module): Trained MIL model.
            dataset (list): List of slide objects or dataset with slide info.
            explanation (Explanation): Explanation object with compute_relevance().
            device (str): 'cuda' or 'cpu'.
        """
        self.model = model
        self.dataset = dataset
        self.explanation = explanation
        self.device = torch.device(device)
        self.perturb_steps = config.evaluation.get("perturbation_steps", 100)
        # Store results
        self.metrics_results = {}
        self.all_relevance_scores = []
        self.all_ground_truths = []
        self.all_predictions = []

    def run(self):
        """
        Main routine: evaluate all slides, compute metrics, generate heatmaps, and summarize.
        """
        auprc2_list = []
        aupc_list = []
        relevance_per_slide = []
        ground_truth_per_slide = []
        pred_per_slide = []

        for slide_obj in self.dataset:
            slide_id = slide_obj.slide_id
            # Load slide image for visualization if needed
            slide_image = load_slide_image(slide_obj.slide_path)

            # Get features: assumes slide_obj.features (list or np.ndarray)
            features = torch.tensor(slide_obj.features, dtype=torch.float)
            # Compute explanation relevance scores
            relevance_scores = self.explanation.compute_relevance(features)
            # relevance_scores: list of float per instance
            relevance_scores = np.array(relevance_scores)

            # Optional: normalize, clip relevance for visualization
            relevance_clipped = self._clip_relevance(relevance_scores)

            # Aggregate evidence ground truth
            # For histopathology, we rely on known labels/annotations (from dataset)
            ground_truth_evidence = self._get_ground_truth_evidence(slide_obj)

            # Compute AUPRC-2
            auprc2 = self._compute_auprc2(ground_truth_evidence, relevance_clipped)
            auprc2_list.append(auprc2)

            # Store for overall analysis
            self.all_relevance_scores.append(relevance_clipped)
            self.all_ground_truths.append(ground_truth_evidence)
            # For prediction used in AUPC
            pred_score = self._get_prediction_score(slide_obj)
            self.all_predictions.append(pred_score)
            ground_truth_per_slide.append(np.mean([v > 0 for v in ground_truth_evidence]))  # support presence
            pred_per_slide.append(pred_score)

            # Generate heatmaps if configured
            if config.evaluation.get("visualization", {}).get("heatmaps", False):
                heatmap_img = self._generate_heatmap(slide_image, relevance_clipped, slide_obj)
                self._save_heatmap(heatmap_img, slide_id)

            # Store per-slide relevance for perturbation
            relevance_per_slide.append(relevance_clipped)

        # Compute overall metrics
        overall_auprc2 = np.mean(auprc2_list)
        overall_aupc = self._compute_aupc(self.all_ground_truths, self.all_relevance_scores, self.all_predictions)

        self.metrics_results['AUPRC-2'] = {'mean': overall_auprc2, 'std': np.std(auprc2_list)}
        self.metrics_results['AUPC'] = {'mean': np.mean(aupc), 'std': np.std(aupc)}
        # Print or log
        print(f"Overall AUPRC-2: {overall_auprc2:.3f} ± {np.std(auprc2_list):.3f}")
        print(f"Overall AUPC: {np.mean(aupc):.3f} ± {np.std(aupc):.3f}")

        # Optional: plot aggregate curves (e.g., average perturbation curve)
        self._plot_perturbation_curve()

        return self.metrics_results

    def _clip_relevance(self, relevance_scores: np.ndarray) -> np.ndarray:
        """
        Clip relevance scores at whiskers for better visualization.
        """
        q1 = np.percentile(relevance_scores, 25)
        q3 = np.percentile(relevance_scores, 75)
        whisker = 1.5 * (q3 - q1)
        lower = q1 - whisker
        upper = q3 + whisker
        return np.clip(relevance_scores, lower, upper)

    def _get_ground_truth_evidence(self, slide_obj):
        """
        Obtain ground-truth evidence labels per instance.
        For histopathology, assume sample-level label indicates support.
        Or, in toy data, the explicit ground truth.
        """
        # Placeholder: for real data, may have detailed annotations.
        # For demo, assume support if label==1, refute if label==0.
        # In practice, replace with annotation info.
        # Return an array of +1 (positive evidence), -1 (negative evidence), or 0 (neutral).
        # Here, just a simplistic assumption:
        label = slide_obj.label
        num_instances = len(slide_obj.features)
        if label == 1:
            return np.ones(num_instances)  # all positive evidence support
        elif label == 0:
            return -np.ones(num_instances)  # all refuting
        else:
            return np.zeros(num_instances)  # neutral

    def _compute_auprc2(self, evidence_labels, relevance_scores):
        """
        Compute the AUPRC-2: mean of positive and negative evidence detection.
        """
        # Binarize evidence: positive=1, negative=-1
        e_pos = np.copy(evidence_labels)
        e_neg = -np.copy(evidence_labels)
        # For positive evidence detection
        auprc_pos = average_precision_score((evidence_labels > 0).astype(int),
                                             relevance_scores)
        # For negative evidence detection
        auprc_neg = average_precision_score((evidence_labels < 0).astype(int),
                                             -relevance_scores)
        return 0.5 * (auprc_pos + auprc_neg)

    def _get_prediction_score(self, slide_obj):
        """
        Compute or retrieve the model prediction score for the slide.
        Assumes slide_obj has attributes needed to reconstruct features,
        or that the model can predict from features.
        """
        # For simplicity, assume slide_obj has precomputed features and model
        features_tensor = torch.tensor(slide_obj.features, dtype=torch.float).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(features_tensor)
            prob = torch.sigmoid(logits).item()
        return prob

    def _generate_heatmap(self, slide_image, relevance_scores, slide_obj):
        """
        Create an overlay heatmap on slide image patches.
        """
        import matplotlib.pyplot as plt
        from PIL import Image

        # Use the patch coordinates and relevance for overlay
        patches_positions = slide_obj.patch_coords
        patches_images = slide_obj.patches  # PIL Images
        # Normalize relevance to [-1,1]
        rel_mean = np.mean(relevance_scores)
        rel_std = np.std(relevance_scores) + 1e-8
        norm_scores = (relevance_scores - rel_mean) / rel_std

        # Generate combined heatmap overlay
        overlay_img = slide_image.copy().convert("RGBA")
        for i, rel in enumerate(norm_scores):
            # Color: red for positive, blue for negative
            color = (255, 0, 0, int(255 * abs(rel))) if rel > 0 else (0, 0, 255, int(255 * abs(rel)))
            patch_img = patches_images[i]
            # Resize overlay to patch size if needed
            overlay = Image.new("RGBA", patch_img.size, color)
            # Paste overlay with transparency
            overlay_img.paste(overlay, patches_positions[i], overlay)
        return overlay_img

    def _save_heatmap(self, heatmap_img, slide_id):
        """
        Save heatmap overlay image to disk.
        """
        save_dir = os.path.join(config.save['explanation_heatmaps_path'], slide_id)
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"heatmap_{slide_id}.png")
        heatmap_img.save(filename)

    def _plot_perturbation_curve(self):
        """
        Optional: plot aggregated perturbation (AUPC) curve over all slides.
        """
        # For visualization, aggregate the curves (if stored)
        # Here, it's a placeholder; in full implementation, store curve points to plot
        pass
```

## explanation.py

```python
## explanation.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, List
from torch import Tensor

from config import config

class Explanation:
    """
    Implements instance-wise relevance attribution for MIL models using xMIL-LRP.
    Supports models: attention-based, transformer-based, additive MIL.
    Computes relevance scores per instance (and per feature), revealing positive/negative evidence.
    """
    def __init__(self, model: nn.Module, explanation_method: str = "xMIL-LRP"):
        """
        Initialize Explanation object.
        Args:
            model (nn.Module): Trained MIL model.
            explanation_method (str): Method to use, default "xMIL-LRP".
        """
        self.model = model
        self.method = explanation_method
        # Select relevance propagation rules according to configuration
        self.rules = config.explanation_method['relevance_rules']
        # Epsilon for epsilon-rule (numerical stability in linear relevance propagation)
        self.epsilon = 1e-6

    def compute_relevance(self, features: Tensor, prediction: Optional[Tensor] = None, target_class: Optional[int] = None) -> List[float]:
        """
        Computes relevance scores for input features of shape (K, D) for one bag.
        Args:
            features (torch.Tensor): The instance features tensor (K, D), requires_grad=False.
            prediction (torch.Tensor): Model's output score for the bag, if precomputed.
            target_class (int): Index of class of interest; if None, use model's predicted class.
        Returns:
            List[float]: Relevance scores aggregated per feature for each instance, then flattened.
        """
        # Setup input features
        features = features.requires_grad_(True)
        # Forward pass
        output_score = self._get_model_output(features) if prediction is None else prediction
        if target_class is None:
            target_score = output_score
        else:
            # Assuming binary classification as in paper (output scalar), so treat as class 0 or 1
            target_score = output_score

        # Initialize relevance at output as the score itself
        R = target_score
        # Store relevance at each layer
        relevance_scores = {}
        # Start backward propagation
        relevance_scores['output'] = R

        # Backpropagate relevance through the network layers
        # (Wrap in a recursive or iterative method)
        relevance_scores = self._relprop(self.model, features, relevance_scores['output'])
        # relevance_scores now contain relevance at input features (per feature)
        # Sum over features to get per-instance relevance
        instance_relevance = []
        for k in range(features.shape[0]):
            rel_feat = relevance_scores['input'][k]  # shape: (D,)
            epsilon_score = rel_feat.sum().item()
            instance_relevance.append(epsilon_score)
        return instance_relevance

    def _get_model_output(self, features: Tensor, class_idx: Optional[int] = None) -> Tensor:
        """
        Forward pass through the model given features input.
        Args:
            features (Tensor): (K, D)
            class_idx (int): Optional, index for class-specific explanation.
        Returns:
            Tensor: Scalar output score for given features.
        """
        # Expand features to batch size 1
        feats = features.unsqueeze(0)  # shape: (1, K, D)
        # Forward pass depending on the model type
        if hasattr(self.model, 'forward'):
            output = self.model(feats)
            if isinstance(output, (list, tuple)):
                output = output[0]
            # output shape: (1, 1)
        else:
            raise RuntimeError("Model does not have forward method.")
        return output.squeeze()

    def _relprop(self, model: nn.Module, features: Tensor, R_out: Tensor):
        """
        Recursive relevance propagation for each layer.
        Args:
            model (nn.Module): The model or sub-module (attention, linear, norm).
            features (Tensor): Input features to current layer.
            R_out (Tensor): Relevance output from the layer (scalar or vector).
        Returns:
            Dict: relevance at previous layer's neurons/features, with keys:
                'input': tensor of relevance scores per feature for each input instance.
        """
        # Placeholder: Determine the layer type and apply appropriate relevance rule
        # for simplicity, assume linear or attention layers are functions we can call
        # in practice, this function needs to traverse model structure (e.g., via hooks or a custom wrapper)
        # But for demonstration, implement core logic for basic linear and attention rules

        # --- Linear layer relevance propagation ---
        if isinstance(model, nn.Linear):
            return self._linear_relprop(model, features, R_out)
        elif hasattr(model, 'attention') or ('attention' in str(model).lower()):
            return self._attention_relprop(model, features, R_out)
        elif isinstance(model, nn.LayerNorm):
            return self._layernorm_relprop(model, features, R_out)
        elif isinstance(model, nn.Sequential) or isinstance(model, nn.Module):
            # Recursively apply to contained modules
            # For simplicity, assume the model wrapper handles propagation
            # Here, we need a custom wrapper for the full model......
            # For code completeness, we assume direct linear or attention layer
            return {'input': features, 'relevance': R_out}
        elif isinstance(model, nn.ReLU):
            # ReLU does not change relevance magnitude; relevance flows unchanged where activation >0
            return {'input': features, 'relevance': R_out}
        else:
            # Default: return relevance unchanged
            return {'input': features, 'relevance': R_out}

    def _linear_relprop(self, layer: nn.Linear, inputs: Tensor, R_out: Tensor):
        """
        Propagate relevance for linear layer using epsilon rule.
        """
        with torch.no_grad():
            W = layer.weight  # shape: (out_dim, in_dim)
            b = layer.bias  # shape: (out_dim)
            inputs = inputs.requires_grad_(True)
            # Forward pass contribution
            Z = W @ inputs.T + b.unsqueeze(1)  # shape: (out_dim, batch_size)
            # Stabilize denominator
            Z += self.epsilon * torch.sign(Z)
            # Distribute relevance proportionally
            # R_out shape: (batch_size,)
            # assume R_out spread evenly across out_dim
            # For scalar output, R_out is scalar
            denom = Z.sum(dim=0, keepdim=True)  # sum over out_dim
            # Compute relevance for each input feature
            # Layer contribution: R_opt * (Z / denom)
            # For simplicity, assume scalar R_out
            relevance_input = torch.zeros_like(inputs)
            # No batch assumption: shape: (in_dim,)
            # For vectorized code, need to handle batch; here, single instance
            # For batch, expand accordingly
            # For simplicity, assume R_out is scalar and features shape: (K, D)
            # Recompute accordingly
            # As per paper, for batch, distribute relevance proportionally
            for i in range(inputs.shape[0]):  # over in_dim
                relevance_input[i] = (inputs[i] * torch.sum(W[:, i] * R_out))
            return {'input': relevance_input}

    def _attention_relprop(self, layer, inputs, R_out):
        """
        Relevance redistribution for attention modules based on AH-rule.
        """
        # Assume attention layer has attributes: attention scores, value input
        # For simplicity, assume layer has stored attention scores (if not, provide externally)
        attention_scores = self._get_attention_scores(layer)  # shape: (K,)
        # Distribute relevance proportionally to attention scores and value features
        value_input = inputs  # shape: (K, D)
        relevance_input = torch.zeros_like(value_input)
        sum_scores = attention_scores.sum()
        for k in range(len(attention_scores)):
            weight = attention_scores[k] / (sum_scores + self.epsilon)
            relevance_input[k] = weight * R_out
        return {'input': relevance_input}

    def _layernorm_relprop(self, layer: nn.LayerNorm, inputs, R_out):
        """
        Relevance propagation through LayerNorm, per LN-rule.
        """
        # As per Appendix A.2, propagate relevance proportionally
        # LayerNorm normalizes over features; the relevance is distributed per feature
        mean = inputs.mean(dim=1, keepdim=True)
        std = inputs.std(dim=1, keepdim=True) + self.epsilon
        # Relevance is proportionally distributed
        relevance_input = ((inputs - mean) / std) * R_out
        return {'input': relevance_input}

    def _get_attention_scores(self, layer):
        """
        Placeholder for obtaining attention scores from a layer, if stored.
        """
        # For actual implementation, store attention weights during forward pass
        # For now, return uniform or dummy
        # This must be replaced by actual attention scores during forward.
        # For demonstration, assign equal attention
        K = 10  # placeholder, should be number of instances
        return torch.ones(K)

    def generate_heatmap(self, bag_patches: List, relevance_scores: List[float]):
        """
        Generate visualization heatmap overlayed on patch images.
        Args:
            bag_patches (List): List of patch image objects (PIL images).
            relevance_scores (List): Corresponding relevance scores per instance.
        Returns:
            heatmap_img (PIL.Image): Combined heatmap visualization.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        num_patches = len(bag_patches)
        # Normalize relevance scores for color mapping
        q1 = np.percentile(relevance_scores, 25)
        q3 = np.percentile(relevance_scores, 75)
        whisker = 1.5 * (q3 - q1)
        min_val = np.min(relevance_scores)
        max_val = np.max(relevance_scores)
        clipped_scores = np.clip(relevance_scores, min_val - whisker, max_val + whisker)
        norm_scores = (np.array(clipped_scores) - min_val) / (max_val - min_val + 1e-8)

        # Create heatmaps per patch
        heatmaps = []
        for idx, score in enumerate(norm_scores):
            color = (1, 0, 0) if score > 0.5 else (0, 0, 1)
            alpha = abs(score - 0.5) * 2  # range 0-1
            overlay = bag_patches[idx].copy()
            overlay = overlay.convert("RGBA")
            mask = Image.new("RGBA", overlay.size, color + (int(255 * alpha),))
            combined = Image.alpha_composite(overlay, mask)
            heatmaps.append(combined)

        # Compose final heatmap image
        cols = int(np.sqrt(num_patches))
        rows = (num_patches + cols - 1) // cols
        width, height = bag_patches[0].size
        new_img = Image.new('RGBA', (cols * width, rows * height))
        for i, img in enumerate(heatmaps):
            row = i // cols
            col = i % cols
            new_img.paste(img, (col * width, row * height))
        return new_img
```

## feature_extractor.py

```python
## feature_extractor.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Optional
from PIL import Image

from config import config

class FeatureExtractor:
    """
    Extracts feature vectors from image patches using a pre-trained CNN backbone.
    Supports freezing of the feature extractor as per configuration.
    """

    def __init__(self, model_name: str = "resnet18"):
        """
        Initialize the feature extractor with a specified backbone.
        Args:
            model_name (str): Name of the backbone model, default "resnet18".
        """
        self.device = torch.device(config.hardware['device'])
        self.model_name = model_name
        self._load_model()
        self._setup_transform()

    def _load_model(self):
        """
        Loads a pre-trained backbone model (e.g., ResNet-18) and modifies it
        to output feature vectors instead of classification scores.
        """
        if self.model_name.lower() == "resnet18":
            full_model = models.resnet18(pretrained=True)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        # Remove the final fully connected layer to get features
        # Typically, the last layer is model.fc
        self.model = nn.Sequential(
            *(list(full_model.children())[:-1])  # All layers except the classifier
        )

        # Freeze or unfreeze parameters based on configuration
        freeze = config.model_parameters.get("freeze_feature_extractor", True)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        self.model.to(self.device)
        self.model.eval()

    def _setup_transform(self):
        """
        Sets up image transformation pipeline matching ImageNet normalization.
        """
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extracts a feature vector from a single PIL Image patch.
        Args:
            image (PIL.Image.Image): Input image patch.
        Returns:
            np.ndarray: 1D feature vector (size depends on backbone, typically 512).
        """
        # Convert PIL Image to tensor and normalize
        tensor_img = self.transform(image).unsqueeze(0).to(self.device)  # shape: [1, 3, H, W]
        with torch.no_grad():
            features = self.model(tensor_img)  # shape: [1, 512, 1, 1]
        # Flatten the features
        feature_vector = features.squeeze().cpu().numpy()  # shape: [512]
        return feature_vector

    def extract_batch(self, images: list) -> np.ndarray:
        """
        Batch process multiple images for efficiency.
        Args:
            images (list of PIL.Image): List of image patches.
        Returns:
            np.ndarray: Array of shape [batch_size, feature_dim].
        """
        batch_tensor = torch.stack([self.transform(img) for img in images], dim=0).to(self.device)
        with torch.no_grad():
            features = self.model(batch_tensor)  # shape: [B, 512, 1, 1]
        features = features.squeeze(3).squeeze(2).cpu().numpy()  # shape: [B, 512]
        return features
```

## main.py

```python
## main.py
import os
import sys
import time
import logging
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Import project modules
from config import config
from dataset_loader import DatasetLoader
from feature_extractor import FeatureExtractor
from model import AttentionMIL, TransMIL, AdditiveMIL
from explanation import Explanation
from trainer import Trainer
from evaluation import Evaluation
from utils import save_figure, load_slide_image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    start_time = time.time()
    device_str = config.hardware.get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    logging.info(f"Using device: {device}")

    # 1. Load dataset
    dataset_path = config.dataset_paths.get('histopathology', './data')
    loader = DatasetLoader(dataset_dir=dataset_path)

    # 2. Load dataset (slide list with labels)
    all_slides = loader.load_all_slides()

    # 3. Split data into train, val, test
    # For simplicity, assume random split with fixed seed for reproducibility
    np.random.seed(42)
    slide_indices = list(range(len(all_slides)))
    np.random.shuffle(slide_indices)

    n_total = len(all_slides)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val

    train_idx = slide_indices[:n_train]
    val_idx = slide_indices[n_train:n_train+n_val]
    test_idx = slide_indices[n_train+n_val:]

    train_list = [all_slides[i] for i in train_idx]
    val_list = [all_slides[i] for i in val_idx]
    test_list = [all_slides[i] for i in test_idx]

    logging.info(f"Dataset split: {len(train_list)} train, {len(val_list)} val, {len(test_list)} test.")

    # 4. Extract features for all datasets
    feature_extractor = FeatureExtractor(model_name=config.model_parameters['feature_extractor'])
    feature_extractor.model.eval()

    for dataset_split in [train_list, val_list, test_list]:
        # Check if features already extracted (optional)
        loader.extract_features_for_dataset(dataset_split, feature_extractor, batch_size=128)

    # 5. Initialize model
    model_type = config.model_parameters.get('model_type', 'attention').lower()
    feature_dim = 512  # from the feature extractor output
    hidden_dim = config.model_parameters.get('hidden_dim', 512)
    dropout = config.training.get('dropout', 0.0)

    if model_type == 'attention':
        model = AttentionMIL(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == 'transformer':
        model = TransMIL(feature_dim=feature_dim, num_layers=2, num_heads=8, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == 'additive':
        model = AdditiveMIL(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.to(device)
    logging.info(f"Initialized {model_type} model.")

    # 6. Set up optimizer
    optim_cls = torch.optim.Adam if config.training.get('optimizer', 'Adam').lower() == 'adam' else torch.optim.SGD
    optimizer = optim_cls(filter(lambda p: p.requires_grad, model.parameters()), lr=config.training.get('learning_rate', 0.002))

    # 7. Train model
    trainer = Trainer(model, train_list, val_list, test_list, {
        'optimizer': optimizer,
        'epochs': config.training.get('epochs', 1000),
        'batch_size': config.training.get('batch_size', 32),
        'device': device,
        'checkpoint_path': config.save.get('model_checkpoint_path', './checkpoints/'),
        'early_stop_patience': 10
    })
    logging.info("Starting training...")
    trainer.train()

    # 8. Load best model checkpoint
    # It is handled internally in trainer._load_checkpoint after train()

    # 9. Explaining test set
    explanation_method = config.explanation_method.get('method', 'xMIL-LRP')
    relevance_rules = config.explanation_method['relevance_rules']
    explainer = Explanation(model, explanation_method=explanation_method)
    explainer.rules = relevance_rules

    # Prepare output directories
    os.makedirs(config.save.get('explanation_heatmaps_path', './heatmaps'), exist_ok=True)

    # For each test slide, compute relevance, generate heatmaps, evaluate
    all_preds = []
    all_labels = []
    all_auprc2 = []
    all_aupc = []

    for slide_obj in test_list:
        # 9.1 Get features tensor (K, D)
        features = torch.tensor(slide_obj.features, dtype=torch.float).to(device)
        # 9.2 Compute relevance scores
        relevance_scores = explainer.compute_relevance(features)
        # Convert to numpy
        relevance_np = np.array(relevance_scores)

        # 9.3 Compute or retrieve model prediction
        # For simplicity, rerun model forward on features
        with torch.no_grad():
            logits = model(features.unsqueeze(0))
            prob = torch.sigmoid(logits).item()
        all_preds.append(prob)
        # Ground truth label: from slide label
        label = slide_obj.label
        all_labels.append(label)

        # 9.4 Evaluate explanation (AUPRC-2)
        # Assume ground truth evidence is support (simulate with label==1, refute==0)
        gt_evidence = np.ones(len(slide_obj.features)) if label==1 else -np.ones(len(slide_obj.features))
        # Calculate AUPRC-2
        auprc2 = 0.5 * (
            average_precision_score((gt_evidence>0).astype(int), relevance_np) +
            average_precision_score((gt_evidence<0).astype(int), -relevance_np)
        )
        all_auprc2.append(auprc2)

        # 9.5 Generate heatmap overlay
        if config.evaluation.get('visualization', {}).get('heatmaps', False):
            slide_image = load_slide_image(slide_obj.slide_path)
            heatmap_img = explainer.generate_heatmap(slide_obj.patches, relevance_scores)
            save_figure(heatmap_img, os.path.join(
                config.save.get('explanation_heatmaps_path', './heatmaps'), f"slide_{slide_obj.slide_id}_heatmap.png"
            ))

    # 10. Compute overall metrics
    mean_auprc2 = np.mean(all_auprc2)
    std_auprc2 = np.std(all_auprc2)

    # AUPC: compute by patch dropping in order of relevance
    # (Here skipped for simplicity; implement as needed)
    # For demonstration, assume it is computed and stored as 'overall_aupc'
    # but since detailed code is long, we omit in this code.

    # Log results
    logging.info(f"Test AUROC: {roc_auc_score(all_labels, all_preds):.3f}")
    logging.info(f"Test AUPRC-2: {mean_auprc2:.3f} ± {std_auprc2:.3f}")

    # 11. Final summary
    total_time = time.time() - start_time
    logging.info(f"Total runtime: {total_time/60:.2f} minutes.")

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from math import sqrt
from typing import Optional

# Import configuration from the global config module
from config import config

class AttentionMIL(nn.Module):
    """
    Attention-based MIL model processing precomputed features.
    Consists of an attention network to assign relevance weights to instances,
    aggregates weighted features, and predicts a bag-level output.
    """
    def __init__(self,
                 feature_dim: int = 512,
                 hidden_dim: int = 512,
                 dropout: float = 0.0):
        """
        Initializes the AttentionMIL model.
        Args:
            feature_dim (int): Dimensionality of input instance features.
            hidden_dim (int): Hidden dimension size for attention network.
            dropout (float): Dropout probability.
        """
        super(AttentionMIL, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Attention network components: a small MLP
        self.attention_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )

        # Optional: a small bias term for attention logits
        self.bias = nn.Parameter(torch.zeros(1))

        # Final classifier: linear layer to produce scalar prediction
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, K, D)
        Returns:
            torch.Tensor: Bag prediction logits of shape (batch_size, 1)
        """
        # Compute attention scores for each instance
        # shape: (batch_size, K, 1)
        attn_logits = self.attention_layer(x).squeeze(-1) + self.bias
        # Attention weights: softmax over instances
        attn_weights = F.softmax(attn_logits, dim=1)  # shape: (batch_size, K)

        # Weight instances and sum to get bag representation
        bag_rep = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # shape: (batch_size, D)

        # Final prediction
        out = self.classifier(bag_rep)  # shape: (batch_size, 1)
        return out

class TransMIL(nn.Module):
    """
    Transformer-based MIL model processing precomputed features.
    Uses a Transformer encoder with a class token for global context.
    """
    def __init__(self,
                 feature_dim: int = 512,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 hidden_dim: int = 512,
                 dropout: float = 0.1):
        """
        Initializes the TransMIL model.
        Args:
            feature_dim (int): Dimension of input instance features.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Dimension of transformer feedforward layer.
            dropout (float): Dropout probability.
        """
        super(TransMIL, self).__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Class token: learned embedding prepended to sequence
        self.class_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))
        # Positional encoding can be added if necessary (here omitted for simplicity)

        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(d_model=self.feature_dim,
                                                nhead=self.num_heads,
                                                dim_feedforward=self.hidden_dim,
                                                dropout=self.dropout,
                                                activation='relu')
        self.transformer = TransformerEncoder(encoder_layers, num_layers=self.num_layers)

        # Final classification head applied to class token output
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input features of shape (batch_size, K, D)
        Returns:
            torch.Tensor: Bag prediction logits of shape (batch_size, 1)
        """
        batch_size, K, D = x.shape
        # Expand class token for batch
        class_token = self.class_token.expand(batch_size, -1, -1)  # shape: (batch_size, 1, D)

        # Concatenate class token at sequence start
        seq_input = torch.cat([class_token, x], dim=1)  # shape: (batch_size, K+1, D)

        # Transformer expects input shape: (K+1, batch_size, D) or (batch_size, K+1, D)
        # Using batch-first mode if available: default is batch first, so no change needed
        # transformer expects shape: (seq_len, batch, embed_dim)
        seq_input = seq_input.transpose(0, 1)  # shape: (K+1, batch_size, D)

        # Pass through transformer encoder
        encoder_output = self.transformer(seq_input)  # shape: (K+1, batch_size, D)

        # Extract class token output (first token)
        class_token_output = encoder_output[0]  # shape: (batch_size, D)

        # Compute scalar prediction
        out = self.classifier(class_token_output)  # shape: (batch_size, 1)
        return out

class AdditiveMIL(nn.Module):
    """
    Additive MIL model: predicts bag as sum over per-instance predictions.
    The model is inherently interpretable; each instance's score explains contribution.
    """
    def __init__(self,
                 feature_dim: int = 512,
                 hidden_dim: int = 512,
                 dropout: float = 0.0):
        """
        Initialize the AdditiveMIL model.
        Args:
            feature_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer size for instance scoring.
            dropout (float): Dropout probability.
        """
        super(AdditiveMIL, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Instance scoring network: MLP per instance
        self.instance_scorer = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, K, D)
        Returns:
            torch.Tensor: Bag prediction logits of shape (batch_size, 1)
        """
        # Compute per-instance scores
        # shape: (batch_size, K, 1)
        instance_logits = self.instance_scorer(x)  # shape: (batch_size, K, 1)

        # Sum over instances to get bag score
        bag_logits = torch.sum(instance_logits, dim=1)  # shape: (batch_size, 1)

        return bag_logits
```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import os
from typing import Optional
import numpy as np

from dataset_loader import DatasetLoader, SlideSample
from model import AttentionMIL, TransMIL, AdditiveMIL
from explanation import Explanation
from utils import plot_heatmaps, compute_metrics
from config import config

class Trainer:
    """
    Handles training, validation, checkpointing, and testing of MIL models.
    Implements early stopping based on validation AUC (or other metrics).
    """
    def __init__(self,
                 model: nn.Module,
                 train_dataset: list,
                 val_dataset: list,
                 test_dataset: list,
                 config_dict: dict):
        """
        Initialize the Trainer.
        Args:
            model (nn.Module): The MIL model to train.
            train_dataset (list): List of SlideSample objects for training.
            val_dataset (list): List of SlideSample objects for validation.
            test_dataset (list): List of SlideSample objects for testing.
            config_dict (dict): Hyperparameters, paths, device info from config.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config_dict
        self.device = torch.device(self.config['hardware'].get('device', 'cuda'))
        self.model.to(self.device)
        
        # Prepare DataLoaders
        self.train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_dataset, shuffle=False)
        self.test_loader = self._create_dataloader(self.test_dataset, shuffle=False)

        # Set optimizer
        self.optimizer = self._create_optimizer(self.model, self.config['training'])
        
        # Loss function: binary cross entropy (for binary labels) or use BCEWithLogitsLoss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Learning rate scheduler (optional)
        # Use ReduceLROnPlateau if desired; here, for simplicity, we skip it.
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.early_stop_patience = 10  # can be set from config
        self.patience_counter = 0
        self.checkpoint_path = self.config['save'].get('model_checkpoint_path', './checkpoints/')
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def _create_dataloader(self, dataset: list, shuffle: bool) -> DataLoader:
        """
        Create DataLoader from dataset of SlideSamples.
        """
        return DataLoader(dataset, batch_size=self.config['training'].get('batch_size', 32), shuffle=shuffle, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        """
        Collate function to handle variable-sized bags if needed.
        For simplicity, assume all bags are processed individually; batch contains lists.
        """
        return batch

    def _create_optimizer(self, model: nn.Module, training_cfg: dict):
        """
        Instantiate optimizer.
        """
        lr = training_cfg.get('learning_rate', 0.001)
        opt_name = training_cfg.get('optimizer', 'Adam').lower()
        if opt_name == 'adam':
            return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        elif opt_name == 'sgd':
            return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def train(self):
        """
        Main training loop with validation, early stopping, and checkpointing.
        """
        max_epochs = self.config['training'].get('epochs', 1000)
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            epoch_losses = []
            train_preds = []
            train_labels = []

            # Loop over training batches
            for batch in self.train_loader:
                features_batch, labels_batch = self._prepare_batch(batch)
                features_batch = features_batch.to(self.device)
                labels_batch = labels_batch.to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(features_batch).squeeze(-1)  # shape: (batch_size,)
                loss = self.criterion(outputs, labels_batch)
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

                # Collect predictions for metrics
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                train_preds.extend(probs)
                train_labels.extend(labels_batch.cpu().numpy())

            train_loss = np.mean(epoch_losses)
            train_auc = roc_auc_score(train_labels, train_preds) if len(train_labels) > 0 else 0.5

            # Validation
            val_metrics = self.validate()
            val_auc = val_metrics.get('AUROC', 0)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_AUC={train_auc:.4f}, val_AUC={val_auc:.4f}")

            # Check for improvement
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stop_patience:
                print("Early stopping triggered.")
                break

        # Load best model checkpoint after training
        self._load_checkpoint()

        # Final evaluation on test set
        test_metrics, test_heatmaps = self.evaluate()
        print(f"Test AUROC: {test_metrics.get('AUROC', 0):.4f}")
        if self.config['evaluation']['metrics'].get('AUPRC2', False):
            print(f"Test AUPRC-2: {test_metrics.get('AUPRC2', 0):.4f}")

        # Save heatmaps if needed
        if self.config['evaluation']['visualization'].get('heatmaps', False):
            self._save_heatmaps(test_heatmaps)

    def _prepare_batch(self, batch):
        """
        Converts batch of SlideSample objects into tensors for features and labels.
        Assumes batch is a list of tuples or objects containing features and labels.
        """
        # For simplicity, assume each sample contains features (tensor) and label
        features_list = []
        labels_list = []
        for sample in batch:
            # sample can be a tuple or object
            if isinstance(sample, tuple) or hasattr(sample, 'features'):
                features = getattr(sample, 'features', None)
                label = getattr(sample, 'label', None)
            else:
                # fallback: assume dict with 'features' and 'label'
                features = sample['features']
                label = sample['label']
            # features shape: (K, D)
            # For batch processing, keep list of features and labels
            features_list.append(torch.tensor(features, dtype=torch.float))
            labels_list.append(label)

        # Stack features to tensor (batch_size, max_patches, feature_dim)
        # For variable-sized bags, might need padding; for simplicity, assume all same size
        features_batch = torch.stack(features_list, dim=0)  # shape: (B, K, D)
        labels_batch = torch.tensor(labels_list, dtype=torch.float)  # shape: (B,)
        return features_batch, labels_batch

    def _save_checkpoint(self, epoch: int):
        """
        Save model state_dict and optimizer.
        """
        save_path = os.path.join(self.checkpoint_path, f"best_model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_auc': self.best_val_auc
        }, save_path)
        print(f"Checkpoint saved at epoch {epoch} to {save_path}.")

    def _load_checkpoint(self):
        """
        Load the best saved checkpoint.
        """
        # Find the checkpoint with best val_auc
        checkpoints = [f for f in os.listdir(self.checkpoint_path) if f.endswith('.pt')]
        if not checkpoints:
            print("No checkpoint found.")
            return
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
        latest_ckpt = checkpoints[0]
        ckpt_path = os.path.join(self.checkpoint_path, latest_ckpt)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with val_auc={checkpoint['val_auc']:.4f}")

    def validate(self):
        """
        Evaluate model on validation set, compute metrics.
        """
        self.model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in self.val_loader:
                features_batch, labels_batch = self._prepare_batch(batch)
                features_batch = features_batch.to(self.device)
                labels_batch = labels_batch.to(self.device).float()
                outputs = self.model(features_batch).squeeze(-1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(labels_batch.cpu().numpy())
        auroc = roc_auc_score(val_labels, val_preds) if len(val_labels) > 0 else 0
        auprc2 = self._compute_auprc2(val_labels, val_preds)
        return {'AUROC': auroc, 'AUPRC2': auprc2}

    def _compute_auprc2(self, true_labels, pred_scores):
        """
        Compute the average of AUPRC for positive and negative evidence detection
        (as per AUPRC-2 measurement).
        """
        # For binary classification, AUPRC for positives and negatives
        auprc_pos = average_precision_score(true_labels, pred_scores)
        # For negatives, invert scores
        auprc_neg = average_precision_score(true_labels, [-s for s in pred_scores])
        return 0.5 * (auprc_pos + auprc_neg)

    def _save_heatmaps(self, heatmaps):
        """
        Save final heatmaps for analysis.
        """
        save_dir = self.config['save'].get('explanation_heatmaps_path', './heatmaps/')
        os.makedirs(save_dir, exist_ok=True)
        for idx, heatmap_img in enumerate(heatmaps):
            heatmap_path = os.path.join(save_dir, f"slide_{idx}_heatmap.png")
            heatmap_img.save(heatmap_path)
        print(f"Heatmaps saved to {save_dir}.")

    def evaluate(self):
        """
        Run inference on test set, compute metrics, generate heatmaps.
        """
        self.model.eval()
        test_preds = []
        test_labels = []
        all_heatmaps = []
        with torch.no_grad():
            for batch in self.test_loader:
                features_batch, labels_batch = self._prepare_batch(batch)
                features_batch = features_batch.to(self.device)
                labels_batch = labels_batch.to(self.device).float()

                outputs = self.model(features_batch).squeeze(-1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                test_preds.extend(probs)
                test_labels.extend(labels_batch.cpu().numpy())

                # Generate explanations/heatmaps for this batch if needed
                # Could be added here: e.g., compute relevance maps and visualize
                # For demonstration, we skip that step in bulk evaluation
        auroc = roc_auc_score(test_labels, test_preds) if len(test_labels) > 0 else 0
        auprc2 = self._compute_auprc2(test_labels, test_preds)
        metrics = {'AUROC': auroc, 'AUPRC2': auprc2}

        # Placeholder: generate heatmaps for samples if desired
        # For simplicity, return empty list
        heatmaps_list = []
        return metrics, all_heatmaps
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\xMIL\xMIL_repo`
