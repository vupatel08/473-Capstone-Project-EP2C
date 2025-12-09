## model.py
import torch
import torch.nn.functional as F
from typing import Optional
import os

# Import HuggingFace transformers for ViT and CLIP, torchvision models for ConvNeXt
from torchvision import models as torchvision_models
from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_huge
from transformers import ViTForImageClassification, ViTFeatureExtractor
# For CLIP, use openclip package
import open_clip

class Model:
    def __init__(
        self,
        architecture: str,
        pretrained: bool = True,
        dataset: str = 'ImageNet-21K',
        pretrained_source: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Model object.
        :param architecture: Model type identifier, e.g., 'ConvNeXt-Huge', 'ViT-L/16', 'CLIP-Large'
        :param pretrained: Whether to load pretrained weights
        :param dataset: Training dataset name (e.g., 'ImageNet-21K', 'LAION-400M')
        :param pretrained_source: Source for pretrained weights ('OpenCLIP' or None for torchvision)
        :param device: 'cpu' or 'cuda'; default is CUDA if available
        """
        self.architecture = architecture
        self.pretrained = pretrained
        self.dataset = dataset
        self.pretrained_source = pretrained_source
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_extractor = None  # For CLIP
        # Load model upon initialization
        self.load_pretrained()

    def load_pretrained(self):
        """
        Load the pretrained model based on architecture.
        Raises errors if model not recognized or loading fails.
        """
        arch = self.architecture.lower()
        # ConvNeXt models
        if 'convnext' in arch:
            if 'tiny' in arch:
                self.model = convnext_tiny(pretrained=self.pretrained)
            elif 'small' in arch:
                self.model = convnext_small(pretrained=self.pretrained)
            elif 'base' in arch:
                self.model = convnext_base(pretrained=self.pretrained)
            elif 'large' in arch:
                self.model = convnext_large(pretrained=self.pretrained)
            elif 'huge' in arch:
                self.model = convnext_huge(pretrained=self.pretrained)
            else:
                raise ValueError(f"Unknown ConvNeXt size in architecture: {self.architecture}")
            self.model = self.model.to(self.device)
            self.model.eval()
            # Normalize parameters are standard in torchvision pretrained models
            self.input_size = 224  # for all ConvNeXt
        # Vision Transformer models
        elif 'vit' in arch:
            # Parse size, e.g., 'vit-s/16'
            # Expected format: 'vit-s/16', 'vit-l/16', 'vit-h/14'
            if 's/16' in arch:
                model_name = 'google/vit-base-patch16-224-in21k'
            elif 'l/16' in arch:
                model_name = 'google/vit-large-patch16-224-in21k'
            elif 'h/14' in arch:
                model_name = 'google/vit-huge-patch14-224-in21k'
            else:
                raise ValueError(f"Unknown ViT size in architecture: {self.architecture}")
            self.model = ViTForImageClassification.from_pretrained(model_name)
            self.model.eval()
            self.model = self.model.to(self.device)
            # Use default feature extractor (for normalization)
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
            self.input_size = 224
        # CLIP models
        elif 'clip' in arch:
            # For CLIP, use open_clip
            if self.pretrained_source != 'OpenCLIP':
                raise ValueError("For CLIP models, pretrained_source must be 'OpenCLIP'")
            if 'large' in arch:
                clip_type = 'ViT-B-32'  # default, check model size more specifically if needed
                # For larger models, you might use 'ViT-H-14' or 'RN50x4' as per open_clip
                # but based on description, select the matching:
                # For 'Large': use 'ViT-B/16' or 'ViT-B/32'
                if 'xl' in self.architecture.lower():
                    clip_type = 'ViT-L-14'  # example for XL
                elif 'huge' in self.architecture.lower():
                    clip_type = 'ViT-H-14'  # if available
                else:
                    clip_type = 'ViT-B-32'  # default
            elif 'xl' in self.architecture.lower() or 'huge' in self.architecture.lower():
                clip_type = 'ViT-L-14' if 'xl' in self.architecture.lower() else 'ViT-H-14'
            else:
                clip_type = 'ViT-B-32'  # fallback
            # Load with open_clip
            self.model, self.preprocess = open_clip.load(clip_type, device=self.device,
                                                         pretrained=True, source='openai' if self.pretrained else None)
            self.model.eval()
            self.input_size = 224
        else:
            raise ValueError(f"Model architecture '{self.architecture}' not recognized.")

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run inference on input images and output logits.
        :param images: tensor of shape (batch_size, 3, H, W), preprocessed
        :return: logits tensor of shape (batch_size, num_classes)
        """
        with torch.no_grad():
            images = images.to(self.device)
            # Handle CLIP differently if needed
            if 'clip' in self.architecture.lower():
                # CLIP's model: output similarity scores
                logits_per_image = self.model.encode_image(images)
                # Normalize to get cosine similarity as logits
                logits = logits_per_image / self.model.l2_norm
            else:
                logits = self.model(images).logits
        return logits

    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to class probabilities
        :param logits: tensor of shape (batch_size, num_classes)
        :return: probabilities tensor
        """
        return F.softmax(logits, dim=1)

    def get_confidence(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Compute maximum class probability for each sample.
        :param probabilities: tensor of shape (batch_size, num_classes)
        :return: tensor of shape (batch_size,)
        """
        return probabilities.max(dim=1).values

    def preprocess_image(self, image: 'PIL.Image') -> torch.Tensor:
        """
        Preprocess input image for model inference, normalize, resize.
        It should be used externally if applying per-image processing.
        """
        if hasattr(self, 'feature_extractor'):
            # For ViT with HuggingFace feature extractor
            inputs = self.feature_extractor(images=image, return_tensors='pt')
            return inputs['pixel_values'].squeeze(0)
        else:
            # For torchvision models
            transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            return transform(image)
