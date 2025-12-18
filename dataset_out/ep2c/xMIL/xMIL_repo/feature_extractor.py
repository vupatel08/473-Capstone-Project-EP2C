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
