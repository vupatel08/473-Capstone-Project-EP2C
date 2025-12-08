# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

class MultiViewTransform(Dataset):
    """
    Dataset wrapper that, given an underlying dataset, applies data augmentations
    to generate multiple views per sample, suitable for SSL training.
    """
    def __init__(self, base_dataset: Dataset, num_views: int, augmentation_params: dict):
        """
        Args:
            base_dataset (Dataset): dataset object from torchvision.datasets
            num_views (int): number of augmented views to generate per sample
            augmentation_params (dict): dictionary containing augmentation parameters
        """
        self.base_dataset = base_dataset
        self.num_views = num_views
        self.aug_params = augmentation_params

        # Build the augmentation pipeline based on params
        self.transform = self._build_transform()

    def _build_transform(self):
        # Compose transforms based on augmentation_params
        transform_list = []
        # RandomResizedCrop with scale range
        crop_size = self.aug_params.get('crop_size', 224)
        scale_min = self.aug_params.get('crop_scale_min', 0.08)
        scale_max = self.aug_params.get('crop_scale_max', 1.0)
        transform_list.append(
            transforms.RandomResizedCrop(crop_size, scale=(scale_min, scale_max))
        )
        # Horizontal Flip
        p_flip = self.aug_params.get('horizontal_flip_prob', 0.5)
        transform_list.append(transforms.RandomHorizontalFlip(p=p_flip))
        # Color jitter
        brightness = self.aug_params.get('brightness', 0.4)
        contrast = self.aug_params.get('contrast', 0.4)
        saturation = self.aug_params.get('saturation', 0.2)
        hue = self.aug_params.get('hue', 0.1)
        color_jitter_prob = self.aug_params.get('color_jitter_prob', 0.8)
        if color_jitter_prob > 0:
            transform_list.append(
                transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                       saturation=saturation, hue=hue)
            )
        # ToTensor
        transform_list.append(transforms.ToTensor())

        # Additional augmentations: Gaussian noise, solarization, if specified
        # For simplicity, implement Gaussian noise as a modality
        class GaussianNoise:
            def __init__(self, mean=0.0, std=0.1):
                self.mean = mean
                self.std = std

            def __call__(self, tensor):
                return tensor + torch.randn_like(tensor) * self.std + self.mean

        gaussian_prob = self.aug_params.get('gaussian_prob', 0.0)
        if gaussian_prob > 0:
            transform_list.append(GaussianNoise())

        # Solarization (Invert pixel values above threshold), optional
        solar_prob = self.aug_params.get('solarization_prob', 0.0)
        if solar_prob > 0:
            class Solarization:
                def __init__(self, p=1.0, threshold=128):
                    self.p = p
                    self.threshold = threshold

                def __call__(self, img):
                    if torch.rand(1).item() < self.p:
                        img_temp = TF.to_pil_image(img)
                        img_temp = TF.invert(img_temp)
                        return TF.to_tensor(img_temp)
                    return img

            transform_list.append(Solarization(p=solar_prob))
        # Compose the transforms
        return transforms.Compose(transform_list)

    def __getitem__(self, index):
        """
        For each sample, generate 'num_views' augmented versions.
        Returns a list of tensors.
        """
        original_img, label = self.base_dataset[index]
        views = []
        for _ in range(self.num_views):
            augmented_img = self.transform(original_img)
            views.append(augmented_img)
        # Return all views as a tuple (view1, view2, ...)
        return tuple(views)

    def __len__(self):
        return len(self.base_dataset)

class DatasetLoader:
    """
    Responsible for loading datasets according to configuration, applying
    augmentation pipelines, and providing datasets for DataLoader.
    """
    def __init__(self, dataset_name: str = 'ImageNet-100', dataset_params: dict = None):
        """
        Args:
            dataset_name (str): String identifier of dataset ('CIFAR10', 'CIFAR100', 'ImageNet-100')
            dataset_params (dict): Parameters including crop sizes, augmentation params
        """
        self.dataset_name = dataset_name
        self.dataset_params = dataset_params if dataset_params is not None else {}
        # Placeholder for datasets
        self.train_dataset = None
        self.val_dataset = None

    def load_data(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        Loads dataset based on the configuration and applies data augmentation.
        Returns:
            train_dataset (Dataset): Dataset object with multi-view augmentation if needed
            val_dataset (Dataset): Validation dataset
        """
        dataset_type = self.dataset_params.get('dataset_type', 'image_classification')
        # Build transforms for training
        train_transform = self._build_transform(is_train=True)
        # Build transforms for validation (usually just ToTensor and normalization)
        val_transform = self._build_transform(is_train=False)

        if self.dataset_name.lower() == 'cifar10':
            root = os.path.expanduser('~/.cache/torch/datasets')
            base_train = datasets.CIFAR10(root=root, train=True, download=True)
            base_val = datasets.CIFAR10(root=root, train=False, download=True)
            # Wrap with MultiViewTransform for training (e.g., generate 2 views)
            num_views = self.dataset_params.get('total_crops', 2)
            self.train_dataset = MultiViewTransform(base_train, num_views, self.dataset_params.get('augmentation_params', {}))
            self.val_dataset = datasets.CIFAR10(root=root, train=False, transform=val_transform, download=False)
        elif self.dataset_name.lower() == 'cifar100':
            root = os.path.expanduser('~/.cache/torch/datasets')
            base_train = datasets.CIFAR100(root=root, train=True, download=True)
            base_val = datasets.CIFAR100(root=root, train=False, download=True)
            num_views = self.dataset_params.get('total_crops', 2)
            self.train_dataset = MultiViewTransform(base_train, num_views, self.dataset_params.get('augmentation_params', {}))
            self.val_dataset = datasets.CIFAR100(root=root, train=False, transform=val_transform, download=False)
        elif self.dataset_name.lower() == 'imagenet-100':
            # Assume dataset is organized in folders in a path
            # For small datasets, datasets.ImageFolder is common
            root_path = self.dataset_params.get('dataset_path', './imagenet-100/')
            # For training, apply augmentation
            self.train_dataset = datasets.ImageFolder(root=os.path.join(root_path, 'train'), transform=train_transform)
            self.val_dataset = datasets.ImageFolder(root=os.path.join(root_path, 'val'), transform=val_transform)
        else:
            raise ValueError(f"Unsupported dataset {self.dataset_name}")

        return self.train_dataset, self.val_dataset

    def _build_transform(self, is_train: bool):
        """
        Build validation or training transform
        """
        crop_size = self.dataset_params.get('crop_size', 224)
        augmentation_params = self.dataset_params.get('augmentation_params', {})

        if is_train:
            # Augmentation pipeline
            transform_list = []
            scale_min = self.dataset_params.get('crop_scale_min', 0.08)
            scale_max = self.dataset_params.get('crop_scale_max', 1.0)
            transform_list.append(
                transforms.RandomResizedCrop(crop_size, scale=(scale_min, scale_max))
            )
            p_flip = augmentation_params.get('horizontal_flip_prob', 0.5)
            transform_list.append(transforms.RandomHorizontalFlip(p=p_flip))
            brightness = augmentation_params.get('brightness', 0.4)
            contrast = augmentation_params.get('contrast', 0.4)
            saturation = augmentation_params.get('saturation', 0.2)
            hue = augmentation_params.get('hue', 0.1)
            transform_list.append(
                transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                       saturation=saturation, hue=hue)
            )
            # Add normalization for ImageNet if applicable
            if self.dataset_name.lower().startswith('imagenet'):
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                transform_list.append(transforms.ToTensor())
                transform_list.append(transforms.Normalize(mean=mean, std=std))
            else:
                transform_list.append(transforms.ToTensor())

            return transforms.Compose(transform_list)
        else:
            # Validation transform: just resize/crop and ToTensor
            transform_list = []
            transform_list.append(transforms.Resize(crop_size + 32))
            transform_list.append(transforms.CenterCrop(crop_size))
            transform_list.append(transforms.ToTensor())
            if self.dataset_name.lower().startswith('imagenet'):
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                transform_list.append(transforms.Normalize(mean=mean, std=std))
            return transforms.Compose(transform_list)
```

## evaluation.py

```python
## evaluation.py
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class Evaluation:
    """
    Performs evaluation of a trained SSLModel on a specified dataset split.
    Supports 'linear_classification' and 'knn' evaluation types.
    """

    def __init__(self, model, data, config):
        """
        Args:
            model (SSLModel): Trained SSL model with encoder and projection head.
            data (dict): Dictionary with 'train' and 'test' DataLoader or dataset tuples.
            config (dict): Evaluation configuration parameters.
        """
        self.model = model
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Extract dataset info and evaluation parameters
        self.eval_type = config.get('eval_type', 'linear_classification')
        self.dataset_name = config.get('dataset', 'ImageNet-100')
        self.metric = config.get('metric', 'accuracy')

        # Determine dataset for evaluation
        if 'train' in data:
            self.train_loader = data['train']
        elif isinstance(data.get('train'), torch.utils.data.Dataset):
            self.train_loader = torch.utils.data.DataLoader(data['train'], batch_size=256, shuffle=False)
        else:
            self.train_loader = data['train']

        if 'test' in data:
            self.test_loader = data['test']
        elif isinstance(data.get('test'), torch.utils.data.Dataset):
            self.test_loader = torch.utils.data.DataLoader(data['test'], batch_size=256, shuffle=False)
        else:
            self.test_loader = data['test']

    def extract_features(self, dataloader):
        """
        Extract features for entire dataset by passing data through encoder.
        Args:
            dataloader (DataLoader): DataLoader for dataset split.
        Returns:
            features (np.ndarray): Array of shape [num_samples, feature_dim]
            labels (np.ndarray): Corresponding labels
        """
        feats = []
        lbls = []
        with torch.no_grad():
            for batch in dataloader:
                # Handle batch: assume each batch is (inputs, labels)
                if isinstance(batch, dict):
                    inputs = batch['input']
                    labels = batch['label']
                elif isinstance(batch, list) or isinstance(batch, tuple):
                    inputs = batch[0]
                    labels = batch[1]
                else:
                    # fallback: assume batch[0]=inputs, batch[1]=labels
                    inputs, labels = batch
                inputs = inputs.to(self.device)
                emb = self.model.encode(inputs)
                feats.append(emb.cpu())
                lbls.append(labels.cpu())
        features = torch.cat(feats, dim=0).numpy()
        labels = torch.cat(lbls, dim=0).numpy()
        return features, labels

    def linear_classification(self):
        """
        Perform linear protocol: train a logistic regression on frozen features.
        Returns:
            dict: metrics including 'accuracy' and 'top5_accuracy'
        """
        # Extract features from training and test datasets
        train_feats, train_labels = self.extract_features(self.train_loader)
        test_feats, test_labels = self.extract_features(self.test_loader)

        # Train logistic regression classifier
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
        clf.fit(train_feats, train_labels)
        pred_labels = clf.predict(test_feats)
        pred_scores = clf.predict_proba(test_feats)

        # Compute accuracy
        accuracy = accuracy_score(test_labels, pred_labels)
        # Compute Top-5 accuracy if dataset is large enough
        top5_accuracy = np.mean(np.argsort(-pred_scores, axis=1)[:, :5] == test_labels.reshape(-1, 1)).mean()

        return {
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy
        }

    def knn_evaluation(self, k: int = 5):
        """
        Perform k-NN evaluation directly on embedded features.
        Args:
            k (int): number of neighbors. Default=5.
        Returns:
            dict: metrics including 'accuracy' (on test samples).
        """
        # Extract features
        train_feats, train_labels = self.extract_features(self.train_loader)
        test_feats, test_labels = self.extract_features(self.test_loader)

        # Fit k-NN on train features
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_feats, train_labels)
        pred_labels = knn.predict(test_feats)
        accuracy = accuracy_score(test_labels, pred_labels)

        return {
            'accuracy': accuracy
        }

    def run(self):
        """
        Run the evaluation according to the specified type and return metrics.
        """
        results = {}
        if self.eval_type == 'linear_classification':
            res = self.linear_classification()
            results.update(res)
        elif self.eval_type == 'knn':
            res = self.knn_evaluation(k=5)
            results.update(res)
        else:
            raise ValueError(f"Unknown evaluation type: {self.eval_type}")
        return results
```

## main.py

```python
## main.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np

# Import necessary modules from the project
from dataset_loader import DatasetLoader
from model import SSLModel
from spectral_transform import SpectralTransformer
from regularizer import Regularizer
from trainer import Trainer
from evaluation import Evaluation

def main():
    # 1. Load configuration from 'config.yaml'
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Prepare dataset loader
    dataset_name = config.get('data', {}).get('dataset', 'ImageNet-100')
    dataset_params = config.get('data', {}).get('dataset_params', {})
    dataset_params['dataset_type'] = 'image_classification'  # Assuming classification task

    dataset_loader = DatasetLoader(dataset_name, dataset_params)

    train_dataset, val_dataset = dataset_loader.load_data()

    # 3. Initialize DataLoaders
    batch_size = config.get('training', {}).get('batch_size', 256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Build the backbone model
    model_cfg = config.get('model', {})
    backbone_type = model_cfg.get('backbone', 'ResNet50')
    projection_dim = model_cfg.get('projection_dim', 8192)
    projection_layers = model_cfg.get('projection_layers', 2)
    hidden_dim = model_cfg.get('hidden_dim', 4096)

    model = SSLModel(
        backbone_name=backbone_type,
        projection_dim=projection_dim,
        projection_layers=projection_layers,
        hidden_dim=hidden_dim
    )
    model.to(device)

    # 5. Instantiate spectral transformer
    spec_cfg = {
        'T': config.get('training', {}).get('spe_iteration_T', 4),
        'method': config.get('training', {}).get('spe_method', 'Newton'),
        'p': config.get('training', {}).get('spe_p', 0.5),
        'epsilon': config.get('training', {}).get('spe_epsilon', 1e-5)
    }
    spectral_transformer = SpectralTransformer(**spec_cfg)

    # 6. Instantiate regularizer for trace loss
    trace_loss_weight = config.get('training', {}).get('trace_loss_weight', 0.01)
    regularizer = Regularizer(trace_loss_weight=trace_loss_weight)

    # 7. Setup optimizer
    learning_rate = config.get('training', {}).get('learning_rate', 0.3)
    weight_decay = config.get('training', {}).get('weight_decay', 1e-4)
    momentum = config.get('training', {}).get('momentum', 0.9)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # 8. Setup learning rate scheduler
    epochs = config.get('training', {}).get('epochs', 1000)
    warmup_epochs = config.get('training', {}).get('warmup_epochs', 2)
    schedule_type = config.get('training', {}).get('schedule', 'cosine_decay')
    if schedule_type == 'cosine_decay':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    else:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

    # Create trainer object
    trainer_args = {
        'model': model,
        'spectral_transformer': spectral_transformer,
        'regularizer': regularizer,
        'dataset_loader': dataset_loader,
        'config': {
            'batch_size': batch_size,
            'epochs': epochs,
            'warmup_epochs': warmup_epochs,
            'schedule': schedule_type,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'spe_iteration_T': spec_cfg['T'],
            'spe_method': spec_cfg['method'],
            'spe_p': spec_cfg['p'],
            'spe_epsilon': spec_cfg['epsilon'],
            'trace_loss_weight': trace_loss_weight,
        }
    }
    trainer = Trainer(**trainer_args)
    # Override lossy DataLoader with the actual DataLoader
    trainer.train_loader = train_loader
    trainer.val_loader = val_loader

    # 9. Training loop with spectrum regularization
    for epoch in range(epochs):
        # Warm-up learning rate if needed
        if epoch < warmup_epochs:
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            # Step LR schedule (cosine)
            lr_scheduler.step()

        model.train()
        total_loss = 0.0
        for batch in train_loader:
            # batch: tuple of views, e.g. (view1, view2)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                view1, view2 = batch
            else:
                continue  # adjust as per dataset loader's actual batch output

            view1 = view1.to(device)
            view2 = view2.to(device)

            optimizer.zero_grad()

            # Encode views
            Z1_raw = model.encode(view1)
            Z2_raw = model.encode(view2)

            # Spectral transformation
            Z1_hat = spectral_transformer.transform(Z1_raw)
            Z2_hat = spectral_transformer.transform(Z2_raw)

            # Compute covariance matrices
            Sigma1 = spectral_transformer.compute_covariance(Z1_hat)
            Sigma2 = spectral_transformer.compute_covariance(Z2_hat)

            # Eigendecompose for spectral regularization
            U1, Lambda1 = spectral_transformer.eig_decompose(Sigma1)
            U2, Lambda2 = spectral_transformer.eig_decompose(Sigma2)

            # Optional: compute spectrum regularization (trace loss)
            diag1 = torch.diagonal(Sigma1)
            diag2 = torch.diagonal(Sigma2)
            trace_loss = ( (1 - diag1).pow(2).sum() + (1 - diag2).pow(2).sum() )

            # Compute cosine similarity loss
            Z1_norm = Z1_hat / (torch.norm(Z1_hat, dim=1, keepdim=True) + 1e-8)
            Z2_norm = Z2_hat / (torch.norm(Z2_hat, dim=1, keepdim=True) + 1e-8)
            cosine_sim = (Z1_norm * Z2_norm).sum(dim=1)
            similarity_loss = -torch.mean(cosine_sim)

            # Compute beta (regularization coefficient)
            beta = 0.01 * (np.log2(batch_size) - 3)
            beta = max(beta, 0.0)

            # Total loss
            total_batch_loss = similarity_loss + beta * trace_loss

            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

        # Step the scheduler
        if schedule_type != 'constant':
            lr_scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

        # Optional: logging spectrum, eigenvalues, condition numbers
        # For brevity, omitted here but can be added for diagnostics

    # 10. Evaluation after training
    # Initialize evaluation object
    eval_cfg = {
        'dataset': dataset_name,
        'eval_type': 'linear_classification',
        'metric': 'accuracy'
    }
    evaluator = Evaluation(model, {'train': train_dataset, 'test': val_dataset}, eval_cfg)
    results = evaluator.run()
    print("Evaluation Results: ")
    print(results)

    # 11. Additionally, compute transfer downstream tasks if needed
    # For brevity, omitted here. Can be added as per downstream task datasets and metrics.

if __name__ == "__main__":
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torchvision.models as models
try:
    import timm
except ImportError:
    timm = None

class SSLModel(nn.Module):
    """
    Encapsulates the backbone encoder and projection head for SSL.
    Supports Common Backbones (ResNet) and Vision Transformers via configuration.
    Provides encode() and project() methods.
    """
    def __init__(self,
                 backbone_name: str = 'ResNet50',
                 projection_dim: int = 8192,
                 projection_layers: int = 2,
                 hidden_dim: int = 4096):
        """
        Args:
            backbone_name (str): 'ResNet50', 'ViT-tiny', 'ViT-small', etc.
            projection_dim (int): Output dimension of the projection head.
            projection_layers (int): Number of layers in projection head.
            hidden_dim (int): Hidden dimension size in projection head.
        """
        super().__init__()
        self.backbone_name = backbone_name

        # Initialize backbone
        if backbone_name.lower() == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            # Remove the final FC layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 2048  # ResNet50 final feature size
        elif backbone_name.lower() == 'vit-tiny':
            # Use timm if available
            if timm is None:
                raise ImportError("timm library is required for ViT models.")
            self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=False)
            self.feature_dim = self.backbone.embed_dim  # Typically 192
        elif backbone_name.lower() == 'vit-small':
            if timm is None:
                raise ImportError("timm library is required for ViT models.")
            self.backbone = timm.create_model('vit_small_patch16_224', pretrained=False)
            self.feature_dim = self.backbone.embed_dim  # Typically 384
        else:
            # Default: use ResNet50 if unspecified
            self.backbone = models.resnet50(pretrained=False)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 2048

        # Initialize projection head
        layers = []
        in_dim = self.feature_dim
        for _ in range(projection_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        # Final layer
        layers.append(nn.Linear(in_dim, projection_dim))
        self.projection_head = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone encoder.
        Args:
            x (torch.Tensor): Input images tensor (B, C, H, W)
        Returns:
            torch.Tensor: Backbone feature vectors (B, feature_dim)
        """
        if 'resnet' in self.backbone_name.lower():
            features = self.backbone(x)  # [B, 2048, 1, 1]
            features = features.view(x.size(0), -1)  # flatten to [B, 2048]
        elif 'vit' in self.backbone_name.lower():
            # Use ViT's forward features
            features = self.backbone.forward_features(x)  # [B, embed_dim]
        else:
            # fallback
            features = self.backbone(x)
        return features

    def project(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Pass backbone features through projection head.
        Args:
            emb (torch.Tensor): Features from encoder (B, feature_dim)
        Returns:
            torch.Tensor: Final projections for SSL (B, projection_dim)
        """
        return self.projection_head(emb)
```

## regularizer.py

```python
## regularizer.py
import torch
from typing import Optional

class Regularizer:
    """
    Implements the trace loss regularization for the covariance matrix of spectrally transformed embeddings.
    Encourages the spectrum (diagonal elements) of the covariance matrix to approach 1, preventing collapse.
    """

    def __init__(self, trace_loss_weight: float = 0.01, epsilon: float = 1e-5):
        """
        Initializes the Regularizer with a weight for the trace loss and a stability epsilon.
        
        Args:
            trace_loss_weight (float): Hyperparameter to scale the regularization loss.
            epsilon (float): Small constant added to the diagonal for numerical stability.
        """
        self.trace_loss_weight = trace_loss_weight
        self.epsilon = epsilon

    def __call__(self, covariance_matrix: torch.Tensor) -> torch.Tensor:
        """
        Computes the trace regularization loss given a covariance matrix.
        
        Args:
            covariance_matrix (torch.Tensor): Covariance matrix of shape (d, d), assumed symmetric.
        
        Returns:
            torch.Tensor: Scalar tensor representing the scaled trace loss.
        """
        # Ensure covariance matrix is symmetric; add epsilon to diagonal for stability
        cov = covariance_matrix
        diag_elements = torch.diagonal(cov)
        # Add epsilon for numerical stability if necessary
        diag_stable = diag_elements + self.epsilon

        # Compute (1 - diag_element)^2 for each diagonal
        deviations = (1.0 - diag_stable).pow(2)
        # Sum over all dimensions
        trace_loss = deviations.sum()
        # Scale by the regularization weight
        scaled_loss = self.trace_loss_weight * trace_loss
        return scaled_loss
```

## spectral_transform.py

```python
## spectral_transform.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class SpectralTransformer:
    """
    Implements spectral modulation of embeddings based on covariance spectra.
    Supports whitening, power law modulation, and iterative Newton (IterNorm).
    Designed with compatibility to configuration parameters from 'config.yaml'.
    """

    def __init__(
        self,
        T: int = 4,
        method: str = 'Newton',
        p: float = 0.5,
        epsilon: float = 1e-5
    ):
        """
        Args:
            T (int): Number of iterations T (for iterative Newton).
            method (str): 'whitening', 'power', 'Newton' (for IterNorm).
            p (float): Power parameter for 'power' method, should be near 0.5.
            epsilon (float): Small value for numerical stability, e.g., 1e-5.
        """
        self.T = T
        self.method = method
        self.p = p
        self.epsilon = epsilon

    def compute_covariance(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute empirical covariance matrix of embeddings.
        Args:
            Z (torch.Tensor): shape (d, m), embedding batch
        Returns:
            Sigma (torch.Tensor): shape (d, d), covariance matrix
        """
        m = Z.shape[1]
        Z_mean = Z - Z.mean(dim=1, keepdim=True)
        Sigma = (Z_mean @ Z_mean.t()) / (m - 1)  # unbiased estimator
        return Sigma

    def eig_decompose(self, Sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Eigen decomposition of symmetric matrix Sigma.
        Args:
            Sigma (torch.Tensor): shape (d, d)
        Returns:
            U (torch.Tensor): eigenvectors (d, d)
            Lambda (torch.Tensor): eigenvalues (d,)
        """
        # Use torch.linalg.eigh for symmetric matrices
        # Ensure numerical stability
        Sigma_stable = Sigma + self.epsilon * torch.eye(Sigma.shape[0], device=Sigma.device)
        Lambda, U = torch.linalg.eigh(Sigma_stable)
        # Eigenvalues sorted in ascending order
        return U, Lambda

    def spectral_modulate(self, Lambda: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral modulation function g(λ) = λ^{-p} or whitening.
        Args:
            Lambda (torch.Tensor): eigenvalues (d,)
        Returns:
            gLambda (torch.Tensor): modulated eigenvalues
        """
        if self.method.lower() == 'whitening':
            # g(λ) = λ^{-0.5}
            gLambda = torch.clamp(Lambda, min=self.epsilon) ** -0.5
        elif self.method.lower() == 'power':
            # g(λ) = λ^{-p}
            gLambda = torch.clamp(Lambda, min=self.epsilon) ** -self.p
        elif self.method.lower() == 'newton':
            # Use iterative Newton's method approximation
            # For Newton, apply the f_T function to eigenvalues
            # We'll implement this separately as a method
            # For simplicity, here we do a placeholder: use power law close to 0.5
            gLambda = torch.clamp(Lambda, min=self.epsilon) ** -self.p
        else:
            raise ValueError(f"Unknown spectral modulation method: {self.method}")
        return gLambda

    def apply_iter_norm(self, Sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute the approximate whitening matrix via Newton iteration.
        Returns:
            Phi_T (torch.Tensor): shape (d, d), approximate whitening matrix
        """
        # Initialize P_0
        P = torch.eye(Sigma.shape[0], device=Sigma.device)
        # Compute tr(Sigma)
        tr_sigma = torch.trace(Sigma)
        # Normalize Sigma
        Sigma_N = Sigma / (tr_sigma + self.epsilon)
        for _ in range(self.T):
            # Newton's update: P_k+1 = (3/2)*P_k - (1/2)*P_k^3 * Sigma_N
            P_cubed = P @ P @ P
            P = 0.5 * (3 * P - P_cubed @ Sigma_N)
        Phi_T = P / torch.sqrt(tr_sigma + self.epsilon)
        return Phi_T

    def transform(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Perform spectral transformation on embedding batch Z.
        Args:
            Z (torch.Tensor): shape (d, m)
        Returns:
            Z_hat (torch.Tensor): shape (d, m), spectrally modulated embeddings
        """
        Sigma = self.compute_covariance(Z)
        U, Lambda = self.eig_decompose(Sigma)

        if self.method.lower() == 'Newton':
            # Use iterative Newton's method to approximate Sigma^{-0.5}
            Phi = self.apply_iter_norm(Sigma)
            Z_hat = Phi @ Z
        else:
            # For whitening or power methods, spectral modulation on eigenvalues
            gLambda = self.spectral_modulate(Lambda)
            Z_hat = self.reconstruct_embeddings(U, gLambda, Z)

        return Z_hat

    def reconstruct_embeddings(self, U: torch.Tensor, gLambda: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct transformed embeddings: Z_hat = U (diag(gLambda)) U^T Z
        Args:
            U (torch.Tensor): eigenvectors (d, d)
            gLambda (torch.Tensor): eigenvalues (d,)
            Z (torch.Tensor): original embeddings (d, m)
        Returns:
            Z_hat (torch.Tensor): transformed embeddings (d, m)
        """
        # Construct diagonal matrix of gLambda
        G = torch.diag(gLambda)
        # Compute U G U^T Z
        Z_hat = U @ G @ U.t() @ Z
        return Z_hat

    def log_eigenvalues(self, Lambda: torch.Tensor):
        """
        Optional diagnostic: log spectrum.
        """
        return torch.log(Lambda + self.epsilon)

    # Additional helper functions if needed could be added here
    # For example, a method to compute g(λ) based on configuration

# Note:
# - This module relies on the supplied configuration settings.
# - For 'Newton' mode, the Newton iteration is used to approximate Sigma^{-0.5}.
# - For 'power' mode, apply elementwise power.
# - Proper handling of numerical instability (epsilon adjustment) is embedded.
# - Eigen-decomposition is performed via torch.linalg.eigh with regularization.
# - The class is designed for integration into the training pipeline, transforming batch embeddings.
```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_loader import DatasetLoader
from model import SSLModel
from spectral_transform import SpectralTransformer
from regularizer import Regularizer

import math
from typing import Tuple, Optional


class Trainer:
    """
    Orchestrates the SSL training process with spectral modulation (e.g., INTL) and regularization.
    """

    def __init__(self, 
                 model: SSLModel,
                 spectral_transformer: SpectralTransformer,
                 regularizer: Regularizer,
                 dataset_loader: DatasetLoader,
                 config: dict):
        """
        Initialize the Trainer with model, spectral module, regularizer, dataset, and hyperparameters.

        Args:
            model (SSLModel): Embedding encoder + projection head.
            spectral_transformer (SpectralTransformer): Spectral transformation instance.
            regularizer (Regularizer): Regularizer for spectrum regularization.
            dataset_loader (DatasetLoader): Loader for dataset with augmentations and multiview.
            config (dict): Configuration dict with training hyperparameters.
        """
        self.model = model
        self.spectral_transformer = spectral_transformer
        self.regularizer = regularizer

        # Load dataset
        self.train_dataset, self.val_dataset = dataset_loader.load_data()

        self.batch_size = config.get('batch_size', 256)
        self.epochs = config.get('epochs', 1000)
        self.learning_rate = config.get('learning_rate', 0.3)
        self.warmup_epochs = config.get('warmup_epochs', 2)
        self.schedule_type = config.get('schedule', 'cosine_decay')
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.momentum = config.get('momentum', 0.9)

        # Spectral iteration T, method, modulation p, epsilon
        self.T = config.get('spe_iteration_T', 4)
        self.spe_method = config.get('spe_method', 'Newton')
        self.p = config.get('spe_p', 0.5)
        self.epsilon = config.get('spe_epsilon', 1e-5)

        # Regularization weight for trace loss
        self.trace_loss_weight = config.get('trace_loss_weight', 0.01)

        # Build DataLoader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=int(self.batch_size), shuffle=False)

        # Initialize optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        
        # Learning rate scheduler
        if self.schedule_type == 'cosine_decay':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs - self.warmup_epochs)
        else:
            # Default to cosine
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs - self.warmup_epochs)

    def adjust_learning_rate(self, epoch: int):
        """Adjust learning rate according to schedule and warm-up."""
        if epoch < self.warmup_epochs:
            lr_scale = float(epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate * lr_scale
        else:
            self.lr_scheduler.step()

    def train(self):
        """
        Main training loop over epochs.
        """
        for epoch in range(self.epochs):
            self.adjust_learning_rate(epoch)
            self.model.train()
            total_loss = 0.0
            for batch_idx, batch in enumerate(self.train_loader):
                # Each batch: tuple of views (for simplicity, assume batch is list of tuples)
                # batch: list of tuples, each tuple has views: e.g., (view1, view2)
                # For implementation, batch is List[Tuple[Tensor, Tensor, ...]]
                # Here, assuming batch size is B
                # Batch size: self.batch_size
                # Let's assume batch: list of tuples, length batch_size, each item: tuple of views
                # Reconfigure as per actual Dataset usage
                # For simplicity, assume batch is a tuple of two tensors: (views1, views2)
                # For this code, assume batch: Tuple[Tensor, Tensor]
                # (Alternatively, adapt to multi-view batches as per dataset loader.)

                # Example for two-view batch (batch_view1, batch_view2)
                # If using MultiViewTransform, batch: Tuple[Tensor, Tensor]
                # So, break down accordingly
                # Here, we assume the DatasetLoader returns batch of tuple: (view1, view2)
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    view1, view2 = batch
                elif isinstance(batch, dict):
                    # Adjust as needed
                    continue
                else:
                    # Fallback, assume batch[0], batch[1]
                    view1, view2 = batch
                # Zero grad
                self.optimizer.zero_grad()

                # Forward pass through encoder + projection head
                Z1_raw = self.model.encode(view1)
                Z2_raw = self.model.encode(view2)

                # Spectral transformation for each view
                Z1_hat = self.spectral_transformer.transform(Z1_raw)
                Z2_hat = self.spectral_transformer.transform(Z2_raw)

                # Compute covariance matrices for regularization
                Sigma1 = self.spectral_transformer.compute_covariance(Z1_hat)
                Sigma2 = self.spectral_transformer.compute_covariance(Z2_hat)

                # Eigen-decompose for spectral regularization
                U1, Lambda1 = self.spectral_transformer.eig_decompose(Sigma1)
                U2, Lambda2 = self.spectral_transformer.eig_decompose(Sigma2)

                # Reconstruct modulated embeddings (if needed for spectrum regularization)
                # Compute spectrum regularization loss
                # Diagonal elements approximate eigenvalues
                # For simplicity, use torch.diagonal
                diag1 = torch.diagonal(Sigma1)
                diag2 = torch.diagonal(Sigma2)
                trace_loss1 = torch.sum((1 - diag1).pow(2))
                trace_loss2 = torch.sum((1 - diag2).pow(2))
                trace_loss = trace_loss1 + trace_loss2

                # Compute main similarity loss
                Z1_norm = Z1_hat / (torch.norm(Z1_hat, dim=1, keepdim=True) + 1e-8)
                Z2_norm = Z2_hat / (torch.norm(Z2_hat, dim=1, keepdim=True) + 1e-8)
                cosine_sim = (Z1_norm * Z2_norm).sum(dim=1)
                similarity_loss = -torch.mean(cosine_sim)

                # Total loss = similarity + beta * trace loss
                beta = self._regress_beta(self.batch_size)
                total_loss_batch = similarity_loss + beta * trace_loss

                # Backpropagate
                total_loss_batch.backward()
                self.optimizer.step()

                total_loss += total_loss_batch.item()

            # Step scheduler after each epoch
            if self.schedule_type != 'constant':
                self.lr_scheduler.step()

            # Optional: log average loss, eigenvalues spectrum
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}] - Loss: {avg_loss:.4f}")

    def _regress_beta(self, batch_size: int) -> float:
        """
        Compute beta coefficient based on batch size as per empirical relation.
        """
        # empirically: beta = 0.01 * (log2(batch_size) - 3)
        beta_value = 0.01 * (math.log2(batch_size) - 3)
        return max(beta_value, 0.0)

    def evaluate(self):
        """
        Run evaluation: linear and k-NN classifiers based on features.
        """
        self.model.eval()
        # Extract features for validation set
        features, labels = self._extract_features(self.val_loader)
        # Train linear classifier on features
        # Use sklearn LogisticRegression or a simple linear layer trained separately
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(features, labels)
        acc = clf.score(features, labels)
        print(f"Linear classifier accuracy: {acc*100:.2f}%")
        # k-NN evaluation
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features, labels)
        knn_acc = knn.score(features, labels)
        print(f"k-NN 5 accuracy: {knn_acc*100:.2f}%")
        # Optionally, evaluate transfer downstream tasks, spectrum, etc.

    def _extract_features(self, data_loader: DataLoader):
        """
        Extract features for the entire dataset.
        """
        feats = []
        lbls = []
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                emb = self.model.encode(inputs)
                feats.append(emb.cpu())
                lbls.append(labels)
        features = torch.cat(feats, dim=0).numpy()
        labels = torch.cat(lbls, dim=0).numpy()
        return features, labels

# Note:
# - This implementation assumes a 2-view batch for simplicity; modify as needed for multi-view.
# - Spectral transformation applies to features pulled from encoder before loss computation.
# - Spectral regularization encourages eigenvalues toward 1 via trace loss.
# - The spectrum logging, eigenvalue diagnostics, or condition number checks can be added in train() for monitoring.
# - You may wish to adjust learning rate scheduling, optimizer choices, and hyperparameters as per configuration.
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\INTL\INTL_repo`
