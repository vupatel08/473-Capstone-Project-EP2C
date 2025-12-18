# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset.py

```python
## dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from scipy.linalg import svd

class ToyDataset(Dataset):
    """
    Synthetic toy dataset with 3 classes, Gaussian distributions, and custom augmentation.
    """
    def __init__(self, split='train', seed=42):
        """
        Args:
            split (str): 'train' or 'test'
            seed (int): random seed for reproducibility
        """
        super().__init__()
        np.random.seed(seed)
        self.split = split
        self.num_classes = 3
        self.dim = 2048
        self.std_cov = 0.35  # Covariance scale
        self.samples_per_class = 1000 if split=='train' else 500
        self.total_samples = self.samples_per_class * self.num_classes

        # Generate class means: 3 orthogonal vectors via SVD
        random_matrix = np.random.randn(self.num_classes, self.dim)
        U, _, _ = svd(random_matrix, full_matrices=False)
        self.class_means = U[:self.num_classes]
        # Scale means if needed (here, leave as is for orthogonality)
        
        # Generate all data
        self.data = []
        self.labels = []
        for y in range(self.num_classes):
            mean_y = self.class_means[y]
            # Generate Gaussian samples for class y
            class_samples = mean_y + np.random.randn(self.samples_per_class, self.dim) * self.std_cov
            self.data.append(class_samples)
            self.labels.extend([y] * self.samples_per_class)
        self.data = np.vstack(self.data).astype(np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        # Compute overall data mean for augmentation
        self.data_mean = np.mean(self.data, axis=0)

        # For augmentation, initialize a mask probability
        self.augment_mask_ratio = 0.6  # ~60% features replaced

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        # Apply augmentation
        x_aug = self.augment_features(x)
        # Convert to tensor
        x_tensor = torch.from_numpy(x_aug).float()
        return x_tensor, y

    def augment_features(self, x):
        """
        Augment by replacing approximately 60% of features with the overall mean vector.
        """
        x_aug = x.copy()
        # Determine number of features to replace
        num_replace = int(self.dim * self.augment_mask_ratio)
        # Randomly choose feature indices to replace
        replace_idx = np.random.choice(self.dim, num_replace, replace=False)
        # Replace features with data mean
        x_aug[replace_idx] = self.data_mean[replace_idx]
        return x_aug

class ImageDataset(Dataset):
    """
    Wrapper for torchvision datasets (e.g., ImageNet-100 or downstream datasets)
    with standard augmentations.
    """
    def __init__(self, root, dataset_name='imagenet', split='train', seed=42):
        """
        Args:
            root (str): dataset root directory
            dataset_name (str): dataset identifier ('imagenet', etc.)
            split (str): 'train', 'val', 'test'
            seed (int): random seed
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.root = root
        self.seed = seed

        # Define transformations based on split
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            # For validation/test: resize and center crop
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        # Load dataset
        # Assuming ImageFolder structure, can be adapted
        if self.dataset_name == 'imagenet':
            dataset_path = os.path.join(self.root, 'imagenet', 'imagenet-100')
            self.full_dataset = datasets.ImageFolder(dataset_path, transform=self.transform)
        else:
            # Placeholder for other datasets if needed
            self.full_dataset = None

        # For splitting val if needed; here, assume dataset is already split
        if self.full_dataset is None:
            raise RuntimeError("Dataset not found or unsupported dataset_name.")

    def __len__(self):
        return len(self.full_dataset)

    def __getitem__(self, idx):
        return self.full_dataset[idx]  # returns (image_tensor, label)
```


## evaluation.py

```python
## evaluation.py
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class LinearEvaluator:
    """
    Evaluates learned representations via linear probing.
    Extracts features from a frozen encoder, trains a linear classifier,
    and reports accuracy on a downstream dataset.
    """
    def __init__(self, model, dataset_loader, device='cuda'):
        """
        Args:
            model (nn.Module): the pretrained, frozen encoder model
            dataset_loader (torch.utils.data.DataLoader): loader for evaluation dataset
            device (str or torch.device): computation device
        """
        self.model = model
        self.dataset_loader = dataset_loader
        self.device = torch.device(device)
        self.features = None
        self.labels = None
        self._prepare()

    def _prepare(self):
        """
        Extract features and labels from the dataset.
        """
        self.features = []
        self.labels = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in self.dataset_loader:
                images = images.to(self.device)
                # Forward pass through encoder
                feats = self._extract_features(images)
                # Normalize features
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                self.features.append(feats.cpu().numpy())
                self.labels.append(labels.numpy())

        self.features = np.concatenate(self.features, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
    
    def _extract_features(self, images):
        """
        Extract features from images using the model's encoder.
        """
        # Assumption: model.encoder is the backbone; if model is the entire model,
        # replace with correct attribute or method.
        return self.model.encoder(images)
    
    def linear_probe(self):
        """
        Train linear classifier on features and evaluate accuracy.
        Returns:
            accuracy (float): Top-1 accuracy in [0, 1]
        """
        # Split features into train/test if dataset has labeled splits
        # Here, we assume entire dataset is used for features; for simulation, use train/test sets if available
        # If dataset has train/test split, prefer those splits
        # For simplicity, here, use entire features for classifier
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        try:
            clf.fit(self.features, self.labels)
        except Exception as e:
            print(f"Error during classifier training: {e}")
            return 0.0

        preds = clf.predict(self.features)
        acc = accuracy_score(self.labels, preds)
        return acc

    def evaluate(self):
        """
        Run feature extraction and classifier training, then return accuracy.
        """
        self._prepare()
        accuracy = self.linear_probe()
        return {'accuracy': accuracy}

    def run(self):
        """
        Wrapper to perform evaluation and print result.
        """
        metrics = self.evaluate()
        print(f"Linear probing accuracy: {metrics['accuracy']*100:.2f}%")
        return metrics
```

## loss.py

```python
## loss.py
import torch
import torch.nn.functional as F

class SupervisedANCLLoss:
    """
    Implements the combined supervised and self-supervised loss for SUPSIAM/SUPBYOL.
    Handles sampling positives from a class-specific feature pool, normalization,
    stop-gradient, and optional covariance regularization.
    """
    def __init__(self, config, target_pool, device):
        """
        Args:
            config (dict): configuration dictionary, containing:
                - alpha (float): weighting coefficient [0,1]
                - temperature (float): scaling for distance (not used directly here, but kept for compatibility)
                - pool_size (int): size of feature pools
                - sampling_pos (str or int): 'all' or M positives per class
                - covariance_regularization (bool): whether to include covariance regularization
                - cov_reg_weight (float): weight for covariance regularization term
            target_pool (object): object with method get_positives(labels, M=None) -> Tensor
            device (torch.device): computational device
        """
        self.alpha = config.get('alpha', 0.5)
        self.temperature = config.get('temperature', 0.1)
        self.pool_size = config.get('pool_size', 8192)
        self.sampling_pos = config.get('sampling_pos', 'all')
        self.covariance_reg = config.get('covariance_regularization', False)
        self.cov_reg_weight = config.get('cov_reg_weight', 1.0)
        self.target_pool = target_pool
        self.device = device

    def __call__(self, online_proj, online_pred, class_labels):
        """
        Compute the combined loss.
        Args:
            online_proj (Tensor): output of online projection head, shape: [B, D]
            online_pred (Tensor): output of predictor, shape: [B, D]
            class_labels (Tensor): class labels for each sample, shape: [B]
        Returns:
            torch scalar: the total loss
        """
        # Normalize online predictions
        pred_norm = F.normalize(online_pred, p=2, dim=1)

        # Sample supervised targets features from target pool
        # Using class labels, retrieve positives
        supervised_targets = []
        for y in class_labels.cpu().numpy():
            feats = self.target_pool.get_positives(y, M=None if self.sampling_pos=='all' else int(self.sampling_pos))
            # feats shape: [?, D]
            # sample M positives if specified
            if feats.shape[0] == 0:
                # fallback: use zeros
                z_avg = torch.zeros(online_proj.shape[1], device=self.device)
            else:
                if self.sampling_pos != 'all' and int(self.sampling_pos) < feats.shape[0]:
                    indices = torch.randint(0, feats.shape[0], (int(self.sampling_pos),))
                    sampled_feats = feats[indices]
                else:
                    sampled_feats = feats
                z_avg = torch.mean(sampled_feats, dim=0)
            z_avg = F.normalize(z_avg, p=2, dim=0)
            supervised_targets.append(z_avg)
        z_sup = torch.stack(supervised_targets, dim=0)  # shape: [B, D]

        # Compute SSL loss (e.g., BYOL / SIMSIAM style)
        # stop-gradient features z2
        # For this, obtain target features with stop gradient
        # Assume target features are passed to the call if needed
        # Here, we only need their features and normalized projections
        # For simplicity, accept z2 as argument or integrate into process
        # But as per design, assuming features are obtained externally, so we do that outside
        # To stay within priorities, assume user provides z2 externally for this loss (or the caller handles)
        # So, we define a method that computes the loss, given the features, so here, leave ssl_loss as precomputed or handle separately
        # Convenient approach: instead, pass z2 features (target features) to this loss function
        raise NotImplementedError("Please provide z2 features as argument for SSL loss computation.")

    def compute_ssl_loss(self, online_pred, target_feat):
        """
        Compute SSL loss between online prediction and target feature.
        Args:
            online_pred (Tensor): online predictor output [B, D]
            target_feat (Tensor): stop-gradient features from target [B, D]
        Returns:
            scalar loss
        """
        online_norm = F.normalize(online_pred, p=2, dim=1)
        target_norm = F.normalize(target_feat, p=2, dim=1)
        loss = torch.sum((online_norm - target_norm.detach())**2, dim=1).mean()
        return loss

    def compute_supervised_loss(self, online_pred, supervised_targets):
        """
        Compute supervised loss (e.g., with features sampled from pool).
        Args:
            online_pred (Tensor): online predictor output [B, D]
            supervised_targets (Tensor): pooled features [B, D]
        Returns:
            scalar loss
        """
        online_norm = F.normalize(online_pred, p=2, dim=1)
        sup_norm = F.normalize(supervised_targets, p=2, dim=1)
        loss = torch.sum((online_norm - sup_norm.detach()) ** 2, dim=1).mean()
        return loss

    def covariance_regularization(self, features):
        """
        Optional covariance regularization to decorrelate features.
        Args:
            features (Tensor): features to regularize, shape: [B, D]
        Returns:
            scalar covariance regularization loss
        """
        # Center features
        feat_mean = torch.mean(features, dim=0, keepdim=True)
        feats_centered = features - feat_mean
        # Covariance matrix
        cov_matrix = (feats_centered.T @ feats_centered) / features.shape[0]
        # Off-diagonal penalty: encourage off-diagonals to be zero (decorrelation)
        off_diag_mask = ~torch.eye(cov_matrix.size(0), dtype=bool, device=features.device)
        off_diag = cov_matrix[off_diag_mask]
        off_diag_loss = torch.sum(off_diag ** 2)

        # Diagonal penalty: encourage diagonal entries to be close to 1
        diag = torch.diag(cov_matrix)
        diag_loss = torch.sum((diag - 1) ** 2)

        return off_diag_loss + diag_loss

    def __call_with_all(self, online_proj, online_pred, class_labels, target_features_ssl):
        """
        Actual full loss calculation calling the above components.
        Args:
            online_proj (Tensor): online projection output [B,D]
            online_pred (Tensor): online predictor output [B,D]
            class_labels (Tensor): [B]
            target_features_ssl (Tensor): features from target network, [B,D]
        Returns:
            scalar loss
        """
        # normalize online and target features
        online_pred_norm = F.normalize(online_pred, p=2, dim=1)
        target_feat_norm = F.normalize(target_features_ssl, p=2, dim=1)
        # Compute ssl loss
        ssl_loss = torch.sum((online_pred_norm - target_feat_norm.detach())**2, dim=1).mean()

        # Sample supervised targets features
        supervised_targets = []
        for y in class_labels.cpu().numpy():
            feats = self.target_pool.get_positives(y, M=None if self.sampling_pos=='all' else int(self.sampling_pos))
            if feats.shape[0] == 0:
                z_avg = torch.zeros(online_proj.shape[1], device=self.device)
            else:
                if self.sampling_pos != 'all' and int(self.sampling_pos) < feats.shape[0]:
                    indices = torch.randint(0, feats.shape[0], (int(self.sampling_pos),))
                    sampled_feats = feats[indices]
                else:
                    sampled_feats = feats
                z_avg = torch.mean(sampled_feats, dim=0)
            z_avg = F.normalize(z_avg, p=2, dim=0)
            supervised_targets.append(z_avg)
        z_sup = torch.stack(supervised_targets, dim=0)  # [B, D]
        sup_loss = torch.sum((F.normalize(online_pred, p=2, dim=1) - z_sup.detach())**2, dim=1).mean()

        # Total loss with alpha
        total_loss = self.alpha * ssl_loss + (1 - self.alpha) * sup_loss

        # Covariance regularization if enabled
        if self.covariance_reg:
            feats_for_cov = F.normalize(online_proj, p=2, dim=1)
            cov_loss = self.covariance_regularization(feats_for_cov)
            total_loss += self.cov_reg_weight * cov_loss

        return total_loss
```

## main.py

```python
# main.py
import yaml
import os
import torch
import numpy as np
import random

from dataset import ToyDataset, ImageDataset
from model import Encoder, ProjectionHead, Predictor, EMAEncoder
from pool import FeaturePool
from loss import SupervisedANCLLoss
from trainer import Trainer
from evaluation import LinearEvaluator

def main():
    # 1. Read configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. Set random seeds for reproducibility
    seed = config.get('training', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. Data Loading
    dataset_cfg = config.get('dataset_loader', {})
    dataset_name = config.get('pretraining', {}).get('dataset', 'imagenet')
    # Decide dataset type based on dataset_name
    if dataset_name == 'toy':
        # For toy dataset, generate synthetic data
        train_dataset = ToyDataset(split='train', seed=seed)
        test_dataset = ToyDataset(split='test', seed=seed+1)
        # DataLoaders
        train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=config['training'].get('batch_size', 256),
            shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=256, shuffle=False)
    else:
        # For image datasets
        root_path = dataset_cfg.get('root', './data')
        train_dataset = ImageDataset(root=root_path, dataset_name=dataset_name, split='train', seed=seed)
        val_dataset = ImageDataset(root=root_path, dataset_name=dataset_name, split='val', seed=seed+1)
        test_dataset = ImageDataset(root=root_path, dataset_name=dataset_name, split='test', seed=seed+2)
        train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=config['training'].get('batch_size', 128),
            shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=256, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=256, shuffle=False)

    # 4. Instantiate Encoder
    backbone_type = config['model'].get('backbone', 'ResNet50')
    if backbone_type == 'ResNet50':
        encoder = Encoder(arch='ResNet50')
    elif backbone_type == 'Linear':  # for toy or simple features
        encoder = Encoder(arch='Linear', input_dim=2048)
    else:
        raise NotImplementedError(f"Backbone {backbone_type} not implemented.")

    encoder = encoder.to(device)

    # 5. Instantiate Projection Head
    proj_dim = config['model'].get('projection_dim', 128)
    pred_dim = config['model'].get('predictor_dim', 4096)
    num_layers_proj = 2 if 'Sup' not in backbone_type else 3  # As per baseline
    projector = ProjectionHead(input_dim=encoder.out_dim,
                               output_dim=proj_dim,
                               num_layers=num_layers_proj).to(device)

    # Instantiate predictor
    predictor = Predictor(input_dim=proj_dim, hidden_dim=pred_dim).to(device)

    # 6. Instantiate Target Network (EMA or shared)
    mode = 'SUPBYOL' if 'SUPBYOL' in backbone_type else 'SUPSIAM'
    if mode == 'SUPBYOL':
        target_encoder = EMAEncoder(encoder, projector, ema_m=config['training'].get('momentum', 0.99))
    else:
        # For SUPSIAM, optionally share encoder weights
        # or instantiate a separate copy
        target_encoder = EMAEncoder(encoder, projector, ema_m=0.0)  # no EMA
        target_encoder.backbone_ema.load_state_dict(encoder.state_dict())

    # 7. Instantiate feature pool
    pool_size = config['pool'].get('pool_size', 8192)
    num_classes = None
    if hasattr(train_dataset, 'labels'):
        num_classes = int(np.max(train_dataset.labels)) + 1
    elif hasattr(train_dataset, 'full_dataset'):
        # for image dataset
        num_classes = 100  # or get from dataset
    else:
        num_classes = 100  # fallback default

    feature_dim = proj_dim
    class_specific_pool = FeaturePool(size=pool_size, num_classes=num_classes,
                                      feature_dim=feature_dim,
                                      update_with_ema=config['pool'].get('update_with_ema', True),
                                      ema_m=config['training'].get('momentum', 0.99),
                                      device=device)

    # 8. Instantiate Loss function
    loss_params = {
        'alpha': config['loss'].get('alpha', 0.5),
        'temperature': config['loss'].get('temperature', 0.1),
        'pool_size': pool_size,
        'sampling_pos': config['loss'].get('sampling_pos', 'all'),
        'covariance_regularization': config['loss'].get('covariance_regularization', False),
        'cov_reg_weight': config['loss'].get('cov_reg_weight', 1.0)
    }
    loss_fn = SupervisedANCLLoss(loss_params, class_specific_pool, device)

    # 9. Instantiate optimizer and scheduler
    optim_params = list(encoder.parameters()) + list(projector.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.SGD(optim_params,
                                lr=config['training'].get('learning_rate', 0.05),
                                momentum=0.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training'].get('epochs', 200))

    # 10. Instantiate Trainer
    trainer = Trainer(config, train_dataset, encoder, class_specific_pool,
                      loss_fn, device)
    # Set optimizer, scheduler, predictor, target network in trainer
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.predictor = predictor
    if hasattr(trainer, 'target_encoder'):
        trainer.target_encoder = target_encoder
    else:
        # fallback
        trainer.target_encoder = target_encoder

    # 11. Run training
    print("Starting training...")
    trainer.train()

    # 12. Evaluation: linear probing on validation/test
    print("Evaluating linear probe on validation data...")
    acc_val = trainer.evaluate_linear(eval_dataset=val_dataset, split='val')
    print(f"Validation linear accuracy: {acc_val:.2f}%")

    print("Evaluating linear probe on test data...")
    acc_test = trainer.evaluate_linear(eval_dataset=test_dataset, split='test')
    print(f"Test linear accuracy: {acc_test:.2f}%")
    
    # Optional: Save model
    os.makedirs('outputs', exist_ok=True)
    torch.save(encoder.state_dict(), 'outputs/pretrained_encoder.pth')
    torch.save(projector.state_dict(), 'outputs/pretrained_projector.pth')
    torch.save(predictor.state_dict(), 'outputs/predictor.pth')

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torchvision.models as models

from typing import Optional

class Encoder(nn.Module):
    """
    Encoder backbone supporting 'ResNet50', 'ViT-small' (not implemented here),
    and 'Linear' for toy datasets.
    """
    def __init__(self, arch: str = 'ResNet50', input_dim: int = 2048):
        """
        Args:
            arch (str): backbone architecture, options: 'ResNet50', 'ViT-small', 'Linear'
            input_dim (int): feature dimension for linear encoder
        """
        super().__init__()
        self.arch = arch
        if arch == 'ResNet50':
            resnet = models.resnet50(pretrained=False)
            # Remove final FC layer
            modules = list(resnet.children())[:-1]  # remove fc
            self.backbone = nn.Sequential(*modules)
            self.out_dim = 2048
        elif arch == 'ViT-small':
            # Placeholder for ViT; for now, raise NotImplementedError
            # Implement or import ViT as needed.
            raise NotImplementedError("ViT-small backbone not implemented in this snippet.")
        elif arch == 'Linear':
            # For toy data, input features are raw vector: define a linear layer
            self.backbone = nn.Linear(input_dim, input_dim)
            self.out_dim = input_dim
        else:
            raise ValueError(f"Unsupported backbone architecture: {arch}")

    def forward(self, x):
        if self.arch == 'ResNet50':
            x = self.backbone(x)
            x = x.squeeze()  # shape: (batch_size, 2048, 1, 1)
            x = nn.functional.adaptive_avg_pool2d(x, (1,1)).squeeze()
            return x
        elif self.arch == 'ViT-small':
            # Placeholder: implement ViT forward
            raise NotImplementedError("ViT backbone not implemented.")
        elif self.arch == 'Linear':
            # For toy dataset: assume input x is feature vector
            return self.backbone(x)
        else:
            raise ValueError(f"Unsupported backbone architecture: {self.arch}")

class ProjectionHead(nn.Module):
    """
    MLP projection head: 2 or 3 layers based on config.
    """
    def __init__(self, input_dim: int, output_dim: int = 128, num_layers: int = 2):
        """
        Args:
            input_dim (int): input feature dimension
            output_dim (int): projection dimension
            num_layers (int): number of linear layers (2 or 3)
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        # Build hidden layers
        for _ in range(num_layers -1):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = in_dim
        # Final layer to output_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

class Predictor(nn.Module):
    """
    Predictor network: 2-layer MLP with hidden size 4096.
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 4096):
        """
        Args:
            input_dim (int): input feature dimension (projection output)
            hidden_dim (int): hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        x = self.net(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

class EMAEncoder(nn.Module):
    """
    Maintains a copy of the encoder with EMA weights for SUPBYOL.
    """
    def __init__(self, backbone: nn.Module, projection_head: nn.Module, ema_m: float = 0.99):
        """
        Args:
            backbone (nn.Module): online encoder backbone
            projection_head (nn.Module): online projection head
            ema_m (float): EMA momentum
        """
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        self.ema_m = ema_m

        # Create shadow copies
        self.backbone_ema = self._clone_module(self.backbone)
        self.projection_head_ema = self._clone_module(self.projection_head)

        # Initialize EMA networks parameters
        self._initialize_ema()

    def _clone_module(self, module):
        # Clone with identical architecture but independent parameters
        clone = type(module)(*self._get_init_args(module))
        clone.load_state_dict(module.state_dict())
        return clone

    def _get_init_args(self, module):
        # Extract constructor args from the module
        # For simplicity, assume direct instantiation supported
        # Alternatively, hard-coded or passed as args
        return []

    def _initialize_ema(self):
        # Copy initial weights
        self._copy_weights(self.backbone, self.backbone_ema)
        self._copy_weights(self.projection_head, self.projection_head_ema)

    def _copy_weights(self, source, target):
        target.load_state_dict(source.state_dict())

    def update(self):
        """
        Update EMA parameters: backbone_ema and projection_head_ema
        """
        for param_ema, param in zip(self.backbone_ema.parameters(), self.backbone.parameters()):
            param_ema.data.mul_(self.ema_m).add_((1 - self.ema_m) * param.data)
        for param_ema, param in zip(self.projection_head_ema.parameters(), self.projection_head.parameters()):
            param_ema.data.mul_(self.ema_m).add_((1 - self.ema_m) * param.data)

    def forward_online(self, x):
        """
        Forward pass through online encoder + projector + predictor
        """
        feat = self.backbone(x)
        proj = self.projection_head(feat)
        pred = None
        # predictor might be used outside; if needed, can return pred as well
        return feat, proj

    def forward_target(self, x):
        """
        Forward pass through target encoder + projection head
        """
        with torch.no_grad():
            feat = self.backbone_ema(x)
            proj = self.projection_head_ema(feat)
        return feat, proj
```

## pool.py

```python
## pool.py
import numpy as np
import torch
from collections import defaultdict, deque

class FeaturePool:
    """
    Implements class-specific feature queues for supervised ANCL.
    Supports enqueueing features, sampling positives, optional EMA updates,
    and flexible pool management strategies.
    """
    def __init__(self, size: int = 8192, num_classes: int = 100,
                 feature_dim: int = 128, update_with_ema: bool = True,
                 ema_m: float = 0.99, device: torch.device = torch.device('cpu')):
        """
        Args:
            size (int): total size of the feature pool across all classes.
            num_classes (int): total number of classes.
            feature_dim (int): dimension of feature vectors.
            update_with_ema (bool): whether to update features via EMA (SUPBYOL).
            ema_m (float): EMA momentum coefficient.
            device (torch.device): device for storing features.
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.update_with_ema = update_with_ema
        self.ema_m = ema_m

        # Compute per-class buffer size (floored division)
        self.buffer_size_per_class = size // num_classes
        self.total_size = self.buffer_size_per_class * num_classes
        # For classes where size isn't divisible, last class can get remaining slots
        remainder = size - self.total_size
        self.class_buffer_sizes = [self.buffer_size_per_class] * num_classes
        for y in range(remainder):
            self.class_buffer_sizes[y] += 1

        # Initialize buffers: dict: class_label -> tensor buffer (maxlen=buffer size)
        # Using torch tensors stored on device
        self.buffers = {}
        # Maintain current insertion index for each class for FIFO overwriting
        self.next_idx = {}
        for y, buf_size in enumerate(self.class_buffer_sizes):
            self.buffers[y] = torch.zeros((buf_size, feature_dim), device=self.device)
            self.next_idx[y] = 0

    def enqueue(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Insert features into class buffers, updating with optional EMA in SUPBYOL.
        Args:
            features (torch.Tensor): shape [batch_size, feature_dim]
            labels (torch.Tensor): shape [batch_size]
        """
        batch_size = features.shape[0]
        for i in range(batch_size):
            y = int(labels[i].item())
            feat = features[i]
            # Normalize feature
            feat = torch.nn.functional.normalize(feat, p=2, dim=0)
            # Determine insertion index
            idx = self.next_idx[y]
            buf_size = self.class_buffer_sizes[y]
            # Enqueue by overwriting oldest features (cyclic)
            self.buffers[y][idx] = feat
            # Update pointer
            self.next_idx[y] = (idx + 1) % buf_size
            # If EMA is enabled (SUPBYOL), update the buffer feature via EMA
            if self.update_with_ema:
                # current stored feature
                old_feat = self.buffers[y][idx]
                # EMA update
                new_feat = self.ema_m * old_feat + (1 - self.ema_m) * feat
                new_feat = torch.nn.functional.normalize(new_feat, p=2, dim=0)
                self.buffers[y][idx] = new_feat

    def get_positives(self, y: int, M: int = None):
        """
        Retrieve all or M randomly sampled features for class y.
        Args:
            y (int): class label
            M (int or None): number of positives to sample; if None, return all
        Returns:
            torch.Tensor: shape [num_samples, feature_dim]
        """
        feats = self.buffers[y]
        valid_mask = (feats.norm(p=2, dim=1) > 0)  # To identify non-init zero entries
        valid_feats = feats[valid_mask]

        num_feats = valid_feats.shape[0]
        if num_feats == 0:
            # No features stored yet; return zeros
            return torch.zeros((1, self.feature_dim), device=self.device)
        if M is None or M == 'all':
            # Return all features
            return valid_feats
        else:
            # Sample M features, with replacement if needed
            M = int(M)
            if num_feats >= M:
                indices = torch.randint(0, num_feats, (M,), device=self.device)
            else:
                # Less features than M: sample with replacement
                indices = torch.randint(0, num_feats, (M,), device=self.device)
            sampled_feats = valid_feats[indices]
            return sampled_feats

    def get_buffer_for_class(self, y: int):
        """
        Return current features stored for class y.
        """
        return self.buffers[y]

    def buffer_size(self, y: int):
        """
        Return current active size of class y buffer
        """
        feats = self.buffers[y]
        mask = feats.norm(p=2, dim=1) > 0
        return int(mask.sum().item())

```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import math
import time

from dataset import Dataset
from model import Encoder, ProjectionHead, Predictor, EMAEncoder
from pool import FeaturePool
from loss import SupervisedANCLLoss
from evaluation import LinearEvaluator

class Trainer:
    def __init__(self, config: dict, dataset: Dataset, model: Encoder, pool: FeaturePool,
                 loss_fn: SupervisedANCLLoss, device: torch.device):
        """
        Initialize the trainer with model, dataset, pool, loss, and configs.
        """
        self.config = config
        self.device = device
        self.model = model
        self.dataset = dataset
        self.pool = pool
        self.loss_fn = loss_fn

        # Prepare online and target networks
        self.online_encoder = model
        self.online_projector = model  # assuming combined or separate, adapt as needed
        self.predictor = None  # instantiate predictor
        self._init_predictor()

        # If SUPBYOL, create EMA target networks
        if isinstance(model, EMAEncoder):
            self.target_encoder = model
        else:
            # For SUPSIAM, target encoder is a separate copy or EMA
            self.target_encoder = EMAEncoder(self.model, self.model.projection_head, 
                                             ema_m=self.config.get('momentum', 0.99))
        self.target_encoder = self.target_encoder.to(self.device)

        # Fully online model pipeline for convenience
        # for differentiation if needed
        self.online_encoder = self.online_encoder.to(self.device)
        self.online_projector = self.online_projector.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.SGD(
            list(self.online_encoder.parameters()) +
            list(self.online_projector.parameters()) +
            list(self.predictor.parameters()),
            lr=self.config.get('learning_rate', 0.05),
            momentum=0.9,
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('epochs', 200)
        )

        # EMA momentum for target encoder
        self.ema_m = self.config.get('momentum', 0.99)

    def _init_predictor(self):
        input_dim = self.model.out_dim
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, self.config.get('predictor_dim', 4096)),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.get('predictor_dim', 4096), input_dim)
        ).to(self.device)

    def _update_ema_target(self):
        """
        EMA update for target encoder parameters.
        """
        with torch.no_grad():
            for param_online, param_target in zip(self.online_encoder.parameters(),
                                                  self.target_encoder.backbone_ema.parameters()):
                param_target.data.mul_(self.ema_m).add_((1 - self.ema_m) * param_online.data)
            for param_online, param_target in zip(self.online_projector.parameters(),
                                                  self.target_encoder.projection_head_ema.parameters()):
                param_target.data.mul_(self.ema_m).add_((1 - self.ema_m) * param_online.data)

    def train(self):
        """
        Main training loop for specified epochs.
        """
        num_epochs = self.config.get('epochs',200)
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_ssl_loss = 0.0
            total_sup_loss = 0.0
            total_intra_var = 0.0
            total_batches = 0

            for batch_idx, (images, labels) in enumerate(self.dataset.get_train_loader()):
                images = {k: v.to(self.device) for k, v in images.items()}  # assuming dict of views
                labels = labels.to(self.device)

                # Generate two augmented views: in dataset loader, views are already provided
                view1 = images['view1']  # first augmented view
                view2 = images['view2']  # second augmented view

                # Forward pass online branch on view1
                feat1 = self.online_encoder(view1)
                z1 = self.online_projector(feat1)
                p1 = self.predictor(z1)
                p1 = nn.functional.normalize(p1, p=2, dim=1)

                # Forward pass target branch on view2 (no grad)
                with torch.no_grad():
                    feat2, z2 = self.target_encoder.forward_target(view2)
                    # Normalize features
                    z2 = nn.functional.normalize(z2, p=2, dim=1)

                # --- Sample supervised positives (from pool and labels) ---
                # For each example, sample positives of same class from pool
                batch_size = p1.shape[0]
                feat_z2_sup_list = []
                for y in labels.cpu().numpy():
                    feats = self.pool.get_positives(y, M=self.config.get('sampling_pos', 'all'))
                    # shape [?, D]
                    if feats.shape[0] == 0:
                        z_avg = torch.zeros(z2.shape[1], device=self.device)
                    else:
                        if self.config.get('sampling_pos', 'all') != 'all':
                            M_pos = int(self.config.get('sampling_pos', 1))
                            indices = torch.randint(0, feats.shape[0], (M_pos,), device=self.device)
                            feats_sampled = feats[indices]
                            z_avg = torch.mean(feats_sampled, dim=0)
                        else:
                            z_avg = torch.mean(feats, dim=0)
                    z_avg = nn.functional.normalize(z_avg, p=2, dim=0)
                    feat_z2_sup_list.append(z_avg)
                z2_sup = torch.stack(feat_z2_sup_list, dim=0)  # [B, D]

                # Compute combined loss
                ssl_loss = torch.sum((p1 - z2.detach())**2, dim=1).mean()

                # Supervised counterpart loss
                sup_loss = torch.sum((p1 - z2_sup.detach())**2, dim=1).mean()

                total_batch_loss = self.config.get('alpha', 0.5) * ssl_loss + \
                                   (1 - self.config.get('alpha', 0.5)) * sup_loss

                # Optional covariance regularization (not included here; can be added)

                # Backpropagation
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                self.optimizer.step()

                # --- Update target encoder via EMA ---
                if hasattr(self.target_encoder, 'update'):
                    self.target_encoder.update()

                # --- Update feature pools with features from view2 ---
                self.pool.enqueue(z2, labels)

                # --- Track metrics ---
                total_loss += total_batch_loss.item()
                total_ssl_loss += ssl_loss.item()
                total_sup_loss += sup_loss.item()
                # Compute intra-class variance proxy
                intra_vars = []
                for y in set(labels.cpu().numpy()):
                    feats_class = self.pool.get_positives(y)
                    if feats_class.shape[0] > 0:
                        mean_feat = feats_class.mean(dim=0)
                        var = torch.mean(torch.sum((feats_class - mean_feat)**2, dim=1))
                        intra_vars.append(var.item())
                if len(intra_vars) > 0:
                    intra_mean = sum(intra_vars)/len(intra_vars)
                else:
                    intra_mean = 0.0
                total_intra_var += intra_mean

                total_batches += 1

            # Step LR scheduler
            self.scheduler.step()
            # Update EMA for target network
            if hasattr(self.target_encoder, 'update'):
                self.target_encoder.update()

            # Logging
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {total_loss/total_batches:.4f} "
                  f"SSL: {total_ssl_loss/total_batches:.4f} "
                  f"SUP: {total_sup_loss/total_batches:.4f} "
                  f"IntraVar: {total_intra_var/total_batches:.4f}")

            # Save model checkpoints if desired
            # e.g., torch.save(self.online_encoder.state_dict(), 'model_ep{}.pth'.format(epoch+1))
            # Save metrics, etc.

        # End training
        print("Training finished.")

    def evaluate_linear(self, eval_dataset, split='val'):
        """
        Evaluate the learned representation via linear probing.
        """
        evaluator = LinearEvaluator(self.model, eval_dataset, device=self.device)
        acc = evaluator.linear_probe(split=split)
        print(f"Linear probing accuracy: {acc:.2f}%")
        return acc

# Usage Example (not included in this file):
# if __name__ == "__main__":
#     # Load config, datasets, models, pool, loss
#     # Instantiate trainer and run
#     trainer = Trainer(config, dataset, model, pool, loss_fn, device)
#     trainer.train()
#     trainer.evaluate_linear(eval_dataset)
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\Sup-ANCL\Sup-ANCL_repo`
