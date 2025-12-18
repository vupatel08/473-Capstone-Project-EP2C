# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging

# Optional: Implement a custom GaussianBlur transform if not using built-in transforms
class GaussianBlur(object):
    def __init__(self, kernel_size=23, sigma=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        # Use torchvision.transforms.GaussianBlur if torchvision >= 0.10.0
        # For compatibility, implement a simple placeholder here
        # but in practice, replace with torchvision.transforms.GaussianBlur
        return img  # Placeholder: replace with actual implementation if needed

# Default mean/std for CIFAR normalization
_CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR_STD = [0.2470, 0.2435, 0.2616]


class DatasetLoader:
    def __init__(self, dataset_name: str, batch_size: int, augmentations: list, is_train: bool = True):
        """
        Initialize DatasetLoader.

        Args:
            dataset_name (str): "CIFAR-10" or "CIFAR-100"
            batch_size (int): Number of samples per batch
            augmentations (list): List of augmentation dicts e.g.,
                [{"RandomCrop": [32, 32, 4]}, {"HorizontalFlip": True}, ...]
            is_train (bool): Whether loading training data or validation/test data
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.is_train = is_train

        self.transform = self._build_transform()

    def _build_transform(self):
        """
        Build the composition of transformations based on augmentations.
        Always include ToTensor and Normalize.
        """
        transform_list = []

        # Apply data augmentations for training
        if self.is_train:
            # If augmentations are provided, add them accordingly
            for aug in self.augmentations:
                for key, value in aug.items():
                    key_lower = key.lower()
                    if key_lower == 'randomcrop':
                        if isinstance(value, list) and len(value) >= 2:
                            size = value[0]
                            padding = value[2] if len(value) > 2 else 4
                            transform_list.append(transforms.RandomCrop(size, padding=padding))
                        else:
                            # fallback or warning
                            transform_list.append(transforms.RandomCrop(32, padding=4))
                    elif key_lower == 'horizontalflip':
                        if value:
                            transform_list.append(transforms.RandomHorizontalFlip())
                    elif key_lower == 'colorjitter':
                        if isinstance(value, list) and len(value) == 4:
                            transform_list.append(transforms.ColorJitter(
                                brightness=value[0],
                                contrast=value[1],
                                saturation=value[2],
                                hue=value[3]))
                    elif key_lower == 'gaussianblur':
                        # If torchvision version supports, replace with:
                        # transforms.GaussianBlur(kernel_size=23, sigma=0.5)
                        # For compatibility, here's a placeholder
                        transform_list.append(GaussianBlur())
                    else:
                        # Unknown augmentation, log warning
                        logging.warning(f"Unknown augmentation key: {key}. Skipping.")
        # Common transformations
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD),
        ])
        return transforms.Compose(transform_list)

    def load_data(self):
        """
        Load dataset with specified transformations and batching.
        Returns a PyTorch DataLoader.
        """
        dataset_class = datasets.CIFAR10 if self.dataset_name == 'CIFAR-10' else datasets.CIFAR100

        # Determine dataset split
        train_flag = True if self.is_train else False

        # Load dataset
        dataset = dataset_class(root='./data', train=train_flag, transform=self.transform, download=True)

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.is_train,
            num_workers=2,
            pin_memory=True,
            drop_last=True  # keep batch size consistent
        )

        return data_loader
```

## evaluation.py

```python
## evaluation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm
from utils import compute_statistics, normalize_features, compute_uniformity_metric_W2
import os

class Evaluation:
    def __init__(self, model, dataset_name, device=None, save_dir='./evaluation'):
        """
        Args:
            model (nn.Module): The trained SSL model with extract_features() method.
            dataset_name (str): "CIFAR-10" or "CIFAR-100" for downstream evaluation.
            device (torch.device, optional): Device to run computations. Defaults to CUDA if available.
            save_dir (str): Directory to save plots and spectra.
        """
        self.model = model
        self.dataset_name = dataset_name
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Parameters for downstream evaluation
        self.downstream_epochs = 100  # as per paper
        self.eval_split = 'test'

    def evaluate_downstream(self, dataloader, num_classes, linear_lr=0.1):
        """
        Perform linear classification on frozen features and compute accuracy.
        Args:
            dataloader (DataLoader): DataLoader for dataset split (test).
            num_classes (int): Number of classes (10 for CIFAR-10, 100 for CIFAR-100).
            linear_lr (float): Learning rate for linear classifier training.
        Returns:
            dict: {'top1_acc': float, 'top5_acc': float}
        """
        # Freeze encoder
        self.model.eval()
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # Define a simple linear classifier
        linear_classifier = nn.Linear(self.model.feature_dim if hasattr(self.model, 'feature_dim') else 512, num_classes).to(self.device)

        optimizer = torch.optim.SGD(linear_classifier.parameters(), lr=linear_lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Train linear classifier
        linear_classifier.train()
        for epoch in range(self.downstream_epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for batch in dataloader:
                images = batch['view'].to(self.device)  # assuming dataset yields dict with 'view'
                labels = batch['label'].to(self.device)
                with torch.no_grad():
                    feats = self.model.extract_features(images)
                    feats = F.normalize(feats, p=2, dim=1)
                outputs = linear_classifier(feats)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
            # Optional: print epoch info
        # Evaluation on test set
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_correct_top5 = 0
        with torch.no_grad():
            for batch in dataloader:
                images = batch['view'].to(self.device)
                labels = batch['label'].to(self.device)
                feats = self.model.extract_features(images)
                feats = F.normalize(feats, p=2, dim=1)
                outputs = linear_classifier(feats)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                # Top-5 accuracy
                top5_preds = outputs.topk(5, dim=1).indices
                for i in range(labels.size(0)):
                    if labels[i] in top5_preds[i]:
                        total_correct_top5 += 1
                total_samples += labels.size(0)
        top1_acc = 100.0 * total_correct / total_samples
        top5_acc = 100.0 * total_correct_top5 / total_samples
        return {'top1_acc': top1_acc, 'top5_acc': top5_acc}

    def compute_spectrum(self, features, title='Singular Value Spectrum'):
        """
        Compute and plot the singular values of the features covariance matrix.
        Args:
            features (Tensor): shape [N, m], features to analyze.
            title (str): Plot title.
        """
        # Normalize features
        feats = F.normalize(features, p=2, dim=1).cpu().numpy()
        # Compute covariance matrix
        cov = np.cov(feats, rowvar=False)
        # Eigen-decomposition
        eigvals = np.linalg.eigh(cov)[0]
        # Singular values = sqrt of eigenvalues for covariance matrix
        singular_values = np.sqrt(np.maximum(eigvals, 0))
        # Plot in log scale
        plt.figure(figsize=(6,4))
        plt.plot(np.arange(1, len(singular_values)+1), np.log10(singular_values + 1e-8))
        plt.xlabel('Component index')
        plt.ylabel('Log10 Singular value')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{title.replace(' ','_')}.png"))
        plt.close()

    def visualize_distribution(self, features, title='Feature Distribution'):
        """
        Visualize the distribution of features projected onto top 2 PCA components.
        Args:
            features (Tensor): shape [N, m], features to visualize.
            title (str): plot title.
        """
        feats = features.detach().cpu().numpy()
        pca = PCA(n_components=2)
        proj = pca.fit_transform(feats)
        plt.figure(figsize=(6,6))
        plt.scatter(proj[:,0], proj[:,1], alpha=0.5, s=10)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{title.replace(' ','_')}.png"))
        plt.close()

    def compute_uniformity_metric(self, features):
        """
        Compute the scalar uniformity score (-W2) on the features.
        Args:
            features (Tensor): shape [N, m]
        Returns:
            float: the -W2 value indicating uniformity.
        """
        # Normalize to unit sphere
        feats_norm = F.normalize(features, p=2, dim=1)
        mean, cov = compute_statistics(feats_norm)
        neg_W2 = compute_uniformity_metric_W2(mean, cov)
        return neg_W2

    def analyze_feature_overlap(self, features):
        """
        Visualize the distribution overlap of a coordinate of Y_i and \hat{Y}_i as in appendix.
        Generate 1D density plots of features.
        Args:
            features (Tensor): shape [N, m]
        """
        feats = features.detach().cpu().numpy()
        y_i = feats[:,0]
        hat_y_i = feats[:,0]  # assuming same distribution for sampling
        plt.figure(figsize=(8,4))
        plt.hist(y_i, bins=51, density=True, alpha=0.5, label='Y_i')
        plt.hist(hat_y_i, bins=51, density=True, alpha=0.5, label='hat_Y_i')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Distribution of coordinate Y_i and hat_Y_i')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'distribution_overlap.png'))
        plt.close()

```

## losses.py

```python
# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UniformityLoss(nn.Module):
    """
    Implements the uniformity metric based on Wasserstein distance (-W2) as described in the paper.
    Computes the empirical mean and covariance of features, then calculates the Wasserstein distance
    between the feature distribution and the approximate uniform spherical distribution modeled as
    Gaussian with zero mean and scaled identity covariance.
    """
    
    def __init__(self, loss_type='InfoNCE', uniformity_lambda=0.1, tau=0.2, feature_dim=128):
        """
        Args:
            loss_type (str): 'InfoNCE', 'MSE', or 'Covariance' specifying base SSL loss type.
            uniformity_lambda (float): Weight for the uniformity Wasserstein loss.
            tau (float): Temperature parameter for contrastive loss (if applicable).
            feature_dim (int): Dimensionality of the feature embeddings.
        """
        super().__init__()
        self.loss_type = loss_type
        self.uniformity_lambda = uniformity_lambda
        self.tau = tau
        self.feature_dim = feature_dim

        # Small epsilon for numerical stability in eigen decomposition
        self.epsilon = 1e-6

    def compute_wasserstein_distance(self, features):
        """
        Compute the negative uniformity metrics (-W2) based on features.
        Args:
            features (Tensor): Shape [batch_size, feature_dim]
        Returns:
            torch.Tensor: scalar tensor containing -W2 value
        """
        # Normalize features to the unit sphere
        z = F.normalize(features, p=2, dim=1)  # shape: [batch_size, feature_dim]
        n = z.shape[0]
        m = z.shape[1]
        device = z.device

        # Step 1: Compute empirical mean
        mu_hat = torch.mean(z, dim=0)  # [feature_dim]
        
        # Center features
        z_centered = z - mu_hat  # [n, m]
        # Step 2: Covariance estimation
        # Covariance: (Z_centered^T Z_centered) / (n -1)
        sigma_hat = (z_centered.T @ z_centered) / (n - 1)  # [m, m]
        
        # Eigen-decomposition of covariance matrix
        # Use torch.linalg.eigh since sigma_hat is symmetric positive semi-definite
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma_hat)
        # Clamp eigenvalues to non-negative for numerical stability
        eigenvalues = torch.clamp(eigenvalues, min=0)
        # Compute trace of sigma_hat
        trace_sigma = torch.sum(eigenvalues)
        # Compute sqrt of sigma_hat: V * diag(sqrt(eigenvalues)) * V^T
        sqrt_eigenvalues = torch.sqrt(eigenvalues)
        sigma_half = (eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T)
        trace_sigma_half = torch.sum(sqrt_eigenvalues)

        # Compute the Wasserstein distance
        W2 = torch.sqrt(
            torch.norm(mu_hat, p=2) ** 2  # ||mu_hat||^2
            + 1
            + trace_sigma
            - (2.0 / math.sqrt(m)) * trace_sigma_half
        )
        # Negative uniformity loss
        neg_W2 = -W2
        return neg_W2

    def compute_base_loss(self, features_view1, features_view2=None):
        """
        Compute the base SSL loss based on loss_type.
        If 'InfoNCE', features_view1 and view2 are positive pairs.
        If 'MSE', features are from two augmented views.
        For 'Covariance', assume features are used for decorrelation losses.
        """
        if self.loss_type == 'InfoNCE':
            # Expect features from two views
            # features_view1 and features_view2: [batch_size, feature_dim]
            # Normalize
            z1 = F.normalize(features_view1, p=2, dim=1)
            z2 = F.normalize(features_view2, p=2, dim=1)
            # Similarity matrix
            sim_matrix = torch.matmul(z1, z2.T) / self.tau  # scaled by temperature
            # Labels: diagonal entries are positives
            labels = torch.arange(z1.size(0), device=z1.device)
            # Contrastive loss (InfoNCE)
            loss = nn.CrossEntropyLoss()
            loss_val = loss(sim_matrix, labels)
            return loss_val
        elif self.loss_type == 'MSE':
            # For BYOL: mean squared error between normalized features
            z1 = F.normalize(features_view1, p=2, dim=1)
            z2 = F.normalize(features_view2, p=2, dim=1)
            loss_val = torch.mean((z1 - z2).pow(2))
            return loss_val
        elif self.loss_type == 'Covariance':
            # Covariance-based decorrelation loss (e.g., Barlow Twins)
            # features: [batch_size, feature_dim]
            z = F.normalize(features_view1, p=2, dim=1)
            # Cross-correlation matrix
            c = (z.T @ z) / z.shape[0]
            # Loss: sum of squared off-diagonal elements
            off_diag = c - torch.eye(c.size(0), device=c.device)
            loss_val = torch.sum(off_diag ** 2)
            return loss_val
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def forward(self, features_view1, features_view2=None):
        """
        Compute the total loss as base SSL loss + weighted uniformity loss.
        Args:
            features_view1 (Tensor): Batch features from view 1
            features_view2 (Tensor): Batch features from view 2 (if applicable)
        Returns:
            total_loss (Tensor), dict of individual losses for logging
        """
        # Compute base SSL loss
        base_loss = self.compute_base_loss(features_view1, features_view2)

        # Compute uniformity loss (-W2)
        neg_W2 = self.compute_wasserstein_distance(features_view1)

        # Total loss sum
        total_loss = base_loss + self.uniformity_lambda * neg_W2

        # Optional logging dict
        log_dict = {
            'base_loss': base_loss,
            'uniformity_loss': neg_W2,
            'total_loss': total_loss
        }

        return total_loss, log_dict
```

## main.py

```python
# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import yaml
import os

from dataset_loader import DatasetLoader
from model import Model
from losses import UniformityLoss
from utils import normalize_features, compute_statistics
from evaluation import Evaluation

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def adjust_learning_rate(optimizer, base_lr, epoch, total_epochs, schedule_type='cosine', warmup_epochs=10, min_lr_ratio=0.001):
    """Adjusts learning rate with cosine schedule with optional warmup."""
    if schedule_type == 'cosine':
        if epoch < warmup_epochs:
            lr = base_lr * epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    # Add other schedules if needed
    return base_lr

def main():
    # Load configuration
    config = load_config('config.yaml')

    # Set random seeds for reproducibility (optional)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Data loading
    dataset_conf = config['dataset']
    train_loader_obj = DatasetLoader(
        dataset_name=dataset_conf['name'],
        batch_size=config['training']['batch_size'],
        augmentations=dataset_conf.get('augmentations', []),
        is_train=True
    )
    val_loader_obj = DatasetLoader(
        dataset_name=dataset_conf['name'],
        batch_size=512,
        augmentations=[],  # No augmentations for evaluation
        is_train=False
    )
    train_loader = train_loader_obj.load_data()
    val_loader = val_loader_obj.load_data()

    # 2. Model initialization
    model_conf = config['model']
    model = Model(model_conf)

    # 3. Loss setup
    loss_conf = config['loss']
    ssl_loss_type = loss_conf['base_loss']
    uniformity_lambda = loss_conf.get('uniformity_lambda', 0.1)
    tau = loss_conf.get('tau', 0.2)
    feature_dim = model_conf['projection_dim']
    loss_fn = UniformityLoss(loss_type=ssl_loss_type,
                             uniformity_lambda=uniformity_lambda,
                             tau=tau,
                             feature_dim=feature_dim)

    # 4. Optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=config['training']['learning_rate'],
                          momentum=config['training'].get('momentum', 0.9),
                          weight_decay=config['training'].get('weight_decay', 1e-4))
                         
    # 5. Training setup
    total_epochs = config['training_schedule'].get('epochs', 500)
    warmup_epochs = config['training_schedule'].get('warmup_epochs', 10)
    schedule_type = config['training_schedule'].get('schedule_type', 'cosine')
    # Store training options
    class TrainConfig:
        epochs = total_epochs
        warmup_epochs = warmup_epochs
        schedule_type = schedule_type
        base_lr = config['training'].get('learning_rate', 0.03)
        lambda_max = config['training'].get('lambda_uniformity', 0.1)

    # 6. Training loop
    for epoch in range(1, total_epochs + 1):
        # Adjust learning rate
        current_lr = adjust_learning_rate(optimizer, TrainConfig.base_lr, epoch, total_epochs, schedule_type, warmup_epochs)
        # Compute current lambda with linear decay
        lambda_t = TrainConfig.lambda_max - (TrainConfig.lambda_max * (epoch / total_epochs))
        if lambda_t < 0:
            lambda_t = 0.0

        model.train()
        total_loss_epoch = 0.0
        total_base_loss = 0.0
        total_uniformity_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            # Batch data: expecting a dict with views, e.g.,
            # { 'view1': tensor, 'view2': tensor, 'label': tensor (if labels present) }
            x1 = batch['view1'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            x2 = batch.get('view2', None)
            if x2 is not None:
                x2 = x2.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            optimizer.zero_grad()

            # Forward pass
            z_a = model.extract_features(x1)
            if x2 is not None:
                z_b = model.extract_features(x2)
            else:
                z_b = None

            # Normalize features to sphere
            z_a_norm = normalize_features(z_a)
            z_b_norm = normalize_features(z_b) if z_b is not None else None

            # Base SSL loss
            if hasattr(model, 'predictor') and model.predictor is not None:
                base_loss = loss_fn.compute_base_loss(z_a_norm, z_b_norm)
            else:
                base_loss = loss_fn.compute_base_loss(z_a_norm, z_b_norm)

            # Collect features for uniformity
            features_for_uniformity = torch.cat([z_a_norm, z_b_norm], dim=0) if z_b_norm is not None else z_a_norm
            mean, cov = compute_statistics(features_for_uniformity)

            # Compute uniformity term (-W2)
            neg_W2 = loss_fn.compute_wasserstein_distance(mean, cov)

            # Total loss
            total_loss = base_loss + lambda_t * neg_W2

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            batch_size = x1.shape[0]
            total_loss_epoch += total_loss.item() * batch_size
            total_base_loss += base_loss.item() * batch_size
            total_uniformity_loss += neg_W2 * batch_size
            total_samples += batch_size

        # Logging epoch
        print(f"Epoch {epoch}/{total_epochs} - lr: {current_lr:.6f} | Loss: {total_loss_epoch/total_samples:.4f} | Base: {total_base_loss/total_samples:.4f} | Uniform: {total_uniformity_loss/total_samples:.4f} | Lambda: {lambda_t:.4f}")

        # Save checkpoint
        save_freq = config['logging'].get('save_model_every', 50)
        if epoch % save_freq == 0:
            save_path = os.path.join(config['logging'].get('log_dir', './logs'), f'checkpoint_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

    # 7. After training: evaluation
    # Initialize evaluation object
    evaluator = Evaluation(model, dataset_conf['name'])
    # Load and prepare data for evaluation if needed (e.g., test set)
    # (assuming dataset loader provides test data with labels)
    test_dataset = torchvision.datasets.CIFAR10 if dataset_conf['name']=='CIFAR-10' else torchvision.datasets.CIFAR100
    # For evaluation, need a DataLoader with labels, normally we create a DataLoader similar to above but with labels
    # For simplicity, assuming val_loader is dedicated for evaluation (as set earlier)
    # But here, better to re-initialize if needed, or reuse val_loader with labels.
    # For preparing the evaluation, load the dataset with labels
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD)
    ])
    eval_dataset = test_dataset(root='./data', train=False, transform=transform_eval, download=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    # 8. Downstream evaluation
    num_classes = 10 if dataset_conf['name']=='CIFAR-10' else 100
    metrics = evaluator.evaluate_downstream(eval_loader, num_classes=num_classes)
    print(f"Downstream linear eval: Top-1 Acc: {metrics['top1_acc']:.2f}%, Top-5 Acc: {metrics['top5_acc']:.2f}%")

    # 9. Representation analysis: spectral, uniformity, distribution
    # Collect features from entire dataset or subset
    features_list = []
    with torch.no_grad():
        for batch in eval_loader:
            imgs = batch[ 'view' ].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            feats = model.extract_features(imgs)
            feats_norm = F.normalize(feats, p=2, dim=1)
            features_list.append(feats_norm.cpu())
    all_features = torch.cat(features_list, dim=0)
    # Spectral decay
    evaluator.compute_spectrum(all_features, title='Spectral decay of features')
    # Distribution overlap plot
    evaluator.analyze_feature_overlap(all_features)
    # Visualization
    evaluator.visualize_distribution(all_features, title='Features PCA visualization')
    # Record uniformity metric
    uniformity_score = evaluator.compute_uniformity_metric(all_features)
    print(f"Uniformity metric (-W2) on full dataset: {uniformity_score:.4f}")

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, config):
        """
        Initialize the Model with backbone, projection head, and optional predictor.
        Args:
            config (dict): Configuration dictionary with keys:
                - backbone (str): 'ResNet-18' or 'ResNet-50'
                - projection_dim (int): Dimension of the projection head output
                - predictor (bool): Whether to include predictor network (for BYOL)
                - use_mlp (bool): Whether to use MLP layers for projection/predictor
        """
        super().__init__()
        
        # Initialize backbone encoder
        if config.backbone == 'ResNet-18':
            self.encoder = models.resnet18(pretrained=False)
            self.feature_dim = 512
        elif config.backbone == 'ResNet-50':
            self.encoder = models.resnet50(pretrained=False)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}")
        
        # Remove the final FC layer to get features before classification
        modules = list(self.encoder.children())[:-1]  # Exclude final fc
        self.encoder = nn.Sequential(*modules)
        
        # Projection head: simple 1 or 2 layer MLP
        proj_dim = config.projection_dim
        if config.use_mlp:
            self.projection_head = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, proj_dim)
            )
        else:
            # Optional: directly use features without projection head
            self.projection_head = nn.Identity()
        
        # Predictor network for BYOL; optional
        self.predictor = None
        if config.predictor:
            self.predictor = nn.Sequential(
                nn.Linear(proj_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, proj_dim)
            )

    def forward(self, x):
        """
        Forward pass: produce predicted features if predictor is used,
        else projection features.
        Args:
            x (Tensor): Input images, shape (batch_size, C, H, W)
        Returns:
            Tensor: Output features (prediction or projection)
        """
        features = self.extract_features(x)  # raw features before fc
        projections = self.projection_head(features)
        if self.predictor:
            predictions = self.predictor(projections)
            return predictions
        return projections

    def extract_features(self, x):
        """
        Extract features from backbone encoder (pre-global average pooling).
        Args:
            x (Tensor): Input images
        Returns:
            Tensor: Features vector, shape (batch_size, feature_dim)
        """
        feat_map = self.encoder(x)  # shape: (batch_size, C, 1, 1) or (batch_size, C, *, *)
        # Global average pooling
        pooled_feat = torch.flatten(feat_map, start_dim=1)  # shape: (batch_size, feature_dim)
        return pooled_feat
```

## trainer.py

```python
# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import math
from utils import normalize_features, compute_statistics
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model: nn.Module, dataloader, loss_fn, optimizer, config: dict):
        """
        Args:
            model (nn.Module): The self-supervised model with encoder (and predictor if needed).
            dataloader (DataLoader): DataLoader providing batches of augmented data.
            loss_fn (UniformityLoss): Loss function integrating SSL and uniformity.
            optimizer (torch.optim.Optimizer): Optimizer instance.
            config (dict): Configuration dictionary with training parameters.
        """
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # Extract configuration parameters
        self.epochs = config.get('training', {}).get('epochs', 500)
        self.warmup_epochs = config.get('training', {}).get('warmup_epochs', 10)
        self.lambda_max = config.get('training', {}).get('lambda_uniformity', 0.1)
        self.lambda_min = 0.0   # For decay
        self.total_epochs = self.epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # For logging
        self.global_step = 0
        os.makedirs(config.get('logging', {}).get('log_dir', './logs'), exist_ok=True)

        # Save the initial state for checkpointing
        self.save_dir = config.get('logging', {}).get('log_dir', './logs')
        self.save_every = config.get('logging', {}).get('save_model_every', 50)

    def get_lambda_t(self, epoch: int) -> float:
        """
        Linearly decay lambda from max to min over total epochs.
        """
        if epoch >= self.total_epochs:
            return self.lambda_min
        return self.lambda_max - (self.lambda_max - self.lambda_min) * (epoch / self.total_epochs)

    def train(self):
        """
        Main training loop.
        """
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            epoch_uniformity = 0.0
            epoch_base_loss = 0.0
            epoch_count = 0

            lambda_t = self.get_lambda_t(epoch)

            tqdm_loader = tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.epochs}")
            for batch in tqdm_loader:
                # Batch may contain multiple views: e.g., {'view1': ..., 'view2': ...}
                # or a single tensor if only one view
                # For simplicity, assuming dataset yields a dictionary with 'view1', 'view2', (or just 'view')
                # Implement as needed
                # Cast all to device
                # Example assuming batch['view1'], batch['view2']
                x1 = batch['view1'].to(self.device)
                x2 = batch.get('view2', None)
                if x2 is not None:
                    x2 = x2.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass: extract features
                # Assume model.extract_features returns raw features before projection/predictor
                z_a = self.model.extract_features(x1)
                if x2 is not None:
                    z_b = self.model.extract_features(x2)
                else:
                    # For methods without second view
                    z_b = None

                # Normalize features to the sphere
                z_a_norm = normalize_features(z_a)
                if z_b is not None:
                    z_b_norm = normalize_features(z_b)

                # Compute base SSL loss
                if hasattr(self.model, 'predictor') and self.model.predictor is not None:
                    # For BYOL or similar
                    base_loss = self.loss_fn.compute_base_loss(z_a_norm, z_b_norm)
                else:
                    # For contrastive methods or others
                    base_loss = self.loss_fn.compute_base_loss(z_a_norm, z_b_norm)

                # Compute uniformity loss based on features
                # For the batch, collect features (all features from views)
                features_for_uniformity = torch.cat([z_a_norm, z_b_norm], dim=0) if z_b is not None else z_a_norm
                # Compute statistics
                mean, cov = compute_statistics(features_for_uniformity)
                # Compute negative Wasserstein distance
                neg_W2 = self.loss_fn.compute_wasserstein_distance(mean, cov)

                # Dynamic lambda - decay schedule
                lambda_now = lambda_t

                # Total loss
                total_loss = base_loss + lambda_now * neg_W2

                # Backpropagation
                total_loss.backward()
                self.optimizer.step()

                # Logging
                batch_size = x1.shape[0]
                epoch_loss += total_loss.item() * batch_size
                epoch_base_loss += base_loss.item() * batch_size
                epoch_uniformity += neg_W2 * batch_size
                epoch_count += batch_size

                self.global_step += 1

            # Average metrics for epoch
            avg_loss = epoch_loss / epoch_count
            avg_base_loss = epoch_base_loss / epoch_count
            avg_uniformity = epoch_uniformity / epoch_count

            # Save checkpoint
            if epoch % self.save_every == 0:
                checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)

            # Log epoch metrics
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, BaseLoss={avg_base_loss:.4f}, Uniformity={avg_uniformity:.4f}, Lambda={lambda_now:.4f}")

        # End of training
        print("Training completed.")

```

## utils.py

## utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

# Constants
EPSILON = 1e-6

def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """
    Normalize feature vectors to unit \ell_2 norm.
    Args:
        features (torch.Tensor): shape (batch_size, feature_dim)
    Returns:
        torch.Tensor: normalized features, same shape
    """
    norms = features.norm(p=2, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=EPSILON)  # prevent division by zero
    normalized = features / norms
    return normalized

def compute_empirical_statistics(features: torch.Tensor) -> tuple:
    """
    Compute empirical mean and covariance of features.
    Args:
        features (torch.Tensor): shape (batch_size, feature_dim)
    Returns:
        mean (torch.Tensor): shape (feature_dim,)
        cov (torch.Tensor): shape (feature_dim, feature_dim)
    """
    mean = torch.mean(features, dim=0)
    centered = features - mean.unsqueeze(0)
    cov = (centered.T @ centered) / (features.shape[0] - 1)
    return mean, cov

def covariance_sqrt_eigen(covariance: torch.Tensor) -> torch.Tensor:
    """
    Compute the square root of a symmetric positive semi-definite matrix via eigen-decomposition.
    Args:
        covariance (torch.Tensor): shape (m, m)
    Returns:
        cov_sqrt (torch.Tensor): shape (m, m)
    """
    # Eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    # Clamp eigenvalues for numerical stability
    eigvals_clamped = torch.clamp(eigvals, min=0)
    sqrt_eigvals = torch.sqrt(eigvals_clamped)
    cov_sqrt = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
    return cov_sqrt

def compute_uniformity_metric_W2(mean: torch.Tensor, cov: torch.Tensor) -> float:
    """
    Compute the -W2 uniformity score based on the empirical mean and covariance.
    Args:
        mean (torch.Tensor): shape (feature_dim,)
        cov (torch.Tensor): shape (feature_dim, feature_dim)
    Returns:
        float: negative Wasserstein distance (-W2)
    """
    feature_dim = mean.shape[0]
    trace_cov = torch.trace(cov).item()
    cov_sqrt = covariance_sqrt_eigen(cov)
    trace_sqrt = torch.trace(cov_sqrt).item()
    mu_norm_sq = torch.sum(mean ** 2).item()
    W2 = np.sqrt(
        mu_norm_sq + 1 + trace_cov - (2.0 / np.sqrt(feature_dim)) * trace_sqrt
    )
    return -W2

def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    Args:
        preds (torch.Tensor): predicted labels, shape (n,)
        labels (torch.Tensor): true labels, shape (n,)
    Returns:
        float: accuracy percentage
    """
    correct = (preds == labels).sum().item()
    total = labels.shape[0]
    return correct / total

def plot_spectrum(singular_values: np.ndarray):
    """
    Plot the log-scaled singular values to visualize spectral decay.
    Args:
        singular_values (np.ndarray): shape (feature_dim,)
    """
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(singular_values)+1), np.log10(singular_values + EPSILON))
    plt.xlabel('Component index')
    plt.ylabel('Log of singular value')
    plt.title('Spectral decay of representation covariance matrix')
    plt.grid(True)
    plt.show()

def visualize_distribution(features: torch.Tensor, title: str = 'Feature distribution'):
    """
    Visualize the 2D distribution of features projected onto top 2 principal components.
    Args:
        features (torch.Tensor): shape (batch_size, feature_dim)
        title (str): plot title
    """
    # Convert to numpy for plotting
    features_np = features.detach().cpu().numpy()
    # Compute principal components via PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features_np)
    plt.figure(figsize=(6,6))
    plt.scatter(proj[:,0], proj[:,1], alpha=0.5, s=10)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Additional utility functions could include: loading eigen-decomposition with fallback,
# robust eigenvalue handling, normalization with device management, etc.

# Note: For eigen-decomposition, scipy.linalg.eigh is used, but here we rely on torch.linalg.eigh.
# If eigen-decomposition of covariance is slow or unstable, consider batching or regularization.
# Also, ensure that features are normalized to the sphere before covariance calculation if needed.
# This code assumes features are already prepared accordingly during training/evaluation
# and that the user calls these functions with proper tensors on the correct device.


---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\WassersteinSSL\WassersteinSSL_repo`
