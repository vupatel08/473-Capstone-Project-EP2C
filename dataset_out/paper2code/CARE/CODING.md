# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import glob

# Utility for loading protein structures (point clouds)
def load_pdb_points(pdb_file):
    """
    Load 3D points from a PDB file.
    Assumes each line after 'ATOM' contains x,y,z coordinates.
    """
    points = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                if len(parts) >= 7:
                    x, y, z = map(float, parts[6:9])
                    points.append([x, y, z])
    return np.array(points, dtype=np.float32)

# Rotation matrix sampling from SO(3)
def random_rotation_matrix(max_angle_deg=15):
    """
    Generate a random rotation matrix with rotation angle up to max_angle_deg.
    """
    angle = np.random.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180.0
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    # Quaternion to rotation matrix
    rot_mat = np.array([
        [1 - 2*(c**2 + d**2), 2*(b*c - d*a), 2*(b*d + c*a)],
        [2*(b*c + d*a), 1 - 2*(b**2 + d**2), 2*(c*d - b*a)],
        [2*(b*d - c*a), 2*(c*d + b*a), 1 - 2*(b**2 + c**2)]
    ], dtype=np.float32)
    return rot_mat

class ImageAugmentation:
    """
    Compose image augmentations consistent with self-supervised contrastive methods.
    """
    def __init__(self, crop_size=32, jitter_std=0.1, color_jitter=True, blur=True, rotation_deg=15):
        transform_list = []
        if crop_size:
            transform_list.append(transforms.RandomResizedCrop(crop_size))
        if color_jitter:
            transform_list.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))
        if blur:
            transform_list.append(transforms.GaussianBlur(kernel_size=3))
        transform_list.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(transform_list)
        self.rotation_deg = rotation_deg

    def __call__(self, img):
        img_aug = self.transform(img)
        # Optional: add small rotation
        angle = np.random.uniform(-self.rotation_deg, self.rotation_deg)
        img_aug = transforms.functional.rotate(img_aug, angle)
        return img_aug

class ProteinAugmentation:
    """
    Apply random 3D rotation to protein point cloud.
    """
    def __init__(self, max_angle_deg=15):
        self.max_angle_deg = max_angle_deg

    def __call__(self, pointcloud):
        rot_mat = random_rotation_matrix(self.max_angle_deg)
        # pointcloud shape: [N_points, 3]
        rotated = np.dot(pointcloud, rot_mat.T)
        return rotated

class ImageDataset(Dataset):
    """
    Wrapper for torchvision datasets with augmentations.
    """
    def __init__(self, dataset_name, root, transform=None, splits='train'):
        """
        dataset_name: str, e.g. 'CIFAR10'
        root: dataset directory path
        transform: torchvision transforms
        splits: 'train' or 'test'
        """
        self.dataset_name = dataset_name
        self.transform = transform
        if dataset_name == 'CIFAR10':
            self.dataset = datasets.CIFAR10(root=root, train=(splits=='train'), download=True)
        elif dataset_name == 'CIFAR100':
            self.dataset = datasets.CIFAR100(root=root, train=(splits=='train'), download=True)
        elif dataset_name == 'STL10':
            self.dataset = datasets.STL10(root=root, split=splits, download=True)
        elif dataset_name == 'ImageNet100':
            # Assume dataset organized in ImageFolder format
            # images should be in root/train and root/val
            dir_path = os.path.join(root, splits)
            self.dataset = datasets.ImageFolder(root=dir_path, transform=self.transform)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, label = self.dataset[idx]
        return {'x': sample, 'label': label}

class ProteinDataset(Dataset):
    """
    Load protein point clouds, supports random rotations.
    Assumes data in a directory with files for each protein, e.g., in PDB format or numpy arrays.
    """
    def __init__(self, root_dir, max_points=1024, augment=True, max_angle_deg=15):
        """
        root_dir: directory containing protein files
        max_points: max number of points (pad or truncate)
        augment: whether to apply random rotation
        """
        self.root_dir = root_dir
        self.file_list = glob.glob(os.path.join(root_dir, '*.pdb')) + glob.glob(os.path.join(root_dir, '*.npy'))
        self.augment = augment
        self.max_points = max_points
        self.auglator = ProteinAugmentation(max_angle_deg)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        if file_path.endswith('.pdb'):
            points = load_pdb_points(file_path)
        elif file_path.endswith('.npy'):
            points = np.load(file_path)
        else:
            raise ValueError(f"Unsupported file extension for {file_path}")

        # Downsample or pad to fixed size
        if points.shape[0] > self.max_points:
            indices = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[indices]
        elif points.shape[0] < self.max_points:
            pad_size = self.max_points - points.shape[0]
            pad = np.zeros((pad_size, 3), dtype=np.float32)
            points = np.vstack((points, pad))
        # Normalize points if needed (e.g., center) - optional
        # Apply augmentation
        if self.augment:
            points_aug = self.auglator(points)
        else:
            points_aug = points
        return torch.tensor(points, dtype=torch.float32), torch.tensor(points_aug, dtype=torch.float32)

def get_dataset(config):
    """
    Factory function to instantiate datasets based on configuration.
    """
    name = config['dataset']['name']
    path = config['dataset']['path']
    if name in ['CIFAR10', 'CIFAR100', 'STL10', 'ImageNet100']:
        # Compose standard transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Additional normalization can be added here if needed
        ])
        dataset = ImageDataset(dataset_name=name, root=path, transform=transform, splits='train')
        test_dataset = ImageDataset(dataset_name=name, root=path, transform=transform, splits='test') if hasattr(datasets, name) else None
        return dataset, test_dataset
    elif name == 'Proteins':
        # For proteins, create dataset with optional augmentation
        dataset = ProteinDataset(root_dir=path, max_points=1024, augment=True)
        return dataset, None
    else:
        raise ValueError(f"Unsupported dataset name: {name}")
```

## evaluation.py

```python
## evaluation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loader import get_dataset

import yaml
import os

# Load configuration for evaluation parameters
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_linear_probe(model, test_loader, dataset_name, is_protein=False):
    """
    Performs linear evaluation:
    - Extract features for train/test data
    - Train logistic regression on train features
    - Evaluate accuracy on test features
    Returns:
        dict with 'top1_acc' and 'top5_acc' if applicable
    """
    model.eval()
    features_list = []
    labels_list = []
    
    # Collect features for train set
    train_dataset, test_dataset = get_dataset(cfg)
    # For consistency, use train/test splits as per dataset
    # Here, assume features can be extracted via model
    # WARNING: For full pipeline, tend to use a separate DataLoader if needed
    for mode, dataloader in [('train', DataLoader(train_dataset, batch_size=512, shuffle=False)),
                             ('test', DataLoader(test_dataset, batch_size=512, shuffle=False))]:
        feats = []
        lbls = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['x']
                    lbl = batch.get('label', None)
                else:
                    x = batch[0]
                    lbl = batch[1] if len(batch) > 1 else None
                x = x.to(device)
                feat = model(x)
                feats.append(feat.cpu())
                lbls.append(lbl)
        feats = torch.cat(feats, dim=0).numpy()
        lbls = np.array(lbls)
        if mode=='train':
            train_feats = feats
            train_labels = lbls
        else:
            test_feats = feats
            test_labels = lbls

    # Train linear classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_feats, train_labels)
    pred = clf.predict(test_feats)
    top1_acc = accuracy_score(test_labels, pred) * 100
    return {'top1_acc': top1_acc}

def visualize_trajectories(model, x, augmentation_list, label=None, save_path=None):
    """
    For a given input x, apply a sequence of augmentations (e.g. rotations),
    compute embeddings, and plot trajectories in 2D (via PCA or t-SNE).
    """
    model.eval()
    with torch.no_grad():
        # Compute embedding for original x
        z0 = model(x.unsqueeze(0).to(device)).cpu().numpy()
        # Collect embeddings along augmentation sequence
        embeddings = [z0]
        for aug in augmentation_list:
            if isinstance(x, torch.Tensor):
                # assume augmentation modifies tensor directly
                x_aug = aug(x.unsqueeze(0)).squeeze(0)
            else:
                # fallback: if aug is a function applied to numpy array
                x_np = x.cpu().numpy()
                x_aug_np = aug(x_np)
                x_aug = torch.tensor(x_aug_np, device=device)
            z_aug = model(x_aug.unsqueeze(0)).cpu().numpy()
            embeddings.append(z_aug)
        # Convert to numpy array
        embeddings = np.concatenate(embeddings, axis=0)
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        embed_2d = pca.fit_transform(embeddings)
        plt.figure(figsize=(6,6))
        plt.plot(embed_2d[:,0], embed_2d[:,1], marker='o')
        if label is not None:
            plt.title(f"Trajectory for input {label}")
        else:
            plt.title("Embedding Trajectory")
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

def plot_cosine_histogram(z_pairs, title='Cosine Similarity Histogram', save_path=None):
    """
    z_pairs: list of tuples of two embedding tensors (both normalized)
    Plot histogram of cosine similarities.
    """
    cos_sims = []
    for z1, z2 in z_pairs:
        # ensure tensors are normalized
        z1_norm = z1 / (z1.norm(p=2, dim=1, keepdim=True) + 1e-8)
        z2_norm = z2 / (z2.norm(p=2, dim=1, keepdim=True) + 1e-8)
        sims = (z1_norm * z2_norm).sum(dim=1).cpu().numpy()
        cos_sims.extend(sims)
    plt.figure(figsize=(6,4))
    plt.hist(cos_sims, bins=50, range=(-1,1))
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def evaluate_equivariance(model, dataset, num_samples=100, max_angle_deg=15):
    """
    Measures the degree of equivariance/sensitivity:
    - Samples random inputs and augmentation parameters
    - Computes embeddings before and after augmentation
    - Solves Wahba's problem to find rotation R_a approximating f(a(x))
    - Measures deviation ||f(a(x)) - R_a f(x)||_F
    """
    model.eval()
    all_deviations = []
    whitening_transform = None
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    with torch.no_grad():
        for batch in tqdm(loader):
            if isinstance(batch, dict):
                x = batch['x']
            else:
                x = batch[0]
            for _ in range(max(1, num_samples//len(loader))):
                # For image datasets, sample random small rotation
                angle_deg = np.random.uniform(-max_angle_deg, max_angle_deg)
                rot_mat = random_rotation_matrix(max_angle_deg=angle_deg)
                # For 3D point clouds, apply rotation
                if isinstance(x, torch.Tensor) and x.ndim==3:
                    # e.g. protein data
                    x_aug_np = np.einsum('ij,bnj->bni', rot_mat, x.cpu().numpy())  # batch processing
                    x_aug = torch.tensor(x_aug_np, device=device)
                else:
                    # For images, simulate as identity (or skip)
                    x_aug = x.clone()
                    # Alternatively, for images, could rotate if preferred
                # Compute embeddings
                z_orig = model(x.to(device))
                z_aug = model(x_aug)
                # Normalize
                z_orig_norm = z_orig / (z_orig.norm(p=2, dim=1, keepdim=True) + 1e-8)
                z_aug_norm = z_aug / (z_aug.norm(p=2, dim=1, keepdim=True) + 1e-8)
                # For each sample in batch, compute R minimizing deviation
                # Using Wahba's problem solution
                R_a_batch = estimate_rotation_wahba(z_orig_norm, z_aug_norm)
                # Compute deviation
                devs = torch.norm(z_aug - torch.matmul(z_orig, R_a_batch.t()), dim=1).cpu().numpy()
                all_deviations.extend(devs)
    mean_dev = np.mean(all_deviations)
    median_dev = np.median(all_deviations)
    print(f"Avg deviation in embedding space (f(a(x)) vs R_a f(x)): {mean_dev:.4f}")
    print(f"Median deviation: {median_dev:.4f}")
    return {'mean_deviation': mean_dev, 'median_deviation': median_dev}

def estimate_rotation_wahba(Zx, Za):
    """
    Estimate the rotation matrix R via SVD solving Wahba's problem: minimize ||Za - R Zx||_F
    Inputs:
        Zx, Za: embeddings of shape (batch_size, d)
    Output:
        R: estimated rotation matrix (d, d)
    """
    # Compute matrix product
    M = torch.matmul(Za.t(), Zx)
    U, _, Vt = torch.svd(M)
    R = torch.matmul(U, Vt)
    # Ensure rotation matrix has determinant +1
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = torch.matmul(U, Vt)
    return R

def main_evaluation():
    """
    Example function to run all evaluation metrics on a trained model.
    """
    # Load model (assumed frozen)
    # For demonstration, suppose model is loaded and test_dataset is available
    # To do: replace with actual model and dataset loading as per your codebase
    from model import ResNetEncoder
    model = ResNetEncoder(cfg['model']).to(device)
    model.eval()
    # load checkpoint if needed
    # model.load_state_dict(torch.load('path_to_checkpoint.pth'))

    # Load dataset
    dataset_name = cfg['dataset']['name']
    dataset_obj, _ = get_dataset(cfg)

    # --- Linear probe evaluation ---
    print("Performing linear evaluation...")
    train_dl = DataLoader(dataset_obj, batch_size=512, shuffle=False)
    results = compute_linear_probe(model, train_dl, dataset_name)
    print(f"Linear probe Top-1 accuracy: {results['top1_acc']:.2f}%")

    # --- Trajectory visualization example ---
    # Pick a sample
    dataset_test = dataset_obj
    sample_idx = 0
    sample_item = dataset_test[sample_idx]
    if isinstance(sample_item, dict):
        x_sample = sample_item['x']
        label_sample = sample_item.get('label', '')
    else:
        x_sample = sample_item[0]
        label_sample = ''
    # Define a sequence of small rotations for visualization
    from functools import partial
    def small_rotation(x):
        angle_deg = np.random.uniform(-cfg['training']['augmentations'].get('rotation_small',5))
        rot_mat = random_rotation_matrix(max_angle_deg=angle_deg)
        if isinstance(x, torch.Tensor) and x.ndim==3:
            # apply rotation to point cloud
            x_np = x.cpu().numpy()
            rotated_np = np.einsum('ij,bnj->bni', rot_mat, x_np)
            return torch.tensor(rotated_np, device=device)
        else:
            # for images, skip
            return x
    augmentation_sequence = [partial(small_rotation)]
    visualize_trajectories(model, x_sample, augmentation_sequence, label=label_sample)

    # --- Cosine similarity histogram ---
    # Collect pairs: original and augmented
    z_original = model(x_sample.unsqueeze(0).to(device))
    if hasattr(dataset_obj, 'max_points'):
        # For protein data, apply rotation augmentation
        rot_mat = random_rotation_matrix(max_angle_deg=cfg['training']['augmentations'].get('rotation_small',5))
        if isinstance(x_sample, torch.Tensor) and x_sample.ndim==3:
            x_aug_np = np.einsum('ij,bnj->bni', rot_mat, x_sample.cpu().numpy())
            x_aug = torch.tensor(x_aug_np, device=device)
        else:
            x_aug = x_sample
        z_aug = model(x_aug.unsqueeze(0))
        plot_cosine_histogram([(z_original, z_aug)], title='Pos Pair Cosine Similarity', save_path='cosine_hist.png')
    else:
        # For images, create a small augmentation
        pass

    # Additional metrics (equivariance deviations) can be computed similarly

if __name__ == "__main__":
    main_evaluation()
```

## loss.py

```python
# loss.py
"""
This module implements multiple loss functions central to CARE—namely, the contrastive (InfoNCE) loss,
the equivariance (angle preservation) loss, the uniformity loss, and their combined form.
All losses should be differentiable and compatible with batch processing to enable efficient training.
"""

import torch
import torch.nn.functional as F
import yaml

# Load configuration for hyperparameters
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract relevant hyperparameters with defaults if not specified
TEMPERATURE_NCE = config.get('training', {}).get('temperature_infonce', 0.5)
LAMBDA_EQUIV = config.get('training', {}).get('lambda_equiv', 0.001)
USE_INVARIANCE = config.get('loss', {}).get('invariance', True)
USE_UNIFORMITY = config.get('loss', {}).get('uniformity', True)

def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMPERATURE_NCE) -> torch.Tensor:
    """
    Computes the InfoNCE contrastive loss for a batch of positive pairs.
    Args:
        z1 (torch.Tensor): Embeddings of first views (batch_size, embedding_dim), assumed normalized.
        z2 (torch.Tensor): Embeddings of second views (batch_size, embedding_dim), assumed normalized.
        temperature (float): Temperature scaling hyperparameter.
    Returns:
        torch.Tensor: Scalar loss value.
    """
    batch_size = z1.shape[0]
    # Compute pairwise cosine similarity scaled by temperature
    similarity_matrix = torch.matmul(z1, z2.t()) / temperature  # shape: (batch_size, batch_size)

    # For stability, subtract max
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    # Labels: diagonal elements are the positive pairs
    labels = torch.arange(batch_size, device=z1.device)

    loss = F.cross_entropy(logits, labels)
    return loss

def equivariance_loss(z_x: torch.Tensor, z_a_x: torch.Tensor) -> torch.Tensor:
    """
    Computes the equivariance loss that enforces the angle-preserving condition:
    inner products of augmented embeddings should match those of original embeddings.
    Args:
        z_x (torch.Tensor): Embeddings of original inputs (batch_size, embedding_dim), assumed normalized.
        z_a_x (torch.Tensor): Embeddings of augmented inputs (batch_size, embedding_dim), assumed normalized.
    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Since embeddings are normalized, inner product is cosine similarity
    # Compute inner products
    # Shape: (batch_size, batch_size)
    inner_ori = torch.mm(z_x, z_x.t())  # similarity between original embeddings
    inner_aug = torch.mm(z_a_x, z_a_x.t())  # similarity between augmented embeddings

    # Our loss: mean squared difference between inner products
    loss = F.mse_loss(inner_aug, inner_ori)
    return loss

def uniformity_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Computes the uniformity loss to prevent collapse.
    Encourages embeddings to be uniformly distributed on the sphere.
    Args:
        z (torch.Tensor): Embeddings (batch_size, embedding_dim), assumed normalized.
    Returns:
        torch.Tensor: Scalar loss value.
    """
    similarity_matrix = torch.matmul(z, z.t())  # shape: (batch_size, batch_size)
    # Exclude diagonal to avoid trivial zero differences
    mask = ~torch.eye(z.size(0), dtype=bool, device=z.device)
    sims = similarity_matrix[mask]
    # Compute the mean of exponentiated similarities
    exp_sims = torch.exp(sims)
    mean_exp = torch.mean(exp_sims)
    # Use negative log to encourage spread
    loss = -torch.log(mean_exp + 1e-8)
    return loss

def compute_total_loss(z1: torch.Tensor, z2: torch.Tensor,
                       z_x: torch.Tensor, z_a_x: torch.Tensor,
                       mode: str = 'train') -> torch.Tensor:
    """
    Computes the total CARE loss combining contrastive, invariance, uniformity, and equivariance.
    Parameters:
        z1, z2 (torch.Tensor): Representations for contrastive loss.
        z_x, z_a_x (torch.Tensor): Representations for equivariance loss.
        mode (str): Mode of training; can be 'train' or 'eval', determines loss components.
    Returns:
        torch.Tensor: Scalar total loss.
    """
    # Contrastive (InfoNCE) loss
    loss_infonc = contrastive_loss(z1, z2, temperature=TEMPERATURE_NCE)

    # Equivariance loss (angle preservation)
    loss_equiv = equivariance_loss(z_x, z_a_x)

    # Invariance loss: optional, encourage similarity of original and augmented within batch
    # Implemented as mean squared difference or cosine similarity
    # Here, we use cosine similarity for invariance measure
    loss_invar = None
    if USE_INVARIANCE:
        # The typical invariance loss is encouraging f(a(x)) ≈ f(x)
        # For simplicity, we can define as mean squared difference or negative cosine similarity
        # But in the paper, it is more like an invariance term encouraging f(a(x)) ≈ f(x)
        # For numerical stability, use cosine similarity
        loss_invar = 1 - torch.mean(torch.sum(z_x * z_a_x, dim=1))
        # Alternatively, implement as MSE between normalized vectors if desired
    else:
        loss_invar = torch.tensor(0.0, device=z1.device)

    # Uniformity loss
    loss_uniform = torch.tensor(0.0, device=z1.device)
    if USE_UNIFORMITY:
        # On the set of embeddings z (either z1, z2, or combined)
        # Here, compound all embeddings for simplicity
        z_all = torch.cat([z1, z2], dim=0)
        loss_uniform = uniformity_loss(z_all)
    else:
        loss_uniform = torch.tensor(0.0, device=z1.device)

    # Final total loss
    total = loss_invar + loss_uniform + LAMBDA_EQUIV * loss_equiv

    return total
```

## main.py

```python
## main.py
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

from dataset_loader import get_dataset
from model import ResNetEncoder, DeepSetEncoder
from loss import compute_total_loss
from trainer import CareTrainer
from evaluation import evaluate_linear_probe, visualize_trajectories, plot_cosine_histogram

def main():
    # Load configuration from YAML
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # Set device, seeds for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Prepare Dataset
    train_dataset, test_dataset = get_dataset(cfg)  # from dataset_loader.py
    batch_size = cfg['training'].get('batch_size', 256)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = (torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        if test_dataset is not None else None)

    # 2. Instantiate Model
    model_type = cfg['model'].get('type', 'resnet50')
    embedding_dim = cfg['model'].get('embedding_dim', 128)
    use_projection = cfg['model'].get('projection_head', True)

    if model_type == 'resnet50':
        model = ResNetEncoder({'embedding_dim': embedding_dim, 'projection_head': use_projection})
    elif model_type == 'deepset':
        model = DeepSetEncoder({'embedding_dim': embedding_dim})
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model.to(device)

    # 3. Setup Optimizer and Scheduler
    train_cfg = cfg['training']
    lr = train_cfg.get('learning_rate', 1e-3)
    wd = train_cfg.get('weight_decay', 1e-6)
    opt_name = train_cfg.get('optimizer', 'Adam').lower()
    if opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {opt_name} not supported.")

    # Optional scheduler
    num_epochs = train_cfg.get('epochs', 400)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 4. Setup Loss components
    lambda_eq = train_cfg.get('lambda_equiv', 0.001)
    temperature_infonce = train_cfg.get('temperature_infonce', 0.5)
    temperature_equiv = train_cfg.get('temperature_equiv', 0.1)
    batch_splits = train_cfg.get('batch_splits', 16)

    # 5. Instantiate the trainer object
    trainer = CareTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
        cfg=cfg,
        lambda_eq=lambda_eq,
        temperature_infonce=temperature_infonce,
        temperature_equiv=temperature_equiv,
        batch_splits=batch_splits
    )

    # 6. Run training loops
    for epoch in range(1, num_epochs + 1):
        print(f"\nStarting epoch {epoch}/{num_epochs}")
        trainer.train_one_epoch(epoch)

        # Step scheduler
        trainer.scheduler.step()

        # 7. Evaluation and visualization at intervals
        if epoch % cfg['evaluation'].get('eval_interval', 10) == 0 or epoch == num_epochs:
            print(f"\nEvaluation at epoch {epoch}:\n")
            # 7a. Linear probing
            if trainer.test_loader is not None:
                probe_results = evaluate_linear_probe(trainer.model, trainer.test_loader, cfg['dataset']['name'])
                print(f"Linear probe accuracy: {probe_results['top1_acc']:.2f}%")
            # 7b. Embedding trajectories visualization
            # Select a sample from test set (or train)
            try:
                sample_idx = 0
                sample_data = None
                if hasattr(train_dataset, '__getitem__'):
                    sample_data = train_dataset[0]
                elif hasattr(test_dataset, '__getitem__'):
                    sample_data = test_dataset[0]
                else:
                    sample_data = None
            except:
                sample_data = None
            if sample_data is not None:
                # Assume for images or proteins
                if isinstance(sample_data, dict):
                    x_sample = sample_data['x']
                    label_sample = sample_data.get('label', '')
                else:
                    x_sample = sample_data[0]
                    label_sample = ''
                # Define a small rotation sequence for visualization
                def small_rotation_func(x):
                    angle_deg = np.random.uniform(-cfg['training']['augmentations'].get('rotation_small', 5))
                    rot_mat = None
                    if hasattr(trainer, 'model'):
                        # For image: rotation in 2D
                        # for point clouds: use generated rot_mat
                        # but here, just a placeholder: no rotation for images
                        pass
                    # For protein point clouds:
                    # rotation matrix implementation
                    # For simplicity, pass x unchanged (or implement rotation if data is point cloud)
                    return x
                # Generate trajectory visualization
                visualize_trajectories(
                    trainer.model,
                    x_sample,
                    augmentation_list=[small_rotation_func],
                    label=label_sample,
                    save_path=os.path.join(cfg['save'].get('logs_path', './logs'), 'trajectory_epoch_{}.png'.format(epoch))
                )
            # 7c. Cosine similarity histograms
            # For simplicity, we skip detailed implementation here, assuming routine calls
            # e.g.,
            # plot_cosine_histogram(z_pairs=[(z1, z2)], title='Pose pairs at epoch {}'.format(epoch),
            #                       save_path=os.path.join(cfg['save']['logs_path'], 'cosine_hist_epoch_{}.png'.format(epoch)))

            # 7d. Save checkpoint
            trainer.save_checkpoint(epoch)

    print("Training completed. Final model saved.")

if __name__ == "__main__":
    main()
```

## model.py

```python
### model.py
"""
This module defines neural network encoder classes for different data modalities:
- ResNetEncoder: a standard ResNet-50 based encoder for images.
- DeepSetEncoder: a permutation-invariant encoder for protein point clouds.

Both encoders output normalized features on the unit sphere, suitable for contrastive and equivariance objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def normalize_features(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Normalize the input tensor along specified dimension to have unit norm.
    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to normalize.
    Returns:
        torch.Tensor: Normalized tensor.
    """
    return F.normalize(x, p=2, dim=dim)


class ResNetEncoder(nn.Module):
    """
    ResNet-50 based encoder for images.
    Outputs normalized features before the projection head.
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary with keys:
                - 'embedding_dim' (int): dimension of the embedding output.
                - 'projection_head' (bool): whether to include a projection MLP.
                - 'pretrained' (bool): whether to load ImageNet pretrained weights.
        """
        super(ResNetEncoder, self).__init__()
        self.embedding_dim = config.get('embedding_dim', 128)
        self.projection_head_enabled = config.get('projection_head', True)
        pretrained = config.get('pretrained', False)

        # Load ResNet-50 backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        # Remove the fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # all layers except fc

        # Define projection head if enabled
        if self.projection_head_enabled:
            self.projection_head = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, self.embedding_dim)
            )
        else:
            self.projection_head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x (torch.Tensor): Input images tensor of shape (batch_size, 3, H, W)
        Returns:
            torch.Tensor: Normalized embedding of shape (batch_size, embedding_dim)
        """
        features = self.backbone(x)  # shape: (batch_size, 2048, 1, 1)
        features = torch.flatten(features, start_dim=1)  # shape: (batch_size, 2048)
        projected = self.projection_head(features)  # shape: (batch_size, embedding_dim)
        normalized = normalize_features(projected, dim=1)
        return normalized


class DeepSetEncoder(nn.Module):
    """
    Permutation-invariant encoder for protein point clouds.
    Uses shared point-wise MLPs followed by a pooling operation.
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary with keys:
                - 'n_points' (int): number of points in the point cloud.
                - 'embedding_dim' (int): output feature dimension.
                - 'use_projection' (bool): whether to include a projection head.
        """
        super(DeepSetEncoder, self).__init__()
        self.n_points = config.get('n_points', 1024)
        self.embedding_dim = config.get('embedding_dim', 128)
        self.use_projection = config.get('use_projection', True)

        # Point-wise embedding MLP
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Final set-level embedding
        self.pooling = nn.AdaptiveAvgPool1d(1)  # pooling over points

        # Optional projection head
        if self.use_projection:
            self.projection_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.embedding_dim)
            )
        else:
            self.projection_head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input point clouds of shape (batch_size, n_points, 3)
        Returns:
            torch.Tensor: Normalized embedding of shape (batch_size, embedding_dim)
        """
        # Process each point individually
        batch_size, n_points, _ = x.shape
        x_flat = x.view(-1, 3)  # (batch_size * n_points, 3)
        point_features = self.point_mlp(x_flat)  # (batch_size * n_points, 256)
        point_features = point_features.view(batch_size, n_points, -1)  # (batch, n_points, 256)
        # Pool over points
        pooled = torch.mean(point_features, dim=1)  # (batch_size, 256)
        projected = self.projection_head(pooled)  # (batch_size, embedding_dim)
        normalized = normalize_features(projected, dim=1)
        return normalized
```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import os

from dataset_loader import get_dataset
from model import ResNetEncoder, DeepSetEncoder
from loss import compute_total_loss
from evaluation import evaluate_linear_probe, visualize_embeddings, plot_cosine_histogram

class CareTrainer:
    def __init__(self, config: dict):
        """
        Initialize the CareTrainer with hyperparameters, datasets, model, loss, optimizer, and logging.

        Args:
            config (dict): Configuration dictionary from 'config.yaml'.
        """
        # Set random seeds for reproducibility
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Load dataset
        self.train_dataset, self.test_dataset = get_dataset(config)
        batch_size = config['training'].get('batch_size', 256)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        if self.test_dataset:
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        else:
            self.test_loader = None

        # Instantiate model
        model_type = config['model'].get('type', 'resnet50')
        embedding_dim = config['model'].get('embedding_dim', 128)
        use_projection = config['model'].get('projection_head', True)
        model_hparams = {'embedding_dim': embedding_dim, 'projection_head': use_projection}
        if model_type == 'resnet50':
            self.model = ResNetEncoder(model_hparams)
        elif model_type == 'deepset':
            self.model = DeepSetEncoder(model_hparams)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Move model to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Set optimizer
        train_config = config['training']
        learning_rate = train_config.get('learning_rate', 1e-3)
        weight_decay = train_config.get('weight_decay', 1e-6)
        optimizer_name = train_config.get('optimizer', 'Adam')
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=train_config.get('epochs', 400))

        # Loss hyperparameters
        self.lambda_equiv = train_config.get('lambda_equiv', 0.001)
        self.temperature_infonce = train_config.get('temperature_infonce', 0.5)
        self.temperature_equiv = train_config.get('temperature_equiv', 0.1)
        self.batch_splits = train_config.get('batch_splits', 16)

        # Prepare augmentations
        aug_params = train_config.get('augmentations', {})
        self.aug_image = None
        if hasattr(self, 'train_dataset') and hasattr(self.train_dataset, '__getitem__'):
            # Loop over dataset definition to instantiate appropriate augmentation
            if hasattr(self.train_dataset, 'dataset_name'):
                name = self.train_dataset.dataset_name
            else:
                name = None
            if name in ['CIFAR10', 'CIFAR100', 'STL10', 'ImageNet100']:
                self.aug_image = self._get_image_augmentation(aug_params)
            elif name == 'Proteins':
                # For proteins, define augmentation
                self.aug_image = None  # Will handle in data loader
            else:
                self.aug_image = None

        # Store config for further uses
        self.config = config

        # Logging setup
        self.checkpoint_dir = config['save'].get('checkpoint_path', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logs_dir = config['save'].get('logs_path', './logs')
        os.makedirs(self.logs_dir, exist_ok=True)

    def _get_image_augmentation(self, aug_params):
        # Construct the image augmentation transform chain
        crop_size = aug_params.get('crop_size', 32)
        jitter_std = aug_params.get('jitter_std', 0.1)
        color_jitter_flag = aug_params.get('color_jitter', True)
        blur_flag = aug_params.get('blur', True)
        rotation_deg = aug_params.get('rotation_degrees', 15)

        transform_list = []
        if crop_size:
            transform_list.append(transforms.RandomResizedCrop(crop_size))
        if color_jitter_flag:
            transform_list.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))
        if blur_flag:
            transform_list.append(transforms.GaussianBlur(3))
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomRotation(rotation_deg))
        return transforms.Compose(transform_list)

    def train(self):
        """
        Main training loop over epochs and batches.
        """
        num_epochs = self.config['training'].get('epochs', 400)
        for epoch in range(1, num_epochs + 1):
            print(f"\nStarting epoch {epoch}/{num_epochs}")
            self._train_one_epoch(epoch)
            self.scheduler.step()
            # Optional: evaluate periodically
            if epoch % 10 == 0 or epoch == num_epochs:
                self._evaluate_and_log(epoch)
                self._save_checkpoint(epoch)

    def _train_one_epoch(self, epoch):
        """
        Executes a single epoch training.
        """
        self.model.train()
        total_loss = 0.0
        loss_iter = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Reset gradients
            self.optimizer.zero_grad()

            # Extract original data batch
            if hasattr(batch, 'x'):
                x_input = batch['x']
            elif isinstance(batch, dict) and 'x' in batch:
                x_input = batch['x']
            else:
                x_input = batch

            # Decide input modality
            if isinstance(x_input, list) or len(x_input.shape) == 4:
                # Image dataset
                x_input = x_input.to(self.device)
            elif isinstance(x_input, tuple):
                x_input, _ = x_input
                x_input = x_input.to(self.device)
            else:
                x_input = x_input.to(self.device)

            # Generate augmentation functions for contrastive
            a1_func = self._sample_image_augmentation()
            a2_func = self._sample_image_augmentation()

            # Create augmented views for contrastive
            x1 = self._apply_augmentation(x_input, a1_func)
            x2 = self._apply_augmentation(x_input, a2_func)

            # Forward pass for contrastive views
            z1_full = self.model(x1)
            z2_full = self.model(x2)

            # Normalize embeddings explicitly (if not globally normalized)
            z1_full = nn.functional.normalize(z1_full, p=2, dim=1)
            z2_full = nn.functional.normalize(z2_full, p=2, dim=1)

            # Initialize equivariance loss and other metrics
            equiv_loss = torch.tensor(0.0, device=self.device)
            # Prepare batch splits for equivariance
            n_splits = self.batch_splits
            batch_size = x_input.shape[0]
            split_size = max(batch_size // n_splits, 1)

            # Split batch into chunks
            chunk_indices = np.array_split(np.arange(batch_size), n_splits)

            # For each chunk, sample new augmentations
            z_aug_1_list = []
            z_aug_2_list = []

            for idxs in chunk_indices:
                c_x = x_input[idxs]
                # Sample new augmentations for this chunk
                a_tilde1 = self._sample_image_augmentation()
                a_tilde2 = self._sample_image_augmentation()
                c_x_aug1 = self._apply_augmentation(c_x, a_tilde1)
                c_x_aug2 = self._apply_augmentation(c_x, a_tilde2)

                # Forward pass for augmented chunk
                z_tilde1 = self.model(c_x_aug1)
                z_tilde2 = self.model(c_x_aug2)

                z_tilde1 = nn.functional.normalize(z_tilde1, p=2, dim=1)
                z_tilde2 = nn.functional.normalize(z_tilde2, p=2, dim=1)

                z_aug_1_list.append(z_tilde1)
                z_aug_2_list.append(z_tilde2)

            # Concatenate all augmented chunk embeddings
            z_aug_1 = torch.cat(z_aug_1_list, dim=0)
            z_aug_2 = torch.cat(z_aug_2_list, dim=0)

            # Compute contrastive loss (InfoNCE)
            infonce_loss = compute_total_loss(
                z1_full, z2_full, z1_full, z2_full,
                mode='train')  # mode can be used if needed; here kept general

            # Compute equivariance loss
            equiv_loss = compute_total_loss(
                z_full_for_equiv=z_aug_1,
                z_aug_for_equiv=z_aug_2,
                z_x=z1_full,  # original embeddings
                z_a_x=z2_full,  # augmented embeddings
                mode='train')
            # But as per 'loss.py', the function expects z1, z2, z_x, z_a_x
            # We get the equivariance loss:
            equiv_loss = compute_total_loss(z_aug_1, z_aug_2, z_aug_1, z_aug_2)

            # Calculate total loss
            total_loss_value = infonce_loss + self.lambda_equiv * equiv_loss

            # Backpropagation
            total_loss_value.backward()
            self.optimizer.step()

            total_loss += total_loss_value.item()
            loss_iter += 1
            pbar.set_postfix(loss=total_loss / max(loss_iter,1),
                             infonce=infonce_loss.item(),
                             equiv=self.lambda_equiv * equiv_loss.item())

        avg_loss = total_loss / max(loss_iter,1)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

    def _apply_augmentation(self, x_batch, aug_func):
        """
        Apply the augmentation function to a batch of inputs.
        Supports image tensors (PIL or tensors) or point clouds.
        """
        if self.aug_image:
            # For image data
            x_aug = []
            for img in x_batch:
                img_pil = transforms.functional.to_pil_image(img.cpu())
                img_aug = aug_func(img_pil)
                img_tensor = transforms.functional.to_tensor(img_aug).to(self.device)
                x_aug.append(img_tensor)
            return torch.stack(x_aug, dim=0).to(self.device)
        elif isinstance(x_batch, torch.Tensor) and x_batch.ndim == 3:
            # For point clouds: x_batch shape: (batch_size, N_points, 3)
            # Apply 3D rotation matrix
            batch_size, N_points, _ = x_batch.shape
            points_np = x_batch.cpu().numpy()
            rotated_points = []
            for pts in points_np:
                pts_rotated = aug_func(pts)
                rotated_points.append(pts_rotated)
            return torch.tensor(np.stack(rotated_points), device=self.device, dtype=torch.float32)
        else:
            # Other modalities: fallback or raise
            raise NotImplementedError("Augmentation application not implemented for this data type.")

    def _sample_image_augmentation(self):
        """
        Return an augmentation function sampled with current augmentation parameters.
        """
        return self.aug_image

    def _evaluate_and_log(self, epoch):
        """
        Run evaluation on test data, compute metrics, produce visualizations.
        """
        print(f"Evaluating at epoch {epoch}...")
        if self.test_loader:
            # Compute linear probe accuracy
            linear_acc = evaluate_linear_probe(self.model, self.test_loader, self.device)
            print(f"Linear probe accuracy: {linear_acc:.2f}%")
            # Optional: visualize embeddings trajectories
            # For visualization, pick some samples and visualize trajectories
            # (Implementation omitted)
        # Optional: visualize embedding trajectories and cosine histograms
        # (Implementation omitted)
        # Save models periodically
        self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        """
        Save model parameters checkpoint.
        """
        save_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), save_path)

# Usage example (not to be included in the module file)
if __name__ == "__main__":
    import yaml
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    trainer = CareTrainer(cfg)
    trainer.train()
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\CARE\CARE_repo`
