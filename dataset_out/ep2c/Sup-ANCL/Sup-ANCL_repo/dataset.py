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
