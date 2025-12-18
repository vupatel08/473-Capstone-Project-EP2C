## datasets.py
import os
from torchvision import datasets, transforms

def load_data(dataset_name, data_dir, batch_size, is_train=True):
    """
    Load dataset and prepare DataLoader with appropriate transformations
    based on dataset type and training/evaluation split.

    Args:
        dataset_name (str): 'CIFAR100' or 'ImageNet'
        data_dir (str): Root directory of dataset
        batch_size (int): Batch size for DataLoader
        is_train (bool): Whether to load training or validation set

    Returns:
        DataLoader: DataLoader object for dataset
    """
    if dataset_name.lower() == 'cifar100':
        # CIFAR-100 normalization constants (standard practice)
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        if is_train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        dataset_obj = datasets.CIFAR100(
            root=data_dir,
            train=is_train,
            download=False,
            transform=transform
        )

    elif dataset_name.lower() == 'imagenet':
        # ImageNet normalization constants
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        dataset_obj = datasets.ImageNet(
            root=data_dir,
            split='train' if is_train else 'val',
            transform=transform
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not supported.")

    dataloader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4,
        pin_memory=True
    )
    return dataloader
