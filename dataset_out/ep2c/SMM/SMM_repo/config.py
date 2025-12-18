# config.py

from typing import List, Tuple, Dict, Union

class Config:
    # Dataset configuration: defines dataset specifics for loading and preprocessing
    dataset: Dict[str, Union[str, List[int], int]] = {
        'name': 'CIFAR10',                     # Dataset name (e.g., CIFAR10, SVHN, etc.)
        'input_size': [32, 32],                # Size images are resized to before feeding into the model
        'train_split': 50000,                  # Total training set size
        'test_split': 10000,                   # Total testing set size
        'batch_size': 128                      # Batch size for training and evaluation
    }

    # Model configuration: specifies backbone architecture and pretrained setting
    model: Dict[str, Union[str, bool]] = {
        'name': 'ResNet50',                     # Model backbone: 'ResNet50' or 'ViT-B32'
        'pretrained': True                      # Whether to load ImageNet-pretrained weights
    }

    # Mask generator architecture hyperparameters
    mask_generator: Dict[str, Union[int, str]] = {
        'architecture_depth': 5,                # Number of convolutional + pooling layers (e.g., 5)
        'kernel_size': 3,                       # Kernel size for conv layers (3x3)
        'filters': 64,                          # Number of filters per conv layer
        'pooling_layers': 2,                     # Number of MaxPooling layers (e.g., 2)
        'output_ratio': '1/8'                   # Downsampling ratio of mask relative to input size
    }

    # Training hyperparameters for LR schedules, epochs, and regularization
    training: Dict[str, Union[str, float, List[int]]] = {
        'optimizer': 'Adam',                    # Optimizer type: 'Adam'
        'learning_rate': 0.01,                   # Initial LR for mask generator parameters (phi)
        'lr_decay_epochs': [100],               # Epochs at which LR decays
        'lr_decay_factor': 0.1,                  # Multiplicative decay factor
        'epochs': 200,                          # Total training epochs
        'pattern_lr': 0.001,                     # LR for the pattern delta
        'pattern_lr_decay_epochs': [100],       # LR decay epochs for pattern lr
        'pattern_lr_decay_factor': 0.1,         # LR decay factor for delta
        'weight_decay': 1e-4,                   # L2 regularization coefficient
        'pattern_init': 'zeros'                  # Pattern initialization method ('zeros' or 'random')
    }

    # Sampling and interpolation configuration
    sampling: Dict[str, Union[List[int], int]] = {
        'image_resize': [32, 32],                # Resize images to given size before processing
        'patch_size': 8                         # Patch size for patch-wise upsampling (e.g., 8)
    }

    # Evaluation settings and visualization toggle
    evaluation: Dict[str, Union[str, bool]] = {
        'metrics': 'accuracy',                  # Metric for evaluation
        'visualize': True                       # Whether to generate visualizations post-training
    }

    # Reproducibility seed
    reproducibility: Dict[str, int] = {
        'seed': 42                              # Random seed for reproducibility
    }

    # Additional method to load all configurations as dictionary (optional)
    def as_dict(self) -> Dict:
        return {
            'dataset': self.dataset,
            'model': self.model,
            'mask_generator': self.mask_generator,
            'training': self.training,
            'sampling': self.sampling,
            'evaluation': self.evaluation,
            'reproducibility': self.reproducibility
        }
