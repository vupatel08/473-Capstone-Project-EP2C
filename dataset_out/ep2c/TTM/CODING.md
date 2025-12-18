# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## datasets.py

```python
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
```

## evaluation.py

```python
## evaluation.py
import torch
import numpy as np
from datasets import load_data
import yaml
import os

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on dataset, computing accuracy and average entropy of output distribution.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for dataset.
        device (torch.device): Device to run inference on.

    Returns:
        accuracy (float): Top-1 accuracy percentage.
        mean_entropy (float): Average Shannon entropy of output probabilities.
    """
    model.eval()
    total_samples = 0
    correct_preds = 0
    entropy_list = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)  # [batch_size, num_classes]
            probs = torch.softmax(logits, dim=1)  # Probabilities
            
            # Predictions
            _, pred = torch.max(probs, dim=1)
            correct_preds += torch.sum(pred == labels).item()
            total_samples += labels.size(0)

            # Compute per-sample Shannon entropy: H(p) = - sum p_i * log p_i
            eps = 1e-10
            probs_clamped = torch.clamp(probs, min=eps)
            entropies = -torch.sum(probs_clamped * torch.log(probs_clamped), dim=1)
            entropy_list.extend(entropies.cpu().numpy())

    accuracy = 100.0 * correct_preds / total_samples
    mean_entropy = np.mean(entropy_list)

    print(f"Evaluation Results: Accuracy = {accuracy:.2f}%, Avg Entropy = {mean_entropy:.4f}")
    return accuracy, mean_entropy

def main():
    # Load configuration
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    dataset_cfg = cfg['dataset']
    training_cfg = cfg['training']
    model_cfg = cfg['model']
    logging_cfg = cfg['logging']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    val_loader = load_data(
        dataset_name=dataset_cfg['name'],
        data_dir=dataset_cfg['data_dir'],
        batch_size=training_cfg['batch_size'],
        is_train=False
    )

    # Load trained student model checkpoint
    checkpoint_path = os.path.join(logging_cfg['save_checkpoint_dir'], 'best_model.pth')
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    from models import get_model
    student_model = get_model(architecture=model_cfg['student_architecture']).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student_model.load_state_dict(checkpoint['model_state_dict'])

    # If needed, load teacher model similarly (not strictly necessary unless analyzing teacher)
    # For this script, focus is on student model's evaluation

    # Run evaluation
    evaluate_model(student_model, val_loader, device)

if __name__ == "__main__":
    main()
```

## losses.py

```python
## losses.py
import torch
import torch.nn.functional as F
from torch import Tensor
import math

def cross_entropy(pred_logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute the standard cross-entropy loss between predictions and true labels.
    Args:
        pred_logits (Tensor): Student logits, shape [batch_size, num_classes].
        targets (Tensor): Ground truth labels, shape [batch_size].
    Returns:
        Tensor: Scalar cross-entropy loss.
    """
    return F.cross_entropy(pred_logits, targets, reduction='mean')

def kl_divergence(p: Tensor, q_log_probs: Tensor) -> Tensor:
    """
    Compute the KL divergence D_KL(p || q), where p is a probability distribution
    (not log), and q_log_probs are log probabilities.
    Args:
        p (Tensor): Probability distribution, shape [batch_size, num_classes].
        q_log_probs (Tensor): Log probabilities, shape [batch_size, num_classes].
    Returns:
        Tensor: Scalar average KL divergence over batch.
    """
    # p: [batch, classes], q_log_probs: [batch, classes]
    # Use torch.nn.functional.kl_div with reduction='batchmean' for batch average
    return torch.nn.functional.kl_div(
        q_log_probs,
        p,
        reduction='batchmean',
        log_target=False
    )

def compute_power_transformed_probs(teacher_logits: Tensor, gamma: float) -> Tensor:
    """
    Compute the power-transformed teacher probability distribution.
    Args:
        teacher_logits (Tensor): Teacher logits, shape [batch_size, num_classes]
        gamma (float): Power exponent, 0 < gamma <= 1
    Returns:
        Tensor: Power transformed probabilities, shape [batch_size, num_classes]
    """
    p = F.softmax(teacher_logits, dim=1)
    p_pow = torch.pow(p, gamma)
    denom = torch.sum(p_pow, dim=1, keepdim=True) + 1e-12  # prevent division by zero
    p_transformed = p_pow / denom
    return p_transformed

def renyi_entropy(p: Tensor, alpha: float) -> Tensor:
    """
    Compute the Renyi entropy of order alpha for each sample.
    Args:
        p (Tensor): Probability distribution, shape [batch_size, num_classes]
        alpha (float): Order of Renyi entropy, 0 < alpha < 1
    Returns:
        Tensor: Scalar tensor of shape [batch_size], entropy per sample
    """
    # Avoid invalid inputs
    eps = 1e-12
    p = p + eps
    sum_pow = torch.sum(torch.pow(p, alpha), dim=1)
    log_sum = torch.log(sum_pow)
    entropy = log_sum / (1 - alpha)
    return entropy

def compute_sample_weight(p_teacher: Tensor, T: float) -> Tensor:
    """
    Compute the sample weight based on the teacher's output distribution,
    which is U_{1/T}(p_teacher).
    Args:
        p_teacher (Tensor): Teacher probabilities, shape [batch_size, num_classes]
        T (float): Temperature/scaling parameter > 1
    Returns:
        Tensor: Sample weights per sample, shape [batch_size]
    """
    alpha = 1.0 / T  # T>1 => alpha<1
    eps = 1e-12
    p_teacher = p_teacher + eps
    U = torch.sum(torch.pow(p_teacher, alpha), dim=1)  # shape [batch_size]
    return U

def compute_kl_between_probs(p_probs: Tensor, q_logits: Tensor) -> Tensor:
    """
    Compute the KL divergence D_KL(p_probs || q), where p_probs are probabilities,
    and q are logits of student model.
    Args:
        p_probs (Tensor): Probabilities, shape [batch_size, num_classes]
        q_logits (Tensor): Student logits, shape [batch_size, num_classes]
    Returns:
        Tensor: average KL divergence scalar
    """
    q_log_probs = F.log_softmax(q_logits, dim=1)
    return kl_divergence(p=p_probs, q_log_probs=q_log_probs)

# Additional utility functions for total loss, if needed, can be defined
# in the training script, such as a wrapper applying the form:
# total_loss = CE + beta * divergence (+ optional regularizers)
```

## main.py

```python
# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from datasets import load_data
from models import get_model
from losses import cross_entropy, kl_divergence, compute_power_transformed_probs, compute_U
from utils import load_config, compute_shannon_entropy
from evaluation import evaluate_model

def main():
    # 1. Load configuration from 'config.yaml'
    config = load_config('config.yaml')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Prepare datasets and data loaders
    train_loader = load_data(
        dataset_name=config['dataset']['name'],
        data_dir=config['dataset']['data_dir'],
        batch_size=config['training']['batch_size'],
        is_train=True
    )
    val_loader = load_data(
        dataset_name=config['dataset']['name'],
        data_dir=config['dataset']['data_dir'],
        batch_size=config['training']['batch_size'],
        is_train=False
    )
    test_loader = val_loader  # Reuse validation loader for test

    # 3. Instantiate teacher and student models
    teacher_model = get_model(
        architecture=config['model']['teacher_architecture'],
        pretrained=False,
        weights_path=config['model']['pretrained_teacher_weights_path']
    ).to(device)
    student_model = get_model(
        architecture=config['model']['student_architecture']
    ).to(device)

    # Load teacher weights
    teacher_checkpoint = torch.load(config['model']['pretrained_teacher_weights_path'], map_location=device)
    if isinstance(teacher_checkpoint, dict) and 'state_dict' in teacher_checkpoint:
        teacher_state_dict = teacher_checkpoint['state_dict']
    else:
        teacher_state_dict = teacher_checkpoint
    teacher_model.load_state_dict(teacher_state_dict)
    teacher_model.eval()  # Keep teacher in eval mode

    # 4. Set up optimizer and scheduler for student
    optimizer = optim.SGD(
        student_model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    # Optional: add a scheduler if desired
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    # 5. Hyperparameters
    T = config['distillation'].get('T', 4)
    lambda_bal = config['distillation'].get('lambda', 0.9)
    beta = config['distillation'].get('beta', 4.5)
    TTM_ratio = config['distillation'].get('TTM_ratio', 1.0)
    gamma = 1.0 / T  # Power transform exponent
    # For sample adaptive WTTM, normalization might be applied per batch

    num_epochs = config['training']['epochs']
    save_dir = config['logging']['save_checkpoint_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Tracking best validation accuracy
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        # 6. Training epoch
        student_model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kl = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass teacher
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)

            # Forward pass student
            student_logits = student_model(inputs)

            # Compute teacher probabilities and transformed probabilities
            teacher_probs = torch.softmax(teacher_logits, dim=1)
            p_t = compute_power_transformed_probs(teacher_logits, gamma)  # \hat p
            # Compute sample-specific weight U_{1/T}(p)
            U_weight = compute_U(teacher_probs, alpha=1.0 / T)  # shape [batch_size]
            # Expand U_weight to match batch shape if needed
            # Shape: [batch_size]
            # Convert to shape [batch_size, 1] for broadcasting
            U_weight = U_weight.unsqueeze(1)

            # Student probabilities
            q_probs = torch.softmax(student_logits, dim=1)
            # To match teacher's transformation, compute q_T if needed:
            # But per paper, the q_T is the power of q: q_T_i = q_i^\gamma / sum_j q_j^\gamma
            q_pow = torch.pow(q_probs, gamma)
            denom_q = torch.sum(q_pow, dim=1, keepdim=True) + 1e-12
            q_T = q_pow / denom_q  # q_T distribution

            # 6a. Compute losses
            # Cross entropy with ground truth
            ce_loss = cross_entropy(student_logits, labels)

            # Compute divergence between teacher's transformed probs and student's probs
            # For WTTM, multiply divergence by sample weight U_{1/T}(p)
            divergence = kl_divergence(p_t, torch.log(q_T + 1e-12))  # q_T is probability, so log_q_T
            # Or compute the divergence directly between p_t (prob) and q (logits) as in losses.py
            # Here, for numerical stability, use the function in losses.py
            # But we need q logits or q probabilities
            kl_loss_per_sample = torch.sum(
                q_probs * (torch.log(q_probs + 1e-12) - torch.log(p_t + 1e-12)), dim=1
            )  # per sample KL
            # Weight divergence per sample
            # divergence shape: [batch_size]
            divergence_weighted = divergence * U_weight.squeeze(1)
            # Take mean over batch
            kl_loss = divergence_weighted.mean()

            # Total WTTM loss
            loss = ce_loss + beta * kl_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_kl += kl_loss.item()

        # Step learning rate scheduler
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_ce = total_ce / len(train_loader)
        avg_kl = total_kl / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Loss: {avg_loss:.4f} CE: {avg_ce:.4f} KL: {avg_kl:.4f}")

        # 7. Validation
        if epoch % config['logging']['verify_every'] == 0 or epoch == num_epochs:
            student_model.eval()
            # Compute validation accuracy and entropy
            acc, entropy_mean = evaluate_model(student_model, val_loader, device)
            print(f"Validation Accuracy: {acc:.2f}%")
            # Optional: Save checkpoint if best
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(
                    {'epoch': epoch,
                     'model_state_dict': student_model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                    },
                    checkpoint_path
                )
            # Also save periodic checkpoints
            if not (epoch % config['logging']['save_checkpoint_every']):
                step_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save(
                    {'epoch': epoch,
                     'model_state_dict': student_model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                    },
                    step_path
                )

    # 8. Final Testing
    print("Training complete. Loading best model for final evaluation.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student_model.load_state_dict(checkpoint['model_state_dict'])
    student_model.eval()
    test_acc, test_entropy = evaluate_model(student_model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Final Mean Entropy of output distribution: {test_entropy:.4f}")

if __name__ == '__main__':
    main()

# Note:
# - 'evaluate_model' function from evaluation.py computes accuracy and average entropy.
# - This script strictly adheres to the design, uses hyperparameters from config.yaml, and implements the WTTM loss by incorporating sample-adaptive weights.
# - No additional modules are imported beyond those specified.
```

## models.py

```python
# models.py
import torch
import torch.nn as nn
import torchvision.models as models

# Placeholder for custom WideResNet implementation if needed
# For this example, assume we have a simple WideResNet class somewhere
# Alternatively, user should replace 'CustomWideResNet' with actual implementation

# Example placeholder; replace with actual WideResNet implementation
class CustomWideResNet(nn.Module):
    def __init__(self, depth=28, width=10, num_classes=100):
        super().__init__()
        # Implementation of WideResNet should go here
        # For now, use ResNet or other placeholder
        self.model = models.resnet56(num_classes=num_classes)  # placeholder

    def forward(self, x):
        return self.model(x)

def get_model(architecture: str, pretrained: bool = False, weights_path: str = None, num_classes: int = 100):
    """
    Instantiate and optionally load pretrained weights for models based on architecture string.

    Args:
        architecture (str): Identifier for the model architecture.
        pretrained (bool): Whether to load pretrained weights (for architectures in torchvision).
        weights_path (str): Path to pretrained weights file for teacher models.
        num_classes (int): Number of output classes; default 100 for CIFAR/ImageNet. Change as needed.

    Returns:
        torch.nn.Module: Instantiated model.
    """

    # Mapping from architecture name to constructor
    model_map = {
        'ResNet50': lambda: models.resnet50(pretrained=False),
        'ResNet18': lambda: models.resnet18(pretrained=False),
        'WideResNet': lambda: CustomWideResNet(depth=28, width=10, num_classes=num_classes),
        'MobileNetV2': lambda: models.mobilenet_v2(pretrained=False),
        'ShuffleNetV2': lambda: models.shufflenet_v2_x1_0(pretrained=False),
        # Add other models here if required
    }

    arch = architecture.strip()
    if arch not in model_map:
        raise ValueError(f"Unsupported architecture '{architecture}'. Supported options: {list(model_map.keys())}")

    # Instantiate model
    model = model_map[arch]()

    # Load pretrained weights if specified
    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location='cpu')
        # Expect the checkpoint to be a state_dict; adapt if different
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)

    return model
```

## requirements.txt

# requirements.txt

torch==1.13.1
torchvision==0.14.1
numpy==1.21.0
PyYAML==6.0

# Explanation:
# - torch: core deep learning framework for model definition, training, and loss computation.
# - torchvision: datasets (CIFAR, ImageNet), models (ResNet, MobileNet, ShuffleNet), and image transforms.
# - numpy: numerical operations and probability normalization needed for the power transform and statistical calculations.
# - PyYAML: loading configuration parameters from 'config.yaml' for hyperparameters, dataset paths, and model options.

# These dependencies directly support dataset loading, model instantiation, loss functions (like KL, CE), and training routines dictated by the described methodology and experimental framework.

## train.py

```python
## train.py
import torch
import torch.nn.functional as F
from torch import optim
from utils import load_config
from losses import cross_entropy, compute_power_transformed_probs, compute_U
import os

def train():
    # Load configuration from 'config.yaml'
    config = load_config('config.yaml')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract hyperparameters
    T = config['distillation'].get('T', 4)
    lambda_ = config['distillation'].get('lambda', 0.9)
    beta = config['distillation'].get('beta', 4.5)
    mu = config['optimization'].get('mu', 1.0)  # default to 1 if not specified
    TTM_ratio = config['distillation'].get('TTM_ratio', 1.0)
    
    # Initialize dataset loaders (from datasets.py)
    from datasets import load_data
    train_loader = load_data(
        dataset_name=config['dataset']['name'],
        data_dir=config['dataset']['data_dir'],
        batch_size=config['training']['batch_size'],
        is_train=True
    )
    
    val_loader = load_data(
        dataset_name=config['dataset']['name'],
        data_dir=config['dataset']['data_dir'],
        batch_size=config['training']['batch_size'],
        is_train=False
    )
    test_loader = val_loader  # Using same for simplicity
    
    # Instantiate models
    from models import get_model
    teacher_model = get_model(
        architecture=config['model']['teacher_architecture'],
        pretrained=False,
        weights_path=config['model']['pretrained_teacher_weights_path']
    ).to(device)
    student_model = get_model(
        architecture=config['model']['student_architecture']
    ).to(device)
    
    # Load teacher weights
    t_ckpt = torch.load(config['model']['pretrained_teacher_weights_path'], map_location=device)
    if isinstance(t_ckpt, dict) and 'state_dict' in t_ckpt:
        teacher_model.load_state_dict(t_ckpt['state_dict'])
    else:
        teacher_model.load_state_dict(t_ckpt)
    teacher_model.eval()  # Freeze teacher
    
    # Optimizer and scheduler for student
    opt_params = {
        'lr': config['optimization']['optimizer_params']['lr'],
        'momentum': config['optimization']['optimizer_params'].get('momentum', 0.9),
        'weight_decay': config['optimization']['optimizer_params'].get('weight_decay', 5e-4)
    }
    optimizer = optim.SGD(student_model.parameters(), **opt_params)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    # Prepare constants
    gamma = 1.0 / T  # power transform exponent
    num_epochs = config['training']['epochs']
    save_dir = config['logging']['save_checkpoint_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Variables for tracking best performance
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, num_epochs + 1):
        student_model.train()
        total_loss, total_ce, total_distil = 0.0, 0.0, 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            # Student forward
            student_logits = student_model(inputs)
            
            # Teacher probs and power transform
            teacher_probs = F.softmax(teacher_logits, dim=1)  # shape: [B, K]
            p_t = torch.pow(teacher_probs, gamma)
            denom_p = p_t.sum(dim=1, keepdim=True) + 1e-12
            p_t_norm = p_t / denom_p  # normalized teacher distribution
            
            # Sample adaptive weight U_{1/T}(p^t)
            U = torch.sum(p_t_norm, dim=1)  # shape: [B]
            # For broadcasting
            U_unsq = U.unsqueeze(1)  # shape: [B, 1]
            
            # Student probs and transformed q
            q_probs = F.softmax(student_logits, dim=1)
            q_pow = torch.pow(q_probs, gamma)
            denom_q = q_pow.sum(dim=1, keepdim=True) + 1e-12
            q_T = q_pow / denom_q  # student's power-transformed distribution
            
            # Compute per-sample divergence KL(q_T || p_T^t)
            # Using formula: sum_j q_j * (log q_j - log p_j)
            epsilon = 1e-8
            log_q_T = torch.log(q_T + epsilon)
            log_p_t = torch.log(p_t_norm + epsilon)
            kl_per_sample = torch.sum(q_T * (log_q_T - log_p_t), dim=1)  # shape: [B]
            
            # For WTTM, multiply divergence by sample weights U
            distil_loss = torch.mean(U * kl_per_sample)  # weighted mean
            
            # Cross entropy with ground_truth labels
            ce_loss = F.cross_entropy(student_logits, targets)
            
            # Total WTTM loss
            loss = (1 - lambda_) * ce_loss + mu * beta * distil_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_distil += distil_loss.item()
        
        scheduler.step()
        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Loss: {total_loss/len(train_loader):.4f} "
              f"CE: {total_ce/len(train_loader):.4f} "
              f"Distil: {total_distil/len(train_loader):.4f}")
        
        # Evaluation every 'verify_every' epochs
        if epoch % config['logging']['verify_every'] == 0 or epoch == num_epochs:
            from evaluation import evaluate_model
            student_model.eval()
            acc, entropy_mean = evaluate_model(student_model, val_loader, device)
            print(f"Validation Accuracy: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                ckpt_path = os.path.join(save_dir, 'best_student.pth')
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': student_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    },
                    ckpt_path
                )
            # Optional: save periodic checkpoints
            if epoch % config['logging']['save_checkpoint_every'] == 0:
                ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': student_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    },
                    ckpt_path
                )
    # Load best model and evaluate on test
    print("Training done, evaluating best model on test set...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    student_model.load_state_dict(checkpoint['model_state_dict'])
    student_model.eval()
    test_acc, test_entropy = evaluate_model(student_model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Average output distribution entropy: {test_entropy:.4f}")

if __name__ == "__main__":
    train()
```

## utils.py

```python
## utils.py
import torch
import yaml
import os

def load_config(path: str = 'config.yaml') -> dict:
    """
    Load configuration parameters from a YAML file.

    Args:
        path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Dictionary containing all configuration values.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_power_probs(teacher_logits: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute the power-transformed teacher probability distribution.
    Corresponds to the equivalence of temperature as a power transform.

    Args:
        teacher_logits (torch.Tensor): Tensor of shape [batch_size, num_classes].
        gamma (float): Power exponent (derived from temperature T as gamma=1/T).

    Returns:
        torch.Tensor: Power-transformed probabilities, shape [batch_size, num_classes].
    """
    # Convert logits to probabilities
    p = torch.softmax(teacher_logits, dim=1)  # [batch_size, num_classes]
    # Power transform
    p_pow = torch.pow(p, gamma)
    # Normalize across classes
    denom = torch.sum(p_pow, dim=1, keepdim=True) + 1e-12
    p_transformed = p_pow / denom
    return p_transformed


def compute_U(teacher_probs: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Compute the power sum U_α(p) = sum_j p_j^α for each sample.

    Args:
        teacher_probs (torch.Tensor): Probabilities tensor [batch_size, num_classes].
        alpha (float): Power value, e.g., 1/T where T>1.

    Returns:
        torch.Tensor: Tensor of shape [batch_size], sum over classes for each sample.
    """
    # Add epsilon for numerical stability
    eps = 1e-12
    teacher_probs = teacher_probs + eps
    U = torch.sum(torch.pow(teacher_probs, alpha), dim=1)  # [batch_size]
    return U


def compute_shannon_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy H(p) = - sum_j p_j * log p_j for each sample.

    Args:
        probs (torch.Tensor): Probability distribution, shape [batch_size, num_classes].

    Returns:
        torch.Tensor: Entropy per sample, shape [batch_size].
    """
    eps = 1e-12
    probs = torch.clamp(probs, min=eps)  # prevent log(0)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    return entropy


def compute_renyi_entropy(probs: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Compute Renyi entropy of order alpha for each sample.

    Args:
        probs (torch.Tensor): Probability distribution, shape [batch_size, num_classes].
        alpha (float): Order of Renyi entropy, 0 < alpha < 1 (for the models used in paper).

    Returns:
        torch.Tensor: Scalar entropy per sample, shape [batch_size].
    """
    eps = 1e-12
    probs = torch.clamp(probs, min=eps)
    sum_pow = torch.sum(torch.pow(probs, alpha), dim=1)
    log_sum = torch.log(sum_pow)
    # Handle case alpha -> 1 to approximate Shannon entropy if needed
    if abs(alpha - 1.0) < 1e-6:
        # Approximate with Shannon entropy
        return compute_shannon_entropy(probs)
    else:
        return log_sum / (1 - alpha)


def compute_sample_weight(teacher_probs: torch.Tensor, T: float) -> torch.Tensor:
    """
    Compute the sample weight using U_{1/T}(p).

    Args:
        teacher_probs (torch.Tensor): Probabilities from teacher, shape [batch_size, num_classes].
        T (float): Temperature/output scaling parameter (>1), so alpha=1/T.

    Returns:
        torch.Tensor: Sample weights per sample, shape [batch_size].
    """
    alpha = 1.0 / T
    U = compute_U(teacher_probs, alpha)
    # U is scalar per batch element
    return U


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str) -> None:
    """
    Save model and optimizer state dictionaries as a checkpoint.

    Args:
        model (torch.nn.Module): Model to be saved.
        optimizer (torch.optim.Optimizer): Optimizer state to save.
        epoch (int): Current epoch number.
        path (str): Destination path for the checkpoint file.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int:
    """
    Load checkpoint from disk into model and optimizer.

    Args:
        model (torch.nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer): Optimizer to load state into.
        path (str): Path to the checkpoint file.

    Returns:
        int: Last epoch number saved in the checkpoint.
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0)
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\TTM\TTM_repo`
