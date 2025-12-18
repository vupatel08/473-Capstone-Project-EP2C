## utils.py
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility across torch, numpy, and random.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # For reproducibility in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """
    Return the available device: CUDA if available, else CPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def send_to_device(tensor: torch.Tensor) -> torch.Tensor:
    """
    Send a tensor to the appropriate device.
    """
    device = get_device()
    return tensor.to(device)

def estimate_bounds(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """
    Estimate initial bounds D and G from model weights and gradients.
    D: Frobenius norm of initial parameters (or reference point).
    G: Max gradient norm over a batch.
    """
    model = model.to(device)
    model.eval()
    # Initialize D as the norm of model parameters
    param_vector = []
    for param in model.parameters():
        param_vector.append(param.detach().cpu().view(-1))
    param_vec = torch.cat(param_vector)
    D = torch.norm(param_vec).item()

    # Use a single batch for gradient norm estimation
    for batch in dataloader:
        # Assume batch is a tuple: (inputs, labels)
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = compute_loss(outputs, labels)
        loss.backward()
        # Compute the gradient norms
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach())
        grad_norms = [torch.norm(g) for g in grads]
        G = max([gn.item() for gn in grad_norms]) if grads else 0.0
        break  # only one batch needed
    return D, G

def compute_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Placeholder loss function: uses CrossEntropyLoss by default.
    """
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, targets)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str) -> None:
    """
    Save model and optimizer state dictionaries to a file.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str) -> None:
    """
    Load model and optimizer state dictionaries from a file.
    """
    checkpoint = torch.load(filename, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def plot_training_curve(metrics: dict, save_path: str = None) -> None:
    """
    Plot training metrics over epochs or steps.
    metrics: dict with keys as {'loss', 'accuracy', ...} and values as lists or arrays.
    save_path: if provided, save plot to this path.
    """
    plt.figure(figsize=(8, 6))
    for metric_name, metric_values in metrics.items():
        plt.plot(metric_values, label=metric_name)
    plt.xlabel('Epochs/Steps')
    plt.ylabel('Metric')
    plt.title('Training Metrics')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def normalize_input(image: torch.Tensor, mean: list = [0.4914, 0.4822, 0.4465], std: list = [0.2023, 0.1994, 0.2010]) -> torch.Tensor:
    """
    Normalize input image tensor using dataset-specific mean and std.
    """
    device = get_device()
    mean_tensor = torch.tensor(mean, device=device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(-1, 1, 1)
    return (image - mean_tensor) / std_tensor

def estimate_gradient_norm(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Perform a forward and backward pass on a batch to estimate max gradient norm for G.
    """
    model = model.to(device)
    model.eval()
    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = compute_loss(outputs, labels)
        loss.backward()
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach())
        max_grad_norm = max([torch.norm(g).item() for g in grads]) if grads else 0.0
        return max_grad_norm
    return 0.0

def estimate_initial_parameter_distance(model: torch.nn.Module, reference: torch.nn.Module = None) -> float:
    """
    Compute the Euclidean norm of the initial model parameters relative to a reference point.
    If reference is None, use zero vector (or initial weights).
    """
    params = list(model.parameters())
    if reference is None:
        ref_params = [torch.zeros_like(p) for p in params]
    else:
        ref_params = list(reference.parameters())
    distance_vector = []
    for p, rp in zip(params, ref_params):
        distance_vector.append((p - rp).detach().cpu().view(-1))
    total_distance = torch.norm(torch.cat(distance_vector))
    return total_distance.item()

def prepare_device_and_seed(seed: int = 42):
    """
    Utility to set seed and get device once, for consistent setup.
    """
    set_seeds(seed)
    device = get_device()
    return device
