## dataset_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import math
import random

# Set a fixed seed for reproducibility across the module
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class SyntheticEnergyDataset(Dataset):
    """
    A dataset class for synthetic energy-based distributions.
    Generates samples based on provided energy functions during on-the-fly sampling.
    """
    def __init__(self, energy_fn, num_samples, dim, seed=SEED):
        """
        :param energy_fn: Callable that takes a tensor of shape (batch_size, dim) and returns energies.
        :param num_samples: Total number of samples to generate.
        :param dim: Dimensionality of the samples.
        :param seed: Random seed for reproducibility.
        """
        self.energy_fn = energy_fn
        self.num_samples = num_samples
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        # Precompute samples for test set purposes
        self.samples = self._generate_samples(num_samples)

    def _generate_samples(self, n):
        # Generate samples directly from the unnormalized distribution
        # using MCMC or rejection sampling can be costly in high dimensions.
        # For low-dimensional tasks, we can use rejection sampling; for high dims, sample from standard normal and weight accordingly.
        # Here, for simplicity, use rejection sampling for low-dimensional energy functions,
        # and for high-dimensional, sample from a proposal (e.g., Gaussian) with weights.
        # To keep it straightforward, we sample standard Gaussian and accept with probability proportional to R(x).
        samples = []
        batch_size = n
        attempt = 0
        max_attempts = n * 10
        while len(samples) < n and attempt < max_attempts:
            attempt += 1
            x = self.rng.normal(size=(batch_size, self.dim))
            x_tensor = torch.from_numpy(x).float()
            energies = self.energy_fn(x_tensor)  # energies = E(x)
            # Convert energies to acceptance probabilities
            # in unnormalized distribution: R(x) = exp(-E(x))
            # To avoid underflow, cap energies
            log_accept_prob = -energies
            accept_probs = np.exp(log_accept_prob.numpy())
            uniform_randoms = self.rng.uniform(size=batch_size)
            accepted_mask = accept_probs > uniform_randoms
            accepted_x = x[accepted_mask]
            samples.extend(accepted_x)
            if len(samples) >= n:
                break
        samples = np.array(samples)[:n]
        return torch.from_numpy(samples).float()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return individual sample
        return self.samples[idx]


def load_real_dataset(dataset_name='MNIST', split='train', download=True, root='./data'):
    """
    Load and preprocess the real dataset (e.g., MNIST)
    :param dataset_name: String, name of the dataset
    :param split: 'train' or 'test'
    :param download: Whether to download if not present
    :param root: Directory to store datasets
    :return: DataLoader providing batches of images
    """
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),  # Keep pixel scale [0, 1]
        ])
        dataset = datasets.MNIST(root=root, train=(split=='train'), download=download, transform=transform)
        return DataLoader(dataset, batch_size=300, shuffle=True, seed=SEED)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")


def get_energy_function(task_name='manywell', input_dim=32):
    """
    Returns the energy function corresponding to the specified task.
    For known synthetic tasks, define inline functions.
    For others, raise error or extend as needed.
    :param task_name: String indicating the dataset/energy type
    :param input_dim: Dimensionality of the task
    :return: Callable energy function
    """
    if task_name.lower() == 'manywell':
        def energy_fn(x):
            """
            Sum of 16 identical 2D double-well potentials
            The total input x is batch_size x input_dim (here input_dim=32)
            Reshape to batch_size x 16 x 2 for component-wise potential
            """
            # Reshape
            batch_size = x.shape[0]
            x_reshaped = x.view(batch_size, 16, 2)
            x1 = x_reshaped[:, :, 0]
            x2 = x_reshaped[:, :, 1]
            # Potential: mu(x1,x2) = exp(-x1^4 + 6x1^2 + 0.5x1 - 0.5x2^2)
            pot = np.exp(-x1.numpy()**4 + 6*x1.numpy()**2 + 0.5*x1.numpy() - 0.5*x2.numpy()**2)
            energies = -np.sum(np.log(pot + 1e-9), axis=1)  # Unnormalized, negative log density
            return torch.from_numpy(energies).float()
        return energy_fn

    elif task_name.lower() == 'funnel':
        def energy_fn(x):
            """
            10D funnel: x[0] ~ N(0,3^2), others conditioned on x[0]
            As a synthetic function, approximate the joint energy
            """
            x_np = x.numpy()
            x0 = x_np[:, 0]
            rest = x_np[:, 1:]
            # y = (x0^2)/2 to simulate the conditional energy
            energy_x0 = 0.5 * (x0/np.sqrt(3.0))**2
            energy_rest = 0.5 * np.sum(rest**2 * np.exp(x0[:, None]), axis=1)
            energies = energy_x0 + energy_rest
            return torch.from_numpy(energies).float()
        return energy_fn

    elif task_name.lower() == '25gmm':
        # 2D Gaussian mixture with 25 modes arranged on grid
        centers = []
        grid_points = [-10, -5, 0, 5, 10]
        for cx in grid_points:
            for cy in grid_points:
                centers.append([cx, cy])
        centers = np.array(centers)

        def energy_fn(x):
            """
            Compute negative log of mixture density:
            R(x) ~ sum of 25 Gaussians centered at grid points with var=0.3^2
            """
            x_np = x.numpy()
            log_probs = []
            for c in centers:
                diff = x_np - c
                norm_sq = np.sum(diff ** 2, axis=1)
                log_prob = -0.5 * norm_sq / 0.3**2
                log_probs.append(log_prob)
            stacked = np.vstack(log_probs)  # shape: 25 x batch_size
            log_sum_exp = scipy.special.logsumexp(stacked, axis=0) + np.log(1/25)
            energies = -log_sum_exp
            return torch.from_numpy(energies).float()

        import scipy.special
        return energy_fn

    else:
        raise ValueError(f"Unknown task name: {task_name}")


def get_dataset_loader(config):
    """
    Instantiate dataset loader based on configuration
    :param config: dict loaded from YAML with dataset configs
    :return: DataLoader for real datasets or synthetic samples generator
    """
    dataset_type = config.get('type', 'synthetic_energy')
    dataset_name = config.get('dataset_name', 'manywell')
    input_dim = config.get('input_dim', 2)
    # For simplicity, only handle synthetic and MNIST
    if dataset_type == 'synthetic_energy':
        energy_fn = get_energy_function(dataset_name, input_dim)
        # Generate full dataset samples (for evaluation) or set up an on-demand sampler
        # For now, create a Dataset object with pre-sampled data for testing
        dataset = SyntheticEnergyDataset(energy_fn=energy_fn, num_samples=10000, dim=input_dim)
        return dataset
    elif dataset_type == 'real_dataset':
        # For real data like MNIST
        loader = load_real_dataset(dataset_name=dataset_name, split='train')
        return loader
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported.")

# Helper function to set seed globally for reproducibility
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Example usage (This wouldn't run as part of module, just for testing purposes)
if __name__ == "__main__":
    # Load dataset according to config (example)
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset_obj = get_dataset_loader(config['dataset'])
    if isinstance(dataset_obj, DataLoader):
        for batch in dataset_obj:
            imgs, labels = batch
            print(f"Batch shape: {imgs.shape}")
            break
    else:
        print(f"Synthetic dataset with {len(dataset_obj)} samples loaded.")
