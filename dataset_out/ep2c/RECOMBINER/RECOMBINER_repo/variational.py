## variational.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VariationalDistribution:
    """
    Variational Gaussian posterior q(w) over INR weights, supporting reparameterization
    via the learned linear transformation A and optional hierarchical Bayesian variables.
    """
    def __init__(self, shape, init_params=None):
        """
        Initialize variational parameters μ, ρ, and linear reparameterization matrix A.
        Optional hierarchical variables can be added if needed.
        
        Args:
            shape (list or tuple): shape of the weights tensor w (e.g., total number of parameters).
            init_params (dict): dictionary with optional initial 'mu' and 'rho' tensors.
        """
        # Set shape
        self.shape = shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize mu and rho (log-variance) for the mean-field Gaussian q(h_w)
        if init_params is not None:
            self.mu = nn.Parameter(init_params.get('mu', torch.zeros(shape, device=device)))
            self.rho = nn.Parameter(init_params.get('rho', torch.full(shape, -12.0, device=device)))  # variance ~ 10^{-6}
        else:
            self.mu = nn.Parameter(torch.zeros(shape, device=device))
            self.rho = nn.Parameter(torch.full(shape, -12.0, device=device))
        
        # Initialize A matrix for linear reparameterization (shape: [shape, shape])
        # For computational efficiency, store as a parameter with shape similar to the weight vector
        # and initialize as identity for simplicity
        self.A = nn.Parameter(torch.eye(shape[0], device=device))  # shape: [shape[0], shape[0]]
        
        # Variational parameters for hierarchical variables can be added here if needed
        # For simplicity, we consider only the local layer-wise q(w)

    def sample(self, num_samples=1):
        """
        Sample weights w from q(w) = N(mu, diag(sigma^2)), with w = h_w A.
        Samples h_w are drawn, then transformed via A.
        
        Args:
            num_samples (int): number of Monte Carlo samples.
        
        Returns:
            Tensor: sampled weights, shape [num_samples, *shape]
        """
        epsilon = torch.randn((num_samples,) + self.shape, device=self.mu.device)
        sigma = torch.exp(0.5 * self.rho)  # std: exp(0.5 * rho)
        h_w_samples = self.mu.unsqueeze(0) + epsilon * sigma.unsqueeze(0)
        # Apply linear reparameterization: w = h_w * A
        # Since shape of h_w: [num_samples, *shape], shape of A: [total_params, total_params]
        # Reshape h_w to 2D for matrix multiplication if necessary
        # Here, treat the last dimension as vector w
        # For each sample, perform: w = h_w * A
        # flatten last dims temporarily
        samples_list = []
        for i in range(num_samples):
            h_w_flat = h_w_samples[i].view(-1, 1)  # shape: [prod(shape), 1]
            w_sample = torch.matmul(self.A, h_w_flat).view(self.shape)
            samples_list.append(w_sample)
        return torch.stack(samples_list, dim=0)  # shape: [num_samples] + shape

    def kl_divergence(self, prior):
        """
        Compute KL divergence D_KL(q(w) || p(w)) between two Gaussians.
        Both are assumed diagonal covariances; the prior can be specified.
        For hierarchical, this can be extended.

        Args:
            prior (dict): a dict with keys 'mu' and 'rho' for the prior distribution.
                          they are tensors of shape matching self.mu and self.rho.

        Returns:
            float: KL divergence (scalar tensor)
        """
        # Variational q: mu_q, sigma_q
        mu_q = self.mu
        sigma_q = torch.exp(0.5 * self.rho)
        # Prior p: mu_p, sigma_p
        mu_p = prior.get('mu', torch.zeros_like(mu_q))
        sigma_p = torch.exp(0.5 * prior.get('rho', torch.zeros_like(mu_q)))

        # Compute KL element-wise for diagonal Gaussians
        # D_KL = 0.5 * [ (sigma_q^2 / sigma_p^2) + ((mu_p - mu_q)^2 / sigma_p^2) - 1 + log(sigma_p^2 / sigma_q^2) ]
        term1 = (sigma_q ** 2) / (sigma_p ** 2)
        term2 = ((mu_p - mu_q) ** 2) / (sigma_p ** 2)
        kl = 0.5 * torch.sum(term1 + term2 - 1 + torch.log((sigma_p ** 2).clamp_min(1e-8)/(sigma_q ** 2).clamp_min(1e-8)))
        return kl

    def update_params(self, new_params):
        """
        Update variational parameters μ and ρ.
        Args:
            new_params (dict): dictionary with 'mu' and 'rho' tensors
        """
        if 'mu' in new_params:
            self.mu.data = new_params['mu']
        if 'rho' in new_params:
            self.rho.data = new_params['rho']

    def get_weight(self):
        """
        Get the mean weight vector w = μ + noise via reparameterization.
        For usage in the functional API or explicit inference.
        """
        sigma = torch.exp(0.5 * self.rho)
        epsilon = torch.randn(self.shape, device=self.mu.device)
        h_w = self.mu + epsilon * sigma
        # Apply linear reparameterization
        w = torch.matmul(self.A, h_w.view(-1, 1)).view(self.shape)
        return w

    def set_A(self, A_new):
        """
        Set the learned linear transform A explicitly (fixed during inference).
        Args:
            A_new (Tensor): new A matrix.
        """
        self.A.data = A_new

    def get_A(self):
        """
        Get the current A matrix.
        """
        return self.A
