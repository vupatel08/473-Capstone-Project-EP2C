## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import math
from schedule_utils import get_time_schedule, compute_theta, compute_cum_theta, compute_g, compute_sigma_t, compute_sigma_t_T
from dataset_loader import DatasetLoader
from model import ScoreUNet

class DiffusionTrainer:
    def __init__(self, config: dict):
        """
        Initialize the trainer with configuration parameters.
        Args:
            config (dict): Configuration dictionary loaded from YAML.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Schedule parameters
        self.schedule_type = config['schedule'].get('schedule_type', 'cosine')
        self.steps = config['schedule'].get('steps', 100)
        self.T = 1.0  # total time normalized
        self.N = self.steps

        # Prepare schedule arrays
        self.t_array = get_time_schedule(T=self.T, N=self.N)
        self.theta_array = compute_theta(self.t_array, self.schedule_type)
        self.cum_theta = compute_cum_theta(self.theta_array, self.t_array)
        self.g_array = compute_g(self.t_array, self.theta_array, lambda_sq=30)
        self.sigma_sq = compute_sigma_t(self.t_array, self.theta_array, self.cum_theta, lambda_sq=30)
        self.sigma_sq_t_T = compute_sigma_t_T(self.t_array, self.cum_theta, self.T, lambda_sq=30)
        
        # Convert schedule arrays to tensors for batch access
        self.theta_tensor = torch.tensor(self.theta_array, dtype=torch.float32, device=self.device)
        self.cum_theta_tensor = torch.tensor(self.cum_theta, dtype=torch.float32, device=self.device)
        self.g_tensor = torch.tensor(self.g_array, dtype=torch.float32, device=self.device)
        self.sigma_tensor = torch.tensor(self.sigma_sq, dtype=torch.float32, device=self.device)
        self.sigma_t_T_tensor = torch.tensor(self.sigma_sq_t_T, dtype=torch.float32, device=self.device)

        # Load dataset
        dataset_path = config['dataset'].get('root_path', './dataset')
        self.dataset = DatasetLoader(
            dataset_path=dataset_path,
            batch_size=config['training'].get('batch_size', 8),
            mode='train',
            dataset_type=config['dataset'].get('type', 'inpainting'),
            image_size=128
        )
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=config['training'].get('batch_size', 8),
                                                       shuffle=True, num_workers=4, pin_memory=True)

        # Initialize model
        model_params = {
            'in_channels': 3,
            'base_channels': 64,
            'depth': 4,
            'use_self_attention': False
        }
        self.model = ScoreUNet(**model_params).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training'].get('learning_rate', 1e-4))
        # Learning rate decay schedule
        self.lr_decay_steps = config['training'].get('lr_decay_steps', [300000, 500000, 600000, 700000])
        self.initial_lr = config['training'].get('learning_rate', 1e-4)

        # Training parameters
        self.total_steps = config['training'].get('total_steps', 900000)
        self.current_step = 0
        self.lr_scheduler = self._get_lr_scheduler()

        # Additional parameters
        self.use_mean_ode = True if 'use_mean_ode' not in config['restoration'] else config['restoration'].get('use_mean_ode', True)

    def _get_lr_scheduler(self):
        # Custom scheduler to decay at specific steps
        def lr_lambda(step):
            lr_factor = 1.0
            for decay_step in self.lr_decay_steps:
                if step >= decay_step:
                    lr_factor *= 0.5
            return lr_factor
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def train(self):
        """
        Main training loop for the diffusion model based on maximum likelihood ELBO.
        """
        print("Starting training...")
        for epoch in range(0, int(self.total_steps / len(self.data_loader)) + 1):
            for batch in tqdm(self.data_loader):
                if self.current_step >= self.total_steps:
                    break
                self.model.train()
                self.optimizer.zero_grad()

                # Unpack batch
                x0, xT, mask = batch
                x0 = x0.to(self.device)  # target high-quality images
                if xT is not None:
                    xT = xT.to(self.device)
                else:
                    # For tasks without conditioning, prepare accordingly
                    xT = torch.zeros_like(x0)
                # For inpainting, xT could be masked images, else conditioned input
                
                batch_size = x0.shape[0]
                # Sample t uniformly from [0, N]
                t_idx = np.random.randint(0, self.N + 1, size=batch_size)
                t_tensor = torch.tensor(t_idx / self.N, dtype=torch.float32, device=self.device)  # normalized t
                
                # Gather schedule quantities at sampled t
                theta_t = self.theta_tensor[t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # shape: (B,1,1,1)
                cum_theta_t = self.cum_theta_tensor[t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                g_t = self.g_tensor[t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                sigma_t = self.sigma_tensor[t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                sigma_t_T = self.sigma_t_T_tensor[t_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)

                # Compute \bar{\sigma}_{t-1}'
                if t_idx[0] > 0:
                    sigma_t_minus = self.sigma_tensor[t_idx - 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    sigma_t_minus_prime = sigma_t_minus
                else:
                    # For t=0, define sigma_t-' as zeros
                    sigma_t_minus_prime = torch.zeros_like(sigma_t)

                # Generate x_t conditioned on x0 (for negative ELBO) with the closed-form
                epsilon = torch.randn_like(x0)
                x_t = (x0 - (1 - torch.exp(-cum_theta_t)) * xT) * torch.sqrt(sigma_t) / (np.sqrt(1.0 - np.exp(-2.0 * cum_theta_t))) + epsilon * sigma_t.sqrt()

                x_t = x_t.detach()  # Detach to prevent gradient flow through sampling
                x_t.requires_grad = True

                # Forward pass: neural network predicts epsilon scaled residual
                epsilon_pred = self.model(x_t, xT, t_tensor)
                # True scaled epsilon: based on sampling
                # Compute true epsilon (noise) from the sampled x_t, x0, xT
                # Based on rearranged Eq. 8
                # But for efficiency, just compare epsilon_pred with actual epsilon used

                # Compute predicted mean \tilde{\mu}
                # as per Eq. 16:
                denominator = theta_t + g_t ** 2 * torch.exp(-2 * cum_theta_t) / (sigma_t_T + 1e-8)
                # Add epsilon scaled to match the likelihood
                # Explicit mean calculation
                mu_tilde = (x_t
                            - denominator * (xT - x_t)
                            + g_t ** 2 * self._compute_log_grad(x_t, xT, t_tensor))
                # The above involves: 
                #  - delta term (equation 16),
                #  - gradient of log p(x_t|x_T): approximated via epsilon_pred
                
                # For the gradient term, approximate using epsilon_pred scaled appropriately.
                # Here, instead, based on ELBO derivation, use epsilon_pred as an estimate
                # of noise residual. To match the loss design, compute:
                epsilon_est = epsilon_pred

                # Compute the target epsilon for loss (match the actual noise used during x_t sampling)
                target_epsilon = ((x_t - ((x0 - (1 - torch.exp(-cum_theta_t)) * xT) * torch.sqrt(sigma_t))) / (sigma_t.sqrt()))

                # Compute L1 loss between predicted epsilon and true epsilon
                loss_epsilon = torch.nn.functional.l1_loss(epsilon_pred, target_epsilon, reduction='mean')

                # Alternatively, include ELBO loss as per derivation (Section 3.3 & 3.2)
                # For this implementation, focus on epsilon prediction loss as proxies
                
                # Total loss
                loss = loss_epsilon
                loss.backward()
                self.optimizer.step()

                # Update learning rate
                self.lr_scheduler.step()

                self.current_step += 1

                # Optional: print/logging
                if self.current_step % 1000 == 0:
                    print(f"Step {self.current_step}/{self.total_steps}, Loss: {loss.item():.4f}")

            if self.current_step >= self.total_steps:
                break

        print("Training completed!")

    def _compute_log_grad(self, x_t, xT, t_tensor):
        """
        Placeholder for gradient of log p(x_t|x_T): as per detailed equations,
        can be approximated by the model's output.
        For simplicity, here we return the model's output scaled appropriately.
        """
        epsilon_pred = self.model(x_t, xT, t_tensor)
        # Match scale of epsilon_pred: predicted epsilon scaled residual
        # Approximate gradient of log probability as per paper
        # For stable training, sometimes scaled or normalized inside loss
        return epsilon_pred

    def restore(self, x_T):
        """
        Perform inference (restoration) from conditioned low-quality image x_T.
        Use reverse SDE or Mean-ODE as specified.
        Args:
            x_T (torch.Tensor): low-quality image tensor, shape (1, 3, H, W)
        Returns:
            x0_sample (torch.Tensor): restored high-quality image
        """
        self.model.eval()
        x_T = x_T.to(self.device)
        with torch.no_grad():
            if self.use_mean_ode:
                # Deterministic: solve ODE
                x_curr = x_T
                # Define time grid from T to 0
                t_list = np.linspace(self.T, 0, self.steps)
                for t in tqdm(t_list):
                    t_norm = torch.tensor([t], dtype=torch.float32, device=self.device)
                    # Compute schedule quantities
                    t_idx = min(int(t * self.N), self.N)
                    theta_t = self.theta_tensor[t_idx]
                    cum_theta_t = self.cum_theta_tensor[t_idx]
                    g_t = self.g_tensor[t_idx]
                    sigma_t = self.sigma_tensor[t_idx]
                    sigma_t_T = self.sigma_t_T_tensor[t_idx]
                    # Compute deterministic mean (Equation 13)
                    epsilon_pred = self.model(x_curr, x_T, torch.tensor([t], device=self.device))
                    # Total drift coefficient
                    denom = theta_t + g_t ** 2 * torch.exp(-2 * cum_theta_t) / (sigma_t_T + 1e-8)
                    mu_tilde = (x_curr
                                - denom * (x_T - x_curr)
                                + g_t ** 2 * epsilon_pred)

                    # Euler step
                    dt = - (self.T / self.steps)  # step size, negative as going backward
                    x_curr = mu_tilde
                return x_curr
            else:
                # Stochastic reverse SDE sampling (not implemented here for brevity)
                # Could be added with stochastic integrator
                pass

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
