# main.py

import os
import yaml
import torch
import numpy as np
import random

from dataset_loader import DatasetLoader
from model import INRModel
from variational import VariationalDistribution
from hierarchical_patch import HierarchicalPatchModel
from trainer import Trainer
from coding import BayesianCoder
from evaluation import Evaluation

def main():
    # Load configuration from YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set random seeds for reproducibility
    seed = config.get('experiment', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets via DatasetLoader
    dataset_loader = DatasetLoader(config)
    dataset_samples = dataset_loader.load_data()

    # Partition dataset into training and test sets
    # For simplicity, assume dataset_samples is ordered; in practice, shuffle or separate datasets
    # For the primary experiments (e.g., CIFAR-10), assume all samples are test
    test_dataset = dataset_samples

    # Instantiate the INR model
    model_cfg = config.get('model', {})
    inr_model = INRModel(model_cfg).to(device)

    # Initialize learnable positional encodings h_z as a parameter if used
    # For simplicity, initialize h_z as a parameter; in practice, may be learned via a separate module
    positional_dim = model_cfg.get('positional_encoding_dim', 128)
    batch_size = 1  # for inference; adjust per dataset
    h_z = torch.randn(batch_size, positional_dim, device=device, requires_grad=True)

    # Initialize the upsampling CNN φ
    phi_cfg = {
        'input_dim': positional_dim,
        'output_dim': positional_dim
    }
    phi = Evaluation(None, None, None, None, None).__class__.evaluate  # placeholder to load architecture
    # As per instructions, define φ as in Appendix B.1: small CNN
    # For simplicity, here define a minimal CNN stub
    from model import PositionalEncodingCNN
    phi_net = PositionalEncodingCNN(input_dim=positional_dim, output_dim=positional_dim).to(device)

    # Initialize variational posteriors q(w) (for all weights)
    total_params = sum(p.numel() for p in inr_model.parameters())
    init_params = {
        'mu': torch.zeros(total_params, device=device),
        'rho': torch.full((total_params,), -12.0, device=device)  # std ~ 1e-6
    }
    variational_qw = VariationalDistribution(shape=(total_params,), init_params=init_params)

    # Initialize the linear reparameterization matrix A (block-diagonal structure)
    A_matrix = torch.eye(total_params, device=device)
    A_matrix = A_matrix.requires_grad_(True)

    # Initialize hierarchical patch model if using patches (from config)
    patch_model = None
    if 'hierarchical_levels' in model_cfg:
        # Determine number of patches based on dataset and patch size
        # For CIFAR-10: 50k train, but in test, assuming patches are given
        # For this example, create a dummy placeholder; replace with actual patch count
        total_patches = sum([s['coordinates'].shape[0] for s in test_dataset])  # placeholder
        total_patches = 100  # in real logic, set actual number
        patch_model = HierarchicalPatchModel(
            config=dict(),  # fill with appropriate hyperparameters if needed
            total_patches=total_patches,
            seed=seed
        )

    # Instantiate optimizer
    params_to_optimize = [p for p in inr_model.parameters()] + \
                         [variational_qw.mu, variational_qw.rho, A_matrix] + \
                         list(phi_net.parameters())

    optimizer = torch.optim.Adam(params_to_optimize, lr=config['training'].get('learning_rate', 0.0001),
                                 betas=(0.9, 0.999), weight_decay=1e-5)

    # Initialize beta scheduling
    beta = config['training'].get('beta_initial', 0.3)
    beta_min = config['training'].get('beta_min', 0.0)
    beta_max = config['training'].get('beta_max', 1.0)
    tau_C = config['training'].get('beta_adjust_step', 0.5)
    target_bpp = config['training'].get('beta_target_bpp', 0.3)  # bits per pixel/atom
    epsilon = 1e-4  # bet for adjustment

    # For training the prior hyperparameters, they might be updated per Algorithm 1
    # For simplicity, establish prior mu, rho as parameters, or keep fixed here
    prior_mu = torch.zeros(total_params, device=device)
    prior_rho = torch.full((total_params,), -12.0, device=device)

    # Setup a DataLoader for batching datasets if needed, or process all data at once
    # For simplicity, process entire dataset per epoch with custom batching
    # Masked here: assuming 'dataset_samples' is small enough for full-batch
    # Otherwise, implement DataLoader accordingly

    # Main training loop
    total_epochs = config['training'].get('epochs', 550)
    for epoch in range(total_epochs):
        optimizer.zero_grad()

        # --- Step 1: Infer q(w) via gradient steps ---
        # For this implementation, we perform a single gradient step per batch
        # For test purposes, perform a light inferring routine
        # For the actual implementation, run multiple gradient steps on each batch
        # To mimic this, perform a forward pass, compute loss, backprop, optimizer step
        total_loss = 0.0

        # Loop over all samples (or patches if subdivided)
        for sample in test_dataset:
            coords = sample['coordinates'].to(device)      # shape: [N_points, coord_dim]
            values = sample['values'].to(device)             # shape: [N_points, channels]
            # Generate positional encodings
            h_z_current = h_z
            pos_encodings = phi_net(h_z_current, coords.unsqueeze(0)).squeeze(0)  # shape: [N_points, dim]

            # Sample latents: for simplicity, one sample; in practice, multiple MC samples
            # Draw small epsilon for h_w
            epsilon_hw = torch.randn(total_params, device=device)
            sigma_hw = torch.exp(0.5 * variational_qw.rho)
            h_w_sample = variational_qw.mu + epsilon_hw * sigma_hw

            # Compute weights via linear reparameterization
            w_sample = torch.matmul(A_matrix, h_w_sample.unsqueeze(-1)).squeeze(-1)  # shape: [total_params]

            # Set model parameters accordingly
            # Map w_sample to model parameters
            def set_model_weights(model, weights_vector):
                offset = 0
                for param in model.parameters():
                    param_numel = param.numel()
                    param.data.copy_(weights_vector[offset:offset+param_numel].view_as(param))
                    offset += param_numel

            set_model_weights(inr_model, w_sample)

            # Forward pass
            preds = inr_model.forward(coords, h_z_current, data=None, params=None)
            # Compute distortion
            dist = torch.nn.functional.mse_loss(preds, values, reduction='mean')
            # Compute KL divergence for q(w) vs p(w)
            kl_qp = variational_qw.kl_divergence({'mu': prior_mu, 'rho': prior_rho})
            # Loss (negative ELBO scaled by beta)
            loss = beta * kl_qp + dist
            total_loss += loss

        # Backpropagate and update all parameters
        total_loss.backward()
        optimizer.step()

        # --- Step 2: Update prior parameters (Equation 7) ---
        # Calculate empirical mean and variance over the variational posteriors (here, approximate)
        with torch.no_grad():
            mu_post = variational_qw.mu.data
            rho_post = variational_qw.rho.data
            prior_mu_new = torch.mean(mu_post)
            prior_sigma_new = torch.mean((mu_post - prior_mu_new) ** 2 + torch.exp(rho_post))
            # Update prior mu and rho
            prior_mu.copy_(prior_mu_new)
            prior_rho.copy_(torch.log(prior_sigma_new.clamp(min=1e-8)))

        # --- Step 3: Calculate estimated bits and adjust beta ---
        with torch.no_grad():
            kl_estimate = variational_qw.kl_divergence({'mu': prior_mu, 'rho': prior_rho}).item()
        if kl_estimate > target_bpp + epsilon:
            beta = min(beta * (1 + tau_C), beta_max)
        elif kl_estimate < target_bpp - epsilon:
            beta = max(beta / (1 + tau_C), beta_min)

        # Optional: print training info
        print(f"Epoch {epoch+1}/{total_epochs} | Loss: {total_loss.item():.4f} | KL_est: {kl_estimate:.4f} | beta: {beta:.4f}")

    # After training:
    # Save final model, A, h_z, and variational params if needed
    torch.save({
        'inr_state_dict': inr_model.state_dict(),
        'A': A_matrix.detach().cpu(),
        'h_z': h_z.detach().cpu(),
        'mu': variational_qw.mu.detach().cpu(),
        'rho': variational_qw.rho.detach().cpu(),
        'prior_mu': prior_mu.detach().cpu(),
        'prior_rho': prior_rho.detach().cpu()
    }, 'trained_model.pth')

    # --- Inference and encoding on test data ---
    # For each test sample:
    posterior_samples = []
    reconstructed_data = []

    for sample in test_dataset:
        coords = sample['coordinates'].to(device)
        values = sample['values'].to(device)

        # Infer q(w) by same process as above but with multiple MC samples if desired
        # or with fixed point estimate for simplicity
        with torch.no_grad():
            # For consistency, do multiple MC samples (e.g., 5), but here just 1 for simplicity
            epsilon_hw = torch.randn(total_params, device=device)
            sigma_hw = torch.exp(0.5 * variational_qw.rho)
            h_w_sample = variational_qw.mu + epsilon_hw * sigma_hw

            # Compute weights
            weights = torch.matmul(A_matrix, h_w_sample.unsqueeze(-1)).squeeze(-1)

            # Store posterior parameters
            posterior_samples.append({
                'mu': variational_qw.mu.cpu(),
                'rho': variational_qw.rho.cpu()
            })

            # --- Bayesian coding: encode q(w) sample ---
            # Apply permutation strategy (random permutation)
            permutation = torch.randperm(total_params)
            # Use BayesianCoder to encode weights
            coder = BayesianCoder()
            # Encode weights
            bits_used = coder.encode_weights(weights, 
                                             {'mu': variational_qw.mu.cpu(), 'rho': variational_qw.rho.cpu()}, 
                                             {'mu': prior_mu.cpu(), 'rho': prior_rho.cpu()}, 
                                             permutation=permutation)
            # Save bits (or store in a list)
            # For this example, just store bits used
            # Save the permutation for decoding
            # Save the sample weight (or its bits) for later decoding
            # Store in a list for later use
            # For brevity, omit bits storage, assume access later

            # --- Decoding: retrieve weights from bits (simulate) ---
            decoded_weights = coder.decode_weights(None, 
                                                 {'mu': variational_qw.mu.cpu(), 'rho': variational_qw.rho.cpu()}, 
                                                 {'mu': prior_mu.cpu(), 'rho': prior_rho.cpu()}, 
                                                 permutation=permutation, shape=(total_params,))
            # Set model weights
            def set_model_weights(model, weights_vector):
                offset = 0
                for param in model.parameters():
                    param_numel = param.numel()
                    param.data.copy_(weights_vector[offset:offset+param_numel].view_as(param))
                    offset += param_numel
            set_model_weights(inr_model, decoded_weights)

        # Reconstruct data
        with torch.no_grad():
            preds = inr_model.forward(coords, h_z)
            reconstructed_data.append(preds.cpu())

    # --- Evaluation ---
    evaluator = Evaluation(test_dataset, reconstructed_data, posterior_samples,
                           prior_mu, prior_rho, modality='image', output_path='results')
    eval_metrics = evaluator.evaluate()
    print('RD Metrics:', eval_metrics)

if __name__ == '__main__':
    main()
