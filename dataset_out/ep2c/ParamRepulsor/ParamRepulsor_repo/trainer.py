## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict
from dataset_loader import DatasetLoader
from model import NeuralNetwork
from pair_sampler import PairSampler
from evaluation import Evaluation

def train(
    model: nn.Module,
    dataset: Tuple[np.ndarray, np.ndarray],
    pair_sampler: PairSampler,
    config: Dict
):
    """
    Trains the ParamRepulsor model with hard negative mining and contrastive loss.

    Args:
        model (nn.Module): Neural network projector.
        dataset (Tuple[np.ndarray, np.ndarray]): Dataset tuple (X, labels).
        pair_sampler (PairSampler): PairSampler with neighbor info.
        config (Dict): Configuration dictionary with training hyperparameters.
    """
    # Extract training parameters from config
    training_cfg = config.get('training', {})
    optimization_cfg = config.get('optimization', {})
    loss_weights_cfg = config.get('loss_weights', {})
    hyperparams_cfg = config.get('hyperparameters', {})
    dataset_cfg = config.get('dataset', {})
    
    lr = training_cfg.get('learning_rate', 0.001)
    batch_size = training_cfg.get('batch_size', 1024)
    num_epochs = training_cfg.get('epochs', 100)
    report_interval = hyperparams_cfg.get('report_interval', 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    w_NB = loss_weights_cfg.get('weight_NN', 1.0)
    w_MN = loss_weights_cfg.get('weight_MN', 0.5)
    w_FP = loss_weights_cfg.get('weight_FP', 0.2)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=optimization_cfg.get('betas', [0.9, 0.999]))
    
    # Unpack dataset
    X_data, labels = dataset
    N = X_data.shape[0]
    feature_dim = X_data.shape[1]
    
    # Set dataset data for pair sampler
    pair_sampler.set_data(X_data)
    
    # Calculate number of batches per epoch
    num_batches = int(np.ceil(N / batch_size))
    
    # For evaluation
    evaluator = Evaluation(model, dataset)
    
    # Prepare index array for shuffling
    all_indices = np.arange(N)
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        permuted_indices = np.random.permutation(all_indices)
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{num_epochs}")
        
        for batch_idx in pbar:
            # Sample batch of points
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N)
            batch_indices = permuted_indices[start_idx:end_idx]
            b = len(batch_indices)
            
            # Generate pairs for the batch
            idx_i, idx_j, pair_types = pair_sampler.generate_pairs(batch_indices, X_data)
            # idx_i, idx_j: index tensors for pairs
            # pair_types: tensor indicating pair type (0=NN, 1=MN, 2=FP)

            # Fetch raw data for pair points
            x_batch = torch.from_numpy(X_data[batch_indices]).to(device).float()
            x_NN = torch.from_numpy(X_data[idx_j[:pair_types==0].cpu()]]).to(device).float()
            x_MN = torch.from_numpy(X_data[idx_j[pair_types==1].cpu()]]).to(device).float()
            x_FP = torch.from_numpy(X_data[idx_j[pair_types==2].cpu()]]).to(device).float()

            # Compute embeddings
            y_batch = model.forward(x_batch)                       # shape: (b, 2)
            # Embeddings for pair points
            y_NN = model.forward(x_NN)    # shape: (num_NN_pairs, 2)
            y_MN = model.forward(x_MN)    # shape: (num_MN_pairs, 2)
            y_FP = model.forward(x_FP)    # shape: (num_FP_pairs, 2)

            # Function to compute pairwise squared distances
            def pairwise_distances(y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
                return torch.sum((y1.unsqueeze(1) - y2.unsqueeze(0))**2, dim=2)

            # Compute distances for pairs
            # For NN pairs
            d2_NN = pairwise_distances(y_batch[idx_i[pair_types==0]], y_NN)
            # For MN pairs
            d2_MN = pairwise_distances(y_batch[idx_i[pair_types==1]], y_MN)
            # For FP pairs
            d2_FP = pairwise_distances(y_batch[idx_i[pair_types==2]], y_FP)

            # Compute similarity functions (Sec. 4, Appendix D)
            # q_NN and q_MN: similar form
            def q_nn_or_mn(d2):
                return torch.exp(- (d2 + 10) / (d2 + 10 + 1e-8))
            # q_FP
            def q_fp(d2):
                return torch.exp(- d2 / (d2 + 1 + 1e-8))
            
            q_NN_vals = q_nn_or_mn(d2_NN)
            q_MN_vals = q_nn_or_mn(d2_MN)
            q_FP_vals = q_fp(d2_FP)

            # For loss calculation: following the theoretical form,
            # attraction for NN pairs, repulsion for FP and MN
            # Loss per pair, with weighting
            # Use the equations from Appendix D
            # To avoid numerical issues, add small epsilon where needed
            epsilon = 1e-8

            # Compute pairwise "loss" contributions
            # attraction for NN pairs
            loss_NN = - torch.log(torch.clamp(q_NN_vals, min=epsilon))
            # repulsive for MN and FP (maximize distances)
            loss_MN = - torch.log(1 - torch.clamp(q_MN_vals, min=epsilon))
            loss_FP = - torch.log(1 - torch.clamp(q_FP_vals, min=epsilon))
            
            # Sum weighted contributions
            total_loss = (
                w_NB * torch.sum(loss_NN) +
                w_MN * torch.sum(loss_MN) +
                w_FP * torch.sum(loss_FP)
            ) / b  # Normalize by batch size

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            # Update progress bar
            pbar.set_postfix(loss=total_loss.item())

        # Report metrics at intervals
        if epoch % report_interval == 0:
            # Compute metrics
            # e.g., NN accuracy, triplet accuracy, or distance correlation
            local_acc = evaluator.compute_local_accuracy()
            triplet_ratio = evaluator.compute_triplet_preservation()
            dist_corr = evaluator.compute_distance_correlation()

            print(f"Epoch {epoch}: Loss={epoch_loss/num_batches:.4f}, "
                  f"NN_Acc={local_acc:.4f}, Triplet={triplet_ratio:.4f}, DistCorr={dist_corr:.4f}")

    # Save model after training
    torch.save(model.state_dict(), config.get('save_model_path', './models/paramreprulsor.pth'))


# Usage example (assuming all modules are imported correctly and config is loaded):
# if __name__ == '__main__':
#     import yaml
#     with open('config.yaml', 'r') as f:
#         config = yaml.safe_load(f)
#     dataset_loader = DatasetLoader(config['dataset'])
#     data = dataset_loader.get_normalized_data()
#     pair_sampler = PairSampler(
#         knn_indices=dataset_loader.get_knn()[0],
#         knn_distances=dataset_loader.get_knn()[1],
#         n_points=dataset_loader.N,
#         config=config.get('pair_sampling', {})
#     )
#     model = NeuralNetwork(
#         input_dim=data.shape[1],
#         output_dim=2,
#         hidden_layers=config['model']['hidden_layers'],
#         neurons_per_layer=config['model']['neurons_per_layer'],
#         activation=config['model'].get('activation', 'relu')
#     ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#     train(model, (dataset_loader.data, dataset_loader.labels), pair_sampler, config)
