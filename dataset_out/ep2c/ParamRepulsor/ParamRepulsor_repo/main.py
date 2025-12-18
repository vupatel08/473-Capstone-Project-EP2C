## main.py
import yaml
import numpy as np
import torch
import os
from dataset_loader import DatasetLoader
from pair_sampler import PairSampler
from model import NeuralNetwork
from trainer import train
from evaluation import Evaluation

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set random seeds for reproducibility
    seed = config.get('misc', {}).get('random_seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess dataset
    dataset_cfg = config['dataset']
    dataset_loader = DatasetLoader(dataset_cfg)
    dataset_loader.load_dataset()
    dataset_loader._preprocess()
    # Use PCA reduction if specified, defaults already handled inside DatasetLoader
    dataset_loader._apply_pca()
    data = dataset_loader.get_pca_data()  # normalized, PCA-reduced features
    N = dataset_loader.N

    # Build neighbor graph (kNN)
    knn_indices, knn_distances = dataset_loader.get_knn()

    # Generate mid-near pairs (MN) as per Sec. 4
    dataset_loader._generate_mn_pairs()  # updates dataset_loader.mn_pairs

    # Generate FP indices (far negatives)
    dataset_loader._generate_fp_indices()

    # Initialize PairSampler with neighbor info
    pair_cfg = config.get('pair_sampling', {})
    pair_sampler = PairSampler(
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        n_points=N,
        config=pair_cfg,
        seed=seed
    )
    pair_sampler.set_data(dataset_loader.get_pca_data())

    # Initialize model
    model_cfg = config['model']
    input_dim = dataset_loader.get_pca_data().shape[1]
    model = NeuralNetwork(
        input_dim=input_dim,
        output_dim=2,  # for visualization
        hidden_layers=model_cfg['hidden_layers'],
        neurons_per_layer=model_cfg['neurons_per_layer'],
        activation=model_cfg.get('activation', 'relu')
    ).to(device)

    # Setup optimizer
    training_cfg = config['training']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg.get('learning_rate', 0.001),
        betas=tuple(config.get('optimization', {}).get('betas', [0.9, 0.999]))
    )

    # Training hyperparameters
    num_epochs = config['hyperparameters'].get('num_epochs', 100)
    batch_size = training_cfg.get('batch_size', 1024)
    report_interval = config['hyperparameters'].get('report_interval', 10)

    # Initialize evaluation object for metrics
    evaluator = Evaluation(model, (dataset_loader.get_pca_data(), dataset_loader.labels))

    # Prepare data indices for batching
    all_indices = np.arange(N)
    n_batches = int(np.ceil(N / batch_size))
    print(f"Total samples: {N}, Batches per epoch: {n_batches}")

    for epoch in range(1, num_epochs + 1):
        np.random.seed(seed + epoch)  # optional: different shuffle each epoch
        permuted_indices = np.random.permutation(all_indices)
        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, N)
            batch_indices = permuted_indices[start:end]
            b = len(batch_indices)

            # Generate pair indices and types for current batch
            idx_i, idx_j, pair_types = pair_sampler.generate_pairs(batch_indices, dataset_loader.get_pca_data())

            # Fetch data
            X_all = dataset_loader.get_pca_data()
            x_batch_np = X_all[batch_indices]
            x_batch = torch.from_numpy(x_batch_np).float().to(device)
            # For the pairs, extract respective features
            # Subset features for pairs
            idx_i_cpu = idx_i.cpu().numpy()
            idx_j_cpu = idx_j.cpu().numpy()

            # Get pair data
            x_NN = torch.from_numpy(X_all[idx_j_cpu[pair_types.cpu().numpy() == 0]]).float().to(device)
            x_MN = torch.from_numpy(X_all[idx_j_cpu[pair_types.cpu().numpy() == 1]]).float().to(device)
            x_FP = torch.from_numpy(X_all[idx_j_cpu[pair_types.cpu().numpy() == 2]]).float().to(device)

            # Compute embeddings
            y_batch = model.forward(x_batch)  # shape (b, 2)
            y_NN = model.forward(x_NN)
            y_MN = model.forward(x_MN)
            y_FP = model.forward(x_FP)

            # Compute pairwise distances in embedding space
            def pairwise_dist(y1, y2):
                return torch.sum((y1.unsqueeze(1) - y2.unsqueeze(0))**2, dim=2)  # shape (len(y1), len(y2))
            # For pairs
            # Note: Need to align pairs with batch points
            # Create mappings:
            # For NN: get embeddings of anchor batch points with their NN
            # For MN and FP: same
            # Extract indices
            # Gather the embedding of anchor points (batch points)
            batch_embs = y_batch
            # NN pairs
            nn_indices = idx_i[pair_types == 0]
            nn_embs = y_NN
            # For simplicity, compute all pairwise in the batch using indexing
            # Alternative: create tensors for pairwise computation for each pair type
            # But since the number of pairs is large, do in small batch.

            # Get number of pairs per type
            nn_mask = (pair_types == 0)
            mn_mask = (pair_types == 1)
            fp_mask = (pair_types == 2)

            # Distances
            if nn_mask.sum() > 0:
                y_anchor_nn = y_batch[nn_indices]
                d2_nn = pairwise_dist(y_anchor_nn, y_NN)
            else:
                d2_nn = torch.tensor([]).to(device)
            if mn_mask.sum() > 0:
                y_anchor_mn = y_batch[idx_i[mn_mask]]
                d2_mn = pairwise_dist(y_anchor_mn, y_MN)
            else:
                d2_mn = torch.tensor([]).to(device)
            if fp_mask.sum() > 0:
                y_anchor_fp = y_batch[idx_i[fp_mask]]
                d2_fp = pairwise_dist(y_anchor_fp, y_FP)
            else:
                d2_fp = torch.tensor([]).to(device)

            # Compute similarity functions
            epsilon = 1e-8

            def q_nn_or_mn(d2):
                return torch.exp(- (d2 + 10) / (d2 + 10 + epsilon))
            def q_fp(d2):
                return torch.exp(- d2 / (d2 + 1 + epsilon))

            q_nn_vals = q_nn_or_mn(d2_nn) if d2_nn.numel() > 0 else torch.tensor([]).to(device)
            q_mn_vals = q_nn_or_mn(d2_mn) if d2_mn.numel() > 0 else torch.tensor([]).to(device)
            q_fp_vals = q_fp(d2_fp) if d2_fp.numel() > 0 else torch.tensor([]).to(device)

            # Compute loss contributions
            loss_NN = -torch.log(torch.clamp(q_nn_vals, min=epsilon)) if d2_nn.numel() > 0 else torch.tensor(0.0).to(device)
            loss_MN = -torch.log(1 - torch.clamp(q_mn_vals, min=epsilon)) if d2_mn.numel() > 0 else torch.tensor(0.0).to(device)
            loss_FP = -torch.log(1 - torch.clamp(q_fp_vals, min=epsilon)) if d2_fp.numel() > 0 else torch.tensor(0.0).to(device)

            # Weights from config
            w_NB = config['loss_weights'].get('weight_NN', 1.0)
            w_MN = config['loss_weights'].get('weight_MN', 0.5)
            w_FP = config['loss_weights'].get('weight_FP', 0.2)

            total_loss = (
                w_NB * torch.sum(loss_NN) +
                w_MN * torch.sum(loss_MN) +
                w_FP * torch.sum(loss_FP)
            ) / b

            # Gradient update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        # Logging and evaluation
        if epoch % report_interval == 0 or epoch == num_epochs:
            # Evaluate local NN accuracy
            acc_nn = evaluator.compute_nn_accuracy(embeddings=None, labels=dataset_loader.labels, k=10)
            # Triplet preservation
            # Generate some triplets (e.g., random triplets from labels or high-D data)
            # For simplicity, we skip triplet sampling here, or implement if needed
            trip_metric = evaluator.compute_triplet_preservation(
                embeddings=evaluator.get_embeddings(), 
                high_dim_data=dataset_loader.data, 
                triplets=[]  # Placeholder: generate triplets if needed
            )
            # Global distance correlation
            dist_corr = evaluator.compute_distance_correlation(
                embeddings=evaluator.get_embeddings(),
                high_dim_data=dataset_loader.data,
                cluster_centroids=None  # or precompute centroids if labels known
            )
            print(f"Epoch {epoch}/{num_epochs}: loss={epoch_loss / n_batches:.4f}, "
                  f"NN_Acc={acc_nn:.4f}, Triplet={trip_metric:.4f}, DistCorr={dist_corr:.4f}")

    # Save the trained model
    save_path = config.get('save_model_path', './models/paramreprulsor.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    # Final embedding for entire dataset (or test set)
    final_embeddings = evaluator.get_embeddings()
    # Save embeddings or visualize
    # e.g., save to file
    np.save('final_embeddings.npy', final_embeddings)
    print("Training and embedding complete. Results saved.")

if __name__ == '__main__':
    main()
