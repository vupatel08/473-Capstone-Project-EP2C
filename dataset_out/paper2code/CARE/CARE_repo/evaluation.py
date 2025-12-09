## evaluation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loader import get_dataset

import yaml
import os

# Load configuration for evaluation parameters
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_linear_probe(model, test_loader, dataset_name, is_protein=False):
    """
    Performs linear evaluation:
    - Extract features for train/test data
    - Train logistic regression on train features
    - Evaluate accuracy on test features
    Returns:
        dict with 'top1_acc' and 'top5_acc' if applicable
    """
    model.eval()
    features_list = []
    labels_list = []
    
    # Collect features for train set
    train_dataset, test_dataset = get_dataset(cfg)
    # For consistency, use train/test splits as per dataset
    # Here, assume features can be extracted via model
    # WARNING: For full pipeline, tend to use a separate DataLoader if needed
    for mode, dataloader in [('train', DataLoader(train_dataset, batch_size=512, shuffle=False)),
                             ('test', DataLoader(test_dataset, batch_size=512, shuffle=False))]:
        feats = []
        lbls = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['x']
                    lbl = batch.get('label', None)
                else:
                    x = batch[0]
                    lbl = batch[1] if len(batch) > 1 else None
                x = x.to(device)
                feat = model(x)
                feats.append(feat.cpu())
                lbls.append(lbl)
        feats = torch.cat(feats, dim=0).numpy()
        lbls = np.array(lbls)
        if mode=='train':
            train_feats = feats
            train_labels = lbls
        else:
            test_feats = feats
            test_labels = lbls

    # Train linear classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_feats, train_labels)
    pred = clf.predict(test_feats)
    top1_acc = accuracy_score(test_labels, pred) * 100
    return {'top1_acc': top1_acc}

def visualize_trajectories(model, x, augmentation_list, label=None, save_path=None):
    """
    For a given input x, apply a sequence of augmentations (e.g. rotations),
    compute embeddings, and plot trajectories in 2D (via PCA or t-SNE).
    """
    model.eval()
    with torch.no_grad():
        # Compute embedding for original x
        z0 = model(x.unsqueeze(0).to(device)).cpu().numpy()
        # Collect embeddings along augmentation sequence
        embeddings = [z0]
        for aug in augmentation_list:
            if isinstance(x, torch.Tensor):
                # assume augmentation modifies tensor directly
                x_aug = aug(x.unsqueeze(0)).squeeze(0)
            else:
                # fallback: if aug is a function applied to numpy array
                x_np = x.cpu().numpy()
                x_aug_np = aug(x_np)
                x_aug = torch.tensor(x_aug_np, device=device)
            z_aug = model(x_aug.unsqueeze(0)).cpu().numpy()
            embeddings.append(z_aug)
        # Convert to numpy array
        embeddings = np.concatenate(embeddings, axis=0)
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        embed_2d = pca.fit_transform(embeddings)
        plt.figure(figsize=(6,6))
        plt.plot(embed_2d[:,0], embed_2d[:,1], marker='o')
        if label is not None:
            plt.title(f"Trajectory for input {label}")
        else:
            plt.title("Embedding Trajectory")
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

def plot_cosine_histogram(z_pairs, title='Cosine Similarity Histogram', save_path=None):
    """
    z_pairs: list of tuples of two embedding tensors (both normalized)
    Plot histogram of cosine similarities.
    """
    cos_sims = []
    for z1, z2 in z_pairs:
        # ensure tensors are normalized
        z1_norm = z1 / (z1.norm(p=2, dim=1, keepdim=True) + 1e-8)
        z2_norm = z2 / (z2.norm(p=2, dim=1, keepdim=True) + 1e-8)
        sims = (z1_norm * z2_norm).sum(dim=1).cpu().numpy()
        cos_sims.extend(sims)
    plt.figure(figsize=(6,4))
    plt.hist(cos_sims, bins=50, range=(-1,1))
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def evaluate_equivariance(model, dataset, num_samples=100, max_angle_deg=15):
    """
    Measures the degree of equivariance/sensitivity:
    - Samples random inputs and augmentation parameters
    - Computes embeddings before and after augmentation
    - Solves Wahba's problem to find rotation R_a approximating f(a(x))
    - Measures deviation ||f(a(x)) - R_a f(x)||_F
    """
    model.eval()
    all_deviations = []
    whitening_transform = None
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    with torch.no_grad():
        for batch in tqdm(loader):
            if isinstance(batch, dict):
                x = batch['x']
            else:
                x = batch[0]
            for _ in range(max(1, num_samples//len(loader))):
                # For image datasets, sample random small rotation
                angle_deg = np.random.uniform(-max_angle_deg, max_angle_deg)
                rot_mat = random_rotation_matrix(max_angle_deg=angle_deg)
                # For 3D point clouds, apply rotation
                if isinstance(x, torch.Tensor) and x.ndim==3:
                    # e.g. protein data
                    x_aug_np = np.einsum('ij,bnj->bni', rot_mat, x.cpu().numpy())  # batch processing
                    x_aug = torch.tensor(x_aug_np, device=device)
                else:
                    # For images, simulate as identity (or skip)
                    x_aug = x.clone()
                    # Alternatively, for images, could rotate if preferred
                # Compute embeddings
                z_orig = model(x.to(device))
                z_aug = model(x_aug)
                # Normalize
                z_orig_norm = z_orig / (z_orig.norm(p=2, dim=1, keepdim=True) + 1e-8)
                z_aug_norm = z_aug / (z_aug.norm(p=2, dim=1, keepdim=True) + 1e-8)
                # For each sample in batch, compute R minimizing deviation
                # Using Wahba's problem solution
                R_a_batch = estimate_rotation_wahba(z_orig_norm, z_aug_norm)
                # Compute deviation
                devs = torch.norm(z_aug - torch.matmul(z_orig, R_a_batch.t()), dim=1).cpu().numpy()
                all_deviations.extend(devs)
    mean_dev = np.mean(all_deviations)
    median_dev = np.median(all_deviations)
    print(f"Avg deviation in embedding space (f(a(x)) vs R_a f(x)): {mean_dev:.4f}")
    print(f"Median deviation: {median_dev:.4f}")
    return {'mean_deviation': mean_dev, 'median_deviation': median_dev}

def estimate_rotation_wahba(Zx, Za):
    """
    Estimate the rotation matrix R via SVD solving Wahba's problem: minimize ||Za - R Zx||_F
    Inputs:
        Zx, Za: embeddings of shape (batch_size, d)
    Output:
        R: estimated rotation matrix (d, d)
    """
    # Compute matrix product
    M = torch.matmul(Za.t(), Zx)
    U, _, Vt = torch.svd(M)
    R = torch.matmul(U, Vt)
    # Ensure rotation matrix has determinant +1
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = torch.matmul(U, Vt)
    return R

def main_evaluation():
    """
    Example function to run all evaluation metrics on a trained model.
    """
    # Load model (assumed frozen)
    # For demonstration, suppose model is loaded and test_dataset is available
    # To do: replace with actual model and dataset loading as per your codebase
    from model import ResNetEncoder
    model = ResNetEncoder(cfg['model']).to(device)
    model.eval()
    # load checkpoint if needed
    # model.load_state_dict(torch.load('path_to_checkpoint.pth'))

    # Load dataset
    dataset_name = cfg['dataset']['name']
    dataset_obj, _ = get_dataset(cfg)

    # --- Linear probe evaluation ---
    print("Performing linear evaluation...")
    train_dl = DataLoader(dataset_obj, batch_size=512, shuffle=False)
    results = compute_linear_probe(model, train_dl, dataset_name)
    print(f"Linear probe Top-1 accuracy: {results['top1_acc']:.2f}%")

    # --- Trajectory visualization example ---
    # Pick a sample
    dataset_test = dataset_obj
    sample_idx = 0
    sample_item = dataset_test[sample_idx]
    if isinstance(sample_item, dict):
        x_sample = sample_item['x']
        label_sample = sample_item.get('label', '')
    else:
        x_sample = sample_item[0]
        label_sample = ''
    # Define a sequence of small rotations for visualization
    from functools import partial
    def small_rotation(x):
        angle_deg = np.random.uniform(-cfg['training']['augmentations'].get('rotation_small',5))
        rot_mat = random_rotation_matrix(max_angle_deg=angle_deg)
        if isinstance(x, torch.Tensor) and x.ndim==3:
            # apply rotation to point cloud
            x_np = x.cpu().numpy()
            rotated_np = np.einsum('ij,bnj->bni', rot_mat, x_np)
            return torch.tensor(rotated_np, device=device)
        else:
            # for images, skip
            return x
    augmentation_sequence = [partial(small_rotation)]
    visualize_trajectories(model, x_sample, augmentation_sequence, label=label_sample)

    # --- Cosine similarity histogram ---
    # Collect pairs: original and augmented
    z_original = model(x_sample.unsqueeze(0).to(device))
    if hasattr(dataset_obj, 'max_points'):
        # For protein data, apply rotation augmentation
        rot_mat = random_rotation_matrix(max_angle_deg=cfg['training']['augmentations'].get('rotation_small',5))
        if isinstance(x_sample, torch.Tensor) and x_sample.ndim==3:
            x_aug_np = np.einsum('ij,bnj->bni', rot_mat, x_sample.cpu().numpy())
            x_aug = torch.tensor(x_aug_np, device=device)
        else:
            x_aug = x_sample
        z_aug = model(x_aug.unsqueeze(0))
        plot_cosine_histogram([(z_original, z_aug)], title='Pos Pair Cosine Similarity', save_path='cosine_hist.png')
    else:
        # For images, create a small augmentation
        pass

    # Additional metrics (equivariance deviations) can be computed similarly

if __name__ == "__main__":
    main_evaluation()
