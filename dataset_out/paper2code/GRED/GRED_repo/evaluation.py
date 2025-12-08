## evaluation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import (
    compute_shortest_paths,
    create_neighborhood_masks,
)
from model import GraphGRED

def load_cfg(cfg_path='config.yaml'):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(cfg, model_checkpoint_path, input_dim, device):
    model = GraphGRED(cfg['model'], input_dim=input_dim)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate(model, dataset, device, metric_name='accuracy'):
    """
    Evaluate the model on the dataset using the specified metric.
    """
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for data in tqdm(dataset, desc="Evaluation", leave=False):
            features = data['features'].to(device)            # (V, d)
            label = data['label'].to(device)                  # single label per graph or node
            masks = data['neighborhood_masks']
            masks = {k: v.to(device) for k, v in masks.items()}

            outputs = model(features, [masks])  # model expects list of masks
            all_outputs.append(outputs)
            all_labels.append(label)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if metric_name == 'accuracy':
        preds = torch.argmax(all_outputs, dim=1)
        acc = (preds == all_labels).float().mean().item()
        return {'accuracy': acc}
    elif metric_name == 'MAE':
        mae = torch.abs(all_outputs.squeeze() - all_labels.float()).mean().item()
        return {'MAE': mae}
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

def plot_eigenvalues(eigenvalues, save_path=None):
    """
    Plot the complex eigenvalues in the complex plane.
    eigenvalues: tensor of shape (d_s, 2), real and imaginary parts.
    """
    re = eigenvalues[:, 0].cpu().numpy()
    im = eigenvalues[:, 1].cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.scatter(re, im, c='blue', marker='o')
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.title('Learned Eigenvalues in Complex Plane')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def main():
    cfg = load_cfg('config.yaml')
    device = get_device()

    # Load dataset
    dataset_cfg = cfg['dataset']
    dataset_name = dataset_cfg.get('name', 'CIFAR10')
    dataset_path = dataset_cfg.get('path', None)
    max_K = cfg['model'].get('neighborhood_K', None)

    # Load dataset and compute neighborhood masks if not precomputed
    # Here, for evaluation, assume dataset loader handles masks.
    dataset = dataset_cfg.get('dataset_obj')  # For demonstration, replace with actual dataset loader if needed

    # Suppose dataset is a list-like object (from dataset_loader.py) with each element:
    # {'graph': ..., 'features': ..., 'label': ..., 'dist_matrix': ..., 'neighborhood_masks': ...}
    # with all precomputed.

    # Load model
    sample_data = dataset[0]
    input_dim = sample_data['features'].shape[1]
    checkpoint_path = 'path/to/trained/model.pt'  # replace with actual path
    model = load_model(cfg, checkpoint_path, input_dim, device)

    # Determine task type based on dataset info or task setting
    # For simplicity, assume classification if number of classes > 2
    # Else, regression
    # Here, using label type:
    sample_label = dataset[0]['label']
    if isinstance(sample_label, torch.Tensor):
        label_dim = sample_label.shape
    else:
        label_dim = torch.tensor(sample_label).shape
    # Determine task
    if len(label_dim) == 0 or label_dim[0] == 1:
        # scalar label, treat as regression
        metric_name = 'MAE'
    else:
        # multi-dimensional label: for node classification, assume class labels
        metric_name = 'accuracy'

    # Evaluate
    results = evaluate(model, dataset, device, metric_name=metric_name)

    print("="*40)
    print(f"Results on dataset '{dataset_name}':")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print("="*40)

    # Plot eigenvalues if possible
    # Assume the model has attribute to retrieve learned eigenvalues
    try:
        eigenvalues = model.layers[0].rnn_encoder.get_eigenvalues()
        plot_eigenvalues(eigenvalues)
    except Exception:
        print("Could not retrieve eigenvalues for visualization.")

if __name__ == '__main__':
    main()
