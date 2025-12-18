## evaluation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

class Evaluation:
    """
    Evaluation class to perform inference on test data, compute metrics,
    and optionally visualize node representations.
    """

    def __init__(self, model, predictor, data, device):
        """
        Initialize with trained diffusion model, predictor, dataset split data, and device.
        
        Args:
            model (DiffusionModel): diffusion-based GNN with inverse capabilities.
            predictor (torch.nn.Module): trained classification head.
            data (tuple): (features, labels, train_mask, val_mask, test_mask)
            device (torch.device): computation device.
        """
        self.model = model
        self.predictor = predictor
        self.features, self.labels, self.train_mask, self.val_mask, self.test_mask = data
        self.device = device
        # Set model to eval mode
        self.model.eval()
        self.predictor.eval()
        # Extract diffusion hyperparameters for inference
        self.diff_TF = getattr(self.model, 'T_F', 1.0)
        self.diff_TR = getattr(self.model, 'T_R', -1.0)
        self.fixed_point_iter = getattr(self.model, 'M', 16)
        # Visualization flag from config could be passed here if needed
        self.visualization_enabled = False

    def evaluate(self, n_splits=10, seed=42, visualize_layers=None):
        """
        Perform inference over multiple splits, compute mean and std of accuracy,
        and produce visualizations if enabled.
        
        Args:
            n_splits (int): number of dataset splits.
            seed (int): for reproducibility.
            visualize_layers (list): layers to visualize; e.g., ['forward', 'reverse', 'concatenate']
        
        Returns:
            metrics_dict (dict): contains mean and std of accuracy over splits.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        accuracies = []

        for split_idx in tqdm(range(n_splits), desc='Evaluating splits'):
            # For reproducibility, ensure dataset splits are consistent only if dataset is
            # configured accordingly; assume test_mask is fixed for each split
            test_mask = self.test_mask
            
            # Run diffusion forward at TF
            with torch.no_grad():
                x_fwd = self.model.forward(self.features)
                # Run inverse diffusion at TR
                x_rev = self.model.inverse(x_fwd)
            
            # For test nodes only
            test_indices = torch.nonzero(test_mask, as_tuple=True)[0]

            # Extract representations for test nodes
            # Depending on "layers to visualize", pick representations
            # For this code, simply use final diffused and inverse features
            # If detailed layer features needed, modify accordingly

            # Get representations
            fwd_repr = x_fwd[test_indices]
            rev_repr = x_rev[test_indices]

            # Concatenate representations
            combined = torch.cat([fwd_repr, rev_repr], dim=1)
            # Pass through predictor
            logits = self.predictor(combined)

            preds = torch.argmax(logits, dim=1)

            # Compute accuracy
            lbls = self.labels[test_indices]
            correct = (preds == lbls).sum().item()
            total = lbls.shape[0]
            accuracy = correct / total
            accuracies.append(accuracy)

            # Visualization if enabled
            if self.visualization_enabled:
                self._visualize_embeddings(
                    fwd_repr.cpu().numpy(),
                    rev_repr.cpu().numpy(),
                    lbls.cpu().numpy(),
                    preds.cpu().numpy(),
                    split_idx,
                    features=self.features.cpu().numpy(),
                    indices=test_indices.cpu().numpy()
                )

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        metrics_dict = {
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "accuracy_list": accuracies
        }
        print(f"\nEvaluation over {n_splits} splits:")
        print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        return metrics_dict

    def _visualize_embeddings(self, fwd_repr, rev_repr, labels, preds, split_idx, features=None, indices=None):
        """
        Generate scatter plot of representations for qualitative analysis.
        
        Args:
            fwd_repr (np.ndarray): forward diffusion features
            rev_repr (np.ndarray): inverse diffusion features
            labels (np.ndarray): true labels
            preds (np.ndarray): predicted labels
            split_idx (int): index of split for saving figure
            features (np.ndarray): original features (optional)
            indices (np.ndarray): node indices (optional)
        """
        plt.figure(figsize=(12, 4))
        # Plot forward representations
        plt.subplot(1, 3, 1)
        self._plot_repr(fwd_repr, labels, title='Forward Diffusion', indices=indices)
        # Plot reverse representations
        plt.subplot(1, 3, 2)
        self._plot_repr(rev_repr, labels, title='Reverse Diffusion', indices=indices)
        # Plot concatenated
        plt.subplot(1, 3, 3)
        concat_repr = np.concatenate([fwd_repr, rev_repr], axis=1)
        self._plot_repr(concat_repr, labels, title='Concatenated', indices=indices)

        plt.tight_layout()
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig(f"visualizations/representation_split_{split_idx}.png")
        plt.close()

    def _plot_repr(self, reprs, labels, title='Representation', indices=None):
        """
        Plot 2D visualization of node representations using t-SNE.
        """
        if reprs.shape[1] > 2:
            embedding = TSNE(n_components=2, random_state=42).fit_transform(reprs)
        else:
            embedding = reprs
        plt.scatter(embedding[:,0], embedding[:,1], c=labels, cmap='tab10', s=15)
        plt.title(title)
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.colorbar()
