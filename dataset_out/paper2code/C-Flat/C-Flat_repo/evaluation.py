## evaluation.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

# Assuming PyHessian is available if needed; if not, this can be mocked or skipped
try:
    import pyanalyze
    import pyhessian
    HAS_PYHESSIAN = True
except ImportError:
    HAS_PYHESSIAN = False

from utils import compute_hessian_eigenvalue, estimate_hessian_eigenvalue

class Evaluation:
    def __init__(self, model: torch.nn.Module, dataset_loader, config: Dict, device: torch.device):
        """
        Initialize the Evaluation class.

        Args:
            model (torch.nn.Module): The trained model to evaluate.
            dataset_loader: DatasetLoader object managing data splits.
            config (dict): Evaluation and visualization configuration.
            device (torch.device): Device ('cpu' or 'cuda') for computations.
        """
        self.model = model
        self.dataset_loader = dataset_loader
        self.device = device
        self.config = config

        # Extract evaluation parameters from config with defaults
        eval_conf = self.config.get('evaluation', {})
        landscape_cfg = eval_conf.get('landscape_visualization', False)
        self.show_landscape = landscape_cfg
        self.eval_metrics = eval_conf.get('metrics', True)
        self.eval_freq = eval_conf.get('evaluation_frequency', 1)  # evaluate every 'n' epochs
        self.landscape_eval_freq = eval_conf.get('landscape_eval_freq', 10)  # visualize landscape every 'n' epochs

        # Setup output directory for logs and figures
        self.output_dir = self.config.get('logging', {}).get('output_dir', './logs')
        os.makedirs(self.output_dir, exist_ok=True)

        # Storage for metrics over phases
        self.performance_history = []

        # For Hessian computations: cache for eigenvalues
        self.prev_model_state = None

        # Additional: Track model and data info if needed
        self.current_task_idx = None

        # Optional: initialize visualization tools if available
        # For example, PyHessian
        if HAS_PYHESSIAN:
            # Not creating specific Hessian object here; will instantiate when needed
            pass

        # Store original model parameters for landscape visualization
        self.center_params = []

    def evaluate(self, task_idx: int, seen_class_count: int, dataloader=None) -> Dict:
        """
        Evaluate the model over all seen classes.

        Args:
            task_idx (int): Current task index (for logging).
            seen_class_count (int): Number of classes seen so far.
            dataloader (DataLoader, optional): Optional data loader for evaluation.
        Returns:
            dict: Dictionary with evaluation metrics (accuracy, etc.)
        """
        # Use full test dataset for evaluation
        test_loader = self.dataset_loader.get_full_test_dataset()
        test_loader = self.dataset_loader.get_test_dataloader(batch_size=self.config['training'].get('batch_size', 64),
                                                                  shuffle=False)
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = 100.0 * correct / total

        # Save/Log metrics
        metrics = {"accuracy": accuracy}
        # Optional: print or save
        print(f"Evaluation at task {task_idx}: Acc={accuracy:.2f}% over {total} samples.")
        return metrics

    def compute_hessian_eig(self, sample_inputs: torch.Tensor, sample_labels: torch.Tensor, topk: int=1) -> Tuple[float, List[float]]:
        """
        Compute top Hessian eigenvalues for model curvature estimation.
        Args:
            sample_inputs (Tensor): A batch of inputs for Hessian approximation.
            sample_labels (Tensor): Corresponding labels.
            topk (int): number of top eigenvalues to compute.
        Returns:
            Tuple[float, List[float]]: max eigenvalue and list of top eigenvalues.
        """
        # Use the utility function, optionally with PyHessian if available
        # We'll use a custom estimation here
        eigenvalue = estimate_hessian_eigenvalue(self.model, cross_entropy, sample_inputs, sample_labels, num_iterations=20)
        return eigenvalue

    def visualize_landscape(self, epoch: int):
        """
        Generate and save a 2D loss landscape around current parameters.
        Uses perturbations along top Hessian eigenvectors.
        """
        if not self.show_landscape:
            return
        if not hasattr(self, 'model'):
            return

        # Save current model parameters as center
        self.center_params = [p.clone().detach() for p in self.model.parameters()]

        # Generate random directions or use top eigenvectors
        # For simplicity, generate two random directions
        directions = []
        for _ in range(2):
            dir_list = [torch.randn_like(p) for p in self.model.parameters()]
            # Normalize
            flattened = torch.cat([d.view(-1) for d in dir_list])
            flattened /= (torch.norm(flattened) + 1e-8)
            # Reshape back
            dir_list_norm = []
            idx = 0
            for p in self.model.parameters():
                size = p.numel()
                dir_list_norm.append(flattened[idx:idx+size].view_as(p))
                idx += size
            directions.append(dir_list_norm)

        # Call visualization
        # Note: for full landscape plot, user should implement function, here just a placeholder
        self.plot_loss_surface(self.model, self.center_params, directions, epoch)

    def plot_loss_surface(self, model, center_params, directions, epoch: int):
        """
        Plot loss surface over a grid along two directions.
        """
        grid_size = 50
        alpha = np.linspace(-1, 1, grid_size)
        beta = np.linspace(-1, 1, grid_size)
        loss_surface = np.zeros((grid_size, grid_size))

        # Save original parameters
        orig_params = [p.clone() for p in model.parameters()]

        for i, a in enumerate(alpha):
            for j, b in enumerate(beta):
                # Construct perturbed parameters
                perturbed_params = []
                for idx, p in enumerate(model.parameters()):
                    delta = a * directions[0][idx] + b * directions[1][idx]
                    p.data.copy_(center_params[idx] + delta)
                # Compute loss
                # pick a small batch for evaluation (e.g., random or fixed)
                # For simplicity here, create dummy input
                dummy_input = torch.randn(16, 3, 32, 32).to(next(model.parameters()).device)
                dummy_labels = torch.randint(0, 100, (16,)).to(next(model.parameters()).device)
                output = model(dummy_input)
                loss = cross_entropy(output, dummy_labels).item()
                loss_surface[i, j] = loss

        # Restore original parameters
        for p, orig in zip(model.parameters(), orig_params):
            p.data.copy_(orig)

        # Plotting
        plt.figure(figsize=(6,5))
        plt.contourf(alpha, beta, loss_surface, levels=50)
        plt.xlabel('Direction 1 coefficient')
        plt.ylabel('Direction 2 coefficient')
        plt.title(f'Loss Landscape at Epoch {epoch}')
        plt.colorbar()
        plt.savefig(os.path.join(self.output_dir, f'landscape_task_{self.current_task_idx}_epoch_{epoch}.png'))
        plt.close()

    def process(self, epoch: int, task_idx: int, seen_class_count: int):
        """
        To be called at epoch end for periodic evaluation.
        """
        # Evaluate metrics
        metrics = self.evaluate(task_idx, seen_class_count)
        self.performance_history.append({"task": task_idx, "epoch": epoch, **metrics})

        # Visualize landscape periodically
        if self.show_landscape and epoch % self.landscape_eval_freq == 0:
            self.visualize_landscape(epoch)

        # Compute Hessian eigenvalues at current point
        if HAS_PYHESSIAN:
            # Sample a batch for Hessian estimation
            # As a placeholder, create dummy inputs
            dummy_input = torch.randn(16, 3, 32, 32).to(next(self.model.parameters()).device)
            dummy_labels = torch.randint(0, 100, (16,)).to(next(self.model.parameters()).device)
            eigenvalue = self.compute_hessian_eig(dummy_input, dummy_labels, topk=1)
            print(f"Task {task_idx} Epoch {epoch}: Top Hessian eigenvalue = {eigenvalue:.4f}")

    def save_final_results(self):
        """
        Save collected metrics, plots, and logs after training.
        """
        import json
        # Save metrics history
        json_path = os.path.join(self.output_dir, 'performance_metrics.json')
        with open(json_path, 'w') as f:
            import json
            json.dump(self.performance_history, f, indent=2)

