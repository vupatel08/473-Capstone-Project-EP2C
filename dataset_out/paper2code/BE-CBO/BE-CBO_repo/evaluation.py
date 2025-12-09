## evaluation.py
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Tuple, Optional
import os

class Evaluation:
    def __init__(
        self,
        config: dict,
        dataset_loader,          # instance of DatasetLoader
        classifier,              # neural ensemble classifier
        gp_model,                # GP surrogate for objective
        seed_list: List[int],    # list of seeds for multiple runs
        output_dir: str = 'results'  # directory to store plots and logs
    ):
        """
        Initialize Evaluation object with configuration and models.
        """
        self.config = config
        self.dataset_loader = dataset_loader
        self.classifier = classifier
        self.gp_model = gp_model
        self.seed_list = seed_list
        self.output_dir = output_dir

        # Create output directory if not exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize history containers
        # Dimensions: list of length len(seed_list), each element is a list
        self.f_best_seeds = [[] for _ in seed_list]
        self.accuracy_seeds = [[] for _ in seed_list]
        self.time_seeds = [[] for _ in seed_list]
        self.boundary_points_seeds = [[] for _ in seed_list]  # For boundary visualization

    def compute_best_feasible(self, f_values: np.ndarray, feas_labels: np.ndarray, seed_idx: int):
        """
        Compute running best feasible objective value.
        """
        feasible_mask = feas_labels == 1
        if np.any(feasible_mask):
            min_value = np.nanmin(f_values[feasible_mask])
        else:
            min_value = np.inf
        self.f_best_seeds[seed_idx].append(min_value)

    def compute_classifier_accuracy(
        self,
        true_labels: np.ndarray,   # shape: (n_evaluated,)
        pred_probs: np.ndarray,    # shape: (n_evaluated,)
        threshold: float = 0.5
    ) -> float:
        """
        Compute balanced accuracy for classifier predictions.
        """
        pred_labels = (pred_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, labels=[0,1]).ravel()
        TPR = tp / (tp + fn + 1e-8)
        TNR = tn / (tn + fp + 1e-8)
        balanced_acc = (TPR + TNR) / 2.0
        return balanced_acc

    def record_evaluation(
        self,
        seed_idx: int,
        iteration: int,
        f_values: np.ndarray,
        feas_labels: np.ndarray,
        eval_times: List[float],
        boundary_points: Optional[np.ndarray] = None
    ):
        """
        Record metrics after each evaluation.
        """
        # Update best feasible value
        self.compute_best_feasible(f_values, feas_labels, seed_idx)

        # Compute classifier predicted probabilities for logged points
        if len(f_values) > 0:
            X_eval = self.dataset_loader.X[-len(f_values):]
            pred_probs = self.classifier.predict_proba(X_eval)
            true_labels = feas_labels[-len(f_values):]
            # Compute balanced accuracy
            acc = self.compute_classifier_accuracy(true_labels, pred_probs)
        else:
            acc = np.nan

        self.accuracy_seeds[seed_idx].append(acc)

        # Record runtime up to current iteration
        total_time = sum(eval_times)
        self.time_seeds[seed_idx].append(total_time)

        # Store boundary points for visualization (if provided)
        if boundary_points is not None:
            self.boundary_points_seeds[seed_idx].append(boundary_points)

    def plot_performance(self):
        """
        Plot mean and std of best feasible objective over evaluations across seeds
        """
        # Collect per seed
        min_len = min([len(s) for s in self.f_best_seeds])
        # Truncate all to same length
        f_best_array = np.array([s[:min_len] for s in self.f_best_seeds])
        mean_fbest = np.mean(f_best_array, axis=0)
        std_fbest = np.std(f_best_array, axis=0)
        eval_counts = np.arange(1, min_len + 1)

        plt.figure(figsize=(8,6))
        plt.plot(eval_counts, mean_fbest, label='Mean Best Feasible Objective')
        plt.fill_between(eval_counts, mean_fbest - std_fbest, mean_fbest + std_fbest, alpha=0.3)
        plt.xlabel('Number of Evaluations')
        plt.ylabel('Best Feasible Objective')
        plt.title('BO Convergence Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'performance.png'))
        plt.close()

    def plot_classifier_accuracy(self):
        """
        Plot mean and std of classifier accuracy over evaluations
        """
        min_len = min([len(s) for s in self.accuracy_seeds])
        acc_array = np.array([s[:min_len] for s in self.accuracy_seeds])
        mean_acc = np.mean(acc_array, axis=0)
        std_acc = np.std(acc_array, axis=0)
        eval_counts = np.arange(1, min_len + 1)

        plt.figure(figsize=(8,6))
        plt.plot(eval_counts, mean_acc, label='Classifier Balanced Accuracy')
        plt.fill_between(eval_counts, mean_acc - std_acc, mean_acc + std_acc, alpha=0.3)
        plt.xlabel('Number of Evaluations')
        plt.ylabel('Balanced Accuracy')
        plt.title('Classifier Accuracy Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'classifier_accuracy.png'))
        plt.close()

    def plot_runtime(self):
        """
        Plot runtime evolution over evaluations
        """
        min_len = min([len(s) for s in self.time_seeds])
        time_array = np.array([s[:min_len] for s in self.time_seeds])
        mean_time = np.mean(time_array, axis=0)
        std_time = np.std(time_array, axis=0)

        eval_counts = np.arange(1, min_len + 1)
        plt.figure(figsize=(8,6))
        plt.plot(eval_counts, mean_time, label='Cumulative Runtime (s)')
        plt.fill_between(eval_counts, mean_time - std_time, mean_time + std_time, alpha=0.3)
        plt.xlabel('Number of Evaluations')
        plt.ylabel('Time (seconds)')
        plt.title('Runtime Over BO Iterations')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'runtime.png'))
        plt.close()

    def plot_boundary_evolution(
        self,
        seed_idx: int,
        boundary_points_list: List[np.ndarray],
        true_boundary_fn: Optional[callable] = None,
        eval_idx: int = -1
    ):
        """
        Visualize the evolution of the estimated boundary.
        For synthetic problems, true_boundary_fn can be used.
        """
        # Generate a grid
        grid_size = 100
        # For 2D only visualization
        # Get data
        if len(boundary_points_list) == 0:
            return
        # Use last boundary points in seed
        boundary_points = boundary_points_list[-1]
        if boundary_points is None or boundary_points.shape[0] == 0:
            return
        # For simplicity, visualize for 2D problems
        if self.dataset_loader.dimension != 2:
            return

        x_min, x_max = self.dataset_loader.lower_bounds[0], self.dataset_loader.upper_bounds[0]
        y_min, y_max = self.dataset_loader.lower_bounds[1], self.dataset_loader.upper_bounds[1]
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
        pts = np.vstack([xx.ravel(), yy.ravel()]).T

        # Get predicted probabilities
        proba_pred = self.classifier.predict_proba(pts).reshape(grid_size, grid_size)

        plt.figure(figsize=(6,6))
        plt.contourf(xx, yy, proba_pred, levels=50, cmap='RdYlBu')
        plt.colorbar(label='Feasibility Probability')
        plt.scatter(self.dataset_loader.X[:,0], self.dataset_loader.X[:,1], c='gray', s=20, label='Evaluated Points')
        plt.scatter(boundary_points[:,0], boundary_points[:,1], c='red', s=50, label='Boundary Points')
        if true_boundary_fn is not None:
            # Plot true boundary
            boundary_vals = true_boundary_fn(pts)
            boundary_mask = (np.abs(boundary_vals) < 0.05).reshape(grid_size, grid_size)
            plt.contour(xx, yy, boundary_mask, levels=[0.5], colors='black', linewidths=2, labels='True Boundary')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'Estimated Boundary Evolution (Seed {seed_idx+1}) Eval {self.config["evaluation"]["total_evaluations"]}')
        plt.legend()
        filename = f'boundary_evolution_seed{seed_idx+1}_eval{self.config["evaluation"]["total_evaluations"]}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def save_metrics_summary(self):
        """
        Save all summaries as numpy arrays or CSV for further analysis.
        """
        # Save best feasibility objectives
        np.save(os.path.join(self.output_dir, 'f_best.npy'), np.array(self.f_best_seeds))
        # Save classifier accuracy
        np.save(os.path.join(self.output_dir, 'classifier_accuracy.npy'), np.array(self.accuracy_seeds))
        # Save runtime
        np.save(os.path.join(self.output_dir, 'runtime.npy'), np.array(self.time_seeds))

    def generate_all_plots(self):
        """Generate all plots at the end of experiments."""
        self.plot_performance()
        self.plot_classifier_accuracy()
        self.plot_runtime()
        # Boundary evolution plots for each seed
        for seed_idx in range(len(self.seed_list)):
            boundary_list = self.boundary_points_seeds[seed_idx]
            self.plot_boundary_evolution(seed_idx, boundary_list)

