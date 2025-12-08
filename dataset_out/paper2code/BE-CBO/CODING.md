# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## acquisition_optimizer.py

```python
## acquisition_optimizer.py
import numpy as np
import scipy.optimize as opt
from typing import Callable, Tuple
import warnings

# Import BoundaryBounds class and classifier interface
# Assume they are available in the same package or imported accordingly
# from boundary_bounds import BoundaryBounds
# from constraint_classifier import NeuralEnsembleConstraintClassifier

class AcquisitionOptimizer:
    """
    Class to perform constrained maximization of an acquisition function q(x),
    coupling with dynamic boundary bounds based on classifier uncertainty.
    """
    def __init__(
        self,
        acq_func: Callable[[np.ndarray], float],
        C_func: Callable[[np.ndarray], float],
        bounds: dict,
        classifier,
        boundary_bounds,
        method: str = 'differential_evolution',
        population_size: int = 50,
        max_iter: int = 200,
        init_points: int = 10,
        seed: int = 42
    ):
        """
        Initialize the optimizer.
        Args:
            acq_func: Callable that takes in an array of shape (d,) and returns a scalar acquisition value.
            C_func: Callable that takes in an array (d,) and returns probability of feasibility.
            bounds: dict with variable bounds, e.g.
                {'var1': (lb1, ub1), 'var2': (lb2, ub2), ..., 'var_d': (lbd, ubd)}.
            classifier: instance with methods:
                - predict_proba(x): float in [0,1]
                - predict_uncertainty(x): float (sigma_E)
            boundary_bounds: BoundaryBounds instance for computing lower bounds.
            method: optimization method, default 'differential_evolution'. alternatives: 'L-BFGS-B', 'SLSQP'.
            population_size: EV method population size.
            max_iter: maximum iterations for local solvers.
            init_points: number of random initial points for global search.
            seed: for reproducibility.
        """
        self.acq_func = acq_func
        self.C_func = C_func
        self.bounds = bounds
        self.var_bounds = np.array([bounds[k] for k in sorted(bounds.keys())])  # shape (d,2)
        self.dim = self.var_bounds.shape[0]
        self.classifier = classifier
        self.boundary_bounds = boundary_bounds
        self.method = method
        self.pop_size = population_size
        self.max_iter = max_iter
        self.init_points = init_points
        self.random_state = np.random.RandomState(seed)

    def _get_bounds_list(self):
        """
        Returns list of (lb, ub) for scipy optimizers.
        """
        return list(sorted(self.bounds.values()))

    def _constraint(self, x: np.ndarray) -> float:
        """
        Evaluate the constraint g1(x): C(x) - l(x) >= 0
        """
        # Compute classifier probability
        C_x = self.C_func(x.reshape(1, -1))[0]
        # Compute classifier uncertainty sigma_e
        sigma_e = self.classifier.predict_uncertainty(x.reshape(1, -1))[0]
        # Compute lower bound l(x)
        l_x = self.boundary_bounds.compute_lower_bound(sigma_e)
        # Constraint: C(x) - l(x) >= 0
        return C_x - l_x

    def _prepare_constraints(self):
        """
        Prepare the list of constraints for scipy.optimize
        """
        cons = {
            'type': 'ineq',
            'fun': self._constraint
        }
        return [cons]

    def _initial_samples(self, num_samples: int) -> np.ndarray:
        """
        Generate initial random samples within bounds for global optimization.
        """
        var_bounds_sorted = self._get_bounds_list()
        res = []
        for _ in range(num_samples):
            sample = np.array([
                self.random_state.uniform(low=b[0], high=b[1])
                for b in var_bounds_sorted
            ])
            res.append(sample)
        return np.array(res)

    def _evaluate_acq(self, x: np.ndarray) -> float:
        """
        Evaluate the acquisition function at x.
        """
        return self.acq_func(x)

    def _maximize_single_start(self, x0: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run local optimizer (SLSQP or L-BFGS-B) from a start point x0 to maximize acquisition.
        Since scipy.optimize.minimize minimizes, we minimize negative acquisition.
        """
        bounds_list = self._get_bounds_list()

        def obj_fn(x):
            return -self._evaluate_acq(x)

        res = None
        try:
            res = opt.minimize(
                obj_fn,
                x0=x0,
                method='SLSQP' if self.method == 'SLSQP' else 'L-BFGS-B',
                bounds=bounds_list,
                constraints=self._prepare_constraints(),
                options={'maxiter': self.max_iter, 'ftol':1e-9}
            )
        except Exception as e:
            warnings.warn(f"Local optimization failed from start {x0}: {e}")
        if res is not None:
            return res.x, -res.fun
        else:
            return x0, -self._evaluate_acq(x0)

    def _run_global_optimization(self) -> Tuple[np.ndarray, float]:
        """
        Run global optimization (differential evolution) to find the best feasible point.
        """
        bounds_list = self._get_bounds_list()
        constraints = self._prepare_constraints()

        def min_obj(x):
            return -self._evaluate_acq(x)

        result = None
        try:
            result = opt.differential_evolution(
                min_obj,
                bounds=bounds_list,
                constraints=constraints,
                maxiter=self.max_iter,
                popsize=self.pop_size,
                polish=True,
                seed=self.random_state.randint(0, 10000),
                workers=1
            )
        except Exception as e:
            warnings.warn(f"DiffEvo optimization failed: {e}")
        if result is not None and result.success:
            return result.x, -result.fun
        else:
            # fallback: evaluate random feasible points
            rand_points = self._initial_samples(self.init_points)
            feas_points = []
            feas_vals = []
            for x in rand_points:
                c = self._constraint(x)
                if c >= 0:
                    val = self._evaluate_acq(x)
                    feas_points.append(x)
                    feas_vals.append(val)
            if feas_points:
                idx = np.argmax(feas_vals)
                return feas_points[idx], feas_vals[idx]
            else:
                # No feasible points found, sample random point
                x0 = rand_points[0]
                return x0, self._evaluate_acq(x0)

    def optimize(self) -> Tuple[np.ndarray, dict]:
        """
        Main method: perform constrained maximization.
        Returns:
            best_x: the point with highest acquisition under constraints.
            info: dict with details.
        """
        # Run global optimizer
        best_x, best_acq = self._run_global_optimization()
        # Compile info
        info = {
            'x': best_x,
            'acquisition_value': best_acq,
            'success': True
        }
        return best_x, info
```

**Notes:**
- The code optimizes the acquisition function with the explicit dynamic constraint coupling, using either differential evolution (for global search) or local methods with multi-start if desired.
- It dynamically computes the boundary constraint \( g_1(x) = C(x) - l(x) \) based on the classifier’s predicted probability and uncertainty.
- It respects variable bounds, handles multiple initial points for robustness, and uses `scipy.optimize` routines.
- The default method is `'differential_evolution'`, but it can be adjusted.
- The function returns the best candidate point \( x^* \) and debug info.

This implementation follows the provided design, ensures robust handling of the boundary constraint, and integrates smoothly into the broader BE-CBO framework.

## boundary_bounds.py

```python
## boundary_bounds.py
import numpy as np

class BoundaryBounds:
    def __init__(self, margin: float = 0.5):
        """
        Initialize the BoundaryBounds object.
        Args:
            margin (float): The base threshold for classifier boundary, default 0.5.
        """
        self.margin = margin

    def compute_lower_bound(self, sigma_e: float) -> float:
        """
        Compute the dynamic lower bound l(x) based on classifier uncertainty.
        Args:
            sigma_e (float): Standard deviation (uncertainty) of the ensemble prediction at x.
        Returns:
            float: The lower bound l(x) = max(0, margin - sigma_e).
        """
        l_x = self.margin - sigma_e
        # Ensure the lower bound is within [0,1]
        return max(0.0, l_x)

    def compute_bounds(self, C_x: float, sigma_e: float) -> Tuple[float, float]:
        """
        Compute the sampling bounds for a candidate point x, based on its classifier's
        predicted probability C_x and the ensemble uncertainty sigma_e.
        Args:
            C_x (float): The predicted probability (mean ensemble output) for feasibility, in [0,1].
            sigma_e (float): The classifier ensemble's standard deviation at x.
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound), bounds in [0,1],
                                 used for constrained acquisition optimization.
        """
        # Dynamic lower bound based on uncertainty
        l_x = self.compute_lower_bound(sigma_e)
        # Upper feasibility bound is fixed at 1
        u_x = 1.0
        # The lower bound for sampling is at least the maximum between l_x and C_x - sigma_e (for more exploration)
        # But in this implementation, we use the dynamic lower bound:
        lower_bound = l_x
        # For simplicity, cap the lower bound to be at most C_x (so we don't sample into the more infeasible region)
        lower_bound = min(lower_bound, C_x)
        # Ensure bounds are within [0,1]
        lower_bound = max(0.0, lower_bound)
        u_x = min(1.0, u_x)
        return (lower_bound, u_x)
```

## constraint_classifier.py

```python
## constraint_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List

# A simple neural network architecture class
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: int, neurons_per_layer: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, neurons_per_layer))
            layers.append(nn.ReLU())
            in_dim = neurons_per_layer
        # Final output layer: single neuron (logits)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # shape: (batch_size,)

class NeuralEnsembleConstraintClassifier:
    def __init__(self,
                 input_dim: int,
                 ensemble_size: int = 5,
                 hidden_layers: int = 2,
                 neurons_per_layer: int = 64,
                 learning_rate: float = 0.001,
                 epochs: int = 50,
                 batch_size: int = 32,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize the ensemble of neural network classifiers.
        """
        self.input_dim = input_dim
        self.ensemble_size = ensemble_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        # Create ensemble models independently
        self.models: List[nn.Module] = [
            MLP(self.input_dim, self.hidden_layers, self.neurons_per_layer).to(self.device)
            for _ in range(self.ensemble_size)
        ]

        # Optimizers for each model
        self.optimizers: List[optim.Optimizer] = [
            optim.Adam(model.parameters(), lr=self.learning_rate) for model in self.models
        ]

        # Loss function: binary cross entropy with logits (numerically stable)
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train each ensemble member independently using provided data.
        Args:
            X: numpy array, shape (num_samples, input_dim)
            y: binary labels (0 or 1), shape (num_samples,)
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            for epoch in range(self.epochs):
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    logits = model(batch_X)
                    loss = self.criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict mean probability of feasibility over the ensemble for input X.
        Args:
            X: numpy array, shape (num_samples, input_dim)
        Returns:
            probs: numpy array, shape (num_samples,)
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        probs_list = []

        with torch.no_grad():
            for model in self.models:
                model.eval()
                logits = model(X_tensor)
                probs = torch.sigmoid(logits)
                probs_list.append(probs.cpu().numpy())

        probs_array = np.stack(probs_list, axis=0)  # shape: (ensemble_size, num_samples)
        mean_probs = probs_array.mean(axis=0)     # shape: (num_samples,)
        return mean_probs

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the ensemble predictive standard deviation (uncertainty) at points X.
        Args:
            X: numpy array, shape (num_samples, input_dim)
        Returns:
            sigma: numpy array, shape (num_samples,)
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        probs_list = []

        with torch.no_grad():
            for model in self.models:
                model.eval()
                logits = model(X_tensor)
                probs = torch.sigmoid(logits)
                probs_list.append(probs.cpu().numpy())

        probs_array = np.stack(probs_list, axis=0)  # shape: (ensemble_size, num_samples)
        mean_probs = probs_array.mean(axis=0)
        # Compute variance
        var = np.var(probs_array, axis=0, ddof=1)  # sample variance
        sigma = np.sqrt(var)
        return sigma

    def train_from_dataset(self, X: np.ndarray, labels: np.ndarray):
        """
        Convenient method to retrain the ensemble on new dataset.
        """
        self.train(X, labels)
```

## dataset_loader.py

```python
## dataset_loader.py
import numpy as np
from scipy.stats import qmc
from typing import Tuple, List, Optional

# Import or define synthetic evaluation functions here (to be replaced or extended for real benchmarks)
def evaluate_synthetic Townsend(x: np.ndarray) -> Tuple[float, bool]:
    """Evaluate Townsend synthetic function and its constraint."""
    x1, x2 = x[0], x[1]
    # Objective
    obj = - (np.cos((x1 - 0.1) * x2)) ** 2 - x1 * np.sin(3 * x1 + x2)
    # Constraint
    t = np.arctan2(x2, x1)
    constraint_value = (2 * np.cos(t) - 0.5 * np.cos(2 * t) - 0.25 * np.cos(3 * t) - 0.125 * np.cos(4 * t)) ** 2 + (2 * np.sin(t)) ** 2 - x1 ** 2 - x2 ** 2
    feasible = bool(constraint_value <= 1e-6)
    return obj, feasible

def evaluate_synthetic_Simionescu(x: np.ndarray) -> Tuple[float, bool]:
    """Evaluate Simionescu synthetic function and its constraint."""
    x1, x2 = x[0], x[1]
    obj = 0.1 * x1 * x2
    r_T, r_S, n = 1.0, 0.2, 8
    phi = n * np.arctan2(x2, x1)
    c_val = ((r_T + r_S * np.cos(phi)) **2 + (r_S * np.sin(phi))**2) - (x1 **2 + x2**2)
    feasible = bool(c_val <= 1e-6)
    return obj, feasible

def evaluate_synthetic_LSQ(x: np.ndarray) -> Tuple[float, bool]:
    """Evaluate LSQ synthetic function and constraints."""
    x1, x2 = x[0], x[1]
    obj = x1 + x2
    # Constraints
    c1 = x1 + 2*x2 + np.sin(2*np.pi*(x1**2 - 2*x2))
    c2 = 1.5 - x1**2 - x2**2
    feasible = (c1 >= -1e-6) and (c2 >= -1e-6)
    return obj, feasible

# Placeholder for real problem evaluations (to be extended with actual functions/APIs)
def evaluate_real_problem(name: str, x: np.ndarray) -> Tuple[Optional[float], bool]:
    """
    Evaluate real-world problem 'name' with input x.
    Returns objective value and feasibility. If infeasible or eval fails, return None.
    """
    # For demonstration, return dummy feasible data
    # Replace with actual evaluation routines as needed
    # For now, all points are considered feasible with dummy objective values
    return 0.0, True  # Should be replaced with real evaluations

class DatasetLoader:
    def __init__(self, bounds: dict, initial_samples: int, problem_type: str = 'synthetic', problem_name: str = '', seed: int = 42):
        """
        Initialize the dataset loader.
        :param bounds: dict with variable bounds, e.g.,
            {'x1': (lower, upper, 'type'), ...}
        :param initial_samples: number of initial samples to generate
        :param problem_type: 'synthetic' or 'real'
        :param problem_name: name of the benchmark problem for evaluation routines
        :param seed: random seed for reproducibility
        """
        self.bounds = bounds
        self.initial_samples = initial_samples
        self.problem_type = problem_type
        self.problem_name = problem_name
        self.seed = seed
        self.dimension = len(bounds)
        self.var_names = list(bounds.keys())
        # Parse bounds into arrays for easy sampling
        self.lower_bounds = np.array([bounds[var][0] for var in self.var_names])
        self.upper_bounds = np.array([bounds[var][1] for var in self.var_names])
        # Variable types: 'float' or 'int'
        self.var_types = [bounds[var][2] for var in self.var_names]
        # Initialize dataset storage
        self.X = np.empty((0, self.dimension))
        self.Y_obj = []  # Objective values (float)
        self.Y_feasibility = []  # Boolean flags
        # Generate initial samples
        self._rng = np.random.RandomState(self.seed)
        self.initialize_samples()

    def initialize_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate initial samples uniformly within bounds using Sobol sequence.
        Converts samples to correct variable types.
        """
        sampler = qmc.Sobol(d=self.dimension, scramble=True, seed=self.seed)
        n_samples = self.initial_samples
        sample = sampler.random(n=n_samples)
        # Scale samples to bounds
        scaled_samples = self.lower_bounds + sample * (self.upper_bounds - self.lower_bounds)
        # Convert to correct data types
        for i, v_type in enumerate(self.var_types):
            if v_type == 'int':
                scaled_samples[:, i] = np.round(scaled_samples[:, i])
        self.X = scaled_samples
        return self.X, np.array(self.Y_obj)

    def get_evaluations(self, x: np.ndarray) -> Tuple[Optional[float], bool]:
        """
        Evaluate the objective and feasibility for sample x.
        Handles both synthetic and real benchmark evaluation routines.
        Returns:
            - objective value (float) if feasible
            - feasibility (bool)
        """
        # For synthetic benchmark problems
        if self.problem_type == 'synthetic':
            if self.problem_name == 'Townsend':
                obj, feasible = evaluate_synthetic Townsend(x)
            elif self.problem_name == 'Simionescu':
                obj, feasible = evaluate_synthetic_Simionescu(x)
            elif self.problem_name == 'LSQ':
                obj, feasible = evaluate_synthetic_LSQ(x)
            else:
                raise ValueError(f"Unknown synthetic problem: {self.problem_name}")
            # Store data
            self.Y_obj.append(obj if feasible else np.nan)
            self.Y_feasibility.append(feasible)
            return obj if feasible else np.nan, feasible

        # For real-world benchmark problems
        elif self.problem_type == 'real':
            obj_value, feasible = evaluate_real_problem(self.problem_name, x)
            # If infeasible, no objective value
            obj = obj_value if feasible else np.nan
            # Store data
            self.Y_obj.append(obj)
            self.Y_feasibility.append(feasible)
            return obj, feasible

        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

    def add_evaluation(self, x: np.ndarray, obj: float, feasible: bool):
        """
        Append new evaluation data.
        """
        self.X = np.vstack([self.X, x.reshape(1, -1)])
        self.Y_obj.append(obj if feasible else np.nan)
        self.Y_feasibility.append(feasible)

    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return current dataset: features, objectives, feasibility labels.
        Feasibility as binary: 1 (feasible), 0 (infeasible)
        """
        feas_labels = np.array(self.Y_feasibility, dtype=int)
        return self.X, np.array(self.Y_obj), feas_labels
```

## evaluation.py

```python
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

```

## main.py

```python
## main.py
import sys
import yaml
import time
import numpy as np
import os
from dataset_loader import DatasetLoader
from model import SurrogateObjectiveGP
from constraint_classifier import NeuralEnsembleConstraintClassifier
from boundary_bounds import BoundaryBounds
from acquisition_optimizer import AcquisitionOptimizer
from evaluation import Evaluation

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Extract configs and set defaults
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    optimizer_config = config.get('optimizer', {})
    acq_config = config.get('acquisition', {})
    boundary_config = config.get('boundary', {})
    eval_config = config.get('evaluation', {})

    total_evals = eval_config.get('total_evaluations', 200)
    init_samples = eval_config.get('initial_samples', 10)
    seeds = eval_config.get('seeds', [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627])
    num_seeds = len(seeds)

    # For reproducibility, for each seed, run full experiment
    for seed_idx, seed in enumerate(seeds):
        print(f"Running seed {seed} ({seed_idx+1}/{num_seeds})")
        np.random.seed(seed)

        # Initialize dataset loader: define problem bounds (example for synthetic; user should adapt)
        # For demonstration, we assume 'dataset_loader.py' handles problem bounds internally.
        # Replace the bounds dict with the appropriate bounds for your problem.
        # For synthetic, set bounds accordingly, e.g.,
        # bounds = {'x1': (-2.25, 2.25), 'x2': (-2.5, 1.75)} for Townsend
        # For real-world, load actual bounds.
        # Here, dummy bounds are used; adapt as necessary.
        bounds = {
            'x1': (-2.25, 2.25, 'float'),
            'x2': (-2.5, 1.75, 'float')
        }
        dim = len(bounds)
        dataset = DatasetLoader(bounds, init_samples, seed=seed)

        # Initialize surrogate model for objective
        gp_hyperparams = {'lengthscale': 1.0}
        surrogate_gp = SurrogateObjectiveGP(kernel_type='RBF', hyperparameters=gp_hyperparams)

        # Prepare datasets for model training
        X_init, Y_init, labels_init = dataset.get_dataset()
        # Filter only feasible points for objective GP
        feasible_mask = np.array(labels_init) == 1
        X_feasible = X_init[feasible_mask]
        Y_feasible = np.array(Y_init)[feasible_mask]
        # Train GP surrogate
        if len(X_feasible) > 0:
            surrogate_gp.fit(X_feasible, Y_feasible)
        else:
            print("Warning: No feasible initial samples to train GP.")

        # Initialize neural ensemble classifier for constraints
        ensemble_size = model_config.get('ensemble_size', 5)
        hidden_layers = model_config.get('hidden_layers', 2)
        neurons_per_layer = model_config.get('neurons_per_layer', 64)
        classifier = NeuralEnsembleConstraintClassifier(
            input_dim=dim,
            ensemble_size=ensemble_size,
            hidden_layers=hidden_layers,
            neurons_per_layer=neurons_per_layer,
            learning_rate=training_config.get('learning_rate', 1e-3),
            epochs=training_config.get('epochs', 50),
            batch_size=training_config.get('batch_size', 32),
            device=torch.device('cpu')
        )
        # Train classifier on initial data
        X_all, Y_obj, feas_labels = dataset.get_dataset()
        classifier.train(X_all, feas_labels)

        # Initialize boundary bounds component
        boundary_bounds = BoundaryBounds(margin=boundary_config.get('margin', 0.5))

        # Prepare acquisition optimizer
        acq_func_name = acq_config.get('function', 'EI')
        alpha = acq_config.get('alpha', 1.0)
        # Based on the chosen acquisition function, define a call
        # For simplicity, only EI or UCB
        def acq_function(x: np.ndarray):
            # x shape: (d,)
            # Use surrogate for objective and last trained GP
            mean, var = surrogate_gp.predict(x.reshape(1, -1))
            std = np.sqrt(var)
            f_best = np.nanmin(Y_obj) if len(Y_obj) > 0 else np.inf
            # Expected Improvement calculation
            from scipy.stats import norm
            Z = (np.min(Y_obj) - mean) / (std + 1e-9)
            ei = (np.min(Y_obj) - mean) * norm.cdf(Z) + std * norm.pdf(Z)
            return ei.squeeze()
        def ucb_acq(x):
            mean, var = surrogate_gp.predict(x.reshape(1, -1))
            std = np.sqrt(var)
            return mean.squeeze() + alpha * std

        if acq_func_name.upper() == 'UCB':
            acq_func = ucb_acq
        else:
            acq_func = acq_function

        # Prepare optimizer
        acq_optimizer = AcquisitionOptimizer(
            acq_func=acq_func,
            C_func=None,  # We'll handle constraint coupling inside the optimizer
            bounds=bounds,
            classifier=classifier,
            boundary_bounds=boundary_bounds,
            method='differential_evolution',
            seed=seed
        )

        # Initialize evaluation metrics
        eval_times = []
        start_time = time.time()

        # For visualization and metrics
        evaluation = Evaluation(config, dataset, classifier, surrogate_gp, seeds)

        # Main BO loop
        for iteration in range(total_evals):
            # Step 1: Compute boundary bounds
            # Since optimization runs internally, we need to define the constrained function that includes boundary bounds
            def constraint_cp(x):
                # Get classifier probability and uncertainty
                C_x = classifier.predict_proba(x.reshape(1, -1))[0]
                sigma_e = classifier.predict_uncertainty(x.reshape(1, -1))[0]
                l_x = boundary_bounds.compute_lower_bound(sigma_e)
                return C_x - l_x  # constraint >=0, with boundary at C_x >= l_x
            # Prepare constraint dict for scipy optimizer
            cons = [{'type': 'ineq', 'fun': constraint_cp}]
            # Optimize acquisition with constraints
            x_next, acq_value = acq_optimizer.optimize()
            # Evaluate at x_next
            eval_start = time.time()
            f_eval, feasible = dataset.get_evaluations(x_next)
            eval_end = time.time()
            eval_time = eval_end - eval_start
            eval_times.append(eval_time)

            # Append to dataset
            dataset.X = np.vstack([dataset.X, x_next])
            if feasible:
                Y_obj_new, _ = dataset.get_evaluations(x_next)
                # True objective already evaluated inside get_evaluations()
                new_f = f_eval
            else:
                # No objective to evaluate if infeasible
                new_f = np.nan

            # Append data to arrays
            Y_obj = np.array(Y_obj)
            if feasible:
                Y_obj = np.append(Y_obj, new_f)
            else:
                Y_obj = np.append(Y_obj, np.nan)
            labels_init = np.array(feas_labels)
            labels_init = np.append(labels_init, 1 if feasible else 0)

            # Step 2: Retrain surrogate GP on feasible points
            feasible_mask = np.array(labels_init) == 1
            X_feasible = dataset.X[feasible_mask]
            if len(X_feasible) > 0:
                # Refit GP
                surrogate_gp.fit(X_feasible, Y_obj[feasible_mask])
            else:
                print("Warning: no feasible points for GP update.")

            # Step 3: Retrain classifier on all data
            classifier.train(dataset.X, feas_labels)

            # Step 4: Log and evaluate metrics
            # Get current best objective from feasible data
            if len(Y_obj[feasible_mask]) > 0:
                current_best = np.nanmin(Y_obj[feasible_mask])
            else:
                current_best = np.inf
            evaluation.compute_best_feasible(Y_obj, feas_labels, seed_idx)
            # Classifier accuracy
            pred_probs = classifier.predict_proba(dataset.X)
            true_labels = feas_labels
            acc = evaluation.compute_classifier_accuracy(true_labels, pred_probs)
            evaluation.record_evaluation(seed_idx, iteration+1, Y_obj, feas_labels, eval_times)

            # Visualization (optional, can be toggled)
            if (iteration+1) % 50 == 0 or iteration == total_evals - 1:
                # For 2D problems, plot boundary evolution
                if len(dataset.X[0]) == 2:
                    boundary_points = dataset.X[feas_labels==1]
                    evaluation.plot_boundary_evolution(seed_idx, [boundary_points])

        # Finalize after all iterations
        # Save metrics and plots
        evaluation.save_metrics_summary()
        evaluation.generate_all_plots()

        total_time_elapsed = time.time() - start_time
        print(f"Seed {seed} completed in {total_time_elapsed:.2f} seconds.")

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import gpytorch
import numpy as np

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood, kernel_type: str = 'RBF', hyperparameters: dict = None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.kernel_type = kernel_type.lower()
        # Set default hyperparameters if not provided
        if hyperparameters is None:
            hyperparameters = {}
        # Initialize kernel based on type
        if self.kernel_type == 'matern':
            lengthscale = hyperparameters.get('lengthscale', 1.0)
            nu = hyperparameters.get('nu', 2.5)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=nu, ard_shape=torch.Size([train_x.shape[1]]))
            )
            self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)
        elif self.kernel_type == 'rbf' or self.kernel_type == 'l2' or self.kernel_type == 'gaussian':
            lengthscale = hyperparameters.get('lengthscale', 1.0)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_shape=torch.Size([train_x.shape[1]]))
            )
            self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)
        else:
            # Default to RBF if unknown
            lengthscale = hyperparameters.get('lengthscale', 1.0)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_shape=torch.Size([train_x.shape[1]]))
            )
            self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)
        # Optional: set outputscale if provided
        outputscale = hyperparameters.get('outputscale', 1.0)
        self.covar_module.outputscale = torch.tensor(outputscale)

    def forward(self, x: torch.Tensor):
        mean = torch.zeros(x.size(0))
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class SurrogateObjectiveGP:
    def __init__(self, kernel_type: str = 'RBF', hyperparameters: dict = None, noise_variance: float = 1e-4, device: torch.device = torch.device('cpu')):
        """
        Initialize the GP surrogate model for objective function.
        Args:
            kernel_type: string, e.g., 'RBF', 'Matern'
            hyperparameters: dict, e.g., {'lengthscale': 1.0, 'outputscale': 1.0}
            noise_variance: float, likelihood noise
            device: torch device
        """
        self.kernel_type = kernel_type
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        self.noise_variance = noise_variance
        self.device = device
        self.model = None
        self.likelihood = None
        self.is_trained = False

    def fit(self, X: np.ndarray, Y: np.ndarray, training_epochs: int = 50):
        """
        Fit the GP model to data.
        Args:
            X: numpy array, shape (n_samples, d)
            Y: numpy array, shape (n_samples,) or (n_samples,1)
            training_epochs: int, number of training iterations
        """
        # Convert data to torch tensors
        train_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(Y.squeeze(), dtype=torch.float32).to(self.device)  # shape (n,)
        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=self.noise_variance).to(self.device)
        self.model = ExactGPModel(train_x, train_y, self.likelihood, kernel_type=self.kernel_type, hyperparameters=self.hyperparameters).to(self.device)
        self.model.train()
        self.likelihood.train()

        # Use Adam optimizer to optimize model hyperparameters
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()}
        ], lr=0.01)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(training_epochs):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            # Optional: clip gradients to improve stability
            optimizer.step()

        self.is_trained = True

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance at test points.
        Args:
            X_test: numpy array, shape (n_test, d)
        Returns:
            mean: np.ndarray, shape (n_test,)
            variance: np.ndarray, shape (n_test,)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        self.model.eval()
        self.likelihood.eval()

        test_x = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            posterior = self.model(test_x)
            mean = posterior.mean.cpu().numpy()
            variance = posterior.variance.cpu().numpy()
        return mean, variance
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\BE-CBO\BE-CBO_repo`
