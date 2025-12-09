# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here's a comprehensive, step-by-step plan for reproducing the Boundary Exploration for Bayesian Optimization with Unknown Constraints (BE-CBO) approach described in the paper. This roadmap covers all key aspects needed to guide an implementation, including understanding the core methodology, datasets, experimental setup, hyperparameters, and evaluation metrics.

---

## 1. Understanding the Methodology of BE-CBO

### Core Principles:
- **Optimization Problems with Unknown Feasibility and Objective:**  
  - Goal: Minimize \(f(x)\) s.t. \(c(x) = 1\) (feasibility). The constraints \(c(x)\) are unknown binary functions.
  - Challenge: Function evaluations are expensive; feasibility and objective are unknown and often only observed if the sample is feasible.

- **Boundary-Limited Optimization Focus:**  
  - Optimal solutions often lie on the boundary between feasible and infeasible regions.
  - Efficient exploration involves explicitly modeling and actively exploring this boundary.

- **Constraint Modeling with Neural Ensembles:**  
  - Use an ensemble of neural networks (Deep Ensembles) instead of Gaussian Processes for better modeling of complex, potentially non-smooth boundaries, especially in high dimensions or with complex data.
  - The ensemble provides a predictive probability (\(C(x)\)) for feasibility.

- **Surrogate Objective Model:**  
  - Model the objective \(f(x)\) using a Gaussian Process (GP) for exploiting the acquisition functions' properties (e.g., Expected Improvement).

- **Boundary Exploration Strategy:**  
  - Query points around the boundary where the feasibility probability is close to 0.5, dictated dynamically by the classifier's uncertainty.
  - Use a bound \(l(x) = 0.5 - \sigma_E(x)\), where \(\sigma_E(x)\) is the ensemble's standard deviation in feasibility probability prediction.
  - Search for the optimum on the boundary (where model uncertainty is high).

- **Acquisition Function and Optimization:**
  - Use Expected Improvement (EI) over the objective \(f(x)\).
  - Incorporate feasibility constraints via the classifier’s probabilistic output \(C(x)\), and optimize:
    \[
    \max_x q(x) \quad \text{s.t.} \quad C(x) \geq 0.5 - \sigma_E(x)
    \]
    and
    \[
    u(x) = 1
    \]
  - The boundary bound \(l(x)\) acts as a lower bound for feasible/infeasible exploration.

- **Constraint Coupling Variants:**
  - Propose variants: naïve (fixed bounds), boundary-based, and with different acquisition functions (EI or UCB).

### Summary of the Implementation:
- Surrogate for \(f(x)\): GP trained on feasible points.
- Classifier for \(c(x)\): Deep ensemble of neural networks.
- Active boundary sampling: Focus near the boundary with an uncertainty-based dynamic bound.
- Optimization: Maximize EI constrained with \(C(x)\) and the boundary bound \(l(x)\).

---

## 2. Dataset and Benchmark Preparation

### Synthetic Functions:
- 2D Townsend, Simionescu, LSQ functions.
- 9D Ackley functions.
- 10D synthetic tasks (e.g., Ackley, multi-product batch plant, SOPWM, etc.).
- 30D Ackley or similar high-dimensional functions.
- Additional shifted or modified functions for robustness testing.

### Real-World Benchmarks:
- 2D to 30D problems:  
  - Mechanical: Three-bar truss, welded beam, pressure vessel, cantilever beam.  
  - Engineering: Gas system, gear train, bearing design, etc.  
  - Dataset: For each, extract the design variables, known bounds, and the form of the objective and constraints.  
- For each problem:
  - Record:
    - Dimensionality \(d\),
    - Unknown constraints \(g\),
    - Number of active constraints \(g^*\),
    - True optimal objective \(f^*\),
    - Feasible region geometry (if available).

### Dataset Requirements:
- Initial samples:
  - Randomly generate (or use Latin Hypercube Sampling or Sobol sequences) in the feasible region bounds.
  - Evaluate only feasible points; infeasible points observed via no objective values.
- For synthetic:
  - Generate samples inside the domain; evaluate \(f(x)\) and \(c(x)\) directly.
- For real-world:
  - Use existing evaluation functions or simulate evaluations as per the benchmark definitions.

---

## 3. Experimental Setup & Hyperparameters

### Initialization:
- Number of initial random samples: 10 (standard) or as per the paper.
- Number of total evaluations: 200 (or datasets with similar budget constraints).
- For ensemble:
  - Number of neural networks: 5 (default), optionally test variants (3, 8, 128).
  - Each network trained independently.
  - Ensemble training: on initial feasible points; update after each batch.

### Neural Ensemble:
- Architecture:
  - Input dimension \(d\),
  - Hidden layers: 1–4 (or as per section 4.2.4),
  - Number of neurons per layer (64 as default), or varied.
- Training:
  - Epochs: sufficient until convergence or a fixed number (e.g., 50–100),
  - Learning rate: test values (1e-3, 1e-4, 1e-5 as per section 4.2.5),
  - Optimizer: Adam or SGD,
  - Loss: binary cross-entropy (for feasibility prediction).
- Uncertainty:
  - Ensemble variance provides \(\sigma_E(x)\) for boundary bounds.

### Surrogate Model for Objective:
- Gaussian Process:
  - Kernel: RBF (squared exponential) or Matérn (section 4.1),
  - Hyperparameters: tune via marginal likelihood,
  - Noise level: small (e.g., 1e-6) or tuned.
- Update: after each batch or evaluation.

### Acquisition Optimization:
- Maximize constrained EI:
  - Use global optimizers like SLSQP, L-BFGS-B with boundary bounds.
  - Boundary bounds \(l(x)\): dynamic based on ensemble uncertainty.

### Stopping Rules:
- Max evaluations (200),
- No improvement over last 10 iterations,
- or convergence of feasible optimum region.

---

## 4. Implementation Details & Procedure

### Algorithm Loop:
1. **Initial Sampling:**
   - Generate an initial seed set (size 10),
   - Evaluate \(f(x)\) (if feasible) and \(c(x)\) (binary, feasible or infeasible).

2. **Surrogate Fitting:**
   - Fit GP for \(f(x)\) on feasible points.
   - Fit neural ensemble classifier on labels \(c(x)\).

3. **Boundary Bound Computation:**
   - For each candidate \(x\), compute \(C(x)\),
   - Calculate \(\sigma_E(x)\),
   - Set bounds \(l(x) = 0.5 - \sigma_E(x)\),
   - Set \(u(x) = 1\).

4. **Acquisition Function Optimization:**
   - Discretize, Latin Hypercube Sample, or use multi-start global optimization,
   - Maximize EI constrained to \(C(x) \ge 0.5 - \sigma_E(x)\),
   - Respect input bounds.

5. **Select Next Sample:**
   - Evaluate at the candidate \(x_{\text{next}}\),
   - Observe \(f(x_{\text{next}})\) if feasible,
   - Observe \(c(x_{\text{next}})\).

6. **Update Models:**
   - Add new points to the dataset,
   - Retrain GP and neural ensemble classifier,
   - Repeat until evaluation limit.

---

## 5. Evaluation Metrics and Results Collection

- **Optimization Performance:**
  - Track best feasible \(f(x)\),
  - Plot the best value vs. number of evaluations,
  - Record mean and standard deviation over 10 seeds.
  
- **Constraint Modeling Quality:**
  - Use Balanced Accuracy (BACC):  
    \[
    \frac{TPR + TNR}{2}
    \]
  - Record and plot accuracy over iterations.

- **Runtime and Efficiency:**
  - Time per iteration (proposal + evaluation),
  - Total runtime comparison.

- **Ablation Studies:**
  - Vary ensemble size,
  - Vary number of hidden layers/neurons,
  - Vary learning rate,
  - Use different acquisition functions (EI vs UCB),
  - Vary ensemble vs GP classifier performance.

- **Qualitative Visualization:**
  - Plot feasible/infeasible points, true boundary contours, and identified boundary (Figures 4, 7, 8, 10, 11, 14, 15, 16, 17, 19, 21).

---

## 6. Additional Considerations

- **Parallel Evaluations:**
  - Optional: implement batch sampling,
  - Update ensemble and GP models asynchronously.

- **Reproducibility:**
  - Set random seeds explicitly,
  - Log all hyperparameters and trained model configurations,
  - Restart from initial seed to validate.

- **Implementation Details Not Explicitly Provided:**
  - Python: NumPy, SciPy, scikit-learn, GPflow or GPyTorch, PyTorch/TensorFlow (for neural ensembles),
  - Optimization routines: scipy.optimize, or custom multi-start,
  - Data handling: pandas or NumPy arrays,
  - Visualization: Matplotlib, seaborn.

---

## 7. Summary of the Roadmap (Checklist)
- [ ] Reproduce synthetic benchmarks (Townsend, Simionescu, LSQ, Ackley functions).
- [ ] Load or formulate real-world benchmarks with specified bounds.
- [ ] Initialize with Latin Hypercube Sampling.
- [ ] Implement neural ensemble for constraint classification:
  - Architecture, training, uncertainty estimation.
- [ ] Model objective with GP.
- [ ] Compute dynamic bounds \(l(x)\) as \(0.5 - \sigma_E(x)\).
- [ ] Optimize constrained EI (or UCB) with boundary bounds.
- [ ] Collect performance and accuracy metrics.
- [ ] Conduct ablation and sensitivity studies for ensemble size, architecture, learning rates.
- [ ] Visualize boundary discovery and optimization trajectories.
- [ ] Save experiment logs, parameters, and models for reproducibility.

---

This roadmap ensures a detailed, systematic approach for implementing BE-CBO, from understanding the core ideas to setting up experiments and evaluation, paving the way for reproducible and validated results.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will develop a modular Python system that implements BE-CBO, utilizing open-source libraries: GPyTorch for GP models, PyTorch for neural ensemble classifier, SciPy for optimization, and NumPy/SciPy for data handling. The system will encompass data handling, model training, acquisition optimization, and experiment orchestration. The main loop will iteratively update the GP for the objective, the neural ensembles for constraints, and perform constrained maximum EI with active boundary bounds, focusing sampling near the estimated boundary, until evaluation budget is exhausted.",
    "File list": [
        "main.py",  
        "dataset_loader.py", 
        "surrogate_gp.py",     
        "constraint_classifier.py",  
        "acquisition_optimizer.py",  
        "boundary_bounds.py",  
        "utils.py",  
        "visualization.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run_experiment()
    }
    class DatasetLoader {
        +__init__(bounds: dict, init_samples: int)
        +initialize_samples() -> Tuple[np.ndarray, np.ndarray]
        +get_evaluations(x: np.ndarray) -> Tuple[float, bool]
    }
    class SurrogateGPModel {
        +__init__(kernel_type: str, hyperparameters: dict)
        +fit(X: np.ndarray, y: np.ndarray) -> None
        +predict(x: np.ndarray) -> Tuple[float, float]
    }
    class ConstraintClassifier {
        +__init__(ensemble_size: int, architecture: dict)
        +train(X: np.ndarray, labels: np.ndarray) -> None
        +predict_proba(x: np.ndarray) -> float
        +predict_uncertainty(x: np.ndarray) -> float
    }
    class AcquisitionOptimizer {
        +__init__(acq_func: callable, bounds: dict)
        +maximize(acq_func: callable, C_func: callable, bounds: dict, boundary_bound: float) -> np.ndarray
    }
    class BoundaryBounds {
        +compute_lower_bound(proba_uncertainty: float) -> float
        +compute_bounds(proba: float, uncertainty: float) -> Tuple[float, float]
    }
    class Experiment {
        +__init__(dataset: DatasetLoader, gp_model: SurrogateGPModel, classifier: ConstraintClassifier, optimizer: AcquisitionOptimizer, boundary_bounds: BoundaryBounds, total_evals: int)
        +execute() -> dict
    }
    Main --> DatasetLoader
    Main --> SurrogateGPModel
    Main --> ConstraintClassifier
    Main --> AcquisitionOptimizer
    Main --> BoundaryBounds
    Main --> Experiment
    SurrogateGPModel --> DatasetLoader
    ConstraintClassifier --> DatasetLoader
    AcquisitionOptimizer --> BoundaryBounds
    Experiment --> [DatasetLoader, SurrogateGPModel, ConstraintClassifier, AcquisitionOptimizer, BoundaryBounds]
",
    "Program call flow": "
sequenceDiagram
    participant M as Main
    participant DL as DatasetLoader
    participant SG as SurrogateGPModel
    participant CC as ConstraintClassifier
    participant AO as AcquisitionOptimizer
    participant BB as BoundaryBounds
    participant EXP as Experiment
    M->>DL: initialize_samples()
    DL-->>M: initial_samples_X, initial_samples_y
    M->>SG: fit(initial_samples_X, initial_samples_y)
    SG-->>M: trained
    M->>CC: train(initial_samples_X, labels(initial_samples))
    CC-->>M: trained
    M->>EXP: execute()
    EXP->> DL: get_evaluations(x)
    DL-->>EXP: value, feasibility
    EXP->>SG: predict(x) // for objective acquisition
    SG-->>EXP: mean, variance
    EXP->>CC: predict_proba(x)
    CC-->>EXP: probability (feasibility)
    EXP->>BB: compute_bounds(CC.predict_uncertainty(x))
    BB-->>EXP: current boundary bounds
    EXP->>AO: maximize(acq_func, C_func, bounds, boundary_bound)
    AO-->>EXP: candidate points
    loop until total evaluations reached
        EXP->>DL: get_evaluations(candidate)
        DL-->>EXP: value, feasibility
        EXP->>SG: update with new data
        EXP->>CC: update with new labels
        EXP->>BB: update boundaries
    end
    EXP-->>Main: report results (best feasible objective, boundary visualization, etc.)
",
    "Anything UNCLEAR": "Clarification needed on dataset format for real-world benchmarks, particularly the exact input variables, bounds, and optional data pre-processing steps. Also, further detail on how to handle integer vs continuous variables within the optimization routines would help."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "scipy==1.7.0",
        "torch==1.9.0",
        "gpytorch==1.1.0",
        "scikit-learn==0.24.2",
        "matplotlib==3.4.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required for non-Python components"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class responsible for generating initial samples, loading datasets, and evaluating samples for both synthetic and real-world benchmark problems. Depends on numpy and scipy for sampling. Inputs: bounds, initial sample count. Outputs: arrays of samples, evaluations."
        ],
        [
            "model.py",
            "Defines SurrogateObjectiveGP class encapsulating GPyTorch-based GP model for the objective function. Needs methods for fitting, predicting mean/variance. Depends on torch, gpytorch. Initialized with kernel type, hyperparameters."
        ],
        [
            "constraint_classifier.py",
            "Defines NeuralEnsembleConstraintClassifier class managing an ensemble of PyTorch networks for constraint modeling. Methods: train, predict_proba, predict_uncertainty. Depends on torch, torch.nn, training data, ensemble size, architecture."
        ],
        [
            "acquisition_optimizer.py",
            "Provides constrained maximization of EI (or alternative acquisition functions) using SciPy’s minimize or differential evolution, respecting boundary bounds and active boundary bounds from boundary_bounds.py."
        ],
        [
            "boundary_bounds.py",
            "Provides BoundaryBounds class with methods to compute the dynamic lower bound l(x) = 0.5 - sigma_E(x), and bounds for sampling around the boundary based on classifier uncertainty. Depends on constraint classifier ensemble."
        ],
        [
            "main.py",
            "Main orchestrator script that initializes all components, runs the iterative Bayesian optimization loop, updates models, manages evaluation budget, and handles visualization. Coordinates data loading, model training, acquisition, exploitation, and evaluation."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class to compute and track optimization metrics: best feasible value, classifier accuracy, runtime, and produces visualization plots for boundary discovery, convergence, and comparison studies."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "constraint_classifier.py",
        "boundary_bounds.py",
        "acquisition_optimizer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0 ... (not applicable, system is internal; if needed can define REST API for experiment control and result reporting)",
    "Shared Knowledge": "The system shares functions for random sampling, Gaussian Process hyperparameter tuning, neural network training routines, and evaluation metrics. These should be implemented as utility modules or functions called from respective classes.",
    "Anything UNCLEAR": "Require clarification on the exact format of dataset inputs for real-world benchmarks, and whether certain components (e.g., constraint models for binary classification) require specialized training procedures or data considerations."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.001  # Default to 1e-3; tune as needed based on model convergence
  batch_size: 32        # Typical size for neural ensemble training
  epochs: 50            # Fixed number of epochs for ensemble training
model:
  ensemble_size: 5      # Number of neural networks in the ensemble
  hidden_layers: 2      # Number of hidden layers in each neural network
  neurons_per_layer: 64 # Number of neurons in each hidden layer
optimizer:
  type: Adam            # Optimizer type for neural network training
  learning_rate: 0.001  # Default to 1e-3; can be adjusted
acquisition:
  function: EI        # Expected Improvement (EI) or can be switched to UCB
  alpha: 1.0          # UCB parameter if UCB used, ignored if EI
boundary:
  margin: 0.5        # Base margin, dynamically adjusted by classifier uncertainty (sigma_e)
evaluation:
  total_evaluations: 200  # Total function evaluations per experiment
  initial_samples: 10     # Number of initial random samples
  seeds: [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]
```

---

**Note:** Full configuration is available in `planning_config.yaml`
