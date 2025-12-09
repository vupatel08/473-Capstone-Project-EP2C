# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## acquisition_optimizer.py

# Logic Analysis for acquisition_optimizer.py

This module is designed to perform constrained maximization of acquisition functions—specifically, EI (Expected Improvement) or alternatives such as UCB—while respecting dynamic boundary bounds derived from the classifier’s uncertainty estimates. The core functionality involves optimizing an acquisition function with constraints that couple model predictions (probability of feasibility) and boundary limits, ensuring sampling focuses on promising regions near the estimated boundary.

Below is a detailed, structured breakdown of the core logic, inputs, processes, constraints, and outputs necessary for implementing this module, all aligned with the overarching BE-CBO framework, the design specifications, and the provided configuration.

---

## 1. Purpose and Scope
- **Main task:** Maximize a specified acquisition function \( q(x) \) over the design space \( \mathcal{X} \), subject to constraints involving the classifier’s predicted probability \( C(x) \) and boundary bounds \( l(x) \).
- **Constraints:**  
  - Probabilistic feasibility constraint: \( C(x) \geq l(x) = 0.5 - \sigma_E(x) \)  
  - Variable bounds: bounds on \( x \) as per problem setup  
- **Optimization methods:**  
  - Global methods like differential evolution (commonly) or local methods such as SLSQP or L-BFGS-B for refinement, both capable of handling nonlinear constraints.

## 2. Inputs

### Required Inputs:
- **acq_func:** Callable function that computes the acquisition value \( q(x) \), given \( x \).  
- **C_func:** Callable function to evaluate the feasibility probability \( C(x) \), given \( x \) and the classifier.  
- **bounds:** Dict or structure containing per-variable minimum and maximum bounds, e.g., `'x_min'`, `'x_max'` or a list of tuples `(lower_bound, upper_bound)` for each variable, aligned with the problem dimension. (Derived from the JSON design, config.yaml, or problem bounds)  
- **boundary_bounds:** Instance of BoundaryBounds class which provides methods:
  - `compute_lower_bound(proba_uncertainty)` — compute a dynamic lower bound \( l(x) \) based on classifier uncertainty \(\sigma_E(x)\).
- **classifier:** Constraint classifier object (Neural Ensembling) capable of returning \( C(x) \) and the uncertainty \( \sigma_E(x) \).

### Optional Inputs:
- **initial_points:** Optional starting points for local search or to define the initial population for global algorithms, if applicable.

## 3. Outputs
- **best_x:** The sample point \( x^* \in \mathcal{X} \) that maximizes the constrained acquisition.  
- **optimization_details:** Data about the optimization process—e.g., optimal value \( q(x^*) \), the constraint satisfaction, convergence info, number of evaluations, etc.

## 4. Process and Logic

### Step 1: Preparation
- **Define constraint functions:**
  - **Feasibility constraint:**  
    \[
    g_1(x) = C(x) - l(x) \geq 0
    \]
    where \( l(x) = 0.5 - \sigma_E(x) \), with \( \sigma_E(x) \) provided by the classifier's uncertainty.  
  - **Variable bounds:** directly from `bounds` (list/tuple of bounds per variable).  
- **Set optimization options:**  
  - Method details, e.g., `method='SLSQP'` or `'L-BFGS-B'`  
  - Tolerance levels, maximum iterations, population size if DE.

### Step 2: Constraint Handling
- **Define nonlinear constraint functions for optimizer:**
  - For `scipy.optimize.minimize` (or `differential_evolution`):  
    - Constraints should be provided in a form compatible with scipy:
      - For `SLSQP`: list of dicts, each with `'type':'ineq'`, `'fun': constraint_function`  
      - For `differential_evolution`: `constraints=` parameter accepting similar definitions (if supported).
      
- **Constraint function \( g_1(x) \):**  
  Returns \( C(x) - l(x) \). Note that \( C(x) \) uses the classifier, and \( l(x) \) depends on classifier uncertainty via `boundary_bounds.compute_lower_bound()`.

### Step 3: Optimization Strategy
- **Choose optimization method:**
  - **Global search (recommended):** use `scipy.optimize.differential_evolution` because it handles nonlinear, bounded, and constrained optimization globally, beneficial for complex acquisition landscapes.  
  - **Local refinement:** optionally, perform multiple local optimizations (multi-start) initialized at promising points, or refine using `SLSQP`.
  
- **Initialization:**
  - Generate multiple starting points within bounds, possibly using Latin Hypercube sampling or uniform sampling.

- **Optimization loop:**
  - For each starting point:
    - Run the optimizer (DE or SLSQP) to maximize `-acq_func(x)` subject to constraints, since scipy’s minimize minimizes functions by default.
    - Keep track of the best feasible candidate and record the best acquisition value.
    
- **Iterate:**
  - Optionally, perform multiple runs with multiple initial points; select the overall best.

### Step 4: Boundary Constraints Implementation
- Implement the constraint \( g_1(x) \geq 0 \):
  - At each candidate \( x \), compute:
    - \( C(x) \): via classifier.predict_proba(x),
    - \( \sigma_E(x) \): via classifier.predict_uncertainty(x),
    - \( l(x) = boundary_bounds.compute_lower_bound(\sigma_E(x)) \).
  
  - Return \( g_1(x) = C(x) - l(x) \).

- These functions are used by the optimizer to restrict the feasible search region near the boundary where the uncertainty suggests proximity to the actual boundary.

### Step 5: Boundary Handling
- During optimization, the constraints ensure the sample remains on or close to the boundary:
  - Using the dynamic lower bound \( l(x) \), the optimizer samples points with the classifier’s uncertainty guiding exploration.
- The explicit dynamic boundary ensures exploration of uncertain boundary regions, balancing exploitation and boundary discovery.

### Step 6: Post-Processing
- After convergence:
  - Identify the candidate \( x^* \) with the highest \( q(x) \) that satisfies the constraints.
  - Return \( x^* \) and optional details (e.g., acquisition value, feasibility status).

## 5. Additional Requirements and Considerations

### Numerical Details:
- **Constraint tolerances:** specify small tolerances for feasibility.
- **Maximum iterations and function evaluations:** adjustable via function options.
- **Parallelization:** can run multiple initializations in parallel for efficiency.

### Robustness:
- Employ multiple random seeds for initial points.
- Monitor constraint satisfaction after each optimization run.
- Handle potential numerical issues or infeasible initial points by reinitializing.

### Compatibility:
- Ensure the API of `acq_func`, `C_func`, and `boundary_bounds` matches usage patterns.
- Encapsulate the boundary constraint logic in a dedicated function used for the optimizer.

---

## 6. Summary of Key Functions

- **`constrained_acquisition_maximize()`**:  
  - **Inputs:** acquisition, classifier, bounds, boundary bounds.  
  - **Outputs:** best \( x^* \), and optional metadata.  
  - **Flow:**
    - Generate initial points.
    - For each initial point, run optimization with constraints.
    - Track and select the best feasible point.
    - Return the global best.

- **`constraint_function(x)`**: returns \( C(x) - l(x) \), computed dynamically via classifier and boundary_bounds.

- **`optimization_method()`**: supports DE, SLSQP, or BFGS, selected per configuration.

---

## 7. Final Note
The module provides a flexible and robust way to perform constrained acquisition function maximization, emphasizing the explicit boundary constraint to guide exploration toward the boundary region where the global minimum is likely located, consistent with the insights and experimental design of BE-CBO.

This detailed logic forms the basis for implementing accurate, efficient, and boundary-aware constrained optimization within the broader Bayesian Optimization pipeline.

## boundary_bounds.py

# Boundary Bounds Logic Analysis

The `boundary_bounds.py` module provides the `BoundaryBounds` class, which is responsible for calculating the dynamic lower boundary \( l(x) \) for active boundary exploration and the associated sampling bounds that guide the search toward the boundary region where the optimal solutions are expected to lie. This class depends on the output of the constraint classifier ensemble and on the boundary exploration strategy described in the paper.

---

## Key Objectives and Responsibilities

1. **Compute the Dynamic Lower Bound \( l(x) \):**  
   - Based on the current estimate of the probability of feasibility \( C(x) \) (which is the mean of the neural ensemble predictions).  
   - Adjust the lower bound according to the classifier's uncertainty \( \sigma_E(x) \), enabling exploration that adapts to the confidence in the boundary estimate.

2. **Compute Constraints on Boundary Sampling:**  
   - For a given input \( x \), determine bounds within which samples are explored around the boundary—both into the feasible and infeasible regions.  
   - These bounds are based on the classifier's uncertainty and are used by the acquisition function optimizer to focus search where the boundary is most uncertain.

3. **Ensure Compatibility with the Broader BE-CBO Process:**  
   - Boundary bounds are integrated into the constrained optimization of the acquisition function (e.g., EI).  
   - Boundary bounds should be dynamically updated each iteration based on the classifier's uncertainty.

---

## Inputs

- **Classifier ensemble predictions:**
  - \( C(x) \): The mean predicted probability of feasibility for a given \( x \), typically in \( [0, 1] \).
  - \( \sigma_E(x) \): The standard deviation (uncertainty) of the ensemble's predictions at \( x \), reflecting confidence in the estimate of \( C(x) \).

- **Hyperparameters (from config.yaml):**
  - `margin` (default 0.5): The core probabilistic margin defining the boundary threshold.
  - The current uncertainty \( \sigma_E(x) \) modulates the lower boundary \( l(x) \).

---

## Outputs

- **Lower boundary \( l(x) \):**  
  - Calculated as \( l(x) = 0.5 - \sigma_E(x) \):  
    - This dynamically widens the exploration window into the infeasible region when uncertainty is high, encouraging exploration around the boundary.
    - As the classifier becomes more confident (\( \sigma_E(x) \to 0 \)), \( l(x) \to 0.5 \), focusing search nearer the boundary.

- **Boundary bounds for sampling:**
  - For a given \( C(x) \):  
    - If \( C(x) \ge l(x) \), sampling into the feasible region is encouraged, bounded by \( u(x) = 1 \).  
    - For exploration into the infeasible side, the lower bound can go down to \( l(x) \), or adjusted values based on the uncertainty, enabling the optimizer to probe regions close to the boundary.

---

## Implementation Details

### 1. Initialization
- The class is initialized with the `margin` parameter (default 0.5), which is the base boundary probability.
- It doesn't require initial data; it operates on the classifier's outputs during each iteration.

### 2. Methods
#### a. `compute_lower_bound(self, sigma_e)`
- **Purpose:** Calculate the dynamic lower bound \( l(x) \).
- **Input:**  
  - `sigma_e`: the uncertainty \( \sigma_E(x) \) at a given point \( x \).  
- **Output:**  
  - `l(x) = max(0, 0.5 - sigma_e)`: Ensure lower bound is in valid probability range.

#### b. `compute_bounds(self, C_x, sigma_e)`
- **Purpose:** For a particular \( x \), determine the feasible and infeasible bounds for constraint sampling based on the classifier's predicted probability \( C(x) \) and its uncertainty.
- **Logic based on classification:**  
  - **Upper bound \( u(x) \):** Set to 1 (normalized probability).  
  - **Lower bound \( l_{x} \):**  
    - At boundary: \( l(x) = \max(0, 0.5 - \sigma_E(x)) \).  
    - For sampling: Depending on the precision in modeling \( C(x) \), the bounds might be expanded or contracted, but the default is as specified.
- **Returns:**  
  - Tuple of `(lower_bound, upper_bound)`

### 3. Boundary Dynamics
- The approach guarantees that when classifier uncertainty is high — early training or boundary regions — \( l(x) \) is wider, encouraging exploration into the infeasible side, thus efficiently discovering the boundary.
- As uncertainty reduces, the boundary focus tightens (closer to \( 0.5 \)).

### 4. Integration with the Larger System
- During each BO iteration, after neural ensemble classification:
  - Compute \( \sigma_E(x) \) for candidate \( x \),
  - Calculate \( l(x) \),
  - Pass these bounds to the acquisition optimizer (e.g., constrained SLSQP),
  - The optimizer samples within these bounds, which dynamically adapt exploration based on classifier confidence.

---

## Summary of Core Functions

| Function | Purpose | Inputs | Outputs | Notes |
|------------|------------|---------|---------|--------|
| `compute_lower_bound(sigma_e)` | Calculate \( l(x) = 0.5 - \sigma_E(x) \) | `sigma_e`: float | float \( l(x) \) | Ensures non-negativity, clip to 0 if negative |
| `compute_bounds(C_x, sigma_e)` | Determine sampling bounds | `C_x`: predicted probability, `sigma_e`: uncertainty | `(lower_bound, upper_bound)` | Lower bound dynamically set by classifier uncertainty |
  
---

## Final Remarks
- The class design emphasizes modularity and ease of updates.
- The main input is the ensemble’s uncertainty prediction; the class then outputs the appropriate dynamic bounds for each candidate query point.
- These bounds are critical in steering the optimization process to explore boundary regions where the global optimum resides, as per the paper's insights.

---

This logic analysis guides the implementation of `BoundaryBounds` in `boundary_bounds.py`, ensuring it aligns with the methodology described in the paper and supports the active exploration strategy.

## constraint_classifier.py

### Logic Analysis for `constraint_classifier.py`

**Objective:**  
Implement the `NeuralEnsembleConstraintClassifier` class that manages an ensemble of neural networks (MLPs) for modeling the unknown binary feasibility constraints \( c(x) \) in Bayesian Optimization. The ensemble provides a probabilistic estimate of feasibility \( C(x) \), as well as a measure of uncertainty, which is critical for active boundary exploration.

---

### Core Components and Responsibilities:

1. **Initialization (`__init__`)**:
    - Accepts parameters: ensemble size, neural network architecture (number and size of hidden layers), training hyperparameters (learning rate, epochs, batch size).
    - Sets up internal data structures to hold multiple neural networks.
    - Uses `torch.nn.Module` to define a neural network class with specified architecture.
    - Each network should have a separate set of weights, initialized independently.
  
2. **Model Architecture**:
    - Input layer size: dictated by the feature dimension \( d \) (number of variables in the design space).
    - Hidden layers: number as defined (`hidden_layers`), each with `neurons_per_layer` neurons.
    - Activation: ReLU between layers.
    - Output layer: single neuron with sigmoid activation, outputting a probability \( p \in [0,1] \), representing the feasibility likelihood.
    - Multiple networks (the ensemble): independent instances trained separately for diversity.

3. **Training (`train`)**:
    - Inputs:
        - Training data: `X` (samples, features), shape: `(num_samples, d)`.
        - Labels: `y` (binary: 1 for feasible, 0 for infeasible), shape: `(num_samples,)`.
    - Procedure:
        - For each model in the ensemble:
          - Initialize model weights independently.
          - Use binary cross-entropy loss (`torch.nn.BCEWithLogitsLoss`) for numerical stability.
          - Use the specified optimizer (`Adam`, per configuration) with the specified learning rate.
          - Run for the fixed `epochs`.
          - Data should be fed in mini-batches (`batch_size`).
        - Optional: shuffle dataset each epoch.
    - Additional:
        - Save trained model parameters for inference.
        - No regularization or dropout needed (per paper), but optional dropout could be added for regularization.

4. **Prediction (`predict_proba`)**:
    - Input: `x`, a feature vector or batch of samples.
    - Process:
        - For each neural network in ensemble:
            - Set in evaluation mode.
            - Forward pass the input to obtain logits.
            - Transform logits with sigmoid to get probabilities.
        - Aggregate:
            - Compute mean probability across models: \(\mu_E(x) = \frac{1}{N} \sum_{i=1}^{N} p_i(x)\).
        - Output:
            - The average probability: an estimate of the probability that `x` is feasible.
    - Note:
        - For batch: process all samples through each model, then average.

5. **Uncertainty Estimation (`predict_uncertainty`)**:
    - Input: `x`, single sample or batch.
    - Process:
        - For each network, get predicted probability \( p_i(x) \).
        - Compute mean \(\mu_E(x)\) as above.
        - Compute variance:
            \[
            \sigma_E^2(x) = \frac{1}{N-1} \sum_{i=1}^N (p_i(x) - \mu_E(x))^2
            \]
        - The standard deviation: \(\sigma_E(x) = \sqrt{\sigma_E^2(x)}\).
    - Output:
        - The uncertainty measure \(\sigma_E(x)\), used to dynamically set the boundary bounds \( l(x) = 0.5 - \sigma_E(x) \).

6. **Model Update (after new data)**:
    - When new labeled data (`X`, `labels`) is available:
        - Call `train()` on the dataset.
        - Retrain all models from scratch or via fine-tuning (preferably from scratch for stability).

---

### Implementation Details and Considerations:

- **Data Types & Shapes**:
  - Inputs:
    - `X`: numpy array or torch tensor, shape `(num_samples, d)`.
    - `y`: numpy array or torch tensor, shape `(num_samples,)`.
  - Model outputs:
    - Probabilities: shape `(num_samples, 1)`.
  
- **Device Management**:
  - Use GPU if available (`torch.device`), else CPU.
  - Efficient inference: batch process during predictions.

- **Hyperparameters and Defaults**:
  - Ensemble size (from config): 5.
  - Hidden layers: 2.
  - Neurons per layer: 64.
  - Learning rate: 0.001 (from config).
  - Epochs: 50.
  - Batch size: 32.

- **Training Stability**:
  - Early stopping not strictly necessary but could be added based on validation split or loss plateaus.
  - Use `model.train()`/`model.eval()` appropriately.
  - Save each model’s state dictionary for inference.

- **Numerical Stability & Performance**:
  - Use `BCEWithLogitsLoss` combined with sigmoid during prediction for better stability.
  - Aggregate predictions across the ensemble efficiently.

- **Ensemble Variance & Uncertainty**:
  - Variance of the outputs across models reflects epistemic uncertainty regarding the feasibility.
  - This uncertainty guides the boundary exploration via the dynamic margin \( l(x) = 0.5 - \sigma_E(x) \).

---

### Summary:
This `NeuralEnsembleConstraintClassifier` class manages multiple independently trained neural network classifiers, each predicting feasibility probability, and provides methods:
- `train(X, labels)`: trains all ensemble members.
- `predict_proba(x)`: returns the mean feasibility probability.
- `predict_uncertainty(x)`: computes the standard deviation of ensemble predictions to gauge uncertainty.

This design emphasizes robustness, flexibility, and computational efficiency, aligning with the boundary exploration strategy critical to the BE-CBO algorithm.

---

**Note:**  
Ensure that model training and inference can handle batch inputs, and properly initialize models with independent weights. Also, this module interfaces with other components (e.g., `boundary_bounds.py`) for uncertainty-based boundary bounds, so the interface should be consistent and compatible with the overall system.

## dataset_loader.py

**Logic Analysis for dataset_loader.py**

---

### Purpose:
The `dataset_loader.py` module defines the `DatasetLoader` class, responsible for generating initial samples, loading or creating datasets, and evaluating sample points for both synthetic and real-world benchmark problems. It should support flexible dataset creation, sampling strategies, and evaluation procedures compatible with the experimental framework described in the paper.

---

### Core Responsibilities:
1. **Initialization:**
   - Input:
     - `bounds`: a dictionary defining variable bounds for each dimension (including data type considerations for integer/continuous variables).
     - `init_samples`: number of initial samples to generate.
   - Actions:
     - Store the bounds.
     - Generate initial sample points within bounds.

2. **Sample Generation:**
   - Method:
     - `initialize_samples() -> Tuple[np.ndarray, np.ndarray]`
       - Generate `init_samples` points uniformly (e.g., Latin Hypercube Sampling or quasi-random Sobol sequences) within the variable bounds.
     - For synthetic functions, generate samples purely within the bounds.
     - For real-world datasets:
       - Either load from pre-existing datasets or generate via sampling methods inside specified bounds.
       - Handle continuous versus integer variables accordingly.
   
3. **Sample Evaluation:**
   - Method:
     - `get_evaluations(x: np.ndarray) -> Tuple[float, bool]`
       - For *synthetic functions*:
         - Evaluate the true objective function `f(x)` using the known analytical form.
         - Evaluate the constraints `c(x)`, returning feasibility (boolean: `True` if feasible, `False` if infeasible).
       - For *real-world benchmark problems*:
         - Call the specific evaluation function or interface (possibly a simulation or surrogate model).
         - Determine feasibility based on the constraints' evaluation.
         - Note: when an evaluation determines infeasibility *before* measuring the objective, return no objective value or a special indicator (e.g., None or np.nan).
   
4. **Data Storage:**
   - Maintain internal data structures:
     - `X`: array of evaluated sample points (shape: n_eval x d)
     - `Y`: array of results, which can be:
       - Objective values for feasible points
       - Feasibility labels (binary) for all evaluated points
   - Update with new evaluations during Bayesian Optimization iterations.

5. **Support for Different Problem Types:**
   - Synthetic benchmarks:
     - Pre-coded functions: Townsend, Simionescu, LSQ, Ackley functions.
     - Functions are fully known; evaluation is straightforward.
   - Real-world benchmarks:
     - Load functions or evaluate via API, simulation, or specific routines.
     - Handle specific variable types, constraints, and bounds as provided in the problem description.
   
6. **Handling Variable Types:**
   - Some variables are *integer*: sample as floats within bounds, convert to int before evaluation.
   - Mixed variables: ensure sampling generates valid data types.
   
7. **Sampling Strategy:**
   - Use quasi-random sequences (Sobol) or Latin Hypercube Sampling for initial points.
   - Ensure coverage across the variable domain.
   - Maintain reproducibility by fixing or setting random seed accordingly (aligned with overall experiment seed).

8. **Reproducibility & Seeds:**
   - Accept an optional seed parameter to ensure consistent initial sampling per seed.
   
9. **Flexibility & Extensibility:**
   - Design to accommodate new synthetic functions or real-world datasets with minimal modification.
   - Support dataset loading from files if needed.
   - Modular evaluation functions.

---

### Implementation Details:
- **Constructor (`__init__`)**
  - Parameters:
    - `bounds`: dict with variable bounds, e.g.,
      ```python
      bounds = {
        'x1': (min, max, type='float'/'int'), ...
      }
      ```
    - `initial_samples`: int
  
  - Actions:
    - Parse bounds, decide on sampling for each variable (uniform, Sobol, etc.).
    - Generate initial points (`X_init`).

- **`initialize_samples()`**
  - Use numpy or scipy to generate `initial_samples` points uniformly across the domain.
  - For Sobol: use `scipy.stats.qmc.Sobol`.
  - For Latin Hypercube: use `scipy.stats.qmc.LatinHypercube`.
  - Convert sampled points to correct data types (float or int).

- **`get_evaluations(x)`**
  - Input:
    - `x`: a 1D numpy array representing a sample.
  - For synthetic functions:
    - Evaluate `f(x)` directly (implement specific function).
    - Evaluate constraints `c(x)` (specific formula), determine feasibility.
  - For real benchmarks:
    - Call evaluation routines (possibly placeholder functions or API calls).
    - Return:
      - `float` objective if feasible,
      - `np.nan` or `None` if infeasible.
      - Also return a boolean indicating feasibility if needed downstream.

- **Data Management:**
  - Store data:
    - `self.X`: numpy array, shape `(n_samples, d)`
    - `self.Y_obj`: list/array of objective values
    - `self.Y_feasibility`: list/array of boolean feasibility labels
  
- **Utilities:**
  - Methods to update datasets after each evaluation.
  - Optional: save and load datasets.

---

### Constraints:
- Ensure consistent sample generation:
  - Respect variable bounds (including integer constraints).
- Ensure reproducibility via seed setting.
- Modular design to support synthetic and real-world data.

---

### Summary:
The `DatasetLoader` class should:
- Generate initial datasets with reproducibility.
- Evaluate samples via problem-specific functions.
- Provide methods for sampling and evaluation that integrate seamlessly with the main optimization loop.
- Support both synthetic benchmark evaluation functions and real-world problem evaluation routines.

This detailed logic will guide the precise implementation of the class, ensuring alignment with the methodology, data requirements, and reproducibility goals outlined in the paper and plan.

## evaluation.py

### Logic Analysis for `evaluation.py`

The purpose of `evaluation.py` is to provide a modular and comprehensive framework for:
- Tracking key metrics during the Bayesian Optimization (BO) process.
- Computing evaluation measures such as the best feasible objective value, classifier accuracy.
- Monitoring runtime performance.
- Generating visualizations that illustrate boundary discovery progress, convergence trends, and comparison results across different algorithms.

Below is a detailed, step-by-step logical breakdown of the core classes, methods, data dependencies, and evaluation procedures for `evaluation.py` aligned with the paper's methodology and the overall plan.

---

### 1. **Class Overview: `Evaluation`**

**Objective:**
- Instantiate with necessary data and configuration.
- Update metrics iteratively during BO.
- Facilitate visualization and reporting at the end.

**Core Responsibilities:**
- Store evaluation history (best feasible objective, classifier accuracy, runtime per iteration).
- Compute metrics after each evaluation.
- Generate plots for boundary and convergence analysis.
- Summarize final performance.

---

### 2. **Key Inputs and Data Sources**

- **Evaluations dataset:**
  - `X`: The array of all evaluated points (shape `(n_samples, d)`).
  - `f_values`: Objective function values `f(x)` (shape `(n_samples,)`), only for feasible points.
  - `feasibility_labels`: Binary labels indicating feasibility `c(x)` as 0/1 or True/False.
  - `timesteps`: The iteration index or evaluation count for each sample.

- **Models/models predictions:**
  - Surrogate models:
    - GP for objective: predictions (mean, variance).
    - Classifier ensemble for feasibility: predictions, uncertainty measures.
  - These are from the main loop, but in this file, we primarily consume the data to compute metrics.

- **Configuration parameters:**
  - Total evaluations (from config).
  - Initial sample count.
  - Seed list (for reproducibility and statistical robustness).

---

### 3. **Metrics to Compute**

#### a. **Best Feasible Objective Value (`f_best`) over iterations**
- **Procedure:**
  - At each evaluation step \(i\), identify the subset of evaluated samples that are feasible: `feasibility_labels[:i]`.
  - Among these, find the minimal objective value `min(f_values[feasibility_labels == 1])`.
  - Keep track of the running best (lowest) feasible value across all evaluated points.

- **Details:**
  - Initialization: set to `np.inf` at start.
  - Update after each new evaluation.
  - Plot: x-axis is number of evaluations, y-axis is best feasible `f(x)` so far.
  - Store this history for comparison and convergence analysis.

#### b. **Classifier Accuracy (`accuracy`)**
- **Metric:**
  - Use *Balanced Accuracy* (from the paper, considering class imbalance).
  - Formula:
    \[
    \text{Balanced Accuracy} = \frac{TPR + TNR}{2}
    \]
  - **TPR:** True positive rate (\(\frac{\text{TP}}{\text{TP} + \text{FN}}\))
  - **TNR:** True negative rate (\(\frac{\text{TN}}{\text{TN} + \text{FP}}\))

- **Procedure:**
  - For each iteration (or evaluation step), compare the predicted feasibility `C(x)` (from classifier ensemble) with the true labels.
  - Calculate confusion matrix components:
    - TP: predicted feasible and actually feasible.
    - TN: predicted infeasible and actually infeasible.
    - FP, FN accordingly.
  - Compute TPR and TNR, then precision (balanced accuracy).

- **Implementation:**
  - Requires storing model predictions and true labels at each step.
  - Report the moving or cumulative accuracy over iterations.

#### c. **Runtime Tracking (`runtime`)**
- **Procedure:**
  - Record the time at each iteration start and after each evaluation.
  - Summarize total time taken to reach all evaluations.
- **Implementation:**
  - Use Python's `time.perf_counter()` or similar to measure durations.
  - Store cumulative runtime history aligned with evaluation counts.

---

### 4. **Visualization Functions**

#### a. **Plot Boundary Discovery (`plot_boundary_evolution`)**
- **Input:**
  - Surrogate classifier predictions (e.g., probability maps)
  - True boundary contours (for synthetic data)
  - Sample points (initial, evaluated)
  - Final boundary points.
- **Purpose:**
  - Illustrate how the estimated boundary aligns with the true boundary.
  - Show how the boundary improves over iterations.

#### b. **Convergence Plot (`plot_performance`)**
- **Input:**
  - `f_best` history over evaluation count.
  - Variance or standard deviation over seeds.
- **Purpose:**
  - Visualize optimization progress.
  - Show the effectiveness of the BO process.

#### c. **Comparison Studies (`plot_algorithm_comparison`)**
- **Input:**
  - Multiple algorithms' `f_best` over iterations.
  - Variance/error bands.
- **Purpose:**
  - Direct visual comparison of performance.

#### d. **Additional Plot Aspects:**
- Plot classifier accuracy over iterations.
- Plot feasibility ratio: ratio of feasible points over total evaluated points.
- Visualize estimated vs. true boundary (if available).

---

### 5. **Supporting Utility Functions**

- `compute_best_feasible()`: To update and store the current best feasible value.
- `compute_classifier_accuracy()`: Given predictions and true labels, compute balanced accuracy.
- `plot_metrics()`: Plot performance metrics across iterations.
- `save_fig()`: Save visualization outputs with clear naming conventions for reproducibility.
- `log_progress()`: Record metrics into logs (CSV or JSON) for cross-run analysis.

---

### 6. **Implementation Details & Usage Flow**

1. **Initialization:**
   - Instantiate `Evaluation` object with configuration parameters and reference to datasets and models.
   - Prepare containers for metrics: `best_feasible`, `accuracy_history`, `runtime_history`.

2. **Per EVALUATION LOOP (called after each BO iteration):**
   - Input: current evaluated data (`X_eval`, `f_eval`, `labels`, timing info).
   - Update:
     - Calculate current `f_best`.
     - Compute classifier predictions on the current evaluated points.
     - Calculate classifier accuracy.
     - Update elapsed runtime.
   - Append current metrics to history lists.
   - Generate optional plots (boundary, convergence).

3. **Finalization:**
   - Summarize global best, overall classifier accuracy, total runtime.
   - Save plots and metrics summaries for reporting.

4. **Optional:**
   - Save full logs in JSON/CSV.
   - Generate comparison plots across different algorithms for datasets with multiple implementations.

---

### 7. **Handling Uncertainty and Variability**
- Since experiments are run with multiple seeds:
  - Store per-seed metrics separately.
  - Aggregate results (mean ± std) for plots.
  - Use statistical analysis if necessary to highlight significance.

---

### 8. **Possible Extensions**
- Incorporate metrics for boundary uncertainty evolution.
- Track and visualize the distribution of evaluated points relative to the true boundary.
- Enable exporting of intermediate boundary visualizations for detailed analysis.

---

### **Summary**
`evaluation.py` must:
- Act as a wrapper to accumulate all relevant metrics at each iteration.
- Provide functions to:
  - Update best feasible objective value.
  - Calculate and track classifier accuracy.
  - Measure runtime.
  - Generate informative visualizations of the boundary, convergence, and comparison results.
- Handle multiple seeds and random runs for robust statistical analysis.
- Be flexible to visualize both synthetic and real-world problem results, with particular attention to boundary evolution over BO iterations.

This logical blueprint should guide the structured implementation of `evaluation.py`, ensuring fidelity to the methodology, reproducibility, and comprehensive performance assessment.

## main.py

# Main.py Logic Analysis

This analysis delineates a comprehensive plan for implementing the core orchestrator script `main.py`, which manages the overall execution of the BE-CBO adaptive Bayesian optimization process. The script will coordinate dataset initialization, model training, acquisition function optimization, evaluation, updating models, and visualizations according to the described methodology, dataset structures, and configuration parameters.

---

## Overview of Responsibilities
- Initialize all components: dataset loader, objective surrogate model, constraint classifier ensemble, boundary bounds, acquisition optimizer, and evaluation metrics.
- Generate initial samples.
- Iteratively:
  - Update surrogate models (`f(x)`, `c(x)`).
  - Compute boundary bounds based on the classifier’s uncertainty.
  - Optimize the constrained acquisition function to propose a new sample.
  - Evaluate the sample:
    - If feasible: evaluate the true objective.
    - If infeasible: do not evaluate objective; record as infeasible.
  - Update models with new data.
  - Track progress: best feasible value, classifier accuracy, uncertainty, runtime, etc.
  - Save/visualize intermediate results.
- Terminate after reaching total evaluation budget.
- Final reporting: best objective found, convergence plots, boundary visualization.

## Step-by-step Logic

### 1. Initialization
- **Read configuration (`config.yaml`)**:
  - Load `training`, `model`, `optimizer`, `acquisition`, `boundary`, `evaluation` parameters.
- **Set random seed(s)** for reproducibility (`seeds` list).
- **Initialize components**:
  - `DatasetLoader`: with problem bounds, initial sample size, seed.
  - `SurrogateObjectiveGP`: with kernel type and hyperparameters (default RBF or Matern).
  - `NeuralEnsembleConstraintClassifier`: with `ensemble_size`, `hidden_layers`, `neurons_per_layer`.
  - `BoundaryBounds`: with base margin (0.5), possibly other parameters.
  - `AcquisitionOptimizer`: with the specified acquisition function (`EI` or `UCB`), bounds, and active boundary bounds logic.
  - `Evaluation metrics` and visualization modules.

### 2. Initial Sampling
- Call `DatasetLoader.initialize_samples()`:
  - Generate initial random samples (e.g., via Sobol or Latin Hypercube).
  - Evaluate each sample:
    - Call `dataset_loader.get_evaluations(x)`:
      - If feasible:
        - Evaluate true objective function `f(x)`.
        - Record `(x, f(x), feasible=True)`.
      - If infeasible:
        - No objective evaluation; record `(x, None, feasible=False)`.
  - Collect data arrays: `X_init`, `Y_init`, labels `labels_init` (feasible/infeasible).

### 3. Model Training
- **Train the GP surrogate for the objective**:
  - Filter feasible samples: `X_feasible`, `Y_feasible`.
  - Fit `SurrogateObjectiveGP` with these data.
- **Train the neural ensemble classifier for constraints**:
  - Use all samples with labels:
    - `X_all`, `labels_all`.
  - Train the neural ensemble classifier.

### 4. Iterative Optimization Loop (for each evaluation up to total_evaluations)
- **Compute boundary bounds**:
  - For each candidate point in the search space (or during optimization), predict constraint feasibility probability `C(x)`:
    - Use NeuralEnsembleConstraintClassifier.predict_proba(x)`:
      - Obtain mean probability \(\mu_E(x)\).
      - Obtain ensemble uncertainty \(\sigma_E(x)\).
  - Calculate dynamic boundary lower bound \(l(x) = 0.5 - \sigma_E(x)\).
  - Set upper bound \(u(x) = 1\) (full feasible probability).
- **Optimization of acquisition function**:
  - Use `AcquisitionOptimizer` to maximize the constrained EI:
    - With constraints:
      \[
      C(x) \geq l(x) = 0.5 - \sigma_E(x)
      \]
    - Subject to input bounds.
    - Possibly multiple starting points for robustness.
  - Obtain the candidate point(s) `x_next`.
- **Sample evaluation**:
  - Evaluate at `x_next`:
    - Call `dataset_loader.get_evaluations(x_next)`:
      - Feasible: evaluate true objective `f(x_next)`.
      - Infeasible: no objective; only record infeasibility.
- **Update datasets**:
  - Append new sample:
    - `X = np.vstack([X, x_next])`
    - `Y = np.append(Y, f(x_next) or placeholder)`
    - Labels for classifier:
      - `labels = np.append(labels, 1 if feasible else 0)`
- **Re-train models**:
  - Fit GP surrogate with all feasible data.
  - Retrain constraint classifier ensemble with all data.
- **Update boundary bounds**:
  - For next iteration, re-compute `l(x)` using the updated classifier with ensemble uncertainty.
- **Logging and visualization**:
  - Record metrics:
    - Best feasible minimal `f(x)`.
    - Classifier balanced accuracy.
    - Ensemble uncertainty levels.
    - Runtime elapsed.
  - Save current sample set, model parameters, and metrics.
  - Visualize current surrogate surface, boundary, and sampled points:
    - Use `visualization.py` functions for plotting.
- **Check termination criteria**:
  - If total evaluations reached, or if convergence criteria met (e.g., no significant improvement), break.

### 5. Finalization
- **Summarize and report**:
  - Prepare plots of best objective vs. evaluations.
  - Visualize the final boundary and sampled points.
  - Export final models and metrics.
  - Save logs and hyperparameters for reproducibility.
- **Additional experiments**:
  - If applicable, rerun with different seeds.
  - Compare variants (e.g., EI vs UCB, boundary vs fixed bounds).

---

## Additional Considerations
- **Parallelization**:
  - Although not required, consider supporting batch sampling via multiple candidate points.
- **Robustness**:
  - Handle potential model failure points (e.g., optimizing boundary constraints).
- **Hyperparameters & Tuning**:
  - Use default or user-specified parameters via `config.yaml`.
- **Reproducibility**:
  - Fix random seed(s).
  - Log all hyperparameters and data.

---

## Summary Table

| Step | Action | Methods/Functions | Output/Result |
|---|---|---|---|
| Initialization | Load config, set seeds | `yaml.safe_load()`, seed setting | Configurations, seed fixed |
| Initial Sampling | Generate `X_init`, evaluate for `Y_init` | `dataset_loader.initialize_samples()` | Initial data set |
| Model Training | Fit GP and constraint ensemble | `fit()` methods | Trained surrogate models |
| Loop (per iteration) | Compute bounds, optimize acquisition, evaluate | Call respective classes/functions | New data points, updated models |
| Collect metrics | Store best, accuracy, uncertainty, runtime | `evaluation.py` functions | Logs, plots |
| Termination | Check via eval count or convergence | Loop break condition | Final statistics and visualizations |

---

## Conclusion
This detailed logical flow provides a solid blueprint for implementing `main.py` as the central orchestration script that systematically executes the BE-CBO workflow, adheres to experimental design, and ensures reproducibility, scalability, and clarity in the implementation.

## model.py

### Logic Analysis for `model.py` (Defines SurrogateObjectiveGP class)

This module is responsible for modeling the objective function \(f(x)\) using Gaussian Process (GP) regression within the BE-CBO framework. It encapsulates building, training, and prediction operations, ensuring the surrogate model adheres to the specified configuration and experimental protocol.

---

#### **Objective:**
- Create a class `SurrogateObjectiveGP` that:
  - Initializes a GP model with specific kernel and hyperparameters.
  - Fits the GP model to provided data.
  - Provides methods to predict the mean and variance of \(f(x)\) at query points.
  - Supports hyperparameter tuning and model management appropriately.

---

#### **Dependencies:**
- `torch` (PyTorch): For tensor operations and model parameters.
- `gpytorch`: For defining and training Gaussian Process models.
- `numpy`: For data conversion and input/output handling.
- `scipy`: (if any hyperparameter tuning or optimization routines are needed; potentially in other modules).

---

#### **Design Reference & Constraints:**
- **Input Data:**
  - Dataset: Input points `X` (shape \((n, d)\)), outputs `Y` (shape \((n, 1)\) or \((n,)\)).
  - `X` and `Y` are likely NumPy arrays on input, converted to torch tensors internally.
- **Kernel specification:**
  - Use kernel type defined in configuration: e.g., RBF (Squared Exponential), Matern, etc.
  - Hyperparameters: lengthscale, outputscale, noise level — initialized from configuration hyperparameters.
- **Hyperparameter Initialization & Tuning:**
  - Use `gpytorch` default hyperparameter initializations.
  - Optional: provide a method for hyperparameter optimization (max likelihood).
- **Prediction:**
  - Given test points `x_test`: output mean \(\hat{f}(x)\) and predictive variance \(\text{Var}[\hat{f}(x)]\).
- **Model Management:**
  - Store trained model for repeated efficient querying.
  - Enable retraining with new data.

---

#### **Implementation Plan:**

**1. Initialization (`__init__`):**

- Accept kernel type and hyperparameter dictionary:
  - `kernel_type`: e.g., 'RBF' or 'Matern'
  - `hyperparameters`: dictionary containing parameters like lengthscale, outputscale, noise_variance
- Initialize:
  - `self.model` to None
  - `self.likelihood` as GaussianLikelihood with initial noise level.
  - `self.gp_model` as a subclass inheriting from `gpytorch.models.ExactGP`.
- Return the model instance ready for training.

**2. Model Definition:**

- Create a custom class `ExactGPModel` (internal or external class) inheriting from `gpytorch.models.ExactGP`:
  - Initialize with training data, likelihood, kernel.
  - Kernel instantiated based on `kernel_type` and configured hyperparameters.
  - Wrap the kernel in a `gpytorch.kernels.ScaleKernel` for output scale.
- This class defines the GP prior and covariance structure.

**3. Fitting (`fit` method):**

- Inputs: training data `X_train`, `Y_train` (both as NumPy arrays).
- Convert to torch tensors.
- Instantiate or update `ExactGPModel`.
- Set the model and likelihood to train mode.
- Use `gpytorch.mlls.ExactMarginalLogLikelihood` for marginal likelihood.
- Optimize hyperparameters via maximum likelihood:
  - Use `torch.optim.Adam`.
  - Maximize the marginal log-likelihood.
  - Run for the number of epochs specified in `config.yaml`.
- Save trained model and hyperparameters for predictions.

**4. Prediction (`predict` method):**

- Inputs: test points `x_test` as NumPy array.
- Convert to torch tensor.
- Set model to eval mode.
- Use `with torch.no_grad()`: query the GP model.
- Obtain posterior distribution:
  - Use `model(x_test)` to get the `MultivariateNormal`.
  - Compute mean and variance:
    - `mean = posterior.mean.detach().cpu().numpy()`
    - `variance = posterior.variance.detach().cpu().numpy()`
- Return the mean and variance as NumPy arrays.

---

#### **Additional Considerations:**

- Hyperparameter flexibility:
  - Hyperparameters should be adjustable via constructor or defaulted to values in `config.yaml`.
- Compatibility with `gpytorch`:
  - Use `ExactGP` class pattern for clean model definition.
- Data management:
  - Handling small datasets typical for BO — ensure numerical stability.
- Model retraining:
  - Support incremental retraining if desired; in BO, usually retrain afresh with accumulated data.
- Model saving and loading:
  - Optional, for experiments with multiple runs.

---

#### **Summary:**

The class `SurrogateObjectiveGP` acts as a wrapper around `gpytorch` for elegant management of GP modeling in Bayesian Optimization. It provides a clear API for training the GP on current data, predicting mean/variance at candidate points, and initializing with kernel and hyperparameter configurations. The model is designed to be flexible (via kernel choice and hyperparameters) and efficient (retrain with latest data) to facilitate iterative BO.

---

This detailed logic analysis ensures that the implementation in `model.py` aligns perfectly with the design, configuration, and methodology described in the paper and plan.

