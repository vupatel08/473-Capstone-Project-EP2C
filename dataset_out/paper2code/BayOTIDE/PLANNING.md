# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Below is a comprehensive, step-by-step plan structured to guide the reproduction of "BayOTIDE," covering the core methodology, experiment design, datasets, hyperparameters, and evaluation metrics. This roadmap is meant to facilitate precise implementation.

---

### 1. **Understand and Implement the Core Methodology**

#### A. **Functional Decomposition Model**
- **Concept**: Model multivariate time series (MTS) X(t) as a weighted sum of trend and seasonal factors:
  \[
  \mathbf{X}(t) = \mathbf{U} \mathbf{V}(t) = [\mathbf{U}_{trend} \quad \mathbf{U}_{season}] \times
  \begin{bmatrix}
  \mathbf{v}_{trend}(t) \\
  \mathbf{v}_{season}(t)
  \end{bmatrix}
  \]
- **Implementation**:
  - Initialize weight matrices \(\mathbf{U}_{trend} \in \mathbb{R}^{D \times D_r}\) and \(\mathbf{U}_{season} \in \mathbb{R}^{D \times D_s}\). For practical purposes, these can be parameters to learn or be initialized randomly.
  - Goal: Given \(\mathbf{U}\), learn the temporal factors \(\mathbf{v}_{trend}(t)\) and \(\mathbf{v}_{season}(t)\).

#### B. **Gaussian Process Priors for Factors**
- **Trend factors** \(\mathbf{v}_{trend}(t)\) modeled with **Matérn kernel**.
- **Seasonality factors** \(\mathbf{v}_{season}(t)\) modeled with **Periodic kernel**.
- **Implementation**:
  - Create a set of GP priors for each factor using the spectral equivalent via **LTI SDEs**.
  - For each kernel, derive the corresponding LTI SDE (e.g., for Matérn \(\nu=3/2\), periodic).
  
#### C. **State-space Representation & Inference**
- Use **LTI SDEs** derived from spectral kernels to form a **linear Gaussian state space model (SSM)**.
- Implement **Kalman filter** for **online inference**:
  - Discretize the SDEs over arbitrary timestamps.
  - Derive and code the **state transition matrices \(\mathbf{A}_n\)**, **process noise covariances \(\mathbf{Q}_n\)**, and **steady-state covariance \(\mathbf{P}_\infty\)**.
- Incorporate **multiple factors** by concatenating their companion states into \(\mathbf{Z}(t)\).
- **Algorithm flow**:
  - For each incoming timestamp, update posterior distribution over factors using Kalman filtering.
  - Use **Rauch-Tung-Striebel (RTS) smoother** after observing entire sequence for full posterior estimates.

#### D. **Online Variational Inference & Moment-matching**
- Use **Conditional Expectation Propagation (CEP)**:
  - Approximate likelihoods \(p(\mathbf{Y} | \mathbf{U}, \mathbf{V}(t), \tau)\) via message passing.
  - Derive closed-form update equations:
    - Update Gaussian approximations \(q(\mathbf{u}^d)\),
    - Gamma distribution \(q(\tau)\),
    - Gaussian distributions for factors \(q(\mathbf{Z}(t))\).
  - These updates involve expectations over the current posterior, computed in closed-form, involving:
    - Mean and variance updates for \(\mathbf{u}^d\),
    - Posterior shape and rate updates for \(\tau\),
    - Factor parameters (\(\hat{\mu}_i\), \(\hat{S}_i\), etc.).

#### E. **Handling Arbitrary Timestamps (Probabilistic Interpolation)**
- Derive the predictive distribution \(q(\mathbf{z}(t^\star))\) for any timestamp \(t^\star \in (t_k, t_{k+1})\),
- Use the **Kalman prediction** equations with the forward-backward messages,
- Compute mean \(\mathbf{m}^\star\) and covariance \(\mathbf{V}^\star\).

---

### 2. **Model Initialization & Hyperparameters**

- **Number of trend factors \(D_r\)**: to be set based on experiments (e.g., 1, 10, 20, 30, 50).
- **Number of seasonal factors \(D_s\)**: similarly set (e.g., 5, 10, 15, 20).
- **Kernel hyperparameters**:
  - Matérn kernel: \(\nu=3/2\), length scale \(\rho\), amplitude \(a\),
  - Periodic kernel: frequency \(p\), length scale,
  - Variances for each GP factor.
- **Prior parameters**:
  - \(\mathbf{U}\): Gaussian prior (mean zero, or specific initialization),
  - Noise precision \(\tau\): Gamma prior (\(a_0, b_0\)), e.g., with hyperparameters tuned per dataset.
- **Optimization/Update parameters**:
  - Number of damping iterations,
  - Inner EP/CEP iterations per timestamp,
  - Damping factors.

---

### 3. **Handling Streaming Data / Online Updates**
- For each new timestamp:
  1. Approximate likelihood messages (equation 12).
  2. Update Gaussian/posterior distributions for \(\mathbf{U}\) and factors.
  3. Run Kalman filter for the current timestamp to update \(\mathbf{Z}(t)\).
- After processing all timestamps:
  - Run **RTS smoother** for full posterior of factors at all timestamps.
- To perform imputation at arbitrary timestamp \(t^\star\):
  - Find nearest observed timestamps;
  - Use the state-space model (via equations 16-17) for probabilistic interpolation.

---

### 4. **Datasets and Experimental Setup**

#### Synthetic Data:
- Generate low-rank \(\mathbf{U}\) and temporal factors \(\mathbf{V}(t)\):
  - \(\mathbf{U}\): random matrix with entries in fixed ranges (e.g., [-1,1]).
  - \(\mathbf{V}(t)\): as sums of sinusoids and polynomial trends.
- Sample irregular timestamps:
  - E.g., 2000 points over [0,1], 70% and 50% observed ratios.
- Add Gaussian noise to observed samples.

#### Real-world Data:
- Use datasets provided:
  - **Traffic-Guangzhou** (214 channels, 500 timestamps),
  - **Solar-Power** (137 channels, 52560 timestamps),
  - **Uber-Move** (7489 channels, 744 timestamps).
- For all:
  - Randomly mask 50% and 70%, separately.
  - Normalize features if needed.
  - Handle irregular sampling by providing timestamps.

---

### 5. **Evaluation Metrics**
- **Deterministic**: MAE, RMSE over the imputed missing values.
- **Probabilistic**: CRPS, Negative Log-Likelihood (NLLK).
- Computation:
  - Use 50 posterior samples per missing point to estimate CRPS, NLLK.
  - Report average over multiple runs (e.g., 5-fold).

---

### 6. **Hyperparameter Tuning & Validation**
- Use validation splits to tune:
  - \(D_r, D_s\),
  - Kernel hyperparameters (\(\rho, p, a\)),
  - Noise prior parameters (\(a_0, b_0\)),
  - Damping and inner iterations for CEP,
  - Number of epochs for inner EP/CEP updates.
- Use grid search or Bayesian optimization if needed, guided by validation RMSE/CRPS.

---

### 7. **Implementation Specifics**
- Modular code:
  - Data preprocessing pipeline (handling irregular timestamps).
  - GP prior modules: functions to generate \(\mathbf{F}, \mathbf{L}, \mathbf{Q}\), from kernel hyperparameters.
  - Online inference engine:
    - Kalman filter and RTS smoother.
    - CEP updates.
  - Imputation routines:
    - For arbitrary timestamps.
- Numerical stability:
  - Damping in CEP updates.
  - Cholesky decompositions for covariance updates.
- Efficient batching:
  - Parallelize message passing per channel/factor.
  - Use vectorized operations where possible.

---

### 8. **Summarize Experimental Procedure**
- For synthetic:
  1. Generate data with known factors.
  2. Mask data partially uniformly.
  3. Initialize hyperparameters.
  4. Run online inference.
  5. Evaluate imputation over fixed and unseen timestamps.
- For real datasets:
  1. Load data and mask.
  2. Tune hyperparameters via validation.
  3. Run online inference.
  4. Evaluate with metrics (RMSE, MAE, CRPS, NLLK).
  5. Compare with baselines.

---

**Final Notes & Possible Clarifications Needed:**

- Precise kernel hyperparameters are dataset-specific; initial values should be set according to the reported hyperparameters in the appendix.
- The number of trend and seasonal factors is to be tuned, but initial grid could follow the paper's recommendation.
- Handling large datasets (like Solar-Power) may require efficient data streaming and batching.
- Implementation of the spectral-to-LTI conversion involves solving Lyapunov equations, which should be pre-coded or use existing libraries.
- The approximate inference methods (CEP, Kalman filter) should be carefully tested for numerical stability before integration.

This roadmap provides a detailed, sequential guide from understanding the model to preparing for implementation and experimentation, aligning closely with the paper's technical and experimental setup.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "We will implement the BayOTIDE method as a modular Python package leveraging PyTorch for neural components, NumPy/SciPy for linear algebra and SDE calculations, and filterpy for Kalman filtering. The system will include components for data handling, spectral-to-LTI kernel conversion, online variational inference with CEP, Kalman filtering, and probabilistic interpolation at arbitrary timestamps. Emphasize efficient numerical routines, parallel processing for message passing, and flexible handling of irregular timestamps.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "spectral_to_LTI.py",
        "gpr_model.py",
        "inference.py",
        "utils.py",
        "evaluation.py",
        "config.py"
    ],
    "Data structures and interfaces": "classDiagram\n    class Main {\n        +__init__(config: dict)\n        +run_experiment()\n    }\n    class DatasetLoader {\n        +__init__(dataset_paths: dict, mask_ratios: list)\n        +load_data() -> tuple\n        +generate_irregular_timestamps() -> np.ndarray\n        +apply_mask(ratio: float)\n    }\n    class SpectralToLTID {\n        +__init__(kernel_type:str, hyperparameters: dict)\n        +construct_matrices() -> dict\n    }\n    class GPRFactorModel {\n        +__init__(U: np.ndarray, D_r: int, D_s: int, kernel_params: dict)\n        +initialize_statespace()\n        +predict(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]\n        +update_posterior(observations: dict)\n        +smooth_full_posterior()\n    }\n    class InferenceEngine {\n        +__init__(model: GPRFactorModel, data: dict)\n        +online_update(new_data: dict) -> None\n        +impute(t_star: float) -> Tuple[np.ndarray, np.ndarray]\n        +run_Kalman_filter() -> None\n        +pose_fixed_timestamps(t_list: np.ndarray) -> dict\n    }\n    class Evaluation {\n        +__init__(ground_truths: dict, imputations: dict)\n        +calculate_metrics() -> dict\n    }\n    Main --> DatasetLoader\n    Main --> SpectralToLTID\n    Main --> GPRFactorModel\n    Main --> InferenceEngine\n    Main --> Evaluation\n    DatasetLoader --> utils\n    SpectralToLTID --> utils\n    GPRFactorModel --> utils\n    InferenceEngine --> utils\n",
    "Program call flow": "sequenceDiagram\n    participant C as Main\n    participant D as DatasetLoader\n    participant S as SpectralToLTID\n    participant M as GPRFactorModel\n    participant I as InferenceEngine\n    participant E as Evaluation\n    C->>D: load_data()\n    D-->>C: dataset_tuple\n    C->>S: construct kernel matrices with hyperparams\n    S-->>C: matrices {'F', 'L', 'Q'}\n    C->>M: initialize U, D_r, D_s, kernel_params\n    M-->>C: set state-space matrices\n    C->>I: initialize inference with model\n    loop over streaming data\n        D->>D: generate/take batch\n        D-->>C: masked data, timestamps\n        C->>I: online_update(new_data)\n        I-->>C: posterior update\n        C->>E: evaluate_imputation()\n    end\n    C->>E: compute_metrics()\n    E-->>C: results\n",
    "Anything UNCLEAR": "Clarify the initial hyperparameter tuning procedure, distribution of synthetic vs. real datasets, and the exact API for the core classes, especially the methods for online CEP updates and probabilistic interpolation, to ensure full coverage of the techniques described."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0",
        "scipy==1.7.0",
        "filterpy==1.4.5",
        "pyro-ppl==1.8.0"  # Optional if using probabilistic modules
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Defines DatasetLoader class: loading raw datasets, masking data according to specified ratios, generating irregular timestamp sequences, and providing data batches for training and evaluation. This module depends on numpy and pandas."
        ],
        [
            "spectral_to_LTI.py",
            "Contains SpectralToLTID class: converts specified kernel types (Matérn, periodic) into state-space matrices (F, L, Q) for the LTI SDE formulation, using SciPy for matrix exponentiation and Lyapunov equations."
        ],
        [
            "gpr_model.py",
            "Implements GPRFactorModel class: initializes with U, D_r, D_s, kernel hyperparameters; constructs the state-space model; provides methods for incremental online updates, Kalman prediction, and smoothing; maintains factor states and posterior parameters."
        ],
        [
            "inference.py",
            "Contains InferenceEngine class: manages online variational inference with CEP, Kalman filtering, and approximation of message updates. Facilitates real-time posterior updates and probabilistic interpolation at arbitrary timestamps."
        ],
        [
            "utils.py",
            "Provides utility functions: mathematical operations, expectation and covariance calculations, initialization routines, and helper methods for matrix operations, ensuring clean code reuse."
        ],
        [
            "evaluation.py",
            "Defines Evaluation class: computes MAE, RMSE, CRPS, and NLLK based on ground truth and imputed values; manages metrics aggregation across datasets and experimental runs."
        ],
        [
            "main.py",
            "Serves as the entry point: reads configuration, initializes components (dataset loader, spectral-to-LTI converter, model, inference engine, evaluator), orchestrates execution flow for data loading, model training/inference, and evaluation. Calls all modules in necessary order, manages experiment repetitions."
        ]
    ],
    "Task list": [
        "requirements.txt",
        "dataset_loader.py",
        "spectral_to_LTI.py",
        "gpr_model.py",
        "inference.py",
        "utils.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0\n  components:\n    schemas:\n      DataInput:\n        type: object\n        properties:\n          data:\n            type: array\n            items:\n              type: object\n              properties:\n                timestamp:\n                  type: number\n                values:\n                  type: array\n                  items:\n                    type: number\n          mask:\n            type: array\n            items:\n              type: boolean\n      ImputationRequest:\n        type: object\n        properties:\n          timestamps:\n            type: array\n            items:\n              type: number\n      Metrics:\n        type: object\n        properties:\n          MAE:\n            type: number\n          RMSE:\n            type: number\n          CRPS:\n            type: number\n          NLLK:\n            type: number\n  paths:\n    /load_data:\n      post:\n        summary: Load dataset with masking and irregular timestamps\n        requestBody:\n          required: true\n          content:\n            application/json:\n              schema:\n                $ref: '#/components/schemas/DataInput'\n        responses:\n          '200':\n            description: Data loaded successfully\n    /impute:\n      post:\n        summary: Perform online imputation at arbitrary timestamps\n        requestBody:\n          required: true\n          content:\n            application/json:\n              schema:\n                $ref: '#/components/schemas/ImputationRequest'\n        responses:\n          '200':\n            description: Imputation results\n            content:\n              application/json:\n                schema:\n                  type: object\n                  properties:\n                    imputed_values:\n                      type: array\n                      items:\n                        type: number\n    /metrics:\n      get:\n        summary: Retrieve evaluation metrics\n        responses:\n          '200':\n            description: Metrics object\n            content:\n              application/json:\n                schema:\n                  $ref: '#/components/schemas/Metrics'",
    "Shared Knowledge": "The system shares hyperparameters, kernel configurations, and the state-space model conversion functions across modules. `utils.py` provides standardized functions for matrix operations, covariance calculations, and initializations used in multiple components. The overall data schema for dataset input, imputation requests, and evaluation results should be consistent across modules.",
    "Anything UNCLEAR": "Clarify the expected data schema details, e.g., dataset formats, missing data representation, and how the irregular timestamps are provided. Also, specify whether to pretrain hyperparameters or tune them dynamically."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: 0.001  # Not specified in the paper, set as a reasonable default
  batch_size: 32        # Not specified in the paper, set as a standard batch size
  epochs: 50            # Not specified; choose a moderate number for convergence

model:
  D_r: 10               # Number of trend factors (hyperparameter to tune; start with 10)
  D_s: 10               # Number of seasonal factors (hyperparameter to tune; start with 10)
  kernel_type_trend: 'matern32'  # Based on paper's 3/2 Matérn kernel
  kernel_type_season: 'periodic' # Periodic kernel for seasonality
  kernel_params:
    matern32:
      length_scale: 0.1     # Initial value, tuning needed
      variance: 1.0         # Initial value, tuning needed
    periodic:
      length_scale: 0.5     # Initial value, tuning needed
      period: 12.0          # Based on typical seasonal period (e.g., daily/weekly)
      variance: 1.0         # Initial value, tuning needed

optimization:
  learning_rate: 0.0005  # Fine-tuned for stable convergence
  damping_epochs: 5      # Inner CEP damping iterations per timestamp
  inner_iterations: 3   # Number of message passing iterations per update
  damping_factor: 0.5   # Damping coefficient for stability

data:
  observed_ratio_train: 0.7  # 70% observed data in training
  observed_ratio_eval: 0.5   # 50% observed data in evaluation
  max_sequence_length: 2000  # For synthetic data; dataset-dependent for real data
  noise_std: 0.1             # Noise standard deviation added to synthetic data

inference:
  online_update: true      # Perform online updates with CEP and Kalman filtering
  interpolation_method: 'gaussian'  # Probabilistic interpolation at arbitrary timestamps
  num_samples: 50          # Number of posterior samples for metrics
  damping_epochs_for_inference: 5
  smoothing: true          # Run RTS smoothing after sequence inference

evaluation:
  metrics: ['MAE', 'RMSE', 'CRPS', 'NLLK']
  evaluation_repeats: 5   # Number of runs for averaging metrics
  evaluation_split_seed: 42  # Seed for reproducibility

dataset:
  synthetic_data:
    num_points: 2000
    timestamp_range: [0, 1]
    pattern_types:
      trend: 'matérn32'
      seasonality: 'periodic'
  real_world:
    datasets:
      - name: 'Traffic-Guangzhou'
        channels: 214
        timestamps: 500
      - name: 'Solar-Power'
        channels: 137
        timestamps: 52560
      - name: 'Uber-Move'
        channels: 7489
        timestamps: 744
```

---

**Note:** Full configuration is available in `planning_config.yaml`
