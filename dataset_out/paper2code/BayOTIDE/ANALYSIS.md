# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## dataset_loader.py

{
  "dataset_loader.py": "The DatasetLoader class serves as the core data handling module, responsible for loading datasets, applying missing data masks, generating irregular timestamp sequences, and providing data batches for training and evaluation. Its primary goal is to simulate realistic multivariate time series with irregular sampling and missing values, in line with the experimental setup described in the paper.\n\n**Key Components & Logical Steps:**\n\n1. Initialization:\n   - Input parameters: dataset_paths (dictionary mapping dataset names to file paths), mask_ratios (list, e.g., [0.5, 0.7]) for evaluating different observed data proportions.\n   - Store parameters for later use.\n\n2. Data Loading:\n   - Load raw data files, which may be in formats such as CSV, HDF5, or other standard formats.\n   - The datasets may include 'traffic', 'solar', 'uber' data, with specified number of channels and timestamps.\n   - Use pandas or numpy to load data arrays: shape should be (channels, time).\n\n3. Data Preprocessing:\n   - Normalize features if necessary (e.g., min-max or z-score normalization), depending on the dataset.\n   - Handle any data-specific cleaning steps.\n\n4. Masking for Missing Data:\n   - For each dataset and for each mask ratio (e.g., 50%, 70%), create a binary mask matrix of shape (D, N): where D is number of channels, N number of timestamps.\n   - Randomly select entries to be observed or missing according to the ratio: set mask entries to 1 or 0.\n   - Store observed and missing indices separately for later evaluation.\n\n5. Generating Irregular Timestamps:\n   - For synthetic data, generate a sequence of timestamp points over [0,1] (or relevant range), e.g., N=2000 points, irregularly sampled.\n   - For real data with existing timestamps, use the inherent timestamps; for simulation, generate irregular timestamps by random sampling or jittering regular grids.\n   - Store timestamps as numpy arrays of shape (N,).\n\n6. Data Batching & Sampling:\n   - Provide methods to return batches of data for training (e.g., batch size = 32).\n   - Each batch should include:\n       - Masked data: observed values with missing entries removed.\n       - Corresponding timestamps (to handle irregular sampling).\n       - For synthetic data, facilitate creation of new random sequences on demand.\n   - Implement functions to retrieve masked data and full/partial sequences for both training and evaluation.\n\n7. Evaluation Data Preparation:\n   - For each dataset, prepare separate evaluation splits:\n       - Use the specified masking ratios.\n       - Keep track of true values for missing entries for error metrics.\n   - Store both observed and missing indices.\n\n8. Utility Functions:\n   - Methods to generate new irregular timestamps for online or continuous imputation testing.\n   - Functions to re-apply masking if needed.\n   - Functions to support data normalization, denormalization for evaluation.\n\n**Relevant Details & Assumptions:**\n- Data format: Should handle CSV or numpy array files; if datasets are provided in other formats, include conversion logic.\n- Masking: Random uniform masking is suitable, matching the paper's masked data ratios.\n- Irregular timestamps: For synthetic data, generate random timestamps uniformly or follow specific pattern; for real data, use existing timestamp data.\n- Data splits for training/evaluation involve masking for missingness only, not necessarily temporal splits.\n- Ensure reproducibility: set random seed internally for masks and timestamp generation.\n- Store data and masks as class attributes: e.g., self.data, self.masks, self.timestamps, self.observed_idx, self.missing_idx.\n\n**Summary:**\nThe DatasetLoader class must enable flexible data ingestion across datasets, simulate realistic missingness and irregular sampling, and provide interfaces for downstream training and evaluation, aligning closely with the experimental protocol and data handling described in the paper."
}

## evaluation.py

**Evaluation.py Logic Analysis**

---

### 1. **Purpose & Responsibilities**
- Purpose: To evaluate and quantify the imputation performance of BayOTIDE (or other baseline methods) by computing various metrics:
  - **MAE (Mean Absolute Error)**
  - **RMSE (Root Mean Square Error)**
  - **CRPS (Continuous Ranked Probability Score)**
  - **NLLK (Negative Log-Likelihood)**
- Responsibilities:
  - Accept ground truth data and imputed data (both possibly probabilistic).
  - Calculate per-sample and aggregated metrics.
  - Support multiple datasets, repeated runs, and multiple metrics.
  - Provide formatted results for comparison, reporting, plotting, and analysis.

### 2. **Input Data & Formats**
- **Ground Truth Data**: `true_values`, shape `(num_samples, D)`.
- **Imputed Data**:
  - For deterministic models: `imputed_means`, shape `(num_samples, D)`.
  - For probabilistic models:
    - `posterior_samples`, shape `(num_samples, num_posterior_samples, D)`, or
    - `posterior_means` + `posterior_stds` for Gaussian posterior approximation.
- **Supporting Data**:
  - Observed mask: To identify missing entries (optional, reused if necessary).
  - Ground truth missing entries: To evaluate imputation specifically at missing places. Usually, provided externally, or explicitly as a subset.

### 3. **Key Considerations**
- **Handling Probabilistic vs. Deterministic Outputs**:
  - **Deterministic**: Direct evaluation with ground truth.
  - **Probabilistic**:
    - **CRPS**: Requires posterior samples or predictive distribution.
    - **NLLK**: Calculated assuming the predictive distribution; typically Gaussian with mean and variance.
  - When posterior samples are available:
    - Compute CRPS via standard formulas (e.g., empirical CDF/quantile).
    - Compute NLLK assuming Gaussian predictive distribution:
      \[
      \text{NLLK} = -\frac{1}{N}\sum_{i=1}^N \log \mathcal{N}(y_i | \mu_i, \sigma_i^2)
      \]
- **Metrics**:
  - MAE and RMSE: Use only the missing entries for evaluation.
  - CRPS: Use posterior samples for each missing entry.
  - NLLK: Use mean and predictive variance at each missing entry.

### 4. **Implementation Details**
- **For each dataset/experiment/repetition**:
  - Input:
    - `ground_truth`: actual true values at missing entries.
    - `imputed_means`: point estimates (e.g., posterior mean).
    - `posterior_samples`: optional, for probabilistic metrics.
  - Output:
    - Scalar metrics: average over all missing entries.

- **Functionality**:
  1. **Filtering missing entries**:
     - Use masked indices, or provided missing indices.
     - Extract ground truth and predictions accordingly.
  2. **Compute MAE**:
     - \(\text{MAE} = \frac{1}{N_{miss}} \sum |\hat{y} - y|\)
  3. **Compute RMSE**:
     - \(\text{RMSE} = \sqrt{\frac{1}{N_{miss}} \sum (\hat{y} - y)^2}\)
  4. **Compute CRPS**:
     - If posterior samples \( \{ y_{sample}^{(i)} \} \) per point are available:
       - Use empirical formulas, e.g.,
         \[
         \text{CRPS} \approx \frac{1}{M} \sum_{i=1}^M | y_{sample}^{(i)} - y_{true} | - \frac{1}{2 M^2} \sum_{i,j=1}^M | y_{sample}^{(i)} - y_{sample}^{(j)} |
         \]
     - For simplicity, a common approximation used in the literature.
  5. **Compute NLLK**:
     - Assuming Gaussian prediction:
       \[
       \text{NLLK} = \frac{1}{N_{miss}}\sum_{i=1}^{N_{miss}} \left( \frac{1}{2} \log (2\pi \sigma_i^2) + \frac{(y_{true,i} - \mu_i)^2}{2 \sigma_i^2} \right)
       \]
     - Where \(\mu_i\), \(\sigma_i\) derived from posterior mean and std.
     - If only posterior samples are available, compute empirical log-likelihoods and average.

- **Metrics Storage & Reporting**:
  - Store computed metrics in a dictionary `Metrics` object:
    \[
    \{ 'MAE': value, 'RMSE': value, 'CRPS': value, 'NLLK': value \}
    \]
  - When multiple datasets or repetitions are evaluated, average results across runs, and report with appropriate formatting.

### 5. **Function Skeleton and Step-by-Step Process**
- `def evaluate(y_true, y_pred_mean, y_pred_samples=None, y_pred_std=None):`
  - Determine if probabilistic:
    - Use `y_pred_samples` for CRPS.
    - Use `y_pred_mean` and `y_pred_std` for NLLK.
  - Select missing entries indices:
    - E.g., via mask or provided indices.
  - Compute:
    - MAE
    - RMSE
    - CRPS (if samples available)
    - NLLK (if std/mean or samples available)
  - Return metrics dictionary.

### 6. **Special Considerations**
- **Batch evaluations**: Handle vectorized computations for efficiency.
- **Multiple runs**: Support cumulative averaging of metrics across repeats for stable evaluation.
- **Missing data indices**:
  - Must be supplied or stored, ensuring evaluation only on missing entries.
- **Numerical stability**:
  - Clip or handle very small std deviations when computing NLLK.
- **Metrics precision**:
  - Use consistent precision and formatting as per the paper (e.g., 3 decimal places).

---

### 7. **Note on Usage**
- The class may initialize with a configuration of datasets, expected metrics.
- Function calls:
  - `add_results(ground_truths, imputations, posterior_samples=None, missing_indices=None)`
  - `get_metrics()`: returns aggregated metrics over all datasets/splits.
- For development:
  - Support for per-dataset, per-run metrics.
  - Support formatting output for reports/visualization.

---

**Summary:**
- `Evaluation.py` defines an `Evaluation` class that:
  - Accepts true and imputed data (deterministic or probabilistic).
  - Computes MAE, RMSE directly.
  - Calculates CRPS using posterior samples.
  - Calculates NLLK assuming Gaussian predictions.
  - Supports metrics aggregation over multiple datasets, repeats.
  - Organized for modular extension and integration into the experiment pipeline.

---

This detailed logic analysis forms a solid foundation for the implementation of `Evaluation.py`, ensuring precise, reproducible, and robust evaluation aligned with the experimental setup described in the paper.

## gpr_model.py

# Logic Analysis for `gpr_model.py`

This analysis details the design and implementation logic for the `GPRFactorModel` class, which encapsulates Gaussian Process (GP) prior modeling of temporal factors, their state-space representations, and state estimation routines critical for the BayOTIDE methodology.

---

## 1. **Purpose and Role**

- To manage the **dynamical modeling** of temporal factors (trend and seasonality) via **spectrally equivalent LTI SDEs**.
- To **initialize, update, and predict** the factor states using **Kalman filtering** and **smoothing**.
- To maintain the set of **posterior parameters** of the factor weights and states, enabling **online inference** and **arbitrary timestamp imputation**.

## 2. **Input Parameters**

- `U`: The weight matrix \(\mathbf{U}\) of shape \((D, D_r + D_s)\) (provided during initialization).
- `D_r`: Number of trend factors (scalar hyperparameter).
- `D_s`: Number of seasonal factors (scalar hyperparameter).
- `kernel_params`: Dictionary containing hyperparameters for the kernels:
  - Trend factors (Matérn): lengthscale, variance.
  - Seasonal factors (Periodic): lengthscale, period, variance.

## 3. **Core Components**

### a. Construction of State-Space Matrices

- For each factor (trend or seasonal), derive the **LTI SDE** matrices:

  - **F**: System matrix, depends on kernel type and hyperparameters.
  - **L**: Input (driving noise) matrix.
  - **Q**: Process noise covariance.
  - **\(P_\infty\)**: Steady-state covariance matrix.
  
- These are **computationally derived using spectral analysis** and **Lyapunov equations**:

  - For **Matérn32** (smoothness \(\nu=3/2\), \(m=1\)):
    - \(F\), \(L\), \(Q\) have specific closed-form expressions, involving \(\lambda = \sqrt{3}/\text{lengthscale}\).
  - For **Periodic**:
    - Construct \(\mathbf{F}_j\), \(\mathbf{L}_j\), \(Q_j\) for each harmonic and sum contributions.

- These matrices are assembled once at initialization per factor using the `spectral_to_LTI.py` logic.

### b. Parameter Storage

- Store **per-factor state-space matrices** (`F`, `L`, `Q`) in lists or dictionaries.
- Store **initial priors**:
  - \(\mathbf{z}(t_1) \sim \mathcal{N}(0, P_\infty)\).

### c. State Variables and Parameters

- Maintain **posterior mean** and **covariance** of the concatenated **factor states** (\(\mathbf{Z}(t)\)):

```python
self.mu_z: np.ndarray  # shape (n_factors, state_dim)
self.Sigma_z: np.ndarray  # shape (n_factors, state_dim, state_dim)
```

- `posterior_params`: for the factors at each timestamp (after update).

- Rooted in **Kalman filter** state, these parameters are updated during online inference.

---

## 4. **Methods**

### a. `initialize_statespace()`

- **Purpose**: Compute and store the **discrete** transition matrices \(\mathbf{A}_n = e^{F \Delta}\),
- For all factors, precompute \(\mathbf{A}_n\) and \(\mathbf{Q}_n\):
  - Use `scipy.linalg.expm()` for matrix exponential.
  - For each timestamp interval \(\Delta\), calculate \(\mathbf{A}_n\) and \(\mathbf{Q}_n\) via integral approximations.

### b. `predict(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`

- **Purpose**: To **predict the factor states distribution** at an arbitrary timestamp \(t^\star\):
  - Find neighboring observed timestamps \(t_k < t^\star < t_{k+1}\).
  - Use **Kalman prediction equations** based on the **discretized transition**:
    - \(\mu_{t^\ast} = \mathbf{A} \mu_{t_k}\),
    - \(V_{t^\ast} = \mathbf{A} \Sigma_{t_k} \mathbf{A}^\top + \mathbf{Q}\).
  - Return predicted mean and covariance for each factor.

### c. `update_posterior(observations: dict)`

- **Purpose**: To **incrementally update** the **posterior distribution** over factor states given new data:
  - Receives posterior parameters (mean, covariance) from previous step.
  - Computes **Kalman update**:
    - For each factor, update \(\mu_{z}\), \(\Sigma_{z}\) via standard measurement update:
      - Innovation \(y - H \hat{z}\),
      - Kalman gain,
      - Posterior mean/covariance.
  - Handles **multiple factors** in parallel if needed.

### d. `run_smoothing()`

- **Purpose**: After a sequence inference, run **RTS smoothing**:
  - Backward pass to refine \(\mu_z\) and \(\Sigma_z\) over the entire sequence.
- Uses stored \(\mathbf{A}_n, \mathbf{Q}_n\) matrices.
- Results stored back into class variables, ready for imputation.

### e. `get_factor_states_at(t: float)`

- Uses `predict()` for probing factors at an arbitrary timestamp.
- Returns the **mean** and **covariance** of each factor at time \(t^\star\), facilitating the probabilistic interpolation.

---

## 5. **Implementation Details**

- **Conversion of kernels** to matrices \(\mathbf{F}\), \(\mathbf{L}\), \(\mathbf{Q}\) is **single responsibility** and to be used in the constructor.
- The **discretization** of continuous-time matrices should respect the timestamp intervals; typically, \(\mathbf{A}_n = e^{F \Delta_n}\), \(\mathbf{Q}_n = \int_0^{\Delta_n} e^{F \tau} L Q L^\top e^{F^\top \tau} d\tau\):
  - Use `scipy.linalg.expm` for \(\mathbf{A}_n\),
  - Approximate \(\mathbf{Q}_n\) via numerical integration or analytical solutions if available.
- The **posterior parameters** are updated in a **recursive Bayesian** manner, with **damping** to ensure numerical stability.

---

## 6. **Summary of Key Variables**

| Variable                                    | Description                                        | Shape / Type                                                  |
|----------------------------------------------|---------------------------------------------------|--------------------------------------------------------------|
| `F_trend`, `F_seasonal`                      | System matrices derived from spectral kernels     | np.ndarray (state_dim, state_dim)                            |
| `L_trend`, `L_seasonal`                      | Input matrices                                    | np.ndarray (state_dim, 1)                                    |
| `Q_trend`, `Q_seasonal`                      | Process noise covariance                          | np.ndarray (state_dim, state_dim)                            |
| `A_trend`, `A_seasonal`                      | Discrete transition matrices per interval       | np.ndarray (state_dim, state_dim)                            |
| `mu_z`, `Sigma_z`                            | Posterior mean and covariance of factor states   | np.ndarray (total_number_of_factors, state_dim), (total, state_dim, state_dim) |
| `hyperparameters`                            | Kernel and prior hyperparameters                 | dict                                                          |

---

## 7. **Conclusion**

Implementing `GPRFactorModel` requires translating spectral kernel properties into exact **discrete-time Kalman filtering** routines, maintaining **posterior states**, and enabling **predictive distributions** for arbitrary timestamps. The class must be robust, efficient, and seamlessly integrated with the online inference pipeline to support BayOTIDE's streaming, uncertainty-aware imputation goals.

---

This completes the thorough logic analysis for `gpr_model.py`.

## inference.py

# Logic Analysis for inference.py

This file is centered on implementing the **InferenceEngine** class, which orchestrates the core online inference operations for the BayOTIDE model. It handles incremental updates of the posterior distribution over model parameters, factor states, and hyperparameters using **Conditional Expectation Propagation (CEP)**, **Kalman filtering**, and smoothing techniques. Additionally, it facilitates probabilistic interpolation to obtain imputed values at arbitrary timestamps.

Below is a detailed, step-by-step logical breakdown of its design, key modules, data flows, and methods, aligned with the paper and configuration.

---

## 1. **Class Overview**

**InferenceEngine**:

- **Purpose**: Manage online variational inference of model parameters, temporal factors, and their associated posteriors; perform Bayesian updates as new data arrives, and provide probabilistic imputation at arbitrary timestamps.
- **Data members**:
  - **Model parameters / state-space matrices**: \(\mathbf{F}, \mathbf{L}, \mathbf{Q}\) for each factor, obtained via spectral-to-LTI conversion.
  - **Posterior distributions**:
    - Gaussian for factor states \(\mathbf{Z}(t)\) (mean \(\mu_i\), covariance \(S_i\))
    - Gaussian for weights \(\mathbf{U}\) (\(q(\mathbf{u}^d)\))
    - Gamma for noise precision \(\tau\) (\(q(\tau)\))
  - **Hyperparameters**: kernel parameters, damping factor, number of inner iterations, damping epochs.
  - **Data buffers**:
    - Recent observations, timestamps.
    - Current prior/posterior states for the factors and weights.

---

## 2. **Initialization**

- **Input**: the model object (GPRFactorModel) which provides the state-space matrices (\(\mathbf{F}, \mathbf{L}, \mathbf{Q}\)), and data associated with initial observations.
- **Actions**:
  - Initialize posteriors:
    - \(\mathbf{Z}\): set \(\mu_i\), \(S_i\) based on prior (steady-state covariance), or from initial observations.
    - Weights \(\mathbf{U}\): Gaussian \(q(\mathbf{u}^d)\), initialized with mean zero covariance \(\sim I\).
    - Noise \(\tau\): Gamma with hyperparameters.
  - Store current data (related timestamps, observed masking) for online updating.

---

## 3. **Online Data Update Workflow**

For each incoming data batch point:

### 3.1. **Data Preparation**

- **Input**:
  - New observation \(\mathbf{Y}_n\)
  - Corresponding timestamp \(t_n\)
  - Mask indicating observed/missing entries
- **Process**:
  - Store or update current data buffers (timestamps, observed values, masks).
  - Identify observed components \(\Omega\).

### 3.2. **Construct Likelihood Message Approximations**

- Use the current posterior estimates to form approximated likelihood messages:
  - For each observed entry \((d, n)\), approximate the likelihood contribution as a product of message factors:
    \[
    p(y_n^d | \mathbf{u}^d, \mathbf{V}(t_n), \tau) \approx \mathcal{Z}_d f_{n+1}^d(\mathbf{Z}(t_{n+1})) f_{n+1}^d(\mathbf{u}^d) f_{n+1}^d(\tau)
    \]
  - **Method**:
    - Compute moments (means, variances) via current posterior estimates (\(\hat{\mathbf{m}}, \hat{\mathbf{V}}\) for \(\mathbf{u}^d\), \(\hat{a}, \hat{b}\) for \(\tau\)), based on equations (12).
    - Use the expectation of the residuals, and the current posterior parameters to compute these moments.

### 3.3. **Update Variational Posteriors of \(\tau\) and \(\mathbf{U}\) (Message Merging)**

- **Equations**:
  \[
  q(\tau) \propto q(\tau)_{prev} \times \prod_d f_{n+1}^d(\tau)
  \]
  \[
  q(\mathbf{u}^d) \propto q(\mathbf{u}^d)_{prev} \times f_{n+1}^d(\mathbf{u}^d)
  \]
- **Method**:
  - Use moment matching to update shape/rate parameters for \(\tau\).
  - Update mean and covariance for each \(\mathbf{u}^d\).

### 3.4. **Update Factor State Distributions via Kalman Filter (Equation 15)**

- **Transition Dynamics**:
  - Use the state transition matrices \(\mathbf{A}_n = \exp(\mathbf{F} \Delta_n)\) obtained from spectral-to-LTI conversion.
  - For the current timestamp, perform **predict** and **update** steps:
    - Prediction:
      \[
      \hat{\mathbf{Z}}_{n|n-1} = \mathbf{A}_n \hat{\mathbf{Z}}_{n-1|n-1}
      \]
      \[
      P_{n|n-1} = \mathbf{A}_n P_{n-1|n-1} \mathbf{A}_n^\top + \mathbf{Q}_n
      \]
    - Update:
      \[
      K_n = P_{n|n-1} C^\top (C P_{n|n-1} C^\top + R)^{-1}
      \]
      \[
      \hat{\mathbf{Z}}_{n|n} = \hat{\mathbf{Z}}_{n|n-1} + K_n (\text{innovation})
      \]
      \[
      P_{n|n} = (I - K_n C) P_{n|n-1}
      \]
  - Update \(\hat{\mu}_i, S_i\) for each factor's state using these equations, damping in inner iterations.

### 3.5. **Inner Damping & Variational Tightening**

- Repeat message passing and Kalman update steps iteratively (inner loop) for inner_iterations, damping (coefficient \(\sim 0.5\)), to encourage stable convergence.

### 3.6. **Posterior Storage**

- Save updated parameters:
  - \(\hat{\mu}_i, S_i\) for each factor at timestamp \(t_{n+1}\).
  - Updated \(\mathbf{U}\) parameters (\(m, V\)).
  - \(\tau\) parameters.
- Keep current posterior for subsequent steps.

---

## 4. **Smoothing After Sequence Completion**

- After processing all data:
  - Run **RTS smoother**: backward pass to obtain full smoothed posterior over all \(\mathbf{Z}(t_i)\).
  - **Purpose**: To refine factor state estimates across entire sequence, allowing better probabilistic interpolation.

---

## 5. **Probabilistic Imputation at Arbitrary Timestamps (Equation 16-17)**

When asked to impute at a timestamp \(t^\star\):

1. **Identify nearest observed timestamps** \(t_k, t_{k+1}\):
   - Find indices \(k\) such that \(t_k < t^\star < t_{k+1}\).

2. **Extract posterior marginals**:
   - Means \(\mathbf{m}_k, \mathbf{m}_{k+1}\),
   - Covariances \(S_k, S_{k+1}\).

3. **Compute Transition Matrices**:
   - \(\mathcal{A}_1, \mathcal{A}_2\),
   - Covariance matrices \(\mathcal{Q}_1, \mathcal{Q}_2\),
   derived from the spectral-to-LTI functions for the relevant kernel.

4. **Calculate interpolated distribution**:
   \[
   q(\mathbf{z}(t^\star)) = \mathcal{N}(\mathbf{m}^\star, \mathbf{V}^\star)
   \]
   with:
   \[
   \mathbf{V}^\star = (\mathcal{Q}_1^{-1} + \mathcal{A}_2^\top \mathcal{Q}_2^{-1} \mathcal{A}_2)^{-1}
   \]
   \[
   \mathbf{m}^\star = \mathbf{V}^\star (\mathcal{Q}_1^{-1} \mathcal{A}_1 \mathbf{m}_k + \mathcal{A}_2^\top \mathcal{Q}_2^{-1} \mathbf{m}_{k+1})
   \]

5. **Construct the multivariate functional estimate**:
   - Recompose the imputed \(\mathbf{X}(t^\star)\):
     \[
     \mathbf{\hat{X}}(t^\star) = \mathbf{U} \mathbf{v}(t^\star)
     \]
   - \(\mathbf{v}(t^\star) \sim \mathcal{N}(\mathbf{m}^\star, \mathbf{V}^\star)\).

6. **Sample from the distribution** if needed, for uncertainty quantification.

---

## 6. **Implementation Details & Considerations**

- **Numerical Stability**:
  - Use Cholesky decompositions in matrix inversions.
  - Implement damping in message updates for stability.
- **Parallelization**:
  - Update message factors for each channel/factor independently.
- **Efficiency**:
  - Leverage vectorized operations.
  - Precompute matrices \(\mathbf{A}_n, \mathbf{Q}_n\) for each timestamp segment.
- **Modularity**:
  - Keep the functions for Kalman filtering, smoothing, message passing, and interpolation separate for clarity and maintainability.
- **Prediction and Imputation**:
  - For missing or unobserved timestamps, rely on the Gaussian predictive distributions for probabilistic imputation.
  - At each imputation, return mean estimates and associated covariance matrices for uncertainty.

---

## 7. **Summary of Method Flow in inference.py**

- Initialize the model posteriors using prior knowledge and initial observations.
- For each new data point:
  - Approximate likelihood messages.
  - Merge messages into variational posteriors (\(\tau, \mathbf{U}\)).
  - Propagate factor states via Kalman filter.
- After all data:
  - Apply RTS smoothing for full posterior refinement.
- At any arbitrary timestamp:
  - Find neighboring observed timestamps.
  - Use derived \(\mathcal{A}_i, \mathcal{Q}_i\) matrices to interpolate the factor posteriors.
  - Reconstruct \(\mathbf{X}(t^\star)\) with \(\mathbf{U}\) and \(\mathbf{v}(t^\star)\).
- Provide uncertainty estimates along with point estimates for imputation.

---

**This detailed methodology ensures the inference.py module effectively implements the online Bayesian inference, variational message passing, Kalman filtering, smoothing, and probabilistic imputation tailored for the BayOTIDE model as described in the paper and aligned with the provided configuration.**

## main.py

# main.py: Logic Analysis for the Entry Point Script of BayOTIDE Implementation

This main.py script acts as the orchestration layer, coordinating dataset loading, model initialization, online inference, and evaluation based on the specified configuration, datasets, and experimental procedures outlined in the paper and plan. The following is a detailed step-by-step logic analysis for implementing main.py, ensuring fidelity to the methodology and experimental design.

---

## 1. Import Required Modules and Packages

- Load necessary packages:
  - Standard: os, sys, logging, argparse (if command line args used).
  - Main components from the system's modules:
    - DatasetLoader (dataset_loader.py)
    - SpectralToLTID (spectral_to_LTI.py)
    - GPRFactorModel (gpr_model.py)
    - InferenceEngine (inference.py)
    - Evaluation (evaluation.py)
    - utils (utils.py)
  - External dependencies:
    - numpy as np
    - torch
    - yaml for configuration loading
    - Any linear algebra or plotting libraries for diagnostics/reporting.

## 2. Load Configuration

- Read "config.yaml" using yaml.safe_load or equivalent.
- Extract hyperparameters:
  - Data parameters: observed ratios, max sequence length, noise level.
  - Model parameters: D_r, D_s, kernel types and parameters.
  - Optimization: learning rate, damping epochs, inner iterations, damping factor.
  - Training/evaluation protocol: number of epochs, number of repeats, seed, etc.

## 3. Initialize Logging & Random Seed

- Set up logging for progress and debugging.
- Set random seeds (numpy, torch, etc.) to ensure reproducibility, based on a seed (e.g., 42).

## 4. Prepare Datasets and Data Loader

- For synthetic data:
  - Generate synthetic dataset with specified pattern types (trend and seasonality) over defined point count and range, using functions from utils.
  - Generate irregular timestamps for training: e.g., randomly sample 2000 points within [0,1] (or as per dataset).
  - Create missing data masks:
    - Mask ~50%, ~70% for evaluation.
    - Apply mask to generate observed/missing matrices.
- For real datasets:
  - Load datasets from file paths or URLs.
  - Normalize features if required.
  - Randomly mask data according to ratio.
  - Generate irregular timestamps if necessary.
- Organize data batches for training/validation/testing.
- Follow the operating procedure: masking and data splits per dataset as in the experiments.

## 5. Instantiate the Spectral-to-LTI Conversion

- Initialize SpectralToLTID class with kernel type and hyperparameters from config.
- Call construct_matrices() to compute \(\mathbf{F}, \mathbf{L}, \mathbf{Q}\) for each kernel:
  - For trend (matérn32) kernel.
  - For seasonal (periodic) kernel.
- These matrices encode the prior SDE models, to be used in subsequent inference.

## 6. Initialize the GPRFactorModel

- Instantiate GPRFactorModel with:
  - U: initialize as zero or small random matrix (D x (D_r + D_s))
  - D_r, D_s: from config
  - Kernel hyperparameters: tuned or default from config.
- Initialize state-space components through the spectral-to-LTI matrices:
  - Build companion form states for trend and seasonal factors.
  - Set initial posterior parameters, e.g., covariance and mean for factors.
  - Set prior for \(\tau\) (noise precision) as Gamma with given a0, b0.

## 7. Instantiate Inference Engine

- Wrap the model with InferenceEngine:
  - Inputs:
    - The model object.
    - Dataset (masked data, timestamps).
    - Hyperparameters for online inference:
      - Damping epochs.
      - Inner EP/CEP iteration count.
      - Smoothing flag.
- The inference engine will manage:
  - Online updates upon arrival of new data.
  - Kalman filter updates for factor states.
  - CEP message passing for variational posterior updates.
  - Probabilistic inference over arbitrary timestamps.

## 8. Run the Data Loading & Initialization Sequence

- Load dataset:
  - For synthetic: generate the data, apply mask, create irregular timestamps.
  - For real datasets: load, normalize, mask, and generate timestamps.
- Log dataset properties: number of channels, points, missing ratio.
- Initialize any data structures for tracking imputation metrics during runtime.

## 9. Perform Online Data Streaming & Imputation Loop

- For each timestamp (or batch):
  - Feed the new data point(s) (masked/uncensored) into InferenceEngine:
    - Call online_update() with current observed data and timestamps.
    - Internally, CEP message approximation and Kalman filtering are performed.
  - At each update:
    - Record posterior parameters, factor states.
    - Impute missing values:
      - For fixed timestamps of evaluation, call the impute() method to get predictive distributions (mean and variance).
      - Store imputed values and posterior samples.

- Optional:
  - Damping iterations per timestamp to ensure numerical stability.
  - Inner iterations for message passing fixed points.
  - Run RTS smoothing after sequence completion for full posterior reconstruction.

## 10. Evaluation & Metrics Calculation

- After sequence inference:
  - For each missing timestamp in the evaluation set:
    - Compare imputed distributions (means, variances) with ground truth.
  - Compute evaluation metrics:
    - MAE, RMSE on point estimates.
    - CRPS: using the multiple posterior samples.
    - NLLK: Negative log-likelihood from the predictive distribution.
- Repeat the entire experiment multiple times (e.g., 5 runs), averaging metrics for robustness.

## 11. Results Recording & Reporting

- Collect metrics:
  - Per dataset, per observed ratio.
  - For different models (e.g., trend-only, fixed weights, full BayOTIDE).
- Save metrics to dictionary or output files.
- Optionally, generate plots:
  - Imputed series with uncertainty intervals.
  - Learned functional factors (trend/seasonality).
  - Online performance curves over time.

## 12. Final Smoothing & Posterior Estimation

- After all timestamps processed:
  - Run RTS smoother to refine full posterior over factors \(\mathbf{Z}(t)\).
  - Store the full smoothed posterior for further analysis or debugging.

## 13. Save & Export Results

- Save posterior estimates, imputed series, and metrics.
- Save hyperparameters and model configurations used.
- Export figures or summaries as needed.

---

## 14. Considerations & Summary Points

- Modular code ensures that each step (dataset handling, spectral kernel conversion, online inference, evaluation) is isolated.
- Ensure that the online updates and message passing conform exactly to the formulas described, especially CEP and Kalman filter equations.
- Default initializations and hyperparameter tuning are guided by the provided config.yaml.
- The entire pipeline is designed to be reproducible, with fixed seeds, consistent data splits, and logs.

---

This detailed analysis guides the implementation of main.py, aligning all components to achieve the experimental fidelity as described in the paper, plan, and configuration.

## requirements.txt

# requirements.txt

# Core numerical computations and linear algebra support
numpy==1.21.0
scipy==1.7.0

# Deep learning framework for neural components (e.g., parameter optimization, tensors)
torch==1.9.0

# Probabilistic and filtering modules
filterpy==1.4.5

# Optional: For advanced probabilistic modeling (e.g., message passing, variational inference)
# (Install if implementing complex EP/CEP algorithms or for additional probabilistic utilities)
pyro-ppl==1.8.0

# Additional packages may be required for data handling, parallel processing, or specific utilities:
# pandas (for dataset loading and preprocessing)
# (If needed, include)
# pandas>=1.3.0

# Note: Do not include packages not directly used in implementation; keep dependencies minimal and aligned with functionality.

# Explanation:
# This set of packages aligns with the implementation plan:
# - numpy and scipy for matrix operations, spectral analysis, solving Lyapunov equations, and SDE conversions.
# - torch for any neural network components, hyperparameter tuning, or automatic differentiation.
# - filterpy for Kalman filtering and smoothing implementations.
# - pyro-ppl as an optional library to facilitate variational inference, message passing, or probabilistic programming if chosen.
# Ensure that all packages are compatible and support the necessary linear algebra and probabilistic routines needed for online Gaussian Process-based inference, state-space modeling, and continuous-time functional decomposition.

## spectral_to_LTI.py

# Spectral-to-LTI Conversion Logic Analysis

## Purpose Overview:
The `spectral_to_LTI.py` module defines the `SpectralToLTID` class, which converts specific stationary kernel functions—namely, Matérn (with \(\nu=3/2\)) and periodic kernels—into their equivalent linear time-invariant (LTI) stochastic differential equations (SDEs), represented in state-space form.

This conversion facilitates scalable, online inference, leveraging classical Kalman filtering, by replacing kernel covariance functions with explicit differential equations characterized by matrices \(\mathbf{F}\), \(\mathbf{L}\), and \(\mathbf{Q}\). These matrices capture the dynamics of the Gaussian process (GP) prior for each factor.

---

## 1. Input Specifications:
- **Kernel type** (`kernel_type`): `'matern32'` or `'periodic'` (from config).
- **Hyperparameters**:
  - For `matern32`:
    - `length_scale` (ℓ)
    - `variance` (\(\sigma^2\))
  - For `periodic`:
    - `length_scale` (ℓ)
    - `period` (p)
    - `variance` (\(\sigma^2\))
- **Output**: Matrices \(\mathbf{F}\), \(\mathbf{L}\), and \(\mathbf{Q}\), defining the SDE:
  \[
  d\mathbf{z}(t) = \mathbf{F}\mathbf{z}(t)dt + \mathbf{L} dw(t)
  \]
  
## 2. Core Components:
- **Class Initialization**:
  - Accepts kernel type and hyperparameters.
  - Validates inputs.
  - Calls appropriate methods to produce \(\mathbf{F}\), \(\mathbf{L}\), \(\mathbf{Q}\).

- **Kernel-specific Matrix Construction**:
  - For `matern32`:
    - Derive \(\mathbf{F}\), \(\mathbf{L}\), \(\mathbf{Q}\) based on spectral analysis.
    - \(\mathbf{F}\) (state transition matrix) typically a \(2 \times 2\) matrix for \(\nu=3/2\).
    - \(\mathbf{L}\): process noise vector.
    - \(\mathbf{Q}\): diffusion term, scaled by the variance and length-scale.

  - For `periodic`:
    - Approximate periodic kernel via a sum of sinusoidal components.
    - Use spectral representation to construct block-diagonal matrices, each block corresponding to a frequency \(\frac{2\pi j}{p}\), \(j=1,\dots,n\).
    - Each sinusoidal component results in a \(2 \times 2\) matrix \(\mathbf{F}_j\), \(\mathbf{L}_j\), and covariance \(\mathbf{Q}_j\).

---

## 3. Computation Details:
### For `matern32`:
- \(\nu=3/2\) kernel:
  - State dimension \(m=1\) (i.e., 2 states: position and velocity).
  - System matrices:
    \[
    \mathbf{F} = 
    \begin{bmatrix}
    0 & 1 \\
    -\lambda^2 & -2\lambda
    \end{bmatrix}
    \]
  - Process noise covariance:
    \[
    \mathbf{Q} = \mathbf{L} q_s \mathbf{L}^T 
    \]
    where:
    \[
    \mathbf{L} = 
    \begin{bmatrix}
    0 \\ 1
    \end{bmatrix}
    \]
    and
    \[
    q_s = 4 \lambda^3 \sigma^2
    \]
    with \(\lambda = \sqrt{3}/\text{length\_scale}\).

- \(\mathbf{P}_\infty\): steady-state covariance; computed via Lyapunov equation:
    \[
    \mathbf{F} \mathbf{P}_\infty + \mathbf{P}_\infty \mathbf{F}^T + \mathbf{L} q_s \mathbf{L}^T = 0
    \]

### For `periodic`:
- Model as a sum of \(n\) harmonic oscillators:
  - Each frequency \(j\) has:
    \[
    \mathbf{F}_j = 
    \begin{bmatrix}
    0 & -\frac{2\pi j}{p} \\
    \frac{2\pi j}{p} & 0
    \end{bmatrix}
    \]
  - Process noise intensity \(q_j^2\):
    \[
    q_j^2 = 2 \mathrm{I}_j (\ell^{-2})/ \exp(\ell^{-2})  \quad \text{(per the spectral approximation)}
    \]
  - Each harmonic contributes a block:
    \[
    \mathbf{Q}_j = q_j^2 \mathbf{I}_2
    \]
- The overall state-space is block-diagonal, combining all \(\mathbf{F}_j\)s and \(\mathbf{Q}_j\)s.

---

## 4. Implementation Details:
- **Matrix Exponentials**:
  - For each \(\mathbf{F}\), compute state transition matrix \(\mathbf{A}_n = \exp(\mathbf{F} \Delta_t)\). Use \(\text{scipy.linalg.expm}\).
- **Lyapunov Equation**:
  - Solve for \(\mathbf{P}_\infty\) using `scipy.linalg.solve_lyapunov`.
- **Handling Parameters**:
  - Convert hyperparameters into \(\lambda\), frequencies, and covariance scalings.
  - Handle optional kernel hyperparameters for periodic kernel.
- **Output**:
  - `get_matrices()` method returns \(\mathbf{F}\), \(\mathbf{L}\), \(\mathbf{Q}\).

## 5. Summary:
- **Input**:
  - Kernel type (`matern32`, `periodic`)
  - Hyperparameters (`length_scale`, `variance`, and `period` for periodic)
- **Output**:
  - \(\mathbf{F}\): state matrix
  - \(\mathbf{L}\): noise input matrix
  - \(\mathbf{Q}\): spectral density scaled covariance
- **Use**:
  - Matrices feed into the discrete transition equations used in Kalman filter and smoother for online GP inference.

---

## Final Notes:
- Ensure proper handling of small or large length scales/powers in the matrices.
- Confirm positive definiteness of \(\mathbf{Q}\) via spectral strategies.
- Modularize code for straightforward extension to other kernels if needed.

This detailed plan will guide the precise implementation of spectral to LTI state-space conversion, ensuring fidelity with theoretical derivations and the paper's methodology.

## utils.py

# Logic Analysis for utils.py

This module is integral for supporting the core components of BayOTIDE, providing efficient, numerically stable, and reusable functions for matrix operations, statistical computations, and model initializations. It should encapsulate common mathematical routines, expectation and covariance calculations, and utility tools tailored to the spectral-to-LTI and probabilistic inference processes.

Below is a detailed, step-by-step analysis of required functionalities, their implementation logic, and your expected behavior.

---

### 1. **Matrix and Linear Algebra Utilities**

- **Matrix Operations**:
  - Cholesky decomposition for positive-definite matrices (with checks and fallback if needed).
  - Matrix inverse and solve routines that are numerically stable.
  - Efficient multiplication and addition routines—preferably leveraging NumPy's optimized functions.

- **Eigen-decompositions / Spectral Functions**:
  - For spectral analysis and constructing state-space matrices, functions for spectral decomposition or eigendecomposition may be necessary, especially for kernels requiring spectral density approximations.
  - Eigenvalues, eigenvectors computations for possibly approximating periodic kernels.

- **Matrix Exponentials**:
  - Compute \(\mathbf{A}_n = \exp(\mathbf{F} \Delta_n)\) for arbitrary \(\mathbf{F}\).
  - Use SciPy's `scipy.linalg.expm()` for small matrices.

- **Lyapunov Equations**:
  - Fast solution for steady-state covariance matrices using `scipy.linalg.solve_lyapunov()`.

- **State-space Construction Utilities**:
  - Functions that, given kernel hyperparameters, generate \(\mathbf{F}\), \(\mathbf{L}\), and \(\mathbf{Q}\) matrices per the appendix; this will be called by spectral_to_LTI.py.

---

### 2. **Probability and Statistical Functions**

- **Expectations and Covariances**:
  - Compute expected values \(\mathbb{E}_q[ \tau ]\), \(\mathbb{E}_q[\mathbf{u}^d]\), \(\mathbb{E}_q[\mathbf{u}^d \mathbf{u}^{dT}]\), and their variances as Gaussian moments.
  - Expectations involving Gaussian distributions, such as \(\mathbb{E}_q[\mathbf{Z}(t)]\), \(\operatorname{Cov}_q[\mathbf{Z}(t)]\).

- **Posterior Moments for Message Passing**:
  - Given prior parameters, compute the updated mean and covariance (or shape/rate for Gamma) after message passing steps.
  - Functions for moments of Gamma distribution: shape \(\alpha\), rate \(\beta\) expected value and variance.

- **CRPS and NLLK Computation**:
  - For probabilistic metrics, implement functions to calculate CRPS given ground truth and Gaussian posterior \(\mathcal{N}(\mu, \sigma^2)\).
  - Compute log-likelihood of observations under the learned Gaussian or Gamma distributions.

---

### 3. **Kernel and Spectral Function Computations**

- **Kernel Matrix Construction**:
  - Functions to compute the kernel \(\kappa(t, t')\) for specified kernel types ('matern32', 'periodic'), hyperparameters as input.
  - Supporting functions to generate the off-diagonal covariance matrices for each factor set, given observed timestamps.

- **Spectral Density to State-Space Conversion**:
  - Functions converting spectral density parameters into \(\mathbf{F}\), \(\mathbf{L}\), and \(\mathbf{Q}\), based on formulas in the appendix.
  - This step involves solving polynomial roots or approximations, which are then formulated as matrix parameters.

---

### 4. **Initializations**

- **Initial Covariance \(\mathbf{P}_\infty\)**:
  - Compute steady-state covariance by solving Lyapunov equations with `scipy.linalg.solve_lyapunov()`.
- **Initial State Vectors**:
  - Initialize posterior means at start (e.g., zeros or small random noise).
- **State Variables**:
  - For each factor, initialize their companion states \(\mathbf{z}(t)\)'s mean and covariance.
- **Noise and Prior Hyperparameters**:
  - Encapsulate defaults for the Gamma prior parameters (\(a_0, b_0\)), e.g., shape and rate.
  - Provide functions for setting or updating priors dynamically if needed.

---

### 5. **Likelihood and Expectation Helpers**

- **Likelihood Functions**:
  - For Gaussian likelihoods `N(y | \mu, \sigma^2)`, functions to evaluate log-likelihoods and generate samples.
- **Prior Sampling / Expectation**:
  - Sampling functions for initial prior distributions (\(\mathbf{U}\), \(\tau\)).
  - Functions to compute expectation over Gaussian/posterior parameters, necessary for message passing and hyperparameter updates.

---

### 6. **Numerical Stability and Efficiency**

- Use `np.clip()` or similar methods to prevent negative or zero variances.
- Damping functions for iterative updates in CEP, e.g., mix old and new parameters.
- Vectorization over channels and factors for speed.
- Batch processing where possible, e.g., multiple Gaussian operations.

---

### 7. **Miscellaneous Utilities**

- **Time index functions**:
  - Convert timestamps into matrix indices, handle irregular sampling.
- **Hyperparameter Management**:
  - Functions for reading, setting, and updating the kernel hyperparameters.
  
- **Data Normalization**:
  - Optional functions to normalize features during data preprocessing.

---

### 8. **Summary & organization**

Partition the utility functions into logical groups:

- `mat_utils`: LM operations, Inversion, matrix exponentials.
- `stats`: Expectation, covariance, likelihood, distribution moments.
- `kernel_utils`: Kernel measurements, spectral density, spectral-to-LTI conversion.
- `state_space`: Functions for creating and solving state-space equations.
- `initialization`: Priors, covariance initializations.
- `numerical`: Damping, clipping, stability checks.

Ensure functions are templated to accept hyperparameters as inputs, enabling flexible kernel configurations and model tuning without code duplication.

---

This detailed analysis provides a concrete foundation for developing `utils.py` that is robust, reusable, and aligned with the core theory and implementation flow of BayOTIDE.

