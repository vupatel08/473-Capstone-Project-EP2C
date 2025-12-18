# Planning Phase

This document contains the complete planning phase output, including the overall plan, architecture design, logic design, and configuration.

## 1. Overall Plan

Certainly! Here is a structured, detailed plan to reproduce the methodology and experiments described in the paper “Task-Agnostic Machine-Learning-Assisted Inference” by Miao et al. The plan covers key methodological steps, experimental design, data needs, hyperparameters, evaluation metrics, and potential implementation details, in a way that will facilitate later code development.

---

## 1. Overall Objective
- Implement the PSPS (PoSt-Prediction Summary-statistics-based) inference framework which allows valid, efficient, task-agnostic inference (e.g., mean, regression, quantile, and non-M-estimation tasks) leveraging ML predictions on *both* labeled and unlabeled data.
- Enable federated, privacy-preserving analysis that uses only summary statistics, not raw data.
- Validate via simulated and real data experiments, assessing coverage, power, FDR, and variance reduction.

---

## 2. Core Methodology of PSPS
### 2.1 Preprocessing and ML Model Predictions
- **Input**:
  - Labeled data \(\mathcal{L} = \{(\mathbf{X}_i, Y_i)\}\),
  - Unlabeled data \(\mathcal{U} = \{\mathbf{X}_j\}\),
  - ML model \(\hat{f}(\cdot)\) trained on \(\mathcal{L}\).
- **Steps**:
  - Generate ML predictions:
    - \(\hat{f}_i = \hat{f}(\mathbf{X}_i)\) for labeled data.
    - \(\hat{f}_j = \hat{f}(\mathbf{X}_j)\) for unlabeled data.
- **Note**: training \(\hat{f}\) should be independent of the specific inference task; no need for retraining per inference task.

### 2.2 Summary Statistics Extraction
- **Apply** existing classical analysis routines (e.g., OLS, logistic regression, quantile estimation) separately on:
  - Labeled data: obtain \([\hat{\theta}_\mathcal{L}, \widehat{\mathrm{Var}}(\hat{\theta}_\mathcal{L})]\),
  - Data with ML-predicted outcomes: \([\hat{\eta}_\mathcal{L}, \widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{L})]\),
  - Unlabeled data ML predictions: \([\hat{\eta}_\mathcal{U}, \widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{U})]\).
- **Sufficient Statistics**:
  - For mean estimation (example): sample mean, sample variance.
  - For regression: estimates of coefficients, covariance matrices.
  - For other tasks: derive or approximate the sufficient statistics as per classical routines.
- **Note**: All these transformations are done at the summary-statistics level, not raw data.

### 2.3 Asymptotic Variance/Covariance Calculation
- **Estimate** these variances/covariances for each summary statistic set via bootstrap or analytical formulas.
- **Key elements**:
  - \(\widehat{\mathrm{Var}}(\hat{\theta}_\mathcal{L})\),
  - \(\widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{L})\),
  - \(\widehat{\mathrm{Cov}}(\hat{\theta}_\mathcal{L}, \hat{\eta}_\mathcal{L})\),
  - \(\widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{U})\).
- Bootstrap should be performed with, e.g., 200 replications on labeled data for variance estimation.

### 2.4 Construction of the Debiased Estimator \(\hat{\theta}_{PSPS}\)
- **Compute weights**:
  - \(\hat{\omega}_0 = (\widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{L}) + \rho \widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{U}))^{-1} \widehat{\mathrm{Cov}}(\hat{\theta}_\mathcal{L}, \hat{\eta}_\mathcal{L})\),
  - \(\rho = n / N\), the ratio of labeled to unlabeled sample sizes.
- **Final inference estimate**:
  \[
  \hat{\theta}_{PSPS} = \hat{\theta}_\mathcal{L} + \hat{\omega}_0^T (\hat{\eta}_\mathcal{U} - \hat{\eta}_\mathcal{L})
  \]
- **Variance of \(\hat{\theta}_{PSPS}\)**:
  - Use the algebraic form provided, estimated via bootstrap or plug-in formulas for the covariance matrices.
- **Asymptotic Normality**:
  - Under large-sample assumptions, \(\hat{\theta}_{PSPS}\) is asymptotically normal around \(\theta^*\).

---

## 3. Experimental Design & Data Requirements
### 3.1 Datasets
- **Simulated Data**:
  - Tasks include mean, linear, logistic, quantile, IV regression, NB regression, debiased Lasso, Wilcoxon tests.
  - Data generated via the specified models with varying labeled/unlabeled sizes.
- **Real Data**:
  - UK Biobank for vQTL analysis (as described).
  - Additional biomedical/genetic datasets for variants of inference tasks.
  
### 3.2 Simulation Specification
- **Data Generation**:
  - Reproduce covariates, outcomes, and ML prediction procedures.
  - Vary sample sizes for labeled (e.g., \(n=500\)) and unlabeled (e.g., 1000, 2500, 5000, 10000).
- **ML Model**:
  - Use RandomForest for outcome prediction.
  - Hyperparameters (e.g., number of trees = 500, depth tuning as per cross-validation).

### 3.3 Experiments
- **Tasks**:
  - Already implemented in software: mean, regression, quantile estimation.
  - Not implemented: IV regression, NB regression, debiased Lasso, Wilcoxon, FDR.
- **Metrics**:
  - Coverage probability of confidence intervals.
  - Width of CIs.
  - Power (for hypothesis tests).
  - FDR control rate (for multiple tests).
- **Repeated runs**:
  - At least 1000 simulated replications.
  - Bootstrap variance estimation with 200 bootstrap samples per run.

### 3.4 Hyperparameters & Tuning
- **Bootstrap**:
  - 200 replications for variance estimation.
- **ML predictions**:
  - Trained once; no task-specific retraining needed.
  - Use cross-validation for hyperparameters when necessary.
- **Debiasing weights**:
  - Estimated via algebraic formulas; no tuning required.

---

## 4. Implementation Details & Considerations
### 4.1 Software
- Use R for simulation (per paper) and Python (e.g., scikit-learn, statsmodels, numpy) for later production.
- Bootstrap: use `boot` package or manual bootstrap loops.
- Variance estimation: bootstrap or analytical asymptotics.
- Classical routines: `lm()`, `glm()`, `quantile()` in R, or equivalent in Python/statsmodels.

### 4.2 Computing Efficiency
- Bootstrap variance estimates are the major computational bottleneck.
- Variance/covariance algebra suffices to avoid repeated fitting.
- For scalability, cache intermediate matrices.

### 4.3 Privacy & Federated Settings
- All steps rely only on summary statistics.
- Simulation should mimic federated data: label in source, unlabeled elsewhere.
- Author-provided code (if available) or mimic schema for data sharing.

---

## 5. Special Notes & Reminders
- Confirm that the analysis routines are correctly applied to both original and ML-predicted outcome data.
- Carefully implement the variance estimation step (bootstrap recommended).
- Ensure the covariance matrices are estimated precisely for the debiasing formulas.
- Be explicit about data splits, bootstrap procedures, and random seeds for reproducibility.
- Validate the asymptotic normality assumptions in finite samples by checking coverage across simulations.
- Document all hyperparameters and software versions for reproducibility.

---

## 6. Summary of Action Items
1. Prepare datasets per experimental plans (simulate or load real).
2. Train ML models \(\hat{f}\) once, store predictions.
3. Run classical analysis routines to extract summary stats.
4. Estimate variances/covariances via bootstrap.
5. Compute the weights \(\hat{\omega}_0\).
6. Generate the PSPS estimator and variance.
7. Conduct inference (confidence intervals, hypothesis tests).
8. Aggregate results: coverage, CI width, power, FDR.
9. Ensure edge cases (small samples, model misspecification) are considered.
10. Document all steps and assumptions explicitly.

---

This roadmap provides a comprehensive guide to implement the PSPS framework and reproduce the experiments described. The key is to treat the process as modular: prediction, summary statistic extraction, variance estimation, and inference calculation, with detailed validation at each step.

## 2. Architecture Design

[CONTENT]
{
    "Implementation approach": "Build a modular Python package that implements the PSPS framework for task-agnostic inference using only summary statistics. Core components include data simulation/loading, ML prediction generation, analysis routines, variance/covariance estimation via bootstrap, and the final debiased inference. Use widely adopted open-source libraries: scikit-learn for ML, numpy/scipy for numerical operations, statsmodels for classic statistical models, and joblib for parallel bootstrap variance estimation. Structure the code into main orchestration, dataset handling, model training/predictions, summary statistic extraction, variance estimation, and inference modules, enabling flexible extension for various tasks.",
    "File list": [
        "main.py",
        "dataset_loader.py",
        "model.py",
        "analysis.py",
        "variance_estimation.py",
        "inference.py",
        "utils.py"
    ],
    "Data structures and interfaces": "
classDiagram
    class Main {
        +__init__(config: dict)
        +run()
    }
    class DatasetLoader {
        +__init__(data_params: dict)
        +load_data() -> dict
    }
    class MLModel {
        +__init__(model_params: dict)
        +train(X: np.ndarray, y: np.ndarray) -> None
        +predict(X: np.ndarray) -> np.ndarray
    }
    class AnalysisRoutine {
        +__init__(model: MLModel, data: dict)
        +compute_summary_statistics() -> dict
    }
    class VarianceEstimator {
        +__init__(data: dict, analysis: AnalysisRoutine)
        +bootstrap_variance(estimator_fn: callable, n_bootstrap: int) -> dict
    }
    class Inference {
        +__init__(summary_stats: dict, variability: dict)
        +compute_debiased_estimate() -> float
        +compute_variance() -> float
        +construct CI(alpha: float) -> tuple
        +hypothesis_test() -> dict
    }
    Main --> DatasetLoader
    Main --> MLModel
    Main --> AnalysisRoutine
    Main --> VarianceEstimator
    Main --> Inference
    AnalysisRoutine --> MLModel
    VarianceEstimator --> AnalysisRoutine
    Inference --> AnalysisRoutine
    Inference --> VarianceEstimator
",
    "Program call flow": "
sequenceDiagram
    participant C as Main
    participant DL as DatasetLoader
    participant M as MLModel
    participant AR as AnalysisRoutine
    participant VE as VarianceEstimator
    participant INF as Inference
    C->>DL: load_data()
    DL-->>C: dataset
    C->>M: initialize(model_params)
    M->>M: train(X_train, y_train)
    M->>DL: predict(X_eval)
    DL-->>C: predictions
    C->>AR: create analysis routine(M, dataset)
    AR->>AR: compute_summary_statistics()
    AR-->>C: summary_stats
    C->>VE: create VarianceEstimator(dataset, AR)
    VE->>VE: bootstrap_variance(analysis_fn, 200)
    VE-->>C: variance estimates
    C->>INF: create Inference(summary_stats, variance)
    INF->>INF: compute_debiased_estimate()
    INF->>INF: compute_variance()
    INF->>INF: construct CI(0.05)
    INF->>INF: hypothesis_test()
    Note over INF: Results: estimate, CI, p-value
",
    "Anything UNCLEAR": "Clarify whether specific models (linear, logistic, quantile) should be integrated, and detail expected input formats for analysis routines and variance estimation for consistent API implementation."
}
[/CONTENT]

## 3. Logic Design & Task List

[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "scipy==1.7.0",
        "scikit-learn==0.24.2",
        "statsmodels==0.13.0",
        "joblib==1.0.1",
        "matplotlib==3.4.3"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "dataset_loader.py",
            "Handles loading datasets, simulating data, and preprocessing. Implements the DatasetLoader class with methods to load real data (e.g., UK Biobank) or generate synthetic data as per specified models. Resolves dependencies on data formats and splitting labeled/unlabeled. Feeds data into subsequent modules and manages data structure formats."
        ],
        [
            "model.py",
            "Defines the ML prediction models used in the pipeline, e.g., RandomForestRegressor from scikit-learn. Implements the MLModel class with methods for training on labeled data and predicting outcomes. This class is shared across analysis routines—imported into analysis.py."
        ],
        [
            "trainer.py",
            "Contains the training pipeline for ML models. Implements a Trainer class that manages training, hyperparameter tuning, and validation. Relies on dataset_loader.py for data input, and outputs trained models for prediction. Supports retraining and cross-validation if needed."
        ],
        [
            "analysis.py",
            "Performs estimation of analysis-specific summary statistics (means, regression coefficients, quantiles, etc.). Implements AnalysisRoutine class with methods to perform classical analysis routines, extracting sufficient statistics for inference, and is called after model predictions are generated."
        ],
        [
            "variance_estimation.py",
            "Responsible for variance and covariance estimation via bootstrap or analytical formulas. Implements VarianceEstimator class with methods to bootstrap variance estimates for each summary statistic, providing inputs for the PSPS correction step."
        ],
        [
            "inference.py",
            "Handles debiased estimation, constructing confidence intervals, and hypothesis tests. Implements Inference class that takes summary statistics and variances, computes the PSPS estimate via the formulas provided, and outputs final point estimates, standard errors, CIs, and p-values."
        ],
        [
            "main.py",
            "The orchestration script. Initializes dataset loader, trains models, runs analysis routines, performs bootstrap variance estimation, and finally executes inference. Coordinates calling sequence of data loading, model training & prediction, analysis, variance estimation, and reporting results."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "analysis.py",
        "variance_estimation.py",
        "inference.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0 ... (not applicable, no API required for core pipeline)",
    "Shared Knowledge": "Functions for bootstrap resampling, evaluation metrics, and numerical utility functions (e.g., matrix algebra, covariance updates) are shared across variance_estimation.py, inference.py, and analysis.py.",
    "Anything UNCLEAR": "Clarify whether the code should be optimized for large-scale or federated setups, and specific data formats expected from real datasets like UK Biobank."
}
[/CONTENT]

## 4. Configuration

## Code: config.yaml
```yaml
## config.yaml
training:
  # As paper does not specify, use typical default for random forest and classic models
  learning_rate: null  # Not applicable for tree-based models; placeholder
  batch_size: null     # Not applicable for scikit-learn models; placeholder
  epochs: null         # Not applicable for scikit-learn models; placeholder

model:
  type: RandomForestRegressor
  n_estimators: 500
  max_depth: null  # Use default, unless hyperparameter tuning is specified later
  random_state: 42

data:
  synthetic: true  # Set to false if loading real datasets (e.g., UK Biobank)
  labeled_size: 500
  unlabeled_sizes: [1000, 2500, 5000, 10000]

variance_estimation:
  bootstrap_samples: 200

analysis:
  method: regression  # Options: 'mean', 'regression', 'quantile', etc. (set based on use case)

experiment:
  repetitions: 1000

output:
  results_dir: "results/"
```

---

**Note:** Full configuration is available in `planning_config.yaml`
