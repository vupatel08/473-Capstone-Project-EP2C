# Analysis Phase

This document contains the detailed logic analysis for each file in the implementation.

## analysis.py

# Logic Analysis for `analysis.py`

This file is responsible for implementing the core analysis routines that extract the relevant summary statistics used in the PSPS framework, based on the specified statistical task (`method`). The class `AnalysisRoutine` will provide methods to perform classical statistical procedures—such as mean estimation, regression, quantile estimation, etc.—using the outcome predictions (for predicted data) and possibly observed outcomes (for labeled data), returning the sufficient statistics necessary for the subsequent variance estimation and debiased inference.

Below is a detailed, step-by-step logical blueprint for `analysis.py`:

---

## 1. Class Design: `AnalysisRoutine`

### 1.1 Initialization (`__init__`)
- **Inputs**:
  - `model`: The trained ML prediction model, used optionally for residual analysis or variance computation if needed.
  - `data`: A dictionary containing datasets:
    - Labeled data: features (`X_lab`), observed outcomes (`Y_lab`).
    - Predicted outcomes for labeled data: (`f_hat_lab`).
    - Unlabeled data: features (`X_unlab`), predictions (`f_hat_unlab`).
  - `method`: String indicating the task type — `'mean'`, `'regression'`, `'quantile'`, `'IV'`, `'NB'`, `'Lasso'`, `'Wilcoxon'`, etc.

- **Purpose**: Store these inputs for use in the computation of the specific summary statistics.

### 1.2 Methods to Implement

For each task, a dedicated method will:
- Perform the traditional analysis routine, e.g.,
  - `compute_mean()` for mean estimation.
  - `compute_regression()` for linear/regression/logistic regression.
  - `compute_quantile()` for quantile estimation.
  - `compute_iv()` for instrumental variable estimation.
  - `compute_nb()` for negative binomial regression.
  - `compute_wilcoxon()` for hypothesis testing (or the median / rank-based estimators).
- Extract relevant point estimates, standard errors, residuals, covariance matrices, etc., as sufficient statistics.
- Return these in a dictionary, e.g.,
  - `{'estimate': ..., 'variance': ...}`
  - For regression, return coefficients and variance-covariance matrices.
  - For mean, just the mean and variance.
  - For quantile, the quantile estimate and its standard error.

### 1.3. compute_summary_statistics()
- Main dispatcher method:
  - Checks the `method` and calls the corresponding routine.
  - Collects all relevant sufficient statistics.
  - Returns a dictionary containing:
    - The point estimate(s),
    - Variance estimate(s),
    - Covariance matrices if needed,
    - Additional info if justified (e.g., residuals, weights).

---

## 2. Implementation Details per Method

### 2.1 Mean Estimation (`compute_mean`)
- **Input**:
  - Both observed outcomes (`Y_lab`) and predicted outcomes (`f_hat_lab`, `f_hat_unlab`).
- **Procedure**:
  - For labeled data:
    - Compute sample mean: `mean_Y_lab = np.mean(Y_lab)`
    - Compute sample variance: `var_Y_lab`
  - For unlabeled data with predicted outcomes:
    - Compute mean of predicted outcomes: `mean_f_unlab = np.mean(f_hat_unlab)`
    - Variance of predicted outcomes: possibly via bootstrap or analytical estimate.
- **Sufficient statistics**:
  - Mean outcome on labeled data.
  - Variance for asymptotic variance estimation.
  - Can also store the mean of predicted outcomes in the unlabeled data.

### 2.2 Regression (linear/logistic)
- **Inputs**:
  - X (features), observed Y
  - Predicted outcomes (for residual calculations or validation)
- **Procedure**:
  - Fit the regression model using the dataset:
    - For linear regression: OLS:
      - Compute coefficients: `beta_hat = (X^T X)^{-1} X^T Y`
      - Covariance matrix of Beta: `(sigma^2)*(X^T X)^{-1}`, with residual variance `sigma^2`.
    - For logistic regression: use `statsmodels` `Logit`:
      - Fit model, extract coefficients, covariance.
  - Sufficient statistics:
    - Coefficients estimate,
    - Covariance matrix,
    - Residual sum of squares or deviance for variance estimation.

### 2.3 Quantile Estimation
- **Procedure**:
  - Use `np.quantile()` on observed outcomes:
    - For observed labeled data (`Y_lab`), directly get the quantile.
    - Variance or standard error of quantile can be estimated via bootstrap or asymptotic formulas.
  - For predictions, quantiles of predicted outcomes in unlabeled data may be used for inference on variability.

### 2.4 IV, NB, Lasso, Wilcoxon tests
- For more complex methods like IV, NB, or high-dimensional Lasso:
  - Use appropriate models/packages (`statsmodels`, `sklearn`, or custom solvers).
  - For IV:
    - Use two-stage least squares (2SLS) with the instrument.
    - Extract estimated coefficient and variance.
  - For NB:
    - Use `statsmodels.discrete.count_model.NegativeBinomial`.
  - For Lasso:
    - Use `sklearn.linear_model` with a debiasing procedure to extract estimates.
  - For Wilcoxon:
    - Use `scipy.stats.mannwhitneyu()` or `ranksums()`.

---

## 3. Variance and Covariance Estimation
- For each analysis output, implement bootstrap procedures:
  - Resample the same dataset many times (~200) with replacement (preferably on labeled data for variance of mean/regression).
  - Recompute summary statistics for each bootstrap replicate.
  - Compute the empirical variance-covariance matrix across bootstrap replications.
- For non-bootstrap methods, derive analytical variance estimates, if available.

## 4. Outputs
- Return a dictionary, e.g.,
```python
{
    'estimate': estimated_value,
    'variance': estimated_variance,
    'covariance': covariance_matrix (if relevant),
    'additional_info': residuals, weights, etc. (optional)
}
```
- These stats are used downstream by the `inference.py` module to compute the PSPS estimate, confidence intervals, and conduct hypothesis tests.

---

## 5. Practical Implementation Notes
- The class should normalize all inputs into compatible numpy arrays or pandas DataFrames.
- For bootstrap:
  - Parallelize if possible, using `joblib` or `multiprocessing`.
  - Store bootstrap estimates for variance computation.
- Modular structure:
  - Implement each analysis method as a separate private method.
  - Include a main method `compute_summary_statistics()` for dispatch based on `method`.
- **Parameter handling**:
  - Use the configuration to set hyperparameters (e.g., bootstrap_samples).
  - Use defaults for models unless specific hyperparameters are provided.

---

## 6. Data Format Assumptions
- Input datasets are dictionaries:

```python
data = {
    'X_lab': np.ndarray,
    'Y_lab': np.ndarray,
    'f_hat_lab': np.ndarray,
    'X_unlab': np.ndarray,
    'f_hat_unlab': np.ndarray
}
```

- Consistent sizes and orders:
```python
len(Y_lab) = n (labeled sample size),
len(f_hat_lab) = n,
len(X_lab) = n,
len(f_hat_unlab) = N,
len(X_unlab) = N.
```

- Ensure proper data types (`float`, `int`) and handle missing or corrupted inputs with validation or assertions.

---

## 7. Validation & Testing
- Build unit tests for each method:
  - Generate synthetic data with known properties.
  - Confirm estimates recover known parameters.
  - Verify bootstrap variance estimates match analytical calculations in simple cases.
- Check that variances decrease with larger sample sizes.
- Confirm implementation is robust to small sample sizes and model misspecification.

---

## Summary
`analysis.py` will define a class `AnalysisRoutine` that:
- Is initialized with data, model, and task type.
- Has dedicated methods per task for classical analysis plus variance estimation.
- Returns sufficient statistics (estimates and variances) in dictionary form.
- Supports bootstrap variance estimation to ensure reliable asymptotic inference as per the PSPS framework.

This structure ensures flexibility across a broad spectrum of statistical tasks, enabling the task-agnostic, summary-statistics-based inference outlined in the paper.

---

This concludes the detailed logic analysis for `analysis.py`.

## dataset_loader.py

{
  "dataset_loader.py": [
    "Purpose and Responsibilities:",
    "Implement a DatasetLoader class responsible for data input, simulation, preprocessing, and data management, serving as the foundational component feeding data into subsequent modules. It must handle both synthetic data generation aligned with the paper's models and loading real datasets (e.g., UK Biobank).",
    "The class should output data in standardized formats compatible with downstream modules: dictionaries or objects containing features matrices, outcome vectors, and possibly precomputed ML predictions if available.",
    "Inputs are controlled via the configuration parameters and experimental context: whether synthetic or real data is used, labeled/unlabeled split sizes, ML prediction models precomputed, etc.",
    "Core Methods and Their Logic:",
    "1. Initialization (__init__):",
    "   - Parse and store configuration parameters: data type (synthetic or real), labeled size, unlabeled sizes list, seed for reproducibility.",
    "   - Load or generate necessary data, preparing data structures for subsequent use.",
    "2. load_data():",
    "   - If real dataset (e.g., UK Biobank):",
    "       - Load dataset files (e.g., CSV, HDF5, or other formats).",
    "       - Data preprocessing: handle missing values, encode categorical variables, normalize or standardize features if needed.",
    "       - Partition into labeled and unlabeled subsets based on specified sizes.",
    "       - Return a dictionary/object: {'X': feature matrix, 'Y': outcome vector, 'labels': boolean or index masks}.",
    "   - If synthetic data:",
    "       - Generate features: for each sample, simulate predictors (X) as per paper's models (normal distributions, correlation structures, etc.).",
    "       - Generate outcomes (Y) according to specified models:",
    "         • For mean regression: linear combination with noise.",
    "         • For regression and classification tasks: include nonlinear terms, noise, and binary thresholds as needed.",
    "       - For unlabeled data: generate features only, or features with pseudo-outcomes if necessary.",
    "       - Save the entire dataset: 'X', 'Y' (labeled), 'X_unlabeled' (unlabeled), possibly 'f_hat' (initial ML predictions if precomputed).",
    "3. Data Splitting and Management:",
    "   - For synthetic data: ensure labeled and unlabeled data are independent samples, matching the sizes dictated by configuration.",
    "   - For real data: implement sample masks or indices to split data into labeled and unlabeled sets artificially or per experimental design.",
    "   - Maintain consistency with the original data structure used in the paper.",
    "4. ML Prediction Integration:",
    "   - Optionally, if precomputed ML predictions are provided (from external models), load these predictions aligned with data samples.",
    "   - Otherwise, prepare methodology to generate predictions on the fly (by calling the model’s predict method). This can be integrated later in 'model.py' and 'main.py'.",
    "5. Data Preprocessing and Standardization:",
    "   - Convert loaded or generated data into numpy arrays for feature matrices (X) and outcome vectors (Y).",
    "   - Handle missingness per the paper (if relevant), or ensure complete data for simulation settings.",
    "   - Standardize features if necessary (though the paper's models may not require explicit scaling).",
    "6. Data Format and Internal Representation:",
    "   - Use numpy.ndarray for matrices and vectors for compatibility with scikit-learn and other analysis routines.",
    "   - Store data in a structured dictionary or class attributes for easy access: e.g., self.data = {'X': X, 'Y': Y, 'X_unlabeled': X_unlabeled, 'f_hat': f_hat}.",
    "   - Using consistent data types and shapes for features (n_samples x n_features), outcomes (n_samples,), unlabeled features (m_samples x n_features), and optionally predicted outcomes.",
    "7. Reproducibility and Random Seeds:",
    "   - Set fixed random seed during simulation to ensure reproducibility of data generation.",
    "   - For synthetic data, document the seed, distribution parameters, and model parameters as per paper’s settings.",
    "8. Output and Interface:",
    "   - The load_data() method returns a clean, ready-to-use data structure.",
    "   - Downstream modules expect standardized inputs, so document the expected format: e.g., dict with keys {'X', 'Y', 'X_unlabeled', 'f_hat'}.",
    "9. Edge Cases and Validation:",
    "   - Check data dimensions align with labeled and unlabeled sizes.",
    "   - Validate that the simulated data match the statistical properties described (e.g., variance explained, correlation structure).",
    "   - For real data, include validity checks: no missing labels, features correctly loaded, data splits consistent.",
    "10. Extensibility and Customization:",
    "    - Architect the class to allow easy extension to other data types, models, or additional simulations.",
    "    - Support optional parameters for hyperparameters, data transformations, noise levels, or correlation structures to match the experiments.",
    "Summary:",
    "- The DatasetLoader must be flexible to support synthetic and real datasets, replicating the paper’s simulation and data management procedures.",
    "- It should produce data in formats compatible with analysis routines and variance estimation modules.",
    "- It should ensure reproducibility, data integrity, and facilitate downstream tasks like predictions, analyses, and variance computations."
  ]
}

## inference.py

# Inference.py Logic Analysis for PSPS-based ML-assisted inference

This analysis documents the detailed logic flow, class structure, functions, inputs, outputs, and computational steps necessary to implement 'inference.py', which executes the core inference tasks—debiasing estimates, constructing confidence intervals, and hypothesis testing—based on the PSPS framework outlined in the paper.

## 1. Purpose and Scope
- Implement an `Inference` class that:
  - Receives summary statistics (estimates and variances) from analysis routines,
  - Applies the PSPS formula to produce debiased point estimates,
  - Estimates the variance (standard errors) of the PSPS estimate,
  - Constructs asymptotically valid confidence intervals,
  - Conducts hypothesis tests (two-sided p-values) for parameters.

This class embodies the final step in the pipeline, translating summary statistics into actionable inference results, using the asymptotic normality results specified in the paper.

---

## 2. Inputs and Data Structures

### Inputs
- `summary_stats`: Dictionary containing the estimated means or regression coefficients and associated covariance matrices, specifically:
  - `theta_hat`: Scalar or vector—point estimate(s) from the analysis routine (\(\hat{\theta}_\mathcal{L}\)),
  - `eta_hat`: Vector—estimated nuisance parameters or summary statistics (\(\hat{\eta}_\mathcal{L}\), \(\hat{\eta}_\mathcal{U}\)),
  - Covariance matrices: 
    - `Cov_theta_eta`: Covariance between `theta_hat` and `eta_hat` (\(\widehat{\mathrm{Cov}}(\hat{\theta}_\mathcal{L}, \hat{\eta}_\mathcal{L})\)),
    - Variances of `eta_hat` (for both labeled and unlabeled),
    - Variance of `theta_hat` (for labeled data).
- Variance matrices: Estimated variances and covariances from bootstrap or analytical formulas, provided as numpy arrays.

### Data class overview
Use a structured dictionary, e.g.,
```python
{
  'theta_hat': np.ndarray or float,
  'eta_hat': np.ndarray,
  'Cov_theta_eta': np.ndarray,
  'Var_eta_L': np.ndarray,
  'Var_eta_U': np.ndarray,
  'Var_theta': float or np.ndarray,
  'n_labeled': int,
  'n_unlabeled': int
}
```

### Additional parameters
- `alpha`: float, significance level for CI (e.g., 0.05).
- `estimate_type`: optional, specify if the estimate is a scalar (mean) or vector (regression coefficient).

---

## 3. Core Components and Step-by-Step Logic

### 3.1. Initialization
- Instantiate class with summary statistics, bootstrap estimates, and configuration parameters as needed.
- Validate input dimensions and completeness.

### 3.2. Compute the Debiased Estimate (\(\hat{\theta}_{PSPS}\))
- Formula:
  \[
  \boxed{
  \hat{\theta}_{PSPS} = \hat{\theta}_\mathcal{L} + \hat{\omega}_0^T (\hat{\eta}_\mathcal{U} - \hat{\eta}_\mathcal{L})
  }
  \]
- **Calculating \(\hat{\omega}_0\)**:
  - Requires:
    - Variance of \(\hat{\eta}_\mathcal{L}\): \(\widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{L})\),
    - Variance of \(\hat{\eta}_\mathcal{U}\): \(\widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{U})\),
    - Covariance between \(\hat{\theta}_\mathcal{L}\) and \(\hat{\eta}_\mathcal{L}\): \(\widehat{\mathrm{Cov}}(\hat{\theta}_\mathcal{L}, \hat{\eta}_\mathcal{L})\),
  - Use:
  \[
  \hat{\omega}_0 = \left( \widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{L}) + \frac{n}{N} \widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{U}) \right)^{-1} \times \widehat{\mathrm{Cov}}(\hat{\theta}_\mathcal{L}, \hat{\eta}_\mathcal{L})
  \]
  - Here, \(n, N\): labeled and unlabeled sample sizes from input.

- Final step:
```python
self.theta_psps = self.theta_labeled + np.dot(self.omega_0, (self.eta_U - self.eta_L))
```

### 3.3. Variance Estimation (Variance of \(\hat{\theta}_{PSPS}\))
- Use the algebraic form:
\[
\mathrm{Var}(\hat{\theta}_{PSPS}) \approx \\
\mathrm{Var}(\hat{\theta}_\mathcal{L}) - \mathbf{V}_{\theta,\eta}^T \left( \mathbf{V}_\eta + \frac{n}{N} \mathbf{V}_\eta^{(U)} \right)^{-1} \mathbf{V}_{\theta,\eta}
\]
- `\(\mathbf{V}_\eta\), \(\mathbf{V}_\eta^{(U)}\)`, \(\(\mathbf{V}_{\theta,\eta}\)\): Variance/covariance matrices estimated via bootstrap or analytical formulas.
- Implement bootstrap:
  - Generate bootstrap replicates of the summary statistics (via stored bootstrap covariance matrices),
  - For each replicate, compute \(\hat{\theta}_{PSPS}\),
  - Variance estimated as empirical variance across bootstrap replicates.

### 3.4. Construct Confidence Interval
- Asymptotic normality:
\[
\hat{\theta}_{PSPS} \sim \mathcal{N}(\theta^*, \mathrm{Var}(\hat{\theta}_{PSPS}))
\]
- CI:
\[
\text{CI}_\alpha = \left( \hat{\theta}_{PSPS} \pm z_{1 - \frac{\alpha}{2}} \times \sqrt{\mathrm{Var}(\hat{\theta}_{PSPS})} \right)
\]
- Use `scipy.stats.norm.ppf(1 - alpha/2)` for critical value `z`.

### 3.5. Hypothesis Testing
- Two-sided p-value:
\[
p = 2 \times \left(1 - \Phi \left( \left| \frac{\hat{\theta}_{PSPS} - \theta_{0}}{\sqrt{\mathrm{Var}(\hat{\theta}_{PSPS})}} \right| \right) \right)
\]
- Typically \(\theta_0 = 0\) or null value depending on task.

### 3.6. Output
- Return a dictionary with:
  - `'estimate'`: debiased point estimate,
  - `'std_error'`: sqrt of estimated variance,
  - `'confidence_interval'`: tuple (lower, upper),
  - `'p_value'`: associated p-value.

---

## 4. Additional Considerations
- **Regularization and numerical stability**:
  - Add small ridge epsilon if matrices are ill-conditioned.
- **Dimensionality**:
  - For scalar quantities (mean), handle as float.
  - For vector parameters (regression), handle as np.ndarray.
- **Multiple parameters**:
  - Loop over each coordinate if vector.
  - Adjust CI construction for multivariate case accordingly.
- **Input validation and error handling**:
  - Ensure covariance matrices are symmetric and positive-definite where needed.
  - Validate dimensions agree.
- **Flexibility**:
  - Implement options for analytical vs bootstrap variance estimation.
  - Allow extension for more complex models or multiple parameters.

---

## 5. Implementation Skeleton (Conceptual)
(Pseudocode in comments, not actual code)

```python
class Inference:
    def __init__(self, summary_stats, n_labeled, n_unlabeled):
        # Initialize with provided summaries and sample sizes
        # Validate inputs
        pass

    def compute_omega_0(self):
        # Calculate omega_0 as per formula using covariance matrices
        pass

    def compute_point_estimate(self):
        # Compute PSPS estimate: theta_psps
        pass

    def bootstrap_variance(self, analysis_fn, n_bootstrap=200):
        # Generate bootstrap replicates of summary statistics
        # For each, compute PSPS estimate
        # Variance is the empirical variance over replicates
        pass

    def estimate_variance(self):
        # Use bootstrap or analytical formula to estimate variance
        pass

    def construct_ci(self, alpha=0.05):
        # Use normal quantile
        pass

    def hypothesis_test(self, theta_null=0):
        # Compute p-value
        pass

    def report(self, alpha=0.05):
        # Output dictionary with estimate, std error, CI, p-value
        pass
```

---

## 6. Summary
- The `Inference` class encapsulates all aspects from estimating the debiased estimate, variance, confidence intervals, to hypothesis tests.
- Leverage the algebraic formulas from the paper and implement variance estimation via bootstrap.
- Ensure modularity and extensibility for varied inference tasks and summary statistic structures.

---

This detailed logic analysis will guide the actual implementation of 'inference.py', ensuring it aligns correctly with the paper’s methodology and theoretical guarantees, and is flexible enough to handle multiple inference tasks in a reproducible, efficient manner.

## main.py

# Logic Analysis: main.py

The purpose of main.py is to serve as the main orchestration script that coordinates the entire pipeline for implementing the PSPS framework on either synthetic or real datasets. It must systematically perform dataset loading, model training, predictions, analysis, variance estimation (via bootstrap), and inference (point estimates, confidence intervals, hypothesis tests). This structure aligns with the described program call flow, class diagram, and pipeline steps.

Below is a comprehensive, detailed breakdown of the logic and sequence of operations needed, referencing the paper, design documents, and configuration file (config.yaml):

---

# 1. Initialization and Configuration
- **Load configuration**:
  - Parse 'config.yaml' to retrieve:
    - Data parameters (synthetic or real). In this case, set `synthetic=True`.
    - Model parameters:
      - Model type: RandomForestRegressor
      - n_estimators: 500
      - max_depth: default
      - random seed: fixed as 42 for reproducibility.
    - Variance estimation parameters:
      - bootstrap_samples: 200 (for bootstrap variance estimation)
    - Data sizes:
      - labeled_size = 500
      - unlabeled_sizes = [1000, 2500, 5000, 10000]
    - Analysis type: e.g., 'regression'
    - Number of repetitions: 1000 for Monte Carlo simulation
    - Results directory: 'results/' (create if not exists)

---

# 2. Dataset Loading / Generation
- **Decision based on 'synthetic' flag**:
  - **If synthetic=True**:
    - Loop over each unlabeled size in ['unlabeled_sizes'].
    - Generate data per the models specified in Appendix D:
      - Use the specified data generation equations:
        - For mean estimation: simulate \(Y_i, X_{1i}, X_{2i}\) with the specified variances and relationships.
        - For regression tasks: generate covariates and outcomes as per the described models.
        - For IV, NB, Lasso, Wilcoxon tasks: generate data accordingly.
      - Store simulated features, labels, and unlabeled features in suitable 'dataset' dictionaries/structures.
  - **If synthetic=False (not asked in task but for completeness)**:
    - Load real datasets (e.g., UK Biobank).
    - Perform preprocessing steps:
      - Data splitting into labeled and unlabeled.
      - Optional feature selection (e.g., top 50 correlated variables).
- **Data structure**:
  - The dataset can be kept in dictionaries or custom classes:
    ```python
    dataset = {
        'X_labeled': np.ndarray of shape (n_labeled, feature_dim),
        'Y_labeled': np.ndarray of shape (n_labeled,),
        'X_unlabeled': np.ndarray of shape (n_unlabeled, feature_dim)
    }
    ```
- **Note**:
  - For each unlabeled size, create a separate dataset instance for independent runs.

---

# 3. ML Model Initialization
- Instantiate MLModel class:
  - Use parameters from config.yaml:
    ```python
    model_params = {
        'type': 'RandomForestRegressor',
        'n_estimators': 500,
        'max_depth': None,  # default
        'random_state': 42
    }
    ```
- Store the model instance for training.

---

# 4. Model Training
- **Step 4.1**:
  - Extract features and labels from labeled data:
    ```python
    X_train = dataset['X_labeled']
    y_train = dataset['Y_labeled']
    ```
- **Step 4.2**:
  - Call `train(X_train, y_train)` method on the model:
    ```python
    model.train(X_train, y_train)
    ```
- **Outcome**:
  - The trained model is used for prediction.

---

# 5. Generate ML Predictions
- **Step 5.1**:
  - For labeled data:
    ```python
    f_hat_labeled = model.predict(X_train)
    ```
- **Step 5.2**:
  - For unlabeled data:
    ```python
    f_hat_unlabeled = model.predict(dataset['X_unlabeled'])
    ```
- **Store prediction arrays**:
  ```python
  predictions = {
      'f_hat_labeled': f_hat_labeled,
      'f_hat_unlabeled': f_hat_unlabeled
  }
  ```

---

# 6. Apply Analysis Routine to Extract Summary Statistics
- Instantiate AnalysisRoutine class, passing:
  - The trained model or directly its predictions.
  - The dataset (or at least the covariates and outcomes).
- **Analysis routines** depend on the task; for example:
  - **Mean estimation**:
    - Compute sample mean of \(Y_i\) (labeled),
    - Compute sample mean of \(\hat{f}_i\) for labeled,
    - Compute sample mean of \(\hat{f}_j\) for unlabeled.
  - **Regression**:
    - Run OLS on labeled data: regress \(Y_i\) on features.
    - Run regressions for other summary statistics as needed.
  - **Quantile estimation**:
    - Compute sample quantile directly.
- **Method**:
  - Call the method like `compute_summary_statistics()`.
- **Output**:
  - A dictionary of estimated parameters and their estimated variances, e.g.,
    ```python
    summary_stats = {
        'theta_hat': estimated_value,
        'var_theta': estimated_variance,
        'eta_hat': vector for ML predictions,
        'var_eta': variance,
        'cov_theta_eta': covariance matrix or vector
    }
    ```

---

# 7. Variance/Covariance Estimation via Bootstrap
- Instantiate VarianceEstimator with:
  - Dataset,
  - The analysis routine.
- Call:
  ```python
  bootstrap_results = variance_estimator.bootstrap_variance(analysis.compute_summary_statistics, n_bootstrap=200)
  ```
- **bootstrap_results** should provide bootstrap estimates of:
  - Variance of \(\hat{\theta}_\mathcal{L}\),
  - Variance of \(\hat{\eta}_\mathcal{L}\), \(\hat{\eta}_\mathcal{U}\),
  - Covariance matrices like \(\mathrm{Cov}(\hat{\theta}_\mathcal{L}, \hat{\eta}_\mathcal{L})\).

**Implementation details**:
- During bootstrap, resample with replacement the labeled data, recompute the analysis routine for each sampled dataset to obtain bootstrap estimates.
- Keep all bootstrap estimates to compute covariance matrices or variance estimates empirically.

---

# 8. Compute Weights for PSPS
- Using the bootstrap variance/covariance estimates, compute:
  \[
  \hat{\omega}_0 = (\widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{L}) + \rho \widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{U}))^{-1} \widehat{\mathrm{Cov}}(\hat{\theta}_\mathcal{L}, \hat{\eta}_\mathcal{L})
  \]
- Store \(\rho = n / N\), where \(n\) = labeled size, \(N\) = unlabeled size, for current iteration.

---

# 9. Debiased Estimation via PSPS
- Calculate the PSPS estimator:
  \[
  \hat{\theta}_{PSPS} = \hat{\theta}_\mathcal{L} + \hat{\omega}_0^T (\hat{\eta}_\mathcal{U} - \hat{\eta}_\mathcal{L})
  \]
- Compute its estimated asymptotic variance \(\widetilde{\mathrm{Var}}\):
  - Use the algebraic formula involving the bootstrap covariance matrices.

---

# 10. Inference: Confidence Intervals and P-values
- Construct \(\alpha\)-level two-sided confidence intervals:
  - \(\mathcal{C}_{\alpha} = (\hat{\theta}_{PSPS} \pm z_{1 - \alpha/2} \times \sqrt{\widetilde{\mathrm{Var}}})\),
  - where \(z_{1 - \alpha/2}\) is the standard normal quantile.
- Calculate p-value:
  \[
  p = 2 \times (1 - \Phi(\left|\frac{\hat{\theta}_{PSPS} - \theta^*}{\sqrt{\widetilde{\mathrm{Var}}}}\right|))
  \]
- **Note**:
  - \(\theta^*\) is unknown; for coverage assessment, simulate or report average properties across repetitions.
  - For reporting real data, provide point estimates, standard errors, confidence intervals, and p-values.

---

# 11. Loop for Multiple Tasks, Validations, or Repetitions
- Wrap entire process in a loop (repetitions = 1000).
- For each repetition:
  - Resample, re-estimate, and store estimates.
- After all repetitions:
  - Compute empirical coverage, CI width averages, power, FDR as required.
- Save or export the result summary for each data size and task to 'results/'.

---

# 12. Additional Considerations
- **Reproducibility**:
  - Set random seeds for data generation, bootstrap, ML model training.
- **Diagnostics**:
  - Check asymptotic normality assumptions,
  - Validate variance estimates via bootstrap convergence,
  - Evaluate coverage over simulations.
- **Output**:
  - Save final estimates, CIs, p-values, variance estimates, and coverage rates.
  - Save model and intermediate results optionally for debugging.

---

# 13. Summary of Key Operations in main.py
- Load config
- Generate/load dataset
- Instantiate ML model
- Train ML model
- Predict outcomes
- Instantiate analysis routines
- Compute summary statistics
- Bootstrap variance/covariance estimates
- Compute weights\(\hat{\omega}_0\)
- Calculate PSPS estimate and variance
- Construct CI and p-value
- Repeat process over simulation iterations
- Aggregate and save results

---

This comprehensive logic analysis provides clear, step-by-step instructions to underpin the implementation in main.py, aligning with the paper, design, and configuration specifications. Each step ensures fidelity to the methodology, with modular design facilitating extensions and debugging.

## model.py

### Logic Analysis for `model.py` — Defining the ML Prediction Model Class

This module is responsible for encapsulating the machine learning (ML) model used within the PSPS framework, specifically implementing the `MLModel` class. The class should provide methods for initialization, training, and prediction, acting as a shared component across data analysis, variance estimation, and inference modules.

---

## 1. Purpose and Scope
- **Objective:** Provide a flexible, reproducible, and task-agnostic interface to initialize, train, and generate predictions from an ML model.
- **Key use case in this paper:** Implementing a RandomForestRegressor with fixed hyperparameters, including number of estimators, maximum depth, and seed for reproducibility.
- **Additional flexibility:** Design to allow future extension to other models if needed, but currently focus on random forest.

---

## 2. Class Specification

### 2.1 Class Name
- `MLModel` — representing the machine learning prediction component.

### 2.2 Core Attributes
- **model_type:** string indicating the type of ML model, e.g., `'RandomForestRegressor'`.
- **model_params:** dictionary of hyperparameters for model initialization:
  - `n_estimators` (e.g., 500)
  - `max_depth` (optional, default to scikit-learn default)
  - `random_state` (fixed seed for reproducibility, e.g., 42)
- **model_instance:** the instantiated ML model object (from scikit-learn) after initialization.

### 2.3 Core Methods
- `__init__(self, model_params)`:
  - Initialize the object with model hyperparameters.
  - Instantiate the scikit-learn model object based on `model_type`.
  - For now, only support `'RandomForestRegressor'`.

- `train(self, X: np.ndarray, y: np.ndarray)`:
  - Fit the ML model to training data (labeled dataset).
  - Validate input types and dimensions.
  - Store the trained model internally.

- `predict(self, X: np.ndarray) -> np.ndarray`:
  - Generate predicted outcomes given feature matrix `X`.
  - Support batch input.
  - Return predictions as a numpy array of the same number of samples, shape `(n_samples,)`.
  - Ensure predictions are in a consistent manner for downstream analysis.

---

## 3. Implementation Constraints & Details
- **Compatibility:** Use `scikit-learn` models.
- **Reproducibility:** Set `random_state=42` (from `config.yaml`) for deterministic behavior.
- **Input validation:** Check `X` is a numpy array with shape `(n_samples, n_features)`; `y` shape `(n_samples,)`.
- **Hyperparameters:** Hard-code or pass via constructor with defaults aligned to config.
- **Extensibility:** Consider designing to support multiple model types in future, but currently fixed to `'RandomForestRegressor'`.
- **No training pipeline:** Focus only on instantiation, training, and prediction; no hyperparameter tuning or cross-validation here.

---

## 4. Flow and Usage
- **Initialization:**
  ```python
  model = MLModel(model_params)
  ```
- **Training:**
  ```python
  model.train(X_train, y_train)
  ```
- **Prediction:**
  ```python
  preds = model.predict(X_eval)
  ```
- The trained `model_instance` object will be used in `analysis.py` and related modules for generating predictions on datasets.

---

## 5. Implementation Considerations
- **On randomness:**
  - Fix `random_state` for deterministic results.
- **Error handling:**
  - Raise informative exceptions if input data shape mismatches.
  - Confirm that the model is trained before prediction; possibly include a flag or check.
- **Model persistence:**
  - Optional extension: save/load model from disk if needed, but not specified here.
- **Testing:**
  - Validate with dummy datasets to ensure consistent training and prediction behavior.

---

## 6. Interface Summary
```python
class MLModel:
    def __init__(self, model_params: dict):
        # Instantiate the ML model based on specified parameters
        pass

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # Fit the model to labeled data
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Generate predictions for input feature matrix
        return predictions
```

---

## 7. Notes on the provided configuration
- Use `model_params` from `config.yaml`:
  ```yaml
  model:
    type: RandomForestRegressor
    n_estimators: 500
    max_depth: null
    random_state: 42
  ```
- With `max_depth: null`, the default scikit-learn value (unlimited depth) will be used unless specified.

---

## 8. Summary of Key Points
- **Reproducibility:** Fixed seed (`random_state=42`)
- **Task-specific support:** Currently only supports regression (RandomForestRegressor).
- **Design:** Encapsulate in a class with clear methods; minimal external dependencies.
- **Interface:** Ensure methods are compatible with the pipeline; predictable input/output.

---

This detailed logical analysis guides the implementation of `model.py` to produce a robust, flexible, and reproducible ML model class that integrates seamlessly into the overall PSPS inference pipeline.

## trainer.py

{
  "trainer.py": [
    "Purpose and Role:\n  - Implement a Trainer class responsible for training the pre-specified ML model (e.g., RandomForestRegressor) using labeled data.\n  - Manage hyperparameter setup, training process, possible cross-validation for hyperparameter tuning, and validation procedures.\n  - Facilitate retraining or hyperparameter tuning if necessary, providing trained model objects for downstream prediction modules.\n\n2. Inputs and Dependencies:\n  - Require dataset loaded via dataset_loader.py, providing access to labeled data: features (\u03a4X) and outcomes (Y).\n  - Depend on configuration parameters (from config.yaml), especially model specifics (type, n_estimators, max_depth).\n  - Use scikit-learn's API, specifically implementing training (fit) and prediction (predict) methods.\n  - May utilize auxiliary functions for validation metrics or hyperparameter tuning.\n\n3. Core Components and Methods:\n  - __init__(self, data, model_config, hyperparam_tuning=False):\n    - Initialize with dataset object/dictionary containing features and labels.\n    - Load model parameters from provided configuration.\n    - Optionally prepare for hyperparameter tuning if flag set.\n\n  - train(self):\n    - Execute the training procedure:\n      - Instantiate the ML model with specified hyperparameters.\n      - Fit the model on features and outcomes: model.fit(X_train, y_train).\n    - Store the trained model internally for prediction.\n\n  - tune_hyperparameters(self):\n    - Implement hyperparameter tuning routine if \u201chyperparam_tuning\u201d flag is enabled.\n    - Use validation sets, grid search, or random search over parameters (e.g., max_depth, min_samples_split).\n    - Select the hyperparameters yielding best validation performance.\n    - Retrain the model with selected hyperparameters.\n\n  - validate(self):\n    - Optional: Compute validation metrics (e.g., R^2, RMSE) on validation subset.\n    - Log or return validation performance for diagnostics.\n\n  - get_trained_model(self):\n    - Return the trained, possibly tuned, model object for prediction.\n\n4. Hyperparameters and Tuning Details:\n  - Use the config.yaml parameters: n_estimators=500, max_depth=None.\n  - For hyperparameter search, consider ranges like max_depth: [None, 10, 20], min_samples_split: [2, 10], if tuning is enabled.\n  - Use scikit-learn's GridSearchCV or RandomizedSearchCV for tuning if necessary.\n  - Choose validation strategy (e.g., k-fold cross-validation) appropriate for dataset size.\n\n5. Data Handling:\n  - Load features (X) and labels (Y) in init.\n  - Split into training and validation sets internally if tuning.\n  - Ensure data is properly formatted as numpy arrays.\n\n6. Training and Validation Workflow:\n  - On method call (train):\n    - If hyperparameter tuning is enabled:\n      - Run tuning routine.\n      - Retrain with best hyperparameters.\n    - Else:\n      - Directly instantiate model with config params.\n      - Fit on entire labeled dataset.\n  - Record fitting time if needed.\n  - After training, the object retains the trained model.\n\n7. Model Saving and Reusability:\n  - Optional: Save trained model object to disk for reuse.\n  - This can be used for later prediction steps without retraining.\n\n8. Error Handling and Robustness:\n  - Check present data shapes, types.\n  - Handle missing values if any (though likely not in synthetic data), or assume dataset_loader handles preprocessing.\n  - Log training status, errors.\n\n9. Integration with Main.py:\n  - Expose interface: instantiate Trainer with dataset, call train() method.\n  - Retrain or tune as needed before passing model to analysis modules.\n\n10. Additional Considerations:\n  - Use consistent random seed (from config) to ensure reproducibility.\n  - Enable verbose logging for debugging and validation.\n  - Modular design for easy extension to other models if needed.\n\nSummary: \nThe trainer.py file will contain a Trainer class encapsulating the steps of loading data, initializing model, optionally hyperparameter tuning with validation, training, and providing access to the fitted model. The focus is on robustness, flexibility, consistent use of configuration parameters, and enabling retraining for various inference tasks in the PSPS pipeline."
  ]
}

## variance_estimation.py

**Variance Estimation Module for PSPS Framework**

---

### Purpose:
Implement a VarianceEstimator class that:

- Accepts input data and summary statistics.
- Provides bootstrap-based (or analytical, if applicable) estimates of variances and covariances necessary for the PSPS correction.
- Supplies accurate estimation of \(\widehat{\mathrm{Var}}\) and \(\widehat{\mathrm{Cov}}\) matrices required for constructing the debiased estimator and its variance.

---

### Core Responsibilities:
1. **Input Handling**:
   - Raw data or precomputed summary statistics:
     - Labeled data: features \( \mathbf{X}_\mathcal{L} \), outcomes \(Y_\mathcal{L}\), ML predictions \(\hat{f}_\mathcal{L}\).
     - Unlabeled data: features \(\mathbf{X}_\mathcal{U}\), ML predictions \(\hat{f}_\mathcal{U}\).
   - Corresponding point estimates: \(\hat{\theta}_\mathcal{L}\), \(\hat{\eta}_\mathcal{L}\), \(\hat{\eta}_\mathcal{U}\).
   - Variance/covariance estimates from analysis routines.

2. **Variance and Covariance Estimation**:
   - Implement bootstrap resampling on labeled data \(\mathcal{L}\) to estimate:
     - \(\mathrm{Var}(\hat{\theta}_\mathcal{L})\),
     - \(\mathrm{Var}(\hat{\eta}_\mathcal{L})\),
     - \(\mathrm{Cov}(\hat{\theta}_\mathcal{L}, \hat{\eta}_\mathcal{L})\).
   - Estimate \(\mathrm{Var}(\hat{\eta}_\mathcal{U})\) using bootstrap on the unlabeled data predictions, or analytical formulas if possible.
   - For simplicity and consistency with paper, bootstrap is recommended due to its flexibility across inference tasks.

3. **Output**:
   - Estimated variance matrices:
     - \(\widehat{\mathrm{Var}}(\hat{\theta}_\mathcal{L})\),
     - \(\widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{L})\),
     - \(\widehat{\mathrm{Cov}}(\hat{\theta}_\mathcal{L}, \hat{\eta}_\mathcal{L})\),
     - \(\widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{U})\).

---

### Step-by-step Logic:

#### 1. Initialization:
- Receive as input:
  - Data required for variance computation, e.g., (or their bootstrap replications)
  - Precomputed estimates if available.
- Use constructor parameters:
  - `bootstrap_samples` (from config): number of bootstrap replications (e.g., 200).
  - Data dictionaries containing required data components.

#### 2. Variance and Covariance via Bootstrap:
- For the labeled data:
  - For each bootstrap replication \(q\):
    - Draw a bootstrap resample of size \(n\) with replacement from the labeled data.
    - Compute the estimator of interest (e.g., regression coefficients, means) on the bootstrap sample using the same analysis routine.
    - Store the bootstrap estimates to form a collection \(\{\hat{\theta}_\mathcal{L}^{(q)}\}\), \(\{\hat{\eta}_\mathcal{L}^{(q)}\}\).

- For the unlabeled data:
  - For each bootstrap replication \(q\):
    - Draw a bootstrap sample of size \(N\) with replacement.
    - Compute the ML predictions for the bootstrap sample.
    - Apply the same analysis routine (or a simplified variance estimator based on prediction variance) to estimate \(\hat{\eta}_\mathcal{U}\).
    - Store these estimates.

- Variance estimation:
  - Use the bootstrap estimates:
    \[
    \widehat{\mathrm{Var}}(\hat{\theta}_\mathcal{L}) \approx \operatorname{SampleVar}(\{\hat{\theta}_\mathcal{L}^{(q)}\}_{q=1}^{Q})
    \]
  - Similar for \(\hat{\eta}_\mathcal{L}\) and \(\hat{\eta}_\mathcal{U}\).

- Covariance estimation:
  \[
  \widehat{\mathrm{Cov}}(\hat{\theta}_\mathcal{L}, \hat{\eta}_\mathcal{L}) \approx \operatorname{SampleCov}(\{\hat{\theta}_\mathcal{L}^{(q)}\}, \{\hat{\eta}_\mathcal{L}^{(q)}\})
  \]

- **Note:** For efficiency, bootstrap resampling should be parallelized if computational resources permit.

#### 3. Analytical Variance Formula (Optional/Alternate):
- When computational cost is high or data is large, approximate variances using analytical methods:
  - Use influence functions or linear approximation.
  - For regression coefficients, approximate via residuals and design matrices.
  - For mean estimators, directly use sample variance.

- But given the emphasis on flexibility across tasks, bootstrap is the primary method.

#### 4. Storage & Return:
- Organize estimates into matrices:
  - \(\widehat{\mathrm{Var}}(\hat{\theta}_\mathcal{L})\): a scalar or matrix.
  - \(\widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{L})\): a matrix of size \(K \times K\).
  - \(\widehat{\mathrm{Cov}}(\hat{\theta}_\mathcal{L}, \hat{\eta}_\mathcal{L})\): a vector of length \(K\).
  - \(\widehat{\mathrm{Var}}(\hat{\eta}_\mathcal{U})\): a matrix of size \(K \times K\).

---

### 5. Integration with Main Pipeline:
- The estimates are used to compute \(\hat{\omega}_0\) and the variance of the PSPS estimator.
- These estimates allow valid confidence intervals and hypothesis tests.

---

### 6. Additional Considerations:
- **Reproducibility:**
  - Set random seeds explicitly for bootstrap reproducibility.
- **Memory Management:**
  - For large datasets, store bootstrap estimates efficiently or compute on-the-fly.
- **Validation:**
  - Optionally include checks for bootstrap convergence.
  - Possible fallback to analytical formulas where feasible.

---

### 7. Summary:
- Input: data, point estimates, bootstrap parameters.
- Output: estimated variance-covariance matrices needed for debiasing.
- Core logic: bootstrap resampling, compute estimates each time, aggregate to estimate variances/covariances.
- Designed to be task-agnostic, flexible, and compatible with diverse analysis routines and inference tasks.

---

This detailed reasoning ensures that the `variance_estimation.py` module provides accurate, reproducible, and flexible variance estimates central to the PSPS inference methodology.

