# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## analysis.py

```python
## analysis.py

import numpy as np
from scipy.stats import norm, mannwhitneyu, rankdata
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.utils import resample

class AnalysisRoutine:
    """
    Class to compute the sufficient summary statistics for a variety of
    statistical tasks using classical estimators.
    """

    def __init__(self, model, data: dict, method: str = "mean"):
        """
        Initialize with trained ML model, dataset, and task method.
        Args:
            model: The pre-trained ML prediction model (not necessarily used directly here).
            data (dict): Data containing keys:
                - 'X_lab': np.ndarray, features for labeled data
                - 'Y_lab': np.ndarray, outcomes for labeled data
                - 'f_hat_lab': np.ndarray, ML predictions for labeled data
                - 'X_unlab': np.ndarray, features for unlabeled data
                - 'f_hat_unlab': np.ndarray, ML predictions for unlabeled data
            method (str): Specifies the task type, e.g., 'mean', 'regression', 'quantile',
                          'IV', 'NB', 'Lasso', 'Wilcoxon'.
        """
        self.model = model
        self.data = data
        self.method = method.lower()

    def compute_summary_statistics(self):
        """
        Dispatch method to compute and return the relevant sufficient statistics.
        Returns:
            dict: Contains point estimate, variance estimate, and possibly covariance matrices.
        """
        if self.method == "mean":
            return self._compute_mean()
        elif self.method == "regression":
            return self._compute_regression()
        elif self.method == "logistic":
            return self._compute_logistic()
        elif self.method == "quantile":
            return self._compute_quantile()
        elif self.method == "iv":
            return self._compute_iv()
        elif self.method == "nb":
            return self._compute_nb()
        elif self.method == "lasso":
            return self._compute_lasso()
        elif self.method == "wilcoxon":
            return self._compute_wilcoxon()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _compute_mean(self):
        """
        Compute sample mean and variance of Y (labeled),
        mean of predicted f_hat for unlabeled,
        as well as bootstrap variance estimates.
        """
        Y_lab = self.data.get('Y_lab')
        f_hat_unlab = self.data.get('f_hat_unlab')
        X_lab = self.data.get('X_lab')

        # Check presence of data
        if Y_lab is None:
            raise ValueError("Y_lab is required for mean estimation")
        if f_hat_unlab is None:
            raise ValueError("f_hat_unlab is required for mean estimation")
        n = len(Y_lab)
        N = len(f_hat_unlab)

        estimate = np.mean(Y_lab)
        var_estimate = np.var(Y_lab, ddof=1) / n  # variance of mean

        f_hat_mean = np.mean(f_hat_unlab)

        # Bootstrap for variance estimation
        boot_samples = 200
        boot_estimates = []
        for _ in range(boot_samples):
            # Bootstrap resample labeled data
            Y_bs, X_bs, f_hat_bs = resample(Y_lab, X_lab, self.data.get('f_hat_lab'), replace=True)
            # Recompute bootstrap mean
            b_mean = np.mean(Y_bs)
            boot_estimates.append(b_mean)
        bootstrap_var = np.var(boot_estimates, ddof=1)

        return {
            'estimate': estimate,
            'variance': var_estimate,
            'bootstrap_variance': bootstrap_var,
            'mean_pred_unlab': f_hat_mean
        }

    def _compute_regression(self):
        """
        Fit linear or logistic regression using features and outcomes,
        extract coefficients, covariance matrix, residual variance, etc.
        """
        X_lab = self.data.get('X_lab')
        Y_lab = self.data.get('Y_lab')
        f_hat_lab = self.data.get('f_hat_lab')
        f_hat_unlab = self.data.get('f_hat_unlab')

        if X_lab is None or Y_lab is None:
            raise ValueError("X_lab and Y_lab are required for regression.")
        n_samples, p_features = X_lab.shape

        # Add intercept
        X_with_const = sm.add_constant(X_lab)

        # Fit regression
        if self.method == "regression":
            # Use ordinary least squares
            model = sm.OLS(Y_lab, X_with_const).fit()
            coef = model.params  # shape (p+1,)
            cov = model.cov_params()  # (p+1, p+1)
            residuals = model.resid
            residual_var = np.var(residuals, ddof=p_features+1)
            estimate = coef
            variance = cov
        elif self.method == "logistic":
            # Use logistic regression
            logit_model = sm.Logit(Y_lab, X_with_const).fit(disp=0)
            coef = logit_model.params
            cov = logit_model.cov_params()
            estimate = coef
            variance = cov
        else:
            raise ValueError(f"Unsupported regression method: {self.method}")

        # Bootstrap variance
        boot_samples = 200
        boot_coef_list = []
        for _ in range(boot_samples):
            Y_bs, X_bs = resample(Y_lab, X_lab, replace=True)
            X_bs_const = sm.add_constant(X_bs)
            try:
                if self.method == "regression":
                    mdl = sm.OLS(Y_bs, X_bs_const).fit()
                else:
                    mdl = sm.Logit(Y_bs, X_bs_const).fit(disp=0)
                boot_coef_list.append(mdl.params.values)
            except Exception:
                continue  # skip failed bootstrap

        boot_coef_array = np.vstack(boot_coef_list)
        bootstrap_cov = np.cov(boot_coef_array, rowvar=False)

        return {
            'estimate': estimate,
            'variance': variance,
            'residual_variance': residual_var,
            'bootstrap_cov': bootstrap_cov
        }

    def _compute_quantile(self, quantile_level=0.75):
        """
        Estimate the specified quantile of the outcomes.
        """
        Y_lab = self.data.get('Y_lab')
        f_hat_unlab = self.data.get('f_hat_unlab')

        if Y_lab is None:
            raise ValueError("Y_lab is required for quantile estimation.")

        q_estimate = np.quantile(Y_lab, quantile_level)

        # Variance estimate via bootstrap
        boot_samples = 200
        estimates = []
        for _ in range(boot_samples):
            Y_bs = resample(Y_lab, replace=True)
            estimates.append(np.quantile(Y_bs, quantile_level))
        variance_estimate = np.var(estimates, ddof=1)

        return {
            'estimate': q_estimate,
            'variance': variance_estimate
        }

    def _compute_iv(self):
        """
        Implement two-stage least squares (2SLS) IV estimation.
        """
        X_lab = self.data.get('X_lab')
        Y_lab = self.data.get('Y_lab')
        Z_lab = self.data.get('f_hat_lab')  # Assuming Z as predicted "instrument" in this context

        if X_lab is None or Y_lab is None or Z_lab is None:
            raise ValueError("X_lab, Y_lab, and instrument (f_hat_lab) are required for IV.")

        # Fit first stage: regress X on Z
        # Assuming Z is adequate as instrument
        from sklearn.linear_model import LinearRegression

        # First stage: regress X (for each feature) on Z
        # For simplicity, consider first feature as example, real implementation could do multivariate IV
        # Here, we implement a simplified 2SLS for a single coefficient of first feature
        X1 = X_lab[:,0].reshape(-1,1)  # first predictor
        Z = Z_lab.reshape(-1,1)

        # First stage: regress X1 on Z
        first_stage = LinearRegression().fit(Z, X1)
        X1_hat = first_stage.predict(Z)

        # Second stage: regress Y on X1_hat
        second_stage = sm.OLS(Y_lab, sm.add_constant(X1_hat)).fit()
        coef = second_stage.params
        cov = second_stage.cov_params()

        # Bootstrap for variance
        boot_samples = 200
        coefs = []
        for _ in range(boot_samples):
            idx = np.random.choice(len(Y_lab), len(Y_lab), replace=True)
            try:
                Z_bs = Z[idx]
                X_bs = X_lab[idx, 0].reshape(-1, 1)
                Y_bs = Y_lab[idx]
                # First stage
                first_stg = LinearRegression().fit(Z_bs, X_bs)
                X_hat_bs = first_stg.predict(Z_bs)
                # Second stage
                mdl2 = sm.OLS(Y_bs, sm.add_constant(X_hat_bs)).fit()
                coefs.append(mdl2.params.values)
            except:
                continue
        if len(coefs) > 0:
            bootstrap_cov = np.cov(np.vstack(coefs).T)
        else:
            bootstrap_cov = cov

        return {
            'estimate': coef,
            'variance': cov,
            'bootstrap_cov': bootstrap_cov
        }

    def _compute_nb(self):
        """
        Fit a Negative Binomial model for count data.
        """
        import statsmodels.api as sm
        from statsmodels.discrete.count_model import NegativeBinomial

        X_lab = self.data.get('X_lab')
        Y_lab = self.data.get('Y_lab')

        if X_lab is None or Y_lab is None:
            raise ValueError("X_lab and Y_lab are required for NB regression.")

        X_with_const = sm.add_constant(X_lab)

        try:
            nb_model = NegativeBinomial(Y_lab, X_with_const).fit(disp=0)
            coef = nb_model.params
            cov = nb_model.cov_params()
        except Exception:
            # fallback: estimate with simpler method or set cov
            coef = np.zeros(X_with_const.shape[1])
            cov = np.eye(X_with_const.shape[1]) * np.var(Y_lab)

        # Bootstrap
        boot_samples = 200
        coefs = []
        for _ in range(boot_samples):
            idxs = np.random.choice(len(Y_lab), len(Y_lab), replace=True)
            try:
                X_bs = X_with_const[idxs]
                Y_bs = Y_lab[idxs]
                nb_bs = NegativeBinomial(Y_bs, X_bs).fit(disp=0)
                coefs.append(nb_bs.params.values)
            except:
                continue
        if len(coefs) > 0:
            bootstrap_cov = np.cov(np.vstack(coefs).T)
        else:
            bootstrap_cov = cov

        return {
            'estimate': coef,
            'variance': cov,
            'bootstrap_cov': bootstrap_cov
        }

    def _compute_lasso(self):
        """
        Fit a Lasso model and perform a basic coefficient estimate.
        For simplicity, just fit Lasso; for debiased estimates, further procedures needed.
        """
        from sklearn.linear_model import Lasso
        X_lab = self.data.get('X_lab')
        Y_lab = self.data.get('Y_lab')
        if X_lab is None or Y_lab is None:
            raise ValueError("X_lab and Y_lab are required for Lasso.")

        lasso = Lasso(alpha=0.1, max_iter=10000)
        lasso.fit(X_lab, Y_lab)
        coef = lasso.coef_
        # Variance estimate should be derived via bootstrap or asymptotics (not provided here)
        # Placeholder: bootstrap variance
        boot_samples = 200
        coefs = []
        for _ in range(boot_samples):
            X_bs, Y_bs = resample(X_lab, Y_lab)
            try:
                lasso_bs = Lasso(alpha=0.1, max_iter=10000).fit(X_bs, Y_bs)
                coefs.append(lasso_bs.coef_)
            except:
                continue
        if len(coefs) > 0:
            bootstrap_cov = np.cov(np.vstack(coefs).T)
        else:
            bootstrap_cov = np.eye(len(coef))
        return {
            'estimate': coef,
            'bootstrap_cov': bootstrap_cov
        }

    def _compute_wilcoxon(self):
        """
        Perform Wilcoxon rank-sum test or median estimator.
        """
        X_lab = self.data.get('X_lab')
        Y_lab = self.data.get('Y_lab')
        if X_lab is None or Y_lab is None:
            raise ValueError("X_lab and Y_lab required for Wilcoxon test.")
        # Assume binary group indicator: e.g., first half - group 1, second half - group 2
        # For simplicity, assume all data belong to one group, so simulate two groups
        # Placeholder: split data for test
        n = len(Y_lab)
        mid = n // 2
        group1 = Y_lab[:mid]
        group2 = Y_lab[mid:]
        stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        # Percentile estimate as median
        median_estimate = np.median(Y_lab)
        # Standard error approximation (bootstrap)
        boot_samples = 200
        bs_medians = []
        for _ in range(boot_samples):
            Y_bs = resample(Y_lab, replace=True)
            bs_medians.append(np.median(Y_bs))
        std_error = np.std(bs_medians, ddof=1)
        # Return estimate and p-value
        return {
            'estimate': median_estimate,
            'p_value': p_value,
            'se': std_error
        }
```

## dataset_loader.py

```python
# dataset_loader.py

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.datasets import make_regression
import os

class DatasetLoader:
    """
    Purpose and Responsibilities:
    - Load either real datasets (e.g., UK Biobank) or generate synthetic data aligned with the paper's models.
    - Manage labeled and unlabeled partitions for experiments.
    - Store and output data in standardized format: dict with keys 'X', 'Y', 'X_unlabeled', 'f_hat' as needed.
    - Ensure reproducibility by fixing random seed.
    """

    def __init__(self, data_params: dict):
        """
        Initialize DatasetLoader with configuration parameters.
        Args:
            data_params (dict): Dictionary containing configuration parameters from 'config.yaml'.
                Expected keys:
                    - 'synthetic' (bool): Whether to generate synthetic data or load real data.
                    - 'labeled_size' (int): Number of labeled samples.
                    - 'unlabeled_sizes' (list): List of unlabeled sample sizes.
                    - 'seed' (int): Random seed for reproducibility.
        """
        self.synthetic = data_params.get('synthetic', True)
        self.labeled_size = data_params.get('labeled_size', 500)
        self.unlabeled_sizes = data_params.get('unlabeled_sizes', [1000, 2500, 5000, 10000])
        self.seed = data_params.get('seed', 42)
        self.random_state = np.random.RandomState(self.seed)

        # Placeholders for data
        self.data = {
            'X': None,
            'Y': None,
            'X_unlabeled': {},  # dict keyed by unlabeled size
            'f_hat': {}  # optional, can be assigned externally if precomputed
        }

        # Load or generate datasets
        self.load_data()

    def load_data(self):
        """
        Load real dataset or generate synthetic data based on the configuration.
        Returns:
            dict: Data dictionary with keys 'X', 'Y', 'X_unlabeled', 'f_hat'.
        """
        if self.synthetic:
            self._generate_synthetic_data()
        else:
            self._load_real_data()

    def _generate_synthetic_data(self):
        """
        Generate synthetic data as per models described in the paper.
        Implements data for mean estimation, regression, classification, and other tasks.
        """
        # Setting common parameters
        n_labeled = self.labeled_size
        total_unlabeled_sizes = self.unlabeled_sizes
        rs = self.random_state

        # Generate predictors: X1, X2 ~ N(0,1), with independence
        X1 = rs.normal(0, 1, n_labeled + max(total_unlabeled_sizes))
        X2 = rs.normal(0, 1, n_labeled + max(total_unlabeled_sizes))

        # Generate outcome Y according to the model for mean regression as an example
        # Variance explained ~81% with the model Y = beta1*X1 + beta2*X2 + noise
        beta1 = np.sqrt(0.08)
        beta2 = np.sqrt(0.08)
        residual_variance = 1 - (np.var(beta1 * X1) + np.var(beta2 * X2))
        # To fix total variance at 1, we set noise variance accordingly
        noise_std = np.sqrt(residual_variance)

        # Generate outcomes with noise
        epsilon = rs.normal(0, noise_std, n_labeled + max(total_unlabeled_sizes))
        Y_full = beta1 * X1 + beta2 * X2 + epsilon

        # Store features and outcomes for labeled data
        self.data['X'] = np.column_stack((X1[:n_labeled], X2[:n_labeled]))
        self.data['Y'] = Y_full[:n_labeled]

        # Generate unlabeled feature data for each size
        for size in total_unlabeled_sizes:
            X_unlab = np.column_stack((
                X1[n_labeled:n_labeled+size],
                X2[n_labeled:n_labeled+size]
            ))
            self.data['X_unlabeled'][size] = X_unlab
            # For unlabeled data, outcomes are not used unless for predicing Y_hat
            # ML predictions can be added later externally

        # Optional: generate initial ML predictions f_hat (placeholder, can be loaded)
        # For simplicity, in synthetic case, we can set f_hat as the true mean or predict as Y mean
        # Here, as placeholder, set to true Y plus small noise
        self.data['f_hat'] = {}
        # For labeled data
        self.data['f_hat']['train'] = self.data['Y']
        # For unlabeled data, predictions can be generated later with models
        for size in total_unlabeled_sizes:
            self.data['f_hat'][f'unlabeled_{size}'] = np.full(
                shape=(size,), fill_value=np.mean(self.data['Y'])
            )

    def _load_real_data(self):
        """
        Placeholder for loading real datasets such as UK Biobank.
        Actual implementation should handle file I/O, preprocessing, and splits.
        """
        # Example: loading CSV files (paths should be configured appropriately)
        # For placeholder purposes, raise NotImplementedError
        raise NotImplementedError("Real data loading not implemented in this mockup. "
                                  "Implement according to dataset specifics.")

    def get_data(self):
        """
        Returns the assembled data dictionary.
        Returns:
            dict: {'X': np.ndarray, 'Y': np.ndarray, 'X_unlabeled': dict, 'f_hat': dict}
        """
        return self.data
```

## inference.py

```python
## inference.py
import numpy as np
from scipy.stats import norm, t, chi2
from scipy.stats import norm as standard_normal
from scipy.stats import t as student_t
from scipy.stats import chi2 as chi_squared

class Inference:
    """
    Implements the final inference step in the PSPS framework.
    Given summary statistics ({theta_hat, eta_hat, covariances}) and estimated variance matrices,
    computes the debiased point estimate, its variance, confidence interval, and p-value.

    Uses asymptotic normality as justified in the paper.
    """

    def __init__(self, summary_stats: dict, n_labeled: int, n_unlabeled: int):
        """
        Initializes Inference object with summary statistics and sample sizes.
        Args:
            summary_stats (dict):
                - 'theta_hat': float or np.ndarray (point estimate from analysis routine)
                - 'eta_hat': np.ndarray (vector of nuisance summary statistic)
                - 'Cov_theta_eta': np.ndarray (covariance matrix between theta_hat and eta_hat)
                - 'Var_eta_L': np.ndarray (covariance matrix of eta_hat_L)
                - 'Var_eta_U': np.ndarray (covariance matrix of eta_hat_U)
                - 'Var_theta': float or np.ndarray (variance of theta_hat)
            n_labeled (int): sample size of labeled data
            n_unlabeled (int): sample size of unlabeled data
        """
        # Validate inputs
        self.theta_hat = summary_stats.get('theta_hat', None)
        self.eta_hat = summary_stats.get('eta_hat', None)
        self.Cov_theta_eta = summary_stats.get('Cov_theta_eta', None)
        self.Var_eta_L = summary_stats.get('Var_eta_L', None)
        self.Var_eta_U = summary_stats.get('Var_eta_U', None)
        self.Var_theta = summary_stats.get('Var_theta', None)
        self.n_labeled = n_labeled
        self.n_unlabeled = n_unlabeled

        # Ensuring numpy array formats
        if isinstance(self.theta_hat, (float, int)):
            self.theta_hat = np.array([self.theta_hat])
        elif isinstance(self.theta_hat, list):
            self.theta_hat = np.array(self.theta_hat)
        if isinstance(self.eta_hat, list):
            self.eta_hat = np.array(self.eta_hat)
        # Covariance matrices
        self.Cov_theta_eta = np.array(self.Cov_theta_eta)
        self.Var_eta_L = np.array(self.Var_eta_L)
        self.Var_eta_U = np.array(self.Var_eta_U)
        # Variance of theta_hat
        self.Var_theta = np.array(self.Var_theta) if isinstance(self.Var_theta, (list, np.ndarray)) else np.array([self.Var_theta])
        
        # Compute ratio
        self.rho = self.n_labeled / self.n_unlabeled

        # Compute omega_0
        self.omega_0 = self._compute_omega_0()

        # Compute the debiased estimate
        self.theta_psps = self._compute_debiased_estimate()

        # Variance of the estimator (will estimate using bootstrap variance later)
        self.var_theta_psps = None

    def _compute_omega_0(self):
        """
        Compute the weight vector omega_0 based on covariance matrices.
        """
        # Sum of variance matrices scaled
        V_eta_sum = self.Var_eta_L + self.rho * self.Var_eta_U
        # Inverse (regularize if needed)
        epsilon = 1e-8
        V_eta_sum_inv = np.linalg.inv(V_eta_sum + epsilon * np.eye(V_eta_sum.shape[0]))
        # covariance between theta and eta (vector)
        Cov_theta_eta = self.Cov_theta_eta.squeeze()
        # Omega_0: (covariance) * (variance inverse)
        omega = V_eta_sum_inv @ Cov_theta_eta
        return omega

    def _compute_debiased_estimate(self):
        """
        Compute Theta_{PSPS} estimate: theta_hat + omega_0^T (eta_U - eta_L)
        """
        delta_eta = self.eta_hat['eta_U'] - self.eta_hat['eta_L']
        # ensure as np.ndarray
        delta_eta = np.array(delta_eta).flatten()
        estimate = self.theta_hat + np.dot(self.omega_0, delta_eta)
        return estimate

    def bootstrap_variance(self, analysis_fn, n_bootstrap=200, random_state=None):
        """
        Estimate the variance of theta_psps via bootstrap of summary statistics.
        Args:
            analysis_fn (callable): Function that takes data dictionary and returns a dict with keys:
                'theta', 'eta', 'eta_unlabeled' estimates.
            n_bootstrap (int): Number of bootstrap samples.
            random_state (int): Random seed.
        """
        rng = np.random.RandomState(random_state)
        bootstrap_estimates = []

        for _ in range(n_bootstrap):
            # Resample with replacement labeled data
            # For summary statistics, resampling pairs is important for covariance estimation.
            # Here, assume summaries are provided, so we bootstrap at the level of the summaries.
            # For simplicity, perform bootstrap by resampling the key summaries directly.
            # More accurate bootstrap of the full summaries could sample data; here, approximate.
            # Alternatively, tile the summaries with added noise based on covariance estimates.
            # For the purpose of illustration, simulate bootstrap same as variance estimation:

            # Bootstrap of theta, eta, eta_unlabeled
            # Sample with replacement from the bootstrap residuals/estimates
            # For simplicity, generate bootstrap summaries around current estimate
            # As per paper, bootstrap of summaries:
            # Add multivariate normal noise with estimated covariance
            # Create bootstrap sample summaries
            summary_vector = np.concatenate([
                np.array([self.theta_hat]),
                self.eta_hat['eta_L'],
                self.eta_hat['eta_U']
            ])
            # Construct covariance matrix for summaries
            # For illustration, approximate with provided variances
            cov_diag = np.zeros_like(summary_vector)
            # Filling covariance matrix accordingly:
            # For simplicity, use identity scaled by variances
            # For accuracy, user can provide bootstrap covariance matrices directly
            cov_matrix = np.diag(np.concatenate([self.Var_theta, np.diag(self.Var_eta_L), np.diag(self.Var_eta_U)]))
            sampled_summary = rng.multivariate_normal(summary_vector, cov_matrix)

            # Decompose sampled summaries back
            # Suppose we know the structure:
            theta_bs = sampled_summary[0]
            eta_L_bs = sampled_summary[1:1+len(self.eta_hat['eta_L'])]
            eta_U_bs = sampled_summary[1+len(self.eta_hat['eta_L']:)]
            # Evaluate analysis function with bootstrap summaries
            bootstrap_summary = {
                'theta': theta_bs,
                'eta': {'eta_L': eta_L_bs, 'eta_U': eta_U_bs}
            }
            est = analysis_fn(bootstrap_summary)
            bootstrap_estimates.append(est['theta'])

        # Variance estimate as empirical variance over bootstrap estimates
        bootstrap_estimates = np.array(bootstrap_estimates)
        self.var_theta_psps = np.var(bootstrap_estimates, ddof=1)

    def estimate_variance(self):
        """
        Compute the variance of theta_psps, via bootstrap estimates.
        """
        if self.var_theta_psps is None:
            raise RuntimeError("Run bootstrap_variance() first to estimate variance.")
        return self.var_theta_psps

    def construct_confidence_interval(self, alpha=0.05):
        """
        Construct (1 - alpha) confidence interval based on asymptotic normality.
        """
        # Use t- or normal-approx, as the sample size is large, normal is justified
        z = norm.ppf(1 - alpha / 2)
        se = np.sqrt(self.var_theta_psps)
        lower = self.theta_psps - z * se
        upper = self.theta_psps + z * se
        return (lower.item(), upper.item())

    def hypothesis_test(self, theta_null=0):
        """
        Conduct two-sided Wald test p-value for H0: theta = theta_null
        """
        se = np.sqrt(self.var_theta_psps)
        z_score = (self.theta_psps - theta_null) / se
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        return {
            'z_score': float(z_score),
            'p_value': float(p_value)
        }

    def report(self, alpha=0.05):
        """
        Generate a report dictionary: estimate, std error, CI, p-value.
        """
        # Ensure variance estimate is available
        if self.var_theta_psps is None:
            raise RuntimeError("Variance not estimated. Run bootstrap_variance() first.")
        se = np.sqrt(self.var_theta_psps)
        ci_lower, ci_upper = self.construct_confidence_interval(alpha)
        pval_info = self.hypothesis_test(0)

        report = {
            'estimate': self.theta_psps,
            'std_error': se,
            'confidence_interval': (ci_lower, ci_upper),
            'p_value': pval_info['p_value']
        }
        return report
```

## main.py

```python
# main.py
import os
import yaml
import numpy as np
from dataset_loader import DatasetLoader
from model import MLModel
from trainer import Trainer
from analysis import AnalysisRoutine
from variance_estimation import VarianceEstimator
from inference import Inference

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set general parameters
    data_params = config.get('data', {})
    model_params = config.get('model', {})
    variance_params = config.get('variance_estimation', {})
    analysis_method = config.get('analysis', {}).get('method', 'regression')
    repetitions = config.get('experiment', {}).get('repetitions', 1000)
    results_dir = config.get('output', {}).get('results_dir', 'results/')

    # Create results directory if not exist
    os.makedirs(results_dir, exist_ok=True)

    # Data configuration
    synthetic = data_params.get('synthetic', True)
    labeled_size = data_params.get('labeled_size', 500)
    unl_sizes = data_params.get('unlabeled_sizes', [1000, 2500, 5000, 10000])
    seed = 42
    np.random.seed(seed)

    # Initialize a list to keep aggregated results
    simulation_results = []

    # Loop over each specified unlabeled size
    for N_unlab in unl_sizes:
        print(f"\nStarting simulation for unlabeled size N={N_unlab}...")

        # Load or generate dataset
        dataset_loader = DatasetLoader({
            'synthetic': synthetic,
            'labeled_size': labeled_size,
            'unlabeled_sizes': [N_unlab],
            'seed': seed
        })
        data_dict = dataset_loader.get_data()
        X_labeled = data_dict['X']
        Y_labeled = data_dict['Y']
        X_unlabeled = data_dict['X_unlabeled'][N_unlab]

        # Initialize and train ML model
        model = MLModel(model_params)
        model.train(X_labeled, Y_labeled)

        # Generate predictions
        f_hat_labeled = model.predict(X_labeled)
        f_hat_unlabeled = model.predict(X_unlabeled)

        # Prepare dataset for analysis routines
        dataset = {
            'X_lab': X_labeled,
            'Y_lab': Y_labeled,
            'f_hat_lab': f_hat_labeled,
            'X_unlab': X_unlabeled,
            'f_hat_unlab': f_hat_unlabeled
        }

        # Storage for metrics per simulation
        coverages = []
        ci_widths = []
        p_values = []
        estimates = []

        for rep in range(repetitions):
            # 1. Instantiate analysis routine based on task
            analysis = AnalysisRoutine(model, dataset, method=analysis_method)
            # 2. Compute summary statistics
            summary_stats = analysis.compute_summary_statistics()

            # 3. Variance estimation via bootstrap
            var_estimator = VarianceEstimator(dataset, summary_stats, {
                'variance_estimation': {'bootstrap_samples': 200}
            })
            var_estimator.bootstrap_variance(analysis.compute_summary_statistics, n_bootstrap=200)
            var_covs = var_estimator.estimate(analysis.compute_summary_statistics)

            # 4. Compute weights for PSPS
            n_nlab = len(X_labeled)
            n_unlab_curr = N_unlab
            rho = n_nlab / n_unlab_curr
            # Variance and covariance matrices from bootstrap
            Var_eta_L = var_covs['Var_eta']
            Var_eta_U = var_covs['Var_eta_unlabeled']
            Cov_theta_eta = var_covs['Cov_theta_eta']

            # Compute omega_0
            # Add small epsilon for numerical stability
            epsilon = 1e-8
            A_inv = np.linalg.inv(Var_eta_L + rho * Var_eta_U + epsilon * np.eye(Var_eta_L.shape[0]))
            omega_0 = A_inv @ Cov_theta_eta.squeeze()

            # 5. Final PSPS estimator
            delta_eta = summary_stats['eta']['eta_U'] - summary_stats['eta']['eta_L']
            delta_eta = np.array(delta_eta).flatten()
            theta_hat = summary_stats['theta_hat']
            theta_psps = theta_hat + np.dot(omega_0, delta_eta)

            # 6. Variance of PSPS estimator
            # Algebraic approximation using bootstrap variances
            Var_theta = var_covs['Var_theta']
            # Compute the variance using the formula from paper
            var_psps = (
                Var_theta
                - np.dot(
                    np.dot(Cov_theta_eta, np.linalg.inv(Var_eta_L + rho * Var_eta_U)),
                    Cov_theta_eta
                )
            )
            var_psps = float(var_psps) if np.isscalar(var_psps) else np.array(var_psps).flatten()[0]
            # To be conservative, we can set variance estimate as bootstrap estimate
            # Here, for simplicity, we use the algebraic approximation

            # 7. Confidence interval and p-value
            se_psps = np.sqrt(var_psps)
            z_alpha = norm.ppf(1 - 0.025)  # 95% CI
            ci_lower = theta_psps - z_alpha * se_psps
            ci_upper = theta_psps + z_alpha * se_psps
            p_value = 2 * (1 - norm.cdf(abs(theta_psps / se_psps)))

            # 8. Save metrics
            coverages.append(1 if (ci_lower <= 0 <= ci_upper) else 0)  # For theta=0 (simulate)
            ci_widths.append(ci_upper - ci_lower)
            p_values.append(p_value)
            estimates.append(theta_psps)

        # After repetitions, compute coverage probability, mean CI width, and power
        coverage_rate = np.mean(coverages)
        mean_width = np.mean(ci_widths)
        # For power, compare p-values against 0.05
        power = np.mean(np.array(p_values) < 0.05)
        # Store results
        result = {
            'unlabeled_size': N_unlab,
            'coverage': coverage_rate,
            'ci_width': mean_width,
            'power': power,
            'estimate_mean': np.mean(estimates),
            'estimate_std': np.std(estimates)
        }
        simulation_results.append(result)
        print(f"Results for N={N_unlab}: coverage={coverage_rate:.3f}, width={mean_width:.3f}, power={power:.3f}")

    # Save results to file
    import json
    results_path = os.path.join(results_dir, 'simulation_summary.json')
    with open(results_path, 'w') as f:
        json.dump(simulation_results, f, indent=2)

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class MLModel:
    """
    MLModel is a wrapper class for an ML prediction model used within the PSPS framework.
    Currently supports RandomForestRegressor with configurable hyperparameters.
    Provides methods to initialize, train, and predict.
    """

    def __init__(self, model_params: dict):
        """
        Initialize the MLModel instance based on provided parameters.
        Args:
            model_params (dict): Dictionary of model hyperparameters.
                Expected keys:
                    - 'type' (str): Supported 'RandomForestRegressor'.
                    - 'n_estimators' (int): Number of trees.
                    - 'max_depth' (Optional[int]): Max depth of trees.
                    - 'random_state' (int): Random seed for reproducibility.
        """
        model_type = model_params.get('type', 'RandomForestRegressor')
        n_estimators = model_params.get('n_estimators', 500)
        max_depth = model_params.get('max_depth', None)
        random_state = model_params.get('random_state', 42)

        if model_type != 'RandomForestRegressor':
            raise ValueError(f"Unsupported model type: {model_type}. Supported: 'RandomForestRegressor'.")

        # Instantiate the scikit-learn model
        self.model_type = model_type
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }
        self.model_instance = RandomForestRegressor(**self.model_params)

        # Internal flag to track if model is trained
        self._is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ML model to the training data.
        Args:
            X (np.ndarray): Features with shape (n_samples, n_features).
            y (np.ndarray): Outcomes with shape (n_samples,).
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X should be a numpy.ndarray, but got {type(X)}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y should be a numpy.ndarray, but got {type(y)}")
        if X.ndim != 2:
            raise ValueError(f"X should be a 2D array, but got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y should be a 1D array, but got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X and y do not match: {X.shape[0]} vs {y.shape[0]}")

        # Fit the RandomForestRegressor
        self.model_instance.fit(X, y)
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for new data points.
        Args:
            X (np.ndarray): Features with shape (n_samples, n_features).
        Returns:
            np.ndarray: Predictions with shape (n_samples,).
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X should be a numpy.ndarray, but got {type(X)}")
        if X.ndim != 2:
            raise ValueError(f"X should be a 2D array, but got shape {X.shape}")

        # Generate predictions
        predictions = self.model_instance.predict(X)
        return predictions
```

## trainer.py

```python
## trainer.py

import logging
import time
from typing import Optional, Dict, Tuple

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from model import MLModel

class Trainer:
    """
    Purpose and Role:
    - Responsible for training a machine learning model (e.g., RandomForestRegressor)
      based on provided labeled data.
    - Supports optional hyperparameter tuning via grid search for improved performance.
    - Manages training workflow, validation, and model storage.
    """

    def __init__(
        self,
        data: dict,
        model_config: dict,
        hyperparam_tuning: bool = False,
        validation_split: float = 0.2,
        random_state: int = 42,
        verbose: bool = False
    ):
        """
        Initialize the Trainer instance.
        Args:
            data (dict): Data dictionary containing 'X' and 'Y' as numpy arrays.
            model_config (dict): Model hyperparameters and type from config.yaml.
            hyperparam_tuning (bool): Whether to perform hyperparameter tuning.
            validation_split (float): Fraction of labeled data used for validation during tuning.
            random_state (int): Seed for reproducibility.
            verbose (bool): Verbosity flag.
        """
        self.data = data
        self.hyperparam_tuning = hyperparam_tuning
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose

        # Extract features and labels
        self.X = data.get('X')
        self.Y = data.get('Y')

        # Validate presence of data
        if self.X is None or self.Y is None:
            raise ValueError("Data dictionary must contain 'X' and 'Y' keys with numpy arrays.")

        # Initialize model parameters
        self.model_type = model_config.get('type', 'RandomForestRegressor')
        self.model_params = {
            'n_estimators': model_config.get('n_estimators', 500),
            'max_depth': model_config.get('max_depth', None),
            'random_state': model_config.get('random_state', 42)
        }

        self.trained_model: Optional[BaseEstimator] = None

    def train(self) -> None:
        """
        Execute the training workflow.
        If hyperparameter tuning is enabled, perform grid search, then retrain with best params.
        Otherwise, train directly with specified parameters.
        """
        if self.hyperparam_tuning:
            self._hyperparameter_tuning()
        else:
            self._train_model(self.model_params)

    def _train_model(self, params: Dict) -> None:
        """
        Instantiate and train the ML model with given hyperparameters.
        Args:
            params (Dict): Hyperparameters for model initialization.
        """
        start_time = time.time()

        # Instantiate and fit model
        if self.model_type == 'RandomForestRegressor':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # Fit the model on entire labeled data
        model.fit(self.X, self.Y)

        self.trained_model = model

        end_time = time.time()
        if self.verbose:
            print(f"Training completed in {end_time - start_time:.2f} seconds.")

    def _hyperparameter_tuning(self) -> None:
        """
        Perform hyperparameter tuning using grid search with validation split.
        """
        # Define hyperparameter grid (can be extended)
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 10]
        }
        from sklearn.model_selection import train_test_split
        # Split data into training and validation for tuning
        X_train, X_val, y_train, y_val = train_test_split(
            self.X,
            self.Y,
            test_size=self.validation_split,
            random_state=self.random_state
        )

        # Initialize model with default params
        model = RandomForestRegressor(
            n_estimators=self.model_params['n_estimators'],
            max_depth=self.model_params['max_depth'],
            random_state=self.model_params.get('random_state', 42)
        )

        # Grid SearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=1 if self.verbose else 0,
            n_jobs=-1
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()

        if self.verbose:
            print(f"Hyperparameter tuning done in {end_time - start_time:.2f} seconds.")
            print(f"Best params: {grid_search.best_params_}")
            best_score = -grid_search.best_score_
            print(f"Validation RMSE: {np.sqrt(best_score):.4f}")

        # Retrain model with best hyperparameters on entire labeled data
        best_params = grid_search.best_params_
        best_params['max_depth'] = best_params['max_depth'] if best_params['max_depth'] != 0 else None

        self._train_model(best_params)

    def validate(self) -> Dict[str, float]:
        """
        Optional validation on held-out validation set.
        Computes and returns validation metrics.
        """
        if self.trained_model is None:
            raise RuntimeError("Model must be trained before validation.")
        # For validation, retrain models if hyperparameter tuning, else skip
        # Here, assume we trained on full data
        # Placeholder: validation on the same data (not ideal), can be extended
        predictions = self.trained_model.predict(self.X)
        rmse = np.sqrt(mean_squared_error(self.Y, predictions))
        r2 = r2_score(self.Y, predictions)
        metrics = {
            'RMSE': rmse,
            'R2': r2
        }
        if self.verbose:
            print(f"Validation RMSE: {rmse:.4f}, R2: {r2:.4f}")
        return metrics

    def get_trained_model(self) -> BaseEstimator:
        """
        Return the trained model object for downstream predictions.
        Raises error if model is not trained.
        """
        check_is_fitted(self.trained_model)
        return self.trained_model
```

## variance_estimation.py

```python
## variance_estimation.py
import numpy as np
from sklearn.utils import resample

class VarianceEstimator:
    """
    VarianceEstimator handles bootstrap-based estimation of variance and covariance matrices
    required for the PSPS framework. It provides bootstrap estimates of:
        - Var(theta_hat_L)
        - Var(eta_hat_L)
        - Cov(theta_hat_L, eta_hat_L)
        - Var(eta_hat_U)
    """

    def __init__(self, data: dict, analysis_results: dict, config: dict):
        """
        Initialize VarianceEstimator.
        Args:
            data (dict): Data dictionary with keys:
                - 'X': np.ndarray of labeled features (n x p)
                - 'Y': np.ndarray outcomes (n,)
                - 'X_unlabeled': dict of unlabeled features keyed by size
                - 'f_hat': dict with predictions for labeled and unlabeled data
            analysis_results (dict): Output from AnalysisRoutine, may include point estimates, etc.
            config (dict): Configuration including number of bootstrap samples.
        """
        self.X = data.get('X')
        self.Y = data.get('Y')
        self.X_unlabeled = data.get('X_unlabeled', {})
        self.f_hat = data.get('f_hat', {})
        self.analysis_results = analysis_results
        self.N_boot = None
        self._parse_config(config)
        # Seed for reproducibility; can be set via config if needed
        self.random_state = np.random.RandomState(42)

        # Bootstrap estimates containers
        self._boot_theta = []
        self._boot_eta = []
        self._boot_eta_u = []

    def _parse_config(self, config: dict):
        """
        Extract bootstrap sample size from config.
        """
        self.N_boot = config.get('variance_estimation', {}).get('bootstrap_samples', 200)

    def bootstrap_variance(self, analysis_fn):
        """
        Perform bootstrap resampling to estimate variance and covariance matrices.
        Args:
            analysis_fn (callable): Function that takes data dict and returns point estimate(s).
                                    For example, a function to compute theta, eta, eta_unlabeled.
        Returns:
            dict: Variance and covariance matrices:
                {
                    'Var_theta': np.ndarray (scalar or vector),
                    'Var_eta': np.ndarray (K x K),
                    'Cov_theta_eta': np.ndarray (K),
                    'Var_eta_unlabeled': np.ndarray (K x K)
                }
        """
        # Initialize bootstrap containers
        boot_theta_list = []
        boot_eta_list = []
        boot_eta_u_list = []

        n_samples = self.X.shape[0]
        # For each bootstrap replicate
        for q in range(self.N_boot):
            # Resample labeled data
            idxs = resample(np.arange(n_samples), replace=True, n_samples=n_samples, random_state=self.random_state)
            X_boot = self.X[idxs]
            Y_boot = self.Y[idxs]
            # For unlabeled data, resample features accordingly
            # For simplicity, assume features are independent samples
            # If necessary, resample unlabeled features for each size
            bootstrap_data = {
                'X': X_boot,
                'Y': Y_boot,
                'X_unlabeled': self.X_unlabeled,
                'f_hat': self.f_hat
            }
            # Run analysis function to get bootstrap estimates
            est = analysis_fn(bootstrap_data)
            # Append estimates: expecting dict with keys 'theta', 'eta', 'eta_unlabeled'
            boot_theta_list.append(est['theta'])
            boot_eta_list.append(est['eta'])
            boot_eta_u_list.append(est['eta_unlabeled'])

        # Convert lists to numpy arrays for variance/covariance calc
        boot_theta_arr = np.array(boot_theta_list)
        boot_eta_arr = np.array(boot_eta_list)
        boot_eta_u_arr = np.array(boot_eta_u_list)

        # Estimate variance/covariance matrices
        Var_theta = np.var(boot_theta_arr, axis=0, ddof=1)
        Var_eta = np.cov(boot_eta_arr, rowvar=False, ddof=1)
        Cov_theta_eta = np.cov(boot_theta_arr.T, boot_eta_arr.T, ddof=1)[0:boot_theta_arr.shape[1], boot_theta_arr.shape[1]:]
        Var_eta_unlabeled = np.cov(boot_eta_u_arr, rowvar=False, ddof=1)

        return {
            'Var_theta': Var_theta,
            'Var_eta': Var_eta,
            'Cov_theta_eta': Cov_theta_eta,
            'Var_eta_unlabeled': Var_eta_unlabeled
        }

    def estimate(self, analysis_fn):
        """
        Run bootstrap and return estimated variance matrices.
        Args:
            analysis_fn (callable): Function to compute point estimator(s) from data.
        Returns:
            dict: Variance and covariance matrices as above.
        """
        return self.bootstrap_variance(analysis_fn)
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\psps\psps_repo`
