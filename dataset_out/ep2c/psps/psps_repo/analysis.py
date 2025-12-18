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
