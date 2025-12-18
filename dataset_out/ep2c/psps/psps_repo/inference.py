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
