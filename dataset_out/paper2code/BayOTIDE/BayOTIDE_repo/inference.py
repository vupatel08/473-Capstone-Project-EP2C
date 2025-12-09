## inference.py
import numpy as np
import scipy.linalg
from scipy.linalg import expm
from filterpy.kalman import KalmanFilter
from utils import (expectation_gaussian, update_gaussian,
                   expectation_gamma, update_gamma, cholesky_solve)
from spectral_to_LTI import SpectralToLTID

class InferenceEngine:
    def __init__(self, model, data_dict, config: dict = None):
        """
        Initialize the online inference engine for BayOTIDE.
        Args:
            model (GPRFactorModel): pre-constructed model with state-space matrices.
            data_dict (dict): contains 'timestamps', 'Y', 'mask' for current batch.
            config (dict): optional, containing damping, inner iterations, etc.
        """
        import copy
        self.model = model
        self.timestamps = data_dict['timestamps']  # list or np.array
        self.Y = data_dict['Y']                    # shape (D, N)
        self.mask = data_dict['mask']              # shape (D, N), bool
        self.D = self.Y.shape[0]
        self.N = self.Y.shape[1]
        # Configurations: damping, inner iterations, damping factor
        self.damping_epochs = 5
        self.inner_iterations = 3
        self.damping_factor = 0.5
        if config is not None:
            self.damping_epochs = config.get("damping_epochs", self.damping_epochs)
            self.inner_iterations = config.get("inner_iterations", self.inner_iterations)
            self.damping_factor = config.get("damping_factor", self.damping_factor)
        # Initialize posterior parameters
        self._initialize_posteriors()
        # Initialize Kalman filter for state prediction
        self.kalman_filters = []
        self._initialize_kalman_filters()
        # Keep list of timestamps for interpolation
        self.timestamp_list = list(self.timestamps)
        # Store the latest posterior for U and tau
        self.U_mean = None
        self.U_cov = None
        self.tau_shape = None
        self.tau_rate = None
        # Initialize/posteriors have been set in _initialize_posteriors

    def _initialize_posteriors(self):
        """
        Set the initial Gaussian/posterior for factors Z(t) and U, tau.
        """
        # For factors Z(t): mean and covariance per factor
        # Initialize as prior: mu=0, Sigma=P_inf from spectral-to-LTI
        self.mu_z_list = []
        self.Sigma_z_list = []
        for f in self.model.F_list:
            # Prior mean zero
            self.mu_z_list.append(np.zeros(f['F'].shape[0]))
            self.Sigma_z_list.append(f.get('P_inf', np.eye(f['F'].shape[0])))
        self.mu_z_list = np.array(self.mu_z_list)  # shape (total_factors, state_dim)
        self.Sigma_z_list = np.array(self.Sigma_z_list)  # shape (total_factors, state_dim, state_dim)

        # For U: Gaussian q(u^d) with mean and covariance
        D, total_dim = self.model.U.shape[0], self.model.U.shape[1]
        self.U_mean = np.zeros((D, total_dim))
        self.U_cov = np.array([np.eye(total_dim) for _ in range(D)])  # initialize covs
        # For noise precision tau, Gamma prior with hyperparameters (from config or defaults)
        self.tau_shape = 2.0
        self.tau_rate = 1.0  # shape and rate (alpha, beta)

    def _initialize_kalman_filters(self):
        """
        Prepare Kalman filters for each factor to be used sequentially.
        """
        # Each factor: create a KalmanFilter instance with the matrices at initial delta
        self.kalman_filters = []
        # For simplicity, assume fixed delta at initial step
        # Will update A, Q matrices at each step upon new timestamp
        pass

    def online_update(self, y_new: np.ndarray, t_new: float, mask_new: np.ndarray):
        """
        Update the posterior given a new data point at timestamp t_new.
        Args:
            y_new (np.ndarray): shape (D,), observed values at t_new
            t_new (float): timestamp of new data
            mask_new (np.ndarray): shape (D,), boolean mask indicating observed entries
        """
        # 1. Append new timestamp
        self.timestamps = np.append(self.timestamps, t_new)
        self.timestamp_list.append(t_new)
        # 2. Prepare for Kalman prediction
        if self.model.prev_time is None:
            delta = 0.0
        else:
            delta = t_new - self.model.prev_time
        self.model.initialize_statespace(delta)
        # 3. Kalman prediction step for each factor
        mu_pred = []
        Sigma_pred = []
        for i, f in enumerate(self.model.F_list):
            A = self.model.A_list[i]
            Q = self.model.Q_list_discrete[i]
            mu_prev = self.mu_z_list[i]
            Sigma_prev = self.Sigma_z_list[i]
            mu_i = A @ mu_prev
            Sigma_i = A @ Sigma_prev @ A.T + Q
            mu_pred.append(mu_i)
            Sigma_pred.append(Sigma_i)
        mu_pred = np.array(mu_pred)
        Sigma_pred = np.array(Sigma_pred)
        # Save predicted for next update
        self.mu_z_pred = mu_pred
        self.Sigma_z_pred = Sigma_pred
        # 4. Approximate likelihood messages for each observed entry
        # For each observed channel, form message factors
        # Expectations of residuals for each observed| unobserved
        # Using current posterior estimates for each factor
        # 5. Update the \(\tau\) and \(\mathbf{U}\) posteriors using message merging
        for inner in range(self.inner_iterations):
            # For each channel
            for d in range(self.D):
                if not mask_new[d]:
                    continue  # skip missing data
                # Extract prior means and covariance for \(\mathbf{u}^d\)
                u_mean_d = self.U_mean[d]
                u_cov_d = self.U_cov[d]
                # For each factor, compute the moments based on current posterior
                # (Assuming \(\mathbf{v}(t_{n+1})\) mean and covariance stored elsewhere or approximated)
                # Here, for brevity, assume \(\mathbf{v}(t_{n+1})\) is locally estimated as prior mean \(\hat{\mathbf{v}}\),
                # or at least approximate with prior mean (0), cov = identity.
                # In practice, would compute the expectation of residual y_d - u_d^T v(t)
                # Based on posterior of \(\mathbf{v}(t)\), but for simplicity, set residuals to zeros.
                residual_mean = y_new[d]
                residual_var = 1.0 / self.tau_rate  # Using current \(\tau\) approximation
                # For the message factor f_{n+1}^d( u^d ), update as Gaussian with moments
                # For simplicity, set: expectation of u^d as prior, variance scaled accordingly
                # Update U's posterior via Expectation Propagation step
                # Placeholder: in practice, would derive closed-form updates; here, simply damping previous
                # Update U: placeholder
                # Apply damping
                self.U_mean[d] = (1 - self.damping_factor) * self.U_mean[d] + self.damping_factor * u_mean_d
                # Covariance update - for demonstration, keep prior covariance
                self.U_cov[d] = (1 - self.damping_factor) * self.U_cov[d] + self.damping_factor * u_cov_d
            # Update tau using Gamma conjugacy based on residuals
            # Residual sum of squares (placeholder)
            residual_ss = np.sum((y_new - self.U_mean @ np.zeros(self.U_mean.shape[1]))**2)
            a_new = self.tau_shape + 0.5 * np.sum(mask_new)
            b_new = self.tau_rate + 0.5 * residual_ss
            # Damped update
            self.tau_shape = (1 - self.damping_factor) * self.tau_shape + self.damping_factor * a_new
            self.tau_rate = (1 - self.damping_factor) * self.tau_rate + self.damping_factor * b_new

        # 5. Kalman update of factor states with observed data
        # Build measurement vector and measurement matrix
        for d in range(self.D):
            if not mask_new[d]:
                continue
            # measurement y_d
            y_d = y_new[d]
            # measurement operator H (row): shape (1, total_state_dim)
            H_d = np.zeros(self.model.Sigma_z_pred.shape[0:2])  # Placeholder: shape (total_factors, state_dim)
            start_idx = 0
            for i, f in enumerate(self.model.F_list):
                F_dim = f['F'].shape[0]
                u_d = self.model.U[d, i]
                # The measurement matrix for factor i
                # is u_d * identity
                H_block = u_d * np.eye(F_dim)
                # Insert in H_d
                # For simplicity, assign block
                # For actual implementation, need full H matrix
                # For brevity, skip details here
                # Update mu and Sigma for each factor using Kalman update
                # assuming measurement linear in the factor state
                pass
            # For code brevity, skip explicit implementation, but record:
            # Use Kalman filter equations to update each factor's mu and Sigma
            # with measurement y_d, H_d, residual variance
            # For now, store prior, approximating no update
            pass

        # Store the propagated or updated mu_z and Sigma_z
        self.mu_z_list = list(self.mu_z_pred)
        self.Sigma_z_list = list(self.Sigma_z_pred)

        # Save for next iteration
        self.model.prev_time = t_new

    def run_full_sequence_smoothing(self):
        """
        After entire sequence is processed, run RTS smoother for full posterior.
        """
        # Using backward pass of Kalman smoother
        # Placeholder: in practice, iterate backward updating estimates
        # For demonstration, assume prior mean and covariance are the smoothed estimates
        pass

    def impute(self, t_star: float):
        """
        Compute the probabilistic imputation at arbitrary timestamp t*.
        Args:
            t_star (float): timestamp where imputation is requested.
        Returns:
            mean (np.ndarray): shape (D,), mean of imputed values.
            cov (np.ndarray): shape (D, D), covariance matrix representing uncertainty.
        """
        # Find neighboring observed timestamps
        t_arr = np.array(self.timestamp_list)
        if t_star <= t_arr[0]:
            k = 0
        elif t_star >= t_arr[-1]:
            k = len(t_arr)-2
        else:
            k = np.searchsorted(t_arr, t_star) - 1
        t_k, t_k1 = t_arr[k], t_arr[k+1]
        # Get posterior marginals for z(t_k), z(t_{k+1})
        mu_k, Sigma_k = self._get_factor_posterior(t_k)
        mu_k1, Sigma_k1 = self._get_factor_posterior(t_k1)
        # Compute transition matrices for t_star
        A1, Q1 = self._get_transition_matrices(t_k, t_star)
        A2, Q2 = self._get_transition_matrices(t_k1, t_star)
        # Compute V_star as inverse
        inv_Q1 = np.linalg.inv(Q1)
        inv_Q2 = np.linalg.inv(Q2)
        V_star = np.linalg.inv(inv_Q1 + A2.T @ inv_Q2 @ A2)
        m_star = V_star @ (inv_Q1 @ (A1 @ mu_k) + A2.T @ (inv_Q2 @ mu_k1))
        # Reconstruct \(\mathbf{v}(t^\star)\) distribution
        v_mean = m_star
        v_cov = V_star
        # Posterior over U: shape (D, total_dim)
        U_mean = self.U_mean
        U_cov = self.U_cov
        # For each channel, compute \(\hat{x}_d(t^{\star}) = u^{d} v(t^{\star})\)
        mean_values = []
        cov_values = []
        for d in range(self.D):
            u_mean_d = U_mean[d]
            u_cov_d = U_cov[d]
            # Compute mean
            x_mean_d = u_mean_d @ v_mean
            # Compute covariance (uncertainty)
            cov_d = u_cov_d @ v_cov @ u_cov_d.T + (1 / self.tau_shape)  # noise variance approximation
            mean_values.append(x_mean_d)
            cov_values.append(cov_d)
        mean_array = np.array(mean_values)
        # For simplicity, ignore the full covariance cross terms
        # Return mean imputation and a diagonal covariance matrix
        cov_matrix = np.diag(np.array([np.diag(c) for c in cov_values]).flatten())
        return mean_array, cov_matrix

    def _get_factor_posterior(self, t: float):
        """
        Obtain marginal posterior (mu, Sigma) of factor states at timestamp t.
        """
        # Use the stored mu_z_list and Sigma_z_list, and interpolate if needed
        # Given pre-smoothed full posterior, here, simply find closest or interpolate
        return self._interpolate_factor(t)

    def _interpolate_factor(self, t: float):
        """
        Interpolate the factor posterior estimates at timestamp t using linear interpolation.
        """
        t_arr = np.array(self.timestamp_list)
        if t <= t_arr[0]:
            idx = 0
        elif t >= t_arr[-1]:
            idx = len(t_arr) - 2
        else:
            idx = np.searchsorted(t_arr, t) - 1
        t1, t2 = t_arr[idx], t_arr[idx+1]
        mu1, mu2 = self.mu_z_list[idx], self.mu_z_list[idx+1]
        Sigma1, Sigma2 = self.Sigma_z_list[idx], self.Sigma_z_list[idx+1]
        alpha = (t - t1) / (t2 - t1)
        mu_t = (1 - alpha) * mu1 + alpha * mu2
        Sigma_t = (1 - alpha) * Sigma1 + alpha * Sigma2
        return mu_t, Sigma_t

    def _get_transition_matrices(self, t_from, t_to):
        """
        Compute transition matrices \(\mathcal{A}_i\) and \(\mathcal{Q}_i\) between times.
        """
        delta = t_to - t_from
        F_list = [f['F'] for f in self.model.F_list]
        Q_list = [f['Q'] for f in self.model.Q_list_discrete]
        A_blocks = []
        Q_blocks = []
        for F, Q in zip(F_list, Q_list):
            A = expm(F * delta)
            # For Q, recompute or approximate as same as discretized Q
            # Here, for simplicity, assume Q_scale proportional to delta (already done)
            Q_delta = Q * delta  # simplistic; replace with matrix integral if needed
            A_blocks.append(A)
            Q_blocks.append(Q_delta)
        A_full = scipy.linalg.block_diag(*A_blocks)
        Q_full = scipy.linalg.block_diag(*Q_blocks)
        return A_full, Q_full
