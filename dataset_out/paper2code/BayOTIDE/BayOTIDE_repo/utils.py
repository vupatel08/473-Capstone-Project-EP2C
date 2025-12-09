## utils.py

import numpy as np
import scipy.linalg
from scipy.linalg import expm
from numpy.linalg import solve
from scipy.stats import gamma, norm

def cholesky_decomposition(A: np.ndarray):
    """
    Perform Cholesky decomposition with numerical stability.
    Args:
        A (np.ndarray): Must be symmetric and positive-definite.
    Returns:
        L (np.ndarray): Lower-triangular matrix such that A = L @ L.T
    """
    # Add jitter if needed for numerical stability
    jitter = 1e-8
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        A_stable = A + jitter * np.eye(A.shape[0])
        L = np.linalg.cholesky(A_stable)
    return L

def matrix_inverse(A: np.ndarray):
    """
    Compute matrix inverse with fallback for singular matrices.
    Args:
        A (np.ndarray): Square matrix.
    Returns:
        A_inv (np.ndarray): Inverse of A.
    """
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse as fallback
        return np.linalg.pinv(A)

def solve_linear_system(A: np.ndarray, B: np.ndarray):
    """
    Solve linear system Ax = B in a numerically stable way.
    Args:
        A (np.ndarray): Square matrix.
        B (np.ndarray): Right-hand side.
    Returns:
        x (np.ndarray): Solution vector.
    """
    return np.linalg.solve(A, B)

def matrix_exponential(F: np.ndarray, delta: float):
    """
    Compute matrix exponential A = expm(F * delta).
    Args:
        F (np.ndarray): State matrix.
        delta (float): time step.
    Returns:
        A (np.ndarray): State transition matrix.
    """
    return expm(F * delta)

def solve_lyapunov(F: np.ndarray, Q: np.ndarray):
    """
    Solve continuous Lyapunov equation: F P + P F^T + Q = 0
    Args:
        F (np.ndarray): State matrix.
        Q (np.ndarray): Process noise covariance.
    Returns:
        P (np.ndarray): Steady-state covariance matrix.
    """
    return scipy.linalg.solve_lyapunov(F, -Q)

def generate_state_space_matrices(kernel_type: str, hyperparameters: dict):
    """
    Generate the state-space matrices (F, L, Q) based on kernel type and hyperparameters.
    Args:
        kernel_type (str): 'matern32' or 'periodic'
        hyperparameters (dict): For 'matern32': {'length_scale', 'variance'}
                                     for 'periodic': {'length_scale', 'period', 'variance'}
    Returns:
        F (np.ndarray): State matrix.
        L (np.ndarray): Input matrix.
        Q (np.ndarray): Process noise covariance.
    """
    if kernel_type == 'matern32':
        return construct_matern32_state_space(hyperparameters)
    elif kernel_type == 'periodic':
        return construct_periodic_state_space(hyperparameters)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def construct_matern32_state_space(hyperparameters: dict):
    """
    Construct state-space (F, L, Q) for Matérn 3/2 kernel.
    """
    l = hyperparameters.get('length_scale', 1.0)
    sigma2 = hyperparameters.get('variance', 1.0)
    lam = np.sqrt(3.0) / l

    F = np.array([[0., 1.],
                  [-lam**2, -2*lam]], dtype=np.float64)
    L = np.array([[0.],
                  [1.]], dtype=np.float64)
    q_s = 4.0 * (lam ** 3) * sigma2
    Q = np.array([[0., 0.],
                  [0., q_s]], dtype=np.float64)
    P_inf = solve_lyapunov(F, -L @ L.T * q_s)
    return F, L, Q, P_inf

def construct_periodic_state_space(hyperparameters: dict):
    """
    Construct state-space matrices for periodic kernel as a sum over harmonics.
    """
    period = hyperparameters.get('period', 12.0)
    length_scale = hyperparameters.get('length_scale', 1.0)
    variance = hyperparameters.get('variance', 1.0)
    D_s = hyperparameters.get('D_s', 10)  # default number of harmonics

    F_blocks = []
    L_blocks = []
    Q_blocks = []

    for j in range(1, D_s + 1):
        omega_j = 2. * np.pi * j / period
        F_j = np.array([[0., -omega_j],
                        [omega_j, 0.]])
        L_j = np.array([[0.],
                        [1.]])
        # Spectral coefficient approximation: decay with j^2
        q_j_squared = variance / (j ** 2)
        Q_j = q_j_squared * np.eye(2)
        F_blocks.append(F_j)
        L_blocks.append(L_j)
        Q_blocks.append(Q_j)

    F = scipy.linalg.block_diag(*F_blocks)
    L = scipy.linalg.block_diag(*L_blocks)
    Q = scipy.linalg.block_diag(*Q_blocks)
    P_inf = None  # Can be computed if needed
    return F, L, Q, P_inf

def clip_positive_definite(A: np.ndarray, min_value=1e-8):
    """
    Ensure a matrix A is positive definite by adding jitter if necessary.
    """
    try:
        np.linalg.cholesky(A)
        return A
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvals(A)
        min_eig = np.min(np.real(eigvals))
        if min_eig > 0:
            return A
        else:
            adjustment = (-min_eig + min_value)
            return A + adjustment * np.eye(A.shape[0])

def compute_expectation_gaussian(mean: float, variance: float):
    """
    Return expectation and variance of Gaussian distribution given mean and variance.
    """
    return mean, variance

def expectation_tau(U_mean: np.ndarray, U_cov: np.ndarray, 
                    residuals: np.ndarray = None):
    """
    Compute the expectation of the noise precision tau under its current Gamma posterior.
    Args:
        U_mean (np.ndarray): shape (D, total_dim)
        U_cov (np.ndarray): shape (D, total_dim, total_dim)
        residuals (np.ndarray): optional, shape (D,) residuals for current data
    Returns:
        E_tau (float): Expectation of tau
    """
    # For approximation, use the posterior parameters
    # E[tau] = a / b for Gamma with shape a, rate b
    # Here, shape and rate are stored or known
    # For simplicity, assume prior / approximate as 1.0
    pass  # Placeholder

def expectation_gamma(shape: float, rate: float):
    """
    Compute expectation and variance for Gamma distribution.
    """
    mean = shape / rate
    var = shape / (rate ** 2)
    return mean, var

def expectation_gaussian_from_params(mu: np.ndarray, sigma2: np.ndarray):
    """
    Compute expectation of Gaussian with mean mu and variance sigma2.
    """
    return mu, sigma2

def expectation_mixed_gaussian(mu_list: list, cov_list: list):
    """
    Compute the mean and covariance of a mixture of Gaussians.
    Args:
        mu_list: list of mean vectors (ndarray)
        cov_list: list of covariance matrices (ndarray)
    Returns:
        mixture_mean: mean vector
        mixture_covariance: covariance matrix
    """
    total_mu = np.mean(mu_list, axis=0)
    total_cov = np.zeros_like(cov_list[0])
    for mu, cov in zip(mu_list, cov_list):
        diff = mu - total_mu
        total_cov += cov + np.outer(diff, diff)
    total_cov /= len(mu_list)
    return total_mu, total_cov

def expectation_CRPS(obs: np.ndarray, mu: float, sigma: float):
    """
    Compute the Continuous Ranked Probability Score (CRPS) for a Gaussian.
    Args:
        obs (float): observed value
        mu (float): predictive mean
        sigma (float): predictive standard deviation
    Returns:
        crps (float)
    """
    u = (obs - mu) / sigma
    crps = sigma * (u * (2 * norm.cdf(u) - 1) + 2 * norm.pdf(u) - 1/np.sqrt(np.pi))
    return crps

def compute_log_likelihood_gaussian(y: np.ndarray, mu: np.ndarray, sigma2: np.ndarray):
    """
    Compute the log-likelihood of y under Gaussian predictions.
    Args:
        y (np.ndarray): observed data.
        mu (np.ndarray): predicted means.
        sigma2 (np.ndarray): predicted variances.
    Returns:
        log_lik (float): log-likelihood value.
    """
    residual = y - mu
    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (residual ** 2) / sigma2)
    return ll

def sample_gaussian(mu: np.ndarray, sigma2: np.ndarray, n_samples: int = 1):
    """
    Generate samples from Gaussian distribution.
    """
    std = np.sqrt(sigma2)
    samples = mu + std * np.random.randn(n_samples, *mu.shape)
    return samples

def expectation_of_gaussian(mu: np.ndarray, sigma2: np.ndarray):
    """
    Return expectation of Gaussian distribution.
    """
    return mu, sigma2

def expectation_of_gamma(shape: float, rate: float):
    """
    Expectation of Gamma distribution.
    """
    mean = shape / rate
    variance = shape / (rate ** 2)
    return mean, variance

def precision_gamma_to_shape_rate(precision: float):
    """
    Convert between alpha, beta of Gamma to shape and rate if needed.
    """
    # Placeholder if hyperparameters need conversion
    pass

def clip_matrix(A: np.ndarray, min_val=1e-8):
    """
    Correct A to be positive definite if necessary.
    """
    return A + min_val * np.eye(A.shape[0])

def safe_logarithm(x: np.ndarray):
    """
    Compute log with safeguards against log(0).
    """
    return np.log(np.clip(x, a_min=1e-10, a_max=None))
