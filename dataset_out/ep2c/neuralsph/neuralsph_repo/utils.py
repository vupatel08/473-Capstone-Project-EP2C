## utils.py
import numpy as np
from scipy.special import erf
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def compute_velocity_std(velocities: np.ndarray, window_size: int = 5) -> float:
    """
    Calculate the isotropic standard deviation (sigma_u) of velocity components
    over the most recent 'window_size' timesteps.

    Args:
        velocities (np.ndarray): Array of shape (N_particles, dim) representing velocities at latest timestep(s).
        window_size (int): Number of recent timesteps to consider for std calculation.

    Returns:
        float: Isotropic standard deviation across particle velocities.
    """
    # velocities shape: (sequence_length, N_particles, dim)
    if velocities.ndim != 3:
        raise ValueError(f"Expected velocities shape (seq_len, N_particles, dim), got {velocities.shape}")
    seq_len = velocities.shape[0]
    start_idx = max(0, seq_len - window_size)
    recent_vels = velocities[start_idx:, ...]  # shape: (window_size, N_particles, dim)
    # Compute std per component over the window and particles
    std_per_component = np.std(recent_vels, axis=0)  # shape: (N_particles, dim)
    # Average over particles
    std_per_component_mean = np.mean(std_per_component, axis=0)  # shape: (dim,)
    # Compute quadratic mean for isotropic sigma
    sigma_u = np.sqrt(np.mean(std_per_component_mean ** 2))
    return float(sigma_u)

def gaussian_convolve_force(force_field: np.ndarray, sigma: float) -> np.ndarray:
    """
    Approximate the convolution of a force field with a Gaussian kernel
    analytically using the error function (erf).

    Args:
        force_field (np.ndarray): Shape (N_particles, dim).
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        np.ndarray: Smoothed force field, same shape as input.
    """
    # For step-like force fields, analytical convolution along each component:
    # convolution with erf: f_smooth(y) = force_component * erf((y - y0)/ (sqrt(2)*sigma))
    # For general force fields, this simplifies to a weighted sum, but here we implement the analytical for step/constant forces.
    # Since force_field is per particle, and force varies per particle, we assume this applies component-wise.
    # For illustration, we smooth each component independently assuming a step function.
    # Alternatively, implement numerical convolution if force_field varies smoothly.
    force_smoothed = np.zeros_like(force_field)
    for d in range(force_field.shape[1]):
        force_smoothed[:, d] = erf(force_field[:, d] / (np.sqrt(2) * sigma))
    # Optional: scale back to force magnitude if force components are normalized
    return force_smoothed * np.max(np.abs(force_field), axis=0)

def effective_force_approximation(
    force_field: np.ndarray,
    velocities: np.ndarray,
    convolution_method: str = 'gaussian',
    sigma_scale: float = 0.025
) -> np.ndarray:
    """
    Compute the effective external force map based on velocity statistics
    and a smoothing kernel, either analytically or numerically.

    Args:
        force_field (np.ndarray): Shape (N_particles, dim), instantaneous external force.
        velocities (np.ndarray): Shape (N_particles, dim) velocity data for std calculation.
        convolution_method (str): 'gaussian' or 'erf'. Determines convolution approach.
        sigma_scale (float): Scaling factor for sigma derivation from velocity std.

    Returns:
        np.ndarray: Smoothed external force field, shape: (N_particles, dim).
    """
    # Compute velocity std deviation per particle
    # If multiple timesteps are available, compute over recent historical data
    sigma_u = np.std(velocities, axis=0)  # shape: (dim,)
    sigma = np.sqrt(np.mean(sigma_u ** 2)) * sigma_scale

    # For spatial smoothing, convolve force with Gaussian, analytical approximation:
    if convolution_method == 'erf':
        # Use erf of (force / (sqrt(2)*sigma))
        smoothed_force = np.zeros_like(force_field)
        for d in range(force_field.shape[1]):
            smoothed_force[:, d] = erf(force_field[:, d] / (np.sqrt(2) * sigma))
        return smoothed_force
    elif convolution_method == 'gaussian':
        # Numerical approximation via kernel sum
        # For simplicity, assume force_field is already spatially sampled
        return gaussian_convolve_force(force_field, sigma)
    else:
        raise ValueError(f"Unknown convolution method: {convolution_method}")

def neighbor_search(positions: np.ndarray, cutoff_radius: float) -> list:
    """
    Identify neighboring particles within cutoff radius using KDTree.

    Args:
        positions (np.ndarray): Shape (N_particles, dim).
        cutoff_radius (float): Radius for neighbor search.

    Returns:
        list: List of lists, where each sublist contains neighbor indices for the corresponding particle.
    """
    tree = KDTree(positions)
    neighbors_list = tree.query_ball_point(positions, r=cutoff_radius)
    return neighbors_list

def sph_kernel_quintic(r: np.ndarray, h: float) -> np.ndarray:
    """
    Evaluate the quintic spline kernel W(r|h) for each inter-particle distance r.

    Args:
        r (np.ndarray): Distances, shape: (num_neighbors,)
        h (float): Smoothing length.

    Returns:
        np.ndarray: Kernel evaluations at each r, shape: (num_neighbors,)
    """
    q = r / h
    W = np.zeros_like(q)

    # Coefficients for the quintic spline kernel (from Monaghan 1993)
    sigma = 7 / (478 * np.pi * h ** 3)  # normalization constant in 3D
    for i, qi in enumerate(q):
        if 0 <= qi < 1:
            W[i] = sigma * ((3 - qi) ** 5 - 6 * (2 - qi) ** 5 + 15 * (1 - qi) ** 5)
        elif 1 <= qi < 2:
            W[i] = sigma * ((3 - qi) ** 5 - 6 * (2 - qi) ** 5)
        elif 2 <= qi < 3:
            W[i] = sigma * (3 - qi) ** 5
        else:
            W[i] = 0.0
    return W

def compute_density(
    positions: np.ndarray,
    neighbor_list: list,
    mass: float,
    h: float,
    rho_min: float = 0.98,
    rho_max: float = 1.02,
    rho_ref: float = 1.0
) -> np.ndarray:
    """
    Calculate the density at each particle via kernel summation (Eq. 1).
    Apply density clipping for free surface correction.

    Args:
        positions (np.ndarray): (N_particles, dim)
        neighbor_list (list): List of neighbor indices per particle
        mass (float): Uniform mass for each particle
        h (float): Kernel support radius
        rho_min, rho_max (float): Clipping thresholds relative to rho_ref
        rho_ref (float): Reference density

    Returns:
        density (np.ndarray): Shape (N_particles,)
    """
    N = positions.shape[0]
    density = np.zeros(N)
    for i in range(N):
        neighbors = neighbor_list[i]
        r_ijs = np.linalg.norm(positions[neighbors] - positions[i], axis=1)  # shape: (num_neighbors,)
        W_vals = sph_kernel_quintic(r_ijs, h)
        density[i] = np.sum(W_vals) * mass
    # Density clipping: enforce minimum and maximum
    lower_bound = rho_ref * rho_min
    upper_bound = rho_ref * rho_max
    density = np.clip(density, lower_bound, upper_bound)
    return density

def compute_pressure(density: np.ndarray, p_ref: float, rho_ref: float = 1.0) -> np.ndarray:
    """
    Compute pressure using the equation of state p = p_ref * (rho / rho_ref - 1).

    Args:
        density (np.ndarray): Particle density (N_particles,)
        p_ref (float): Reference pressure coefficient
        rho_ref (float): Reference density

    Returns:
        pressure (np.ndarray): Shape (N_particles,)
    """
    pressure = p_ref * (density / rho_ref - 1.0)
    return pressure

def pressure_clamp(pressure: np.ndarray, rho_ref: float = 1.0, clip_min: float=0.98, clip_max: float=1.02) -> np.ndarray:
    """
    Clamp pressure values to prevent tensile instability, based on thresholds.

    Args:
        pressure (np.ndarray): Unclamped pressure values
        rho_ref (float): Reference density
        clip_min, clip_max (float): Clipping thresholds relative to rho_ref

    Returns:
        np.ndarray: Clamped pressure values
    """
    min_val = rho_ref * clip_min
    max_val = rho_ref * clip_max
    return np.clip(pressure, min_val, max_val)

def density_at_surface_correction(raw_density: np.ndarray, rho_ref: float=1.0, tol_lower=0.98, tol_upper=1.02) -> np.ndarray:
    """
    Correct densities at free surfaces by clipping and enforcing threshold bounds.

    Args:
        raw_density (np.ndarray): Raw density from summation
        rho_ref (float): Reference density
        tol_lower, tol_upper (float): Tolerance bounds for density clipping

    Returns:
        corrected_density (np.ndarray): Density after correction
    """
    corrected_density = np.copy(raw_density)
    # Set densities below threshold to rho_ref
    corrected_density[corrected_density < rho_ref * tol_lower] = rho_ref
    # Clip densities to upper threshold
    corrected_density[corrected_density > rho_ref * tol_upper] = rho_ref * tol_upper
    return corrected_density

def boundary_condition_wall(pressure: np.ndarray, neighbors: list, wall_mask: np.ndarray) -> np.ndarray:
    """
    Enforce wall boundary conditions, setting wall particle pressures to average of neighbors,
    avoiding penetration and modeling impermeability.

    Args:
        pressure (np.ndarray): Current pressures per particle
        neighbors (list): List of neighbor indices per particle
        wall_mask (np.ndarray): Boolean array: True for wall particles

    Returns:
        np.ndarray: Enforced pressure array
    """
    pressure_enforced = np.copy(pressure)
    for i, is_wall in enumerate(wall_mask):
        if is_wall:
            neighbor_indices = neighbors[i]
            # Only consider neighboring fluid particles for pressure averaging
            neighbor_pressures = pressure[neighbor_indices]
            # Set wall particle pressure to average neighbor pressure
            pressure_enforced[i] = np.mean(neighbor_pressures)
    return pressure_enforced

def compute_dirichlet_energy(
    density: np.ndarray,
    positions: np.ndarray,
    h: float
) -> float:
    """
    Calculate Dirichlet energy of the density field to quantify clustering and instability.

    Args:
        density (np.ndarray): Particle densities
        positions (np.ndarray): Particle positions, shape (N, dim)
        h (float): Kernel smoothing length used for gradient approximation

    Returns:
        float: Total Dirichlet energy
    """
    N, dim = positions.shape
    # Approximate gradient of density via kernel derivatives
    energy = 0.0
    for i in range(N):
        neighbors = neighbor_search(positions, h)[i]
        r_ij = positions[neighbors] - positions[i]
        r_norm = np.linalg.norm(r_ij, axis=1) + 1e-8
        # Derivative of kernel w.r.t. r
        grad_W = sph_kernel_gradient(r_ij, r_norm, h)
        # Sum over neighbors
        grad_density = np.sum(grad_W * (density[neighbors][:, np.newaxis] / density[i]), axis=0)
        energy += np.linalg.norm(grad_density) ** 2
    return energy / N

def sph_kernel_gradient(r_vectors: np.ndarray, r_norm: np.ndarray, h: float) -> np.ndarray:
    """
    Compute the gradient of the quintic spline kernel with respect to position.

    Args:
        r_vectors (np.ndarray): Vector differences, shape (num_neighbors, dim)
        r_norm (np.ndarray): Norms of r_vectors, shape: (num_neighbors,)
        h (float): Smoothing length

    Returns:
        np.ndarray: Gradient vectors, shape (num_neighbors, dim)
    """
    q = r_norm / h
    # Compute derivative of W with respect to q
    # Derivative of quintic spline
    dW_dq = np.zeros_like(q)
    # Implement piecewise derivatives similar to sph_kernel_quintic
    for i, qi in enumerate(q):
        if 0 <= qi < 1:
            dW_dq[i] = (1/h) * ( ( -5 * (3 - qi) ** 4 + 30 * (2 - qi) ** 4 - 75 * (1 - qi) ** 4))
        elif 1 <= qi < 2:
            dW_dq[i] = ( -5 * (3 - qi) ** 4 + 30 * (2 - qi) ** 4)
        elif 2 <= qi < 3:
            dW_dq[i] = (5 * (3 - qi) ** 4)
        else:
            dW_dq[i] = 0.0
    # Gradient: dW/dx = (dW/dq) * (r_vector / r_norm) / h
    grad_W = (dW_dq / (r_norm + 1e-8))[:, np.newaxis] * r_vectors
    return grad_W

def visualize_particle_field(
    positions: np.ndarray,
    density: Optional[np.ndarray] = None,
    title: str = "Particle Field"
):
    """
    Generate and display scatter plot of particles, optionally colored by density.

    Args:
        positions (np.ndarray): Particle positions, shape (N, dim)
        density (np.ndarray, optional): Particle densities for coloring.
        title (str): Plot title.
    """
    plt.figure(figsize=(6,6))
    if density is not None:
        plt.scatter(positions[:, 0], positions[:, 1], c=density, cmap='viridis', s=20)
        plt.colorbar(label='Density')
    else:
        plt.scatter(positions[:, 0], positions[:, 1], c='blue', s=20)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.show()
