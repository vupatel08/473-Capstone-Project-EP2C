## synthetic_data.py

import numpy as np
from scipy.stats import bernoulli

# Configuration defaults (can be overridden externally)
DEFAULT_N_ALTERNATIVES = 100  # number of alternatives
DEFAULT_NOISE_LEVEL = 0.0     # 0 for noiseless preference, >0 for stochastic
DEFAULT_Z_DISTRIBUTION = 'bernoulli'  # 'bernoulli' or 'uniform'
DEFAULT_P_Z = 0.5             # Bernoulli parameter p
DEFAULT_Z_RANGE = (0, 1)      # uniform range for z if used

# Define the data structure for comparison pairs
class ComparisonPair:
    def __init__(self, a, b, preference, z=None):
        self.a = a
        self.b = b
        self.preference = preference  # 1 if a preferred, 0 if b preferred
        self.z = z  # latent context sample, optional

def generate_alternatives(n):
    """
    Generate n alternatives evenly spaced in [0, 1].
    """
    return np.linspace(0, 1, n)

def sample_hidden_context(size, distribution='bernoulli', p=0.5, z_range=(0, 1)):
    """
    Sample hidden context variable z for each comparison.
    Supports Bernoulli or uniform distributions.
    """
    if distribution == 'bernoulli':
        # Bernoulli with parameter p
        return np.random.binomial(1, p, size)
    elif distribution == 'uniform':
        low, high = z_range
        return np.random.uniform(low, high, size)
    else:
        raise ValueError(f"Unsupported Z distribution: {distribution}")

def true_utility(a, z):
    """
    True utility function u(a, z):
    - For a < 0.8: utility = a
    - For a >= 0.8: utility = 2 * a * z
    """
    if np.isscalar(a):
        a = float(a)
        if a < 0.8:
            return a
        else:
            return 2 * a * z
    else:
        # a is array
        util = np.zeros_like(a)
        mask = a >= 0.8
        util[~mask] = a[~mask]
        util[mask] = 2 * a[mask] * z
        return util

def preference_outcome(a, b, z, noise=False):
    """
    Simulate preference between a and b given hidden context z.
    Uses probabilistic Bradley-Terry model.
    """
    util_a = true_utility(a, z)
    util_b = true_utility(b, z)
    prob_a_pref = np.exp(util_a) / (np.exp(util_a) + np.exp(util_b))
    if noise:
        # Add stochasticity: prefer a with probability p
        return np.random.rand() < prob_a_pref
    else:
        # Deterministic: highest utility preferred
        return util_a > util_b

def generate_comparison_pair(alternatives):
    """
    Generate a single pair of alternatives with associated preference outcome.
    """
    a, b = np.random.choice(alternatives, size=2, replace=False)
    # Sample hidden context z for the comparison
    z_samples = sample_hidden_context(1)
    z = z_samples[0]
    # Generate preference outcome (no noise for ground truth)
    preference = int(preference_outcome(a, b, z, noise=False))
    return ComparisonPair(a=a, b=b, preference=preference, z=z)

def generate_dataset(alternatives, num_pairs, dist='bernoulli', p=0.5, z_range=(0,1)):
    """
    Generate a dataset of comparison pairs with hidden context.
    """
    dataset = []
    for _ in range(num_pairs):
        pair = generate_comparison_pair(alternatives)
        dataset.append(pair)
    return dataset

# Additional helper: generate synthetic dataset with known true utilities and optional noise
def generate_synthetic_data(config):
    """
    Generate synthetic preference dataset based on config parameters.
    """
    n = config.get('synthetic_size', DEFAULT_N_ALTERNATIVES)
    alternatives = generate_alternatives(n)
    num_pairs = n * (n - 1) // 2  # all pairs or can be adjusted
    dist = config.get('z_distribution', DEFAULT_Z_DISTRIBUTION)
    p_z = config.get('p_z', DEFAULT_P_Z)
    z_range = config.get('z_range', DEFAULT_Z_RANGE)
    dataset = generate_dataset(alternatives, num_pairs, dist, p_z, z_range)
    return alternatives, dataset

# Example usage (can be removed or commented out in production)
if __name__ == "__main__":
    # Generate synthetic data with default parameters
    alternatives, dataset = generate_synthetic_data({'synthetic_size': 1000})
    # Print some sample data
    for idx, pair in enumerate(dataset[:5]):
        print(f"Pair {idx}: a={pair.a:.3f}, b={pair.b:.3f}, preference={pair.preference}, z={pair.z}")
