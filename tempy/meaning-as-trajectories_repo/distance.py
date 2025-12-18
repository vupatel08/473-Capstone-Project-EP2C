# distance.py
import numpy as np
from typing import List, Dict, Tuple, Optional
from math import log
from scipy.spatial.distance import cosine

class DistanceEvaluator:
    """
    Implements methods to compute semantic distances and relations between prompts
    by sampling trajectories, estimating likelihood functions, and applying
    distributional comparison algorithms as described in the paper.
    """
    def __init__(self, n_samples: int = 20, metric: str = "log-l1"):
        """
        Initialize DistanceEvaluator with sampling count and distance metric.
        Args:
            n_samples (int): Number of trajectories to sample per prompt.
            metric (str): Distance metric, e.g., "log-l1".
        """
        self.n_samples = n_samples
        self.metric = metric

    def compute_distance(self,
                         prompt_u: str,
                         prompt_v: str,
                         sampler,
                         likelihood_calculator) -> float:
        """
        Estimate the semantic distance between two prompts u and v using sampled trajectories.
        Args:
            prompt_u (str): First prompt string.
            prompt_v (str): Second prompt string.
            sampler (Sampler): Sampler object for trajectory sampling.
            likelihood_calculator (LikelihoodCalculator): For likelihood computations.
        Returns:
            float: Estimated distance value.
        """
        # Sample trajectories for both prompts
        T_u = sampler.sample_trajectories(prompt_u)
        T_v = sampler.sample_trajectories(prompt_v)

        # Compute log likelihoods for each trajectory
        log_mu_u: Dict[str, float] = {}
        log_mu_v: Dict[str, float] = {}
        for t in T_u:
            log_mu_u[t] = likelihood_calculator.compute_log_prob(t, prompt_u)
        for t in T_v:
            log_mu_v[t] = likelihood_calculator.compute_log_prob(t, prompt_v)

        # Build union set of trajectories for expectation estimation
        T_union = list(set(T_u).union(T_v))

        # For each trajectory in union, get corresponding logs, default to -inf if missing
        total_abs_log_diff = 0.0
        count = 0
        for t in T_union:
            log_u = log_mu_u.get(t, None)
            log_v = log_mu_v.get(t, None)
            # If trajectory not sampled for a prompt, skip or assign a large penalty
            if log_u is None or log_v is None:
                continue
            total_abs_log_diff += abs(log_u - log_v)
            count += 1

        if count == 0:
            return float('inf')  # or a large number, if no shared trajectories observed

        # Average absolute difference
        avg_abs_diff = total_abs_log_diff / count

        # Return based on the chosen metric
        if self.metric == "log-l1":
            return avg_abs_diff
        elif self.metric == "log-l2":
            return (avg_abs_diff ** 2) ** 0.5
        elif self.metric == "cosine":
            # For cosine, prepare vectors of logs
            vec_u = np.array([log_mu_u.get(t, -1e12) for t in T_union])
            vec_v = np.array([log_mu_v.get(t, -1e12) for t in T_union])
            # Handle zero vectors
            denom = (np.linalg.norm(vec_u) * np.linalg.norm(vec_v))
            if denom == 0:
                return 1.0  # maximal distance
            return cosine(vec_u, vec_v)
        else:
            # Default fallback
            return avg_abs_diff

    def compute_relation(self,
                         prompt_u: str,
                         prompt_v: str,
                         sampler,
                         likelihood_calculator,
                         relation_type: str = "entailment",
                         threshold: float = 0.5) -> bool:
        """
        Infer a relation (entailment, hyponym/hypernym) based on the distances of conjunctions.
        Args:
            prompt_u (str): First prompt string.
            prompt_v (str): Second prompt string.
            sampler (Sampler): Trajectory sampler.
            likelihood_calculator (LikelihoodCalculator): Likelihood inference.
            relation_type (str): Type of relation: "entailment" or "hyponym".
            threshold (float): Decision threshold for relation inference.
        Returns:
            bool: True if relation holds, False otherwise.
        """
        # Sample trajectories
        T_u = sampler.sample_trajectories(prompt_u)
        T_v = sampler.sample_trajectories(prompt_v)

        # Compute likelihood logs
        log_mu_u: Dict[str, float] = {}
        log_mu_v: Dict[str, float] = {}
        for t in T_u:
            log_mu_u[t] = likelihood_calculator.compute_log_prob(t, prompt_u)
        for t in T_v:
            log_mu_v[t] = likelihood_calculator.compute_log_prob(t, prompt_v)

        # For set operation, sample common trajectories or re-use, but here just pick union
        T_union = list(set(T_u).union(T_v))
        min_log_mu_u: Dict[str, float] = {}
        min_log_mu_v: Dict[str, float] = {}
        for t in T_union:
            log_u = log_mu_u.get(t, -1e12)
            log_v = log_mu_v.get(t, -1e12)
            min_log_mu_u[t] = log_u
            min_log_mu_v[t] = log_v

        # Compute distances for sets involving conjunctions
        dist_u = self._estimate_distance_for_set(log_mu_u, log_mu_v, T_union)
        dist_v = self._estimate_distance_for_set(log_mu_u, log_mu_v, T_union, negate_relation=True)

        # Apply relation inference rule based on distances
        # For entailment, if d(M_u∧M_v, M_v) < d(M_u∧M_v, M_u), then u entails v
        if relation_type == "entailment":
            return dist_u < dist_v - threshold
        elif relation_type == "hyponym":
            # Here, v as hyponym of u is similar logic
            return dist_u < dist_v - threshold
        else:
            # Default fallback, could extend
            return False

    def _estimate_distance_for_set(self,
                                    log_mu_u: Dict[str, float],
                                    log_mu_v: Dict[str, float],
                                    trajectories: List[str],
                                    negate_relation: bool=False) -> float:
        """
        Helper to estimate set-based distances (for set operations).
        If negate_relation=True, inverts the relation logic.
        """
        total = 0.0
        count = 0
        for t in trajectories:
            log_u = log_mu_u.get(t, -1e12)
            log_v = log_mu_v.get(t, -1e12)
            total += abs(log_u - log_v)
            count += 1
        if count == 0:
            return float('inf')
        avg = total / count
        return avg
