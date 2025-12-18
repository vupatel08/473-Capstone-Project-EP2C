## calibration.py
""" 
This module performs binary search-based calibration of mechanism noise parameters
to satisfy specified operational privacy risk constraints (advantage, FPR, FNR),
using precomputed or on-the-fly computed trade-off curves from PLRVs.
"""

import numpy as np
from typing import Callable, Tuple
from utils import interpolate_curve  # Assuming interpolation utilities are in utils.py

class Calibrator:
    """
    Implements calibration routines to find the minimal noise parameter (e.g., sigma)
    such that the mechanism's operational privacy risk constraint is satisfied.
    """

    def __init__(
        self,
        plrv_supports: dict,
        plrv_probs: dict,
        support_support: np.ndarray,
        mode: str = "advantage",  # or "fpr_fnr"
        target_value: float = 0.0,
        alpha: float = 0.01,  # For fpr/fnr calibration, the fixed FPR or FNR level
        epsilon_bounds: Tuple[float, float] = (1e-2, 10.0),
        tolerance: float = 1e-3,
        max_iterations: int = 50,
        mechanism_type: str = "Gaussian",
        sigma: float = 1.0,
        # For mechanisms with privacy accountant, optional function to get f_omega(alpha)
        get_f_omega: Optional[Callable[[float, float], float]] = None,  
        # Function to construct PLRV for a given omega: returns X,Y's support & pmf arrays
        construct_plrv: Optional[Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None,
        # Function to evaluate the operational risk metric, given PLRVs support & pmfs
        evaluate_metric: Optional[Callable[[dict], float]] = None
    ):
        """
        Args:
            plrv_supports: dict with 'X_support', 'Y_support' support points arrays.
            plrv_probs: dict with 'X_probs', 'Y_probs' probability arrays.
            support_support: array, support including infinities.
            mode: 'advantage' or 'fpr_fnr' for risk constraint type.
            target_value: value for advantage, or for fpr/fnr constraints.
            alpha: fixed FPR or FNR level for fpr/fnr based calibration.
            epsilon_bounds: tuple, bounds for omega (sigma or mechanism-specific parameter).
            tolerance: binary search precision.
            max_iterations: max number of binary search steps.
            mechanism_type: 'Gaussian' or others.
            sigma: initial noise level for Gaussian.
            get_f_omega: optional function to get f_omega(alpha) for an omega, if available.
            construct_plrv: function to construct PLRV (X,Y) from omega.
            evaluate_metric: function to compute advantage, or f_omega(alpha), from PLRVs support & pmfs.
        """
        self.plrv_supports = plrv_supports
        self.plrv_probs = plrv_probs
        self.support_support = support_support
        self.mode = mode
        self.target_value = target_value
        self.alpha = alpha
        self.epsilon_bounds = epsilon_bounds
        self.tolerance = tolerance
        self.max_iter = max_iterations
        self.mechanism_type = mechanism_type
        self.sigma = sigma
        self.get_f_omega = get_f_omega
        self.construct_plrv = construct_plrv
        self.evaluate_metric = evaluate_metric

    def _compute_tradeoff_curve(self, omega: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs the trade-off curve f_omega(alpha) for a given omega (noise parameter).
        Uses Algorithm 1 with PLRVs derived from the mechanism at omega.
        Returns:
            alphas: discretized list of alpha points.
            f_alpha: corresponding values of f_omega(alpha).
        """
        # If a specific PLRV constructor is given (e.g., for Gaussian), use it
        if self.construct_plrv is not None:
            X_support, X_probs, Y_support, Y_probs = self.construct_plrv(omega)
        else:
            # Otherwise, assume the support & pmf are provided directly via PLRV
            # with support_support, plrv_supports, and plrv_probs
            X_support = self.plrv_supports['X_support']
            X_probs = self.plrv_probs['X_probs']
            Y_support = self.plrv_supports['Y_support']
            Y_probs = self.plrv_probs['Y_probs']

        # Evaluate the tradeoff curve T(P,Q) via Algorithm 1
        t_curve = TradeOffCurve(
            plrv_supports={'X_support': X_support, 'Y_support': Y_support},
            plrv_probs={'X_probs': X_probs, 'Y_probs': Y_probs},
            support_support=self.support_support
        )
        alphas, f_alpha = t_curve.evaluate_curve()

        return alphas, f_alpha

    def _compute_risk_metric(self, omega: float) -> float:
        """
        Computes the operational privacy risk measure (advantage or f_omega(alpha))
        for a given omega, using the associated PLRVs and/or curve evaluation.
        """
        if self.construct_plrv is None or self.evaluate_metric is None:
            raise RuntimeError("Must specify construct_plrv and evaluate_metric functions.")

        # Build PLRV supports & pmfs for current omega
        X_support, X_probs, Y_support, Y_probs = self.construct_plrv(omega)

        # Compute the necessary metric (advantage or f_alpha at fixed alpha)
        plrv_dict = {'X_support': X_support, 'Y_support': Y_support, 'X_probs': X_probs, 'Y_probs': Y_probs}
        metric_value = self.evaluate_metric(plrv_dict)

        return metric_value

    def _binary_search(self, target_constraint: float, 
                        is_advantage: bool = True) -> float:
        """
        Performs binary search over omega bounds to find minimal omega satisfying the constraint.
        Args:
            target_constraint: desired advantage (eta) or f_omega(alpha) threshold.
            is_advantage: True if calibrating advantage, False if calibrating f_omega(alpha).
        Returns:
            omega_star: the minimal omega satisfying the constraint within tolerance.
        """
        low, high = self.epsilon_bounds
        for _ in range(self.max_iter):
            omega_mid = (low + high) / 2.0
            metric = self._compute_risk_metric(omega_mid)
            if is_advantage:
                # For advantage: metric should be <= target_eta
                if metric <= target_constraint:
                    high = omega_mid
                else:
                    low = omega_mid
            else:
                # For f_omega(alpha): metric should be >= target_beta
                if metric >= target_constraint:
                    high = omega_mid
                else:
                    low = omega_mid
            if abs(high - low) < self.tolerance:
                break
        return (low + high) / 2.0

    def calibrate_advantage(self) -> float:
        """
        Calibrates the mechanism to achieve the target advantage by binary searching omega.
        """
        omega_star = self._binary_search(self.target_value, is_advantage=True)
        return omega_star

    def calibrate_fpr_fnr(self) -> float:
        """
        Calibrates the mechanism to achieve target fpr or fnr using f_omega(alpha).
        """
        omega_star = self._binary_search(self.target_value, is_advantage=False)
        return omega_star

    def run_calibration(self) -> float:
        """
        Main method to run calibration depending on mode.
        Returns:
            calibrated omega (sigma or mechanism parameter)
        """
        if self.mode == "advantage":
            return self.calibrate_advantage()
        elif self.mode == "fpr_fnr":
            return self.calibrate_fpr_fnr()
        else:
            raise ValueError(f"Unknown mode {self.mode}")

# Note:
# - The functions construct_plrv, evaluate_metric should be provided by the user.
# - For Gaussian mechanisms, construct_plrv() can be implemented with analytical formulas.
# - evaluate_metric() for advantage computes P[Y>0] - P[X>0] from PLR support.
# - For f_omega(alpha), evaluate_metric() interpolates the computed curve at given alpha.

