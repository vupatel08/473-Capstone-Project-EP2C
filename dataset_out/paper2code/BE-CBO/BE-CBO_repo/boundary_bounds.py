## boundary_bounds.py
import numpy as np

class BoundaryBounds:
    def __init__(self, margin: float = 0.5):
        """
        Initialize the BoundaryBounds object.
        Args:
            margin (float): The base threshold for classifier boundary, default 0.5.
        """
        self.margin = margin

    def compute_lower_bound(self, sigma_e: float) -> float:
        """
        Compute the dynamic lower bound l(x) based on classifier uncertainty.
        Args:
            sigma_e (float): Standard deviation (uncertainty) of the ensemble prediction at x.
        Returns:
            float: The lower bound l(x) = max(0, margin - sigma_e).
        """
        l_x = self.margin - sigma_e
        # Ensure the lower bound is within [0,1]
        return max(0.0, l_x)

    def compute_bounds(self, C_x: float, sigma_e: float) -> Tuple[float, float]:
        """
        Compute the sampling bounds for a candidate point x, based on its classifier's
        predicted probability C_x and the ensemble uncertainty sigma_e.
        Args:
            C_x (float): The predicted probability (mean ensemble output) for feasibility, in [0,1].
            sigma_e (float): The classifier ensemble's standard deviation at x.
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound), bounds in [0,1],
                                 used for constrained acquisition optimization.
        """
        # Dynamic lower bound based on uncertainty
        l_x = self.compute_lower_bound(sigma_e)
        # Upper feasibility bound is fixed at 1
        u_x = 1.0
        # The lower bound for sampling is at least the maximum between l_x and C_x - sigma_e (for more exploration)
        # But in this implementation, we use the dynamic lower bound:
        lower_bound = l_x
        # For simplicity, cap the lower bound to be at most C_x (so we don't sample into the more infeasible region)
        lower_bound = min(lower_bound, C_x)
        # Ensure bounds are within [0,1]
        lower_bound = max(0.0, lower_bound)
        u_x = min(1.0, u_x)
        return (lower_bound, u_x)
