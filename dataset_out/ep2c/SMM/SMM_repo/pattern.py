## pattern.py
import torch
import torch.nn as nn
from typing import Tuple

class Pattern(nn.Module):
    """
    Represents the learnable prompt pattern (delta) in the reprogramming framework.
    This tensor is shared across all samples and is optimized during training.
    """

    def __init__(self,
                 shape: Tuple[int, int, int],
                 init_type: str = 'zeros'):
        """
        Initializes the Pattern tensor.

        Args:
            shape (Tuple[int, int, int]): (channels, height, width) of the pattern.
            init_type (str): Initialization method; default is 'zeros'.
                             Can be extended to other methods if needed.
        """
        super(Pattern, self).__init__()
        c, h, w = shape

        if init_type == 'zeros':
            pattern_tensor = torch.zeros(c, h, w)
        elif init_type == 'random':
            pattern_tensor = torch.randn(c, h, w)
        else:
            raise ValueError(f"Unsupported init_type: {init_type}")

        # Register as a trainable parameter
        self.pattern = nn.Parameter(pattern_tensor)

    def get_pattern(self) -> torch.Tensor:
        """
        Returns the current pattern tensor.

        Returns:
            torch.Tensor: The learnable pattern tensor of shape (C, H, W).
        """
        return self.pattern

    def reset(self, init_type: str = 'zeros') -> None:
        """
        Reinitializes the pattern tensor.

        Args:
            init_type (str): Reinitialization method; default is 'zeros'.
        """
        with torch.no_grad():
            if init_type == 'zeros':
                self.pattern.copy_(torch.zeros_like(self.pattern))
            elif init_type == 'random':
                self.pattern.copy_(torch.randn_like(self.pattern))
            else:
                raise ValueError(f"Unsupported init_type: {init_type}")
