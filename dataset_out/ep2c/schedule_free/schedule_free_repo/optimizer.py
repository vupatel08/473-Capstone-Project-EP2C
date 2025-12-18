## optimizer.py
import torch
from torch.optim import Optimizer
from typing import List, Dict, Optional

class ScheduleFreeOptimizer(Optimizer):
    """
    Implements the Schedule-Free optimization strategy as described in the paper.
    Maintains sequences z_t (the optimizer trajectory) and x_t (interpolated average),
    updating z_t via a base optimizer (e.g., AdamW or SGD) at each step,
    and updating x_t with decreasing weights (c_t ~ 1/t).
    
    Attributes:
        params: Iterable of model parameters (must be references to model parameters).
        optimizer: The inner optimizer (e.g., AdamW) used for z_t updates.
        beta: Coupling parameter between x_t and z_t, typically around 0.9.
        D: Estimated initial distance bound (used to set large learning rate).
        G: Gradient norm bound (used to set large learning rate).
        eta: Fixed learning rate (approximate D / G), set during init.
        iteration: Keeps track of current iteration step.
        z_params: List of tensors representing z_t parameters.
        x_params: List of tensors representing x_t parameters.
        buffers for optimizer state if needed (AdamW's m_t and v_t).
    """

    def __init__(
        self,
        model_params: List[torch.nn.Parameter],
        optimizer_type: str = "AdamW",
        lr_scale: float = 1.0,  # user-defined scale for eta, to be set as D/G
        beta: float = 0.9,
        D: float = 1.0,
        G: float = 1.0,
        eta: Optional[float] = None,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        betas: tuple = (0.9, 0.999),
        max_grad_norm: Optional[float] = None,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize the ScheduleFreeOptimizer.
        Args:
            model_params (List[torch.nn.Parameter]): model parameters to optimize.
            optimizer_type (str): type of base optimizer ("AdamW" or "SGD").
            lr_scale (float): optional scale; default 1.0, usually set as D / G.
            beta (float): coupling parameter, e.g., 0.9.
            D (float): distance bound, estimate of initial parameter distance.
            G (float): gradient norm bound.
            eta (float): fixed large learning rate; if None, set to D / G.
            weight_decay (float): weight decay coefficient.
            eps (float): epsilon for AdamW.
            betas (tuple): betas for AdamW.
            max_grad_norm (float): optional, for gradient clipping.
            device (torch.device): device to run computations.
        """
        # Store hyperparameters
        self.params = model_params
        self.beta = beta
        self.D = D
        self.G = G
        # Set fixed learning rate
        self.eta = eta if eta is not None else D / G if G > 1e-8 else D
        self.iteration = 1  # start from 1 for 1-based indexing
        self.device = device

        # Initialize z_t as clone of model parameters
        self.z_params = [p.clone().detach().to(device) for p in model_params]
        # Initialize x_t as clone of model parameters (start same as initial)
        self.x_params = [p.clone().detach().to(device) for p in model_params]
        # Initialize optimizer for z_t
        if optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self._parameters_to_optimizer_params(),
                lr=self.eta,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
        elif optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(
                self._parameters_to_optimizer_params(),
                lr=self.eta,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        # Initialize optimizer's internal state
        self.optimizer.zero_grad()

    def _parameters_to_optimizer_params(self) -> List[torch.nn.Parameter]:
        """
        Convert z_params list to a list suitable for optimizer.
        """
        return self.z_params

    def step(self, data=None, target=None, gradient_eval_fn=None):
        """
        Perform a single inner update step:
        - Compute gradient at y_t
        - Update z_t according to optimizer
        - Update x_t as weighted average with c_t
        - Update y_t
        Args:
            data, target: optional, for gradient computation
            gradient_eval_fn: optional function to compute gradients, if external
        """
        # 1. Compute y_t
        y_t = []
        for x_p, z_p in zip(self.x_params, self.z_params):
            y_t.append((1.0 - self.beta) * x_p + self.beta * z_p)
        # 2. Evaluate gradients at y_t
        # User provides gradient_eval_fn, or we assume external gradient computation
        if gradient_eval_fn is None:
            raise RuntimeError("gradient_eval_fn must be provided to perform gradient evaluation.")
        # The eval fn should set gradients on model parameters, or return grads
        grads = gradient_eval_fn(y_t, data, target)

        # 3. Update z_t using optimizer step with grads at y_t
        # Assign computed grads to z_params
        for p, g in zip(self.z_params, grads):
            p.grad = g
        # Step optimizer to update z_t
        self.optimizer.step()
        # Save z_t parameters after update
        # Copy current z_t as clone
        for idx, p in enumerate(self.z_params):
            self.z_params[idx] = p.clone().detach()

        # 4. Update x_t following the decreasing weight schedule c_t
        c_t = 1.0 / self.iteration  # c_t ~ 1/t
        for idx, p in enumerate(self.x_params):
            # x_{t+1} = (1 - c_t) x_t + c_t z_{t+1}
            self.x_params[idx] = (1 - c_t) * p + c_t * self.z_params[idx]

        # Increment iteration count
        self.iteration += 1

        # The optimizer's internal state has been updated; no need to return
        return

    def get_current_x_params(self) -> List[torch.nn.Parameter]:
        """
        Return the current interpolated parameters x_t.
        """
        return self.x_params

    def load_state_dict(self, state_dict: Dict):
        """
        Load state dict for optimizer and internal z,x buffers.
        """
        self.optimizer.load_state_dict(state_dict['optimizer'])
        # Load z_params
        z_state = state_dict.get('z_params', None)
        if z_state is not None:
            self.z_params = [p.clone().detach() for p in z_state]
        # Load x_params
        x_state = state_dict.get('x_params', None)
        if x_state is not None:
            self.x_params = [p.clone().detach() for p in x_state]
        self.iteration = state_dict.get('iteration', self.iteration)

    def state_dict(self) -> Dict:
        """
        Return the state dict including optimizer state, z_params, x_params, iteration.
        """
        return {
            'optimizer': self.optimizer.state_dict(),
            'z_params': self.z_params,
            'x_params': self.x_params,
            'iteration': self.iteration
        }

    def zero_grad(self):
        """
        Zero out gradients of z_params (not X parameters).
        """
        self.optimizer.zero_grad()

    def clip_gradients(self, max_norm: float):
        """
        Optional gradient clipping for z_t parameters.
        """
        torch.nn.utils.clip_grad_norm_(self.z_params, max_norm)

    def to(self, device: torch.device):
        """
        Move optimizer's internal tensors to device as needed.
        """
        self.device = device
        for idx, p in enumerate(self.z_params):
            self.z_params[idx] = p.to(device)
        for idx, p in enumerate(self.x_params):
            self.x_params[idx] = p.to(device)
        self.optimizer.to(device)

