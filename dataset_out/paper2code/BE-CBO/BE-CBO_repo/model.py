## model.py
import torch
import gpytorch
import numpy as np

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood, kernel_type: str = 'RBF', hyperparameters: dict = None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.kernel_type = kernel_type.lower()
        # Set default hyperparameters if not provided
        if hyperparameters is None:
            hyperparameters = {}
        # Initialize kernel based on type
        if self.kernel_type == 'matern':
            lengthscale = hyperparameters.get('lengthscale', 1.0)
            nu = hyperparameters.get('nu', 2.5)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=nu, ard_shape=torch.Size([train_x.shape[1]]))
            )
            self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)
        elif self.kernel_type == 'rbf' or self.kernel_type == 'l2' or self.kernel_type == 'gaussian':
            lengthscale = hyperparameters.get('lengthscale', 1.0)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_shape=torch.Size([train_x.shape[1]]))
            )
            self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)
        else:
            # Default to RBF if unknown
            lengthscale = hyperparameters.get('lengthscale', 1.0)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_shape=torch.Size([train_x.shape[1]]))
            )
            self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)
        # Optional: set outputscale if provided
        outputscale = hyperparameters.get('outputscale', 1.0)
        self.covar_module.outputscale = torch.tensor(outputscale)

    def forward(self, x: torch.Tensor):
        mean = torch.zeros(x.size(0))
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class SurrogateObjectiveGP:
    def __init__(self, kernel_type: str = 'RBF', hyperparameters: dict = None, noise_variance: float = 1e-4, device: torch.device = torch.device('cpu')):
        """
        Initialize the GP surrogate model for objective function.
        Args:
            kernel_type: string, e.g., 'RBF', 'Matern'
            hyperparameters: dict, e.g., {'lengthscale': 1.0, 'outputscale': 1.0}
            noise_variance: float, likelihood noise
            device: torch device
        """
        self.kernel_type = kernel_type
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        self.noise_variance = noise_variance
        self.device = device
        self.model = None
        self.likelihood = None
        self.is_trained = False

    def fit(self, X: np.ndarray, Y: np.ndarray, training_epochs: int = 50):
        """
        Fit the GP model to data.
        Args:
            X: numpy array, shape (n_samples, d)
            Y: numpy array, shape (n_samples,) or (n_samples,1)
            training_epochs: int, number of training iterations
        """
        # Convert data to torch tensors
        train_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(Y.squeeze(), dtype=torch.float32).to(self.device)  # shape (n,)
        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=self.noise_variance).to(self.device)
        self.model = ExactGPModel(train_x, train_y, self.likelihood, kernel_type=self.kernel_type, hyperparameters=self.hyperparameters).to(self.device)
        self.model.train()
        self.likelihood.train()

        # Use Adam optimizer to optimize model hyperparameters
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()}
        ], lr=0.01)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(training_epochs):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            # Optional: clip gradients to improve stability
            optimizer.step()

        self.is_trained = True

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance at test points.
        Args:
            X_test: numpy array, shape (n_test, d)
        Returns:
            mean: np.ndarray, shape (n_test,)
            variance: np.ndarray, shape (n_test,)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        self.model.eval()
        self.likelihood.eval()

        test_x = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            posterior = self.model(test_x)
            mean = posterior.mean.cpu().numpy()
            variance = posterior.variance.cpu().numpy()
        return mean, variance
