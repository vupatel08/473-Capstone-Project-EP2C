## model.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class MLModel:
    """
    MLModel is a wrapper class for an ML prediction model used within the PSPS framework.
    Currently supports RandomForestRegressor with configurable hyperparameters.
    Provides methods to initialize, train, and predict.
    """

    def __init__(self, model_params: dict):
        """
        Initialize the MLModel instance based on provided parameters.
        Args:
            model_params (dict): Dictionary of model hyperparameters.
                Expected keys:
                    - 'type' (str): Supported 'RandomForestRegressor'.
                    - 'n_estimators' (int): Number of trees.
                    - 'max_depth' (Optional[int]): Max depth of trees.
                    - 'random_state' (int): Random seed for reproducibility.
        """
        model_type = model_params.get('type', 'RandomForestRegressor')
        n_estimators = model_params.get('n_estimators', 500)
        max_depth = model_params.get('max_depth', None)
        random_state = model_params.get('random_state', 42)

        if model_type != 'RandomForestRegressor':
            raise ValueError(f"Unsupported model type: {model_type}. Supported: 'RandomForestRegressor'.")

        # Instantiate the scikit-learn model
        self.model_type = model_type
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }
        self.model_instance = RandomForestRegressor(**self.model_params)

        # Internal flag to track if model is trained
        self._is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ML model to the training data.
        Args:
            X (np.ndarray): Features with shape (n_samples, n_features).
            y (np.ndarray): Outcomes with shape (n_samples,).
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X should be a numpy.ndarray, but got {type(X)}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y should be a numpy.ndarray, but got {type(y)}")
        if X.ndim != 2:
            raise ValueError(f"X should be a 2D array, but got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y should be a 1D array, but got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X and y do not match: {X.shape[0]} vs {y.shape[0]}")

        # Fit the RandomForestRegressor
        self.model_instance.fit(X, y)
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for new data points.
        Args:
            X (np.ndarray): Features with shape (n_samples, n_features).
        Returns:
            np.ndarray: Predictions with shape (n_samples,).
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X should be a numpy.ndarray, but got {type(X)}")
        if X.ndim != 2:
            raise ValueError(f"X should be a 2D array, but got shape {X.shape}")

        # Generate predictions
        predictions = self.model_instance.predict(X)
        return predictions
