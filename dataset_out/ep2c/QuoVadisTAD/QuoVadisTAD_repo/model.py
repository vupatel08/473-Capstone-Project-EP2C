# model.py

import torch
import torch.nn as nn
import torch.optim as optim

class SimpleMLP(nn.Module):
    """
    A simple one-layer linear autoencoder model for time series reconstruction and anomaly scoring.
    This model consists of a single linear encoder and decoder, without activation functions,
    matching the description in the paper.
    """
    def __init__(self, input_dim: int, hidden_size: int = 32):
        """
        Initializes the SimpleMLP model.
        Args:
            input_dim (int): Dimensionality of input feature vectors.
            hidden_size (int): Size of the hidden layer. Default is 32.
        """
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_dim)

        # Initialize weights uniformly for reproducibility and stability
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        self.encoder.bias.data.fill_(0.0)
        self.decoder.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Reconstructed output, same shape as input.
        """
        # No activation functions; pure linear layers
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_model(self,
                    train_data: np.ndarray,
                    epochs: int = 50,
                    batch_size: int = 512,
                    learning_rate: float = 0.001,
                    early_stopping_patience: int = 10,
                    device: str = 'cpu',
                    verbose: bool = True) -> None:
        """
        Train the autoencoder on training data.
        Args:
            train_data (np.ndarray): Training feature data, shape (N_samples, input_dim).
            epochs (int): Max number of epochs. Default 50.
            batch_size (int): Batch size for optimizer. Default 512.
            learning_rate (float): Learning rate for Adam optimizer. Default 0.001.
            early_stopping_patience (int): Patience epochs for early stopping. Default 10.
            device (str): 'cpu' or 'cuda'. Default 'cpu'.
            verbose (bool): If True, print training progress.
        """
        self.to(device)
        self.train()

        # Convert training data to tensor
        train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)

        # Prepare DataLoader for batching
        dataset = torch.utils.data.TensorDataset(train_tensor, train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        best_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                output = self.forward(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(train_tensor)

            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.6f}")

            # Early stopping based on training loss (could be adapted to validation if available)
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                epochs_no_improve = 0
                # Save checkpoint in case needed
                self.best_state_dict = self.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    # Load best model state
                    self.load_state_dict(self.best_state_dict)
                    break

    def predict(self, test_data: np.ndarray, device: str = 'cpu') -> np.ndarray:
        """
        Generate predictions (reconstructions) for test data.
        Args:
            test_data (np.ndarray): Test feature data, shape (N_samples, input_dim).
            device (str): 'cpu' or 'cuda'. Default 'cpu'.
        Returns:
            np.ndarray: Reconstructed outputs, shape (N_samples, input_dim).
        """
        self.to(device)
        self.eval()
        with torch.no_grad():
            inputs = torch.tensor(test_data, dtype=torch.float32).to(device)
            outputs = self.forward(inputs)
            return outputs.cpu().numpy()

    def compute_error(self, test_data: np.ndarray, device: str='cpu') -> np.ndarray:
        """
        Compute per-sample error scores (e.g., maximum absolute difference).
        Args:
            test_data (np.ndarray): Original test data, shape (N_samples, input_dim).
            device (str): 'cpu' or 'cuda'.
        Returns:
            np.ndarray: Error scores per sample, shape (N_samples,).
        """
        preds = self.predict(test_data, device)
        errors = np.abs(preds - test_data)
        # For anomaly detection, typically use Frobenius norm per sample
        # Since data is 2D: (samples, features), take max or sum accordingly
        # Here, following the paper, we can use max absolute difference across features
        # Alternatively, MSE: (errors ** 2).mean(axis=1)
        # We'll use max absolute difference per sample
        error_scores = np.max(errors, axis=1)
        return error_scores
