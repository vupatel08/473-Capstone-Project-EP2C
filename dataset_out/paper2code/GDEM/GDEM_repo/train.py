# train.py
import torch
import torch.nn.functional as F
import os
import numpy as np
from typing import Optional
from utils import set_random_seed
from torch_geometric.data import Data
from model import GNNModel

class Trainer:
    def __init__(self,
                 model: GNNModel,
                 train_data: Data,
                 val_data: Optional[Data],
                 test_data: Optional[Data],
                 labels_train: torch.Tensor,
                 labels_val: Optional[torch.Tensor],
                 labels_test: Optional[torch.Tensor],
                 config: dict):
        """
        Initialize the trainer with model, datasets, and hyperparameters.

        Args:
            model (GNNModel): The GNN model to train.
            train_data (Data): Training data object.
            val_data (Data, optional): Validation data.
            test_data (Data, optional): Test data.
            labels_train (torch.Tensor): Ground truth labels for training nodes.
            labels_val (torch.Tensor, optional): Validation labels.
            labels_test (torch.Tensor, optional): Test labels.
            config (dict): Hyperparameters and settings.
        """
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.labels_train = labels_train
        self.labels_val = labels_val
        self.labels_test = labels_test

        # Hyperparameters from config with defaults
        self.epochs = config.get("training", {}).get("epochs", 2000)
        self.lr = config.get("training", {}).get("learning_rate", 0.001)
        self.weight_decay = config.get("training", {}).get("weight_decay", 5e-4)
        self.batch_size = config.get("training", {}).get("batch_size", 128)
        self.validation_interval = config.get("training", {}).get("validation_interval", 10)
        self.early_stopping_rounds = config.get("training", {}).get("early_stopping_rounds", None)

        # Set seed for reproducibility
        seed = config.get("reproducibility", {}).get("random_seed", 42)
        set_random_seed(seed)

        # Initialize optimizer
        self.optimizer = self.model.get_optimizer(self.lr, self.weight_decay)
        # Loss criterion
        self.criterion = torch.nn.CrossEntropyLoss()
        # Variables to track best validation
        self.best_val_acc = 0.0
        self.best_model_path = "best_model.pth"
        self.no_improve_counter = 0

    def train(self):
        """
        Execute the training loop with validation and checkpoint saving.
        """
        self.model.to(self.model.device)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            # Prepare input data
            data_input = self.train_data.to(self.model.device)
            labels_input = self.labels_train.to(self.model.device)

            # Forward pass
            out = self.model(data_input)
            loss = self.criterion(out, labels_input)

            # Backward and optimize
            loss.backward()
            self.optimizer.step()

            # Logging training metrics
            if epoch % 50 == 0 or epoch == 1:
                with torch.no_grad():
                    pred = out.argmax(dim=1)
                    correct = (pred == labels_input).sum().item()
                    acc = correct / labels_input.size(0) * 100
                print(f"Epoch [{epoch}/{self.epochs}] - Loss: {loss.item():.4f} | Train Acc: {acc:.2f}%")

            # Validation
            if self.val_data is not None and epoch % self.validation_interval == 0:
                val_metrics = self.validate()
                val_acc = val_metrics.get('accuracy', 0.0)
                print(f"Validation at epoch {epoch} - Accuracy: {val_acc:.2f}%")
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_model(self.best_model_path)
                    self.no_improve_counter = 0
                else:
                    self.no_improve_counter += 1
                # Early stopping condition
                if self.early_stopping_rounds is not None:
                    if self.no_improve_counter >= self.early_stopping_rounds:
                        print("Early stopping triggered.")
                        break

    def validate(self):
        """
        Evaluate the model on validation data.
        Returns:
            dict: Dictionary with validation accuracy.
        """
        self.model.eval()
        with torch.no_grad():
            data_input = self.val_data.to(self.model.device)
            labels_input = self.labels_val.to(self.model.device)
            out = self.model(data_input)
            pred = out.argmax(dim=1)
            correct = (pred == labels_input).sum().item()
            acc = correct / labels_input.size(0) * 100
        return {
            'accuracy': acc
        }

    def evaluate(self):
        """
        Load the best model and evaluate on test data.
        Returns:
            dict: Evaluation metrics like accuracy.
        """
        self.load_model(self.best_model_path)
        self.model.eval()
        with torch.no_grad():
            data_input = self.test_data.to(self.model.device)
            labels_input = self.labels_test.to(self.model.device)
            out = self.model(data_input)
            pred = out.argmax(dim=1)
            correct = (pred == labels_input).sum().item()
            acc = correct / labels_input.size(0) * 100
        print(f"Test Accuracy: {acc:.2f}%")
        return {'accuracy': acc}

    def save_model(self, filepath: str):
        """
        Save current model state.
        """
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath: str):
        """
        Load model state from file.
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.model.device))
        

def main():
    import yaml
    # Load config file
    import sys
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Example usage: Assume dataset, model, and data are set up
    # Initialize DatasetLoader for loading data
    from dataset_loader import DatasetLoader
    dataset_config = config['dataset']
    # For the purposes of this script, we assume dataset loader is ready to produce data objects
    # Alternatively, load preprocessed data or customize here
    loader = DatasetLoader(dataset_paths=dataset_config, K=500)
    data = loader.get_data()

    # Extract labels for training / validation / test
    labels_train = torch.tensor(data.y[data.train_mask], dtype=torch.long)
    labels_val = torch.tensor(data.y[data.val_mask], dtype=torch.long) if hasattr(data, 'val_mask') else None
    labels_test = torch.tensor(data.y[~data.train_mask & ~data.test_mask], dtype=torch.long) if hasattr(data, 'test_mask') else None

    # Init model with architecture and hyperparameters
    architecture_type = 'GCN'  # Could be set from config, for illustration fixed
    model_params = {'hidden_units': 256, 'layers': 2, 'num_classes': int(torch.max(data.y).item() + 1)}
    model = GNNModel(architecture_type, model_params)

    # Initialize Trainer
    trainer = Trainer(model=model,
                      train_data=data,
                      val_data=data if hasattr(data, 'val_mask') else None,
                      test_data=data,
                      labels_train=labels_train,
                      labels_val=labels_val,
                      labels_test=labels_test,
                      config=config)

    # Run training
    trainer.train()

    # Final evaluation
    final_metrics = trainer.evaluate()
    print(f"Final Test Accuracy: {final_metrics['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
