"""evaluation.py

This module defines the Evaluation class which handles post-training evaluation of
the GAN. It performs qualitative evaluation by generating and visualizing MNIST-like
images from the Generator, and quantitative evaluation by computing a metric analogous
to the Inception Score using a pretrained MNIST classifier.

Note:
  This implementation assumes that the GANModel instance provided has a build_generator()
  method that returns a Generator with trained weights. In a full reproducible system,
  you may want to load the generator’s checkpoint before evaluation.
"""

import os
import logging
from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class SimpleClassifier(nn.Module):
    """A simple classifier for MNIST images.

    This classifier is used to compute a quantitative evaluation metric (analogous to
    the Inception Score) for generated samples. It is a small fully connected network
    that maps the flattened 784-dimensional MNIST image to 10 class logits.
    """

    def __init__(self) -> None:
        super(SimpleClassifier, self).__init__()
        self.fc1: nn.Linear = nn.Linear(784, 128)
        self.relu: nn.ReLU = nn.ReLU()
        self.fc2: nn.Linear = nn.Linear(128, 10)  # MNIST has 10 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 784).

        Returns:
            torch.Tensor: Logits for 10 classes.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Evaluation:
    """Evaluation class handles post-training assessment of the GAN.

    It performs:
      1. Qualitative evaluation by generating a fixed grid of images using the Generator.
      2. Quantitative evaluation by computing a metric analogous to the Inception Score
         via a pretrained MNIST classifier.
    """

    def __init__(self, model: Any, data: Tuple[Any, Any], config: Dict[str, Any]) -> None:
        """
        Initializes the Evaluation class with the GAN model, data loaders, and configuration.

        Args:
            model (Any): Instance of GANModel containing Generator and Discriminator.
            data (Tuple[Any, Any]): Tuple of (train_loader, test_loader) as provided by DatasetLoader.
            config (Dict[str, Any]): Configuration parameters (loaded from config.yaml).
        """
        self.model: Any = model
        self.config: Dict[str, Any] = config if config is not None else {}
        self.train_loader, self.test_loader = data

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use default noise dimension of 100 (common choice for GANs)
        self.noise_dim: int = 100

        # Set up logger for evaluation
        self.logger = logging.getLogger("Evaluation")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger.info("Evaluation initialized on device: %s", self.device)

    def evaluate(self) -> Dict[str, Any]:
        """
        Performs qualitative and quantitative evaluation of the GAN.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "generated_image_path": Path to the saved generated image grid.
                - "inception_score": The computed inception score (quantitative metric).
        """
        # Qualitative Evaluation: generate a grid of images
        generated_image_path: str = self._qualitative_evaluation()

        # Quantitative Evaluation: compute inception score using a pretrained classifier
        inception_score: float = self._quantitative_evaluation()

        evaluation_results: Dict[str, Any] = {
            "generated_image_path": generated_image_path,
            "inception_score": inception_score,
        }
        self.logger.info("Evaluation results: %s", evaluation_results)
        return evaluation_results

    def _qualitative_evaluation(self) -> str:
        """
        Generates a fixed set of images using the GAN generator and saves them as a grid.

        Returns:
            str: The file path where the generated image grid is saved.
        """
        self.logger.info("Starting qualitative evaluation (image generation).")
        # Obtain the generator from the provided GANModel.
        # Note: It is assumed that the generator has been trained.
        generator = self.model.build_generator().to(self.device)
        generator.eval()

        # Generate a fixed noise vector for reproducible results
        num_samples: int = 16  # For a 4x4 grid of images
        fixed_noise: torch.Tensor = torch.randn(num_samples, self.noise_dim, device=self.device)
        with torch.no_grad():
            fake_images: torch.Tensor = generator(fixed_noise)
        # Reshape the output (batch_size, 784) to (batch_size, 28, 28)
        fake_images = fake_images.view(num_samples, 28, 28).cpu().numpy()

        # Create directory for results if it does not exist
        results_dir: str = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Plot a grid of generated images using matplotlib
        grid_rows: int = 4
        grid_cols: int = 4
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols, grid_rows))
        for idx, ax in enumerate(axes.flat):
            if idx < num_samples:
                ax.imshow(fake_images[idx], cmap="gray")
                ax.axis("off")
            else:
                ax.remove()
        plt.tight_layout()

        image_path: str = os.path.join(results_dir, "generated_grid.png")
        try:
            plt.savefig(image_path)
            self.logger.info("Generated image grid saved at %s", image_path)
        except Exception as e:
            self.logger.error("Error saving generated image grid: %s", str(e))
        plt.close(fig)
        return image_path

    def _quantitative_evaluation(self) -> float:
        """
        Computes a quantitative evaluation metric (similar to the Inception Score) for the GAN.

        This is achieved by generating a large set of fake images, obtaining class predictions
        from a pretrained MNIST classifier, and computing the KL divergence between the conditional
        class distributions and the marginal distribution.

        Returns:
            float: The computed inception score.
        """
        self.logger.info("Starting quantitative evaluation (inception score computation).")
        # Retrieve or train a simple MNIST classifier
        classifier: nn.Module = self._get_classifier()
        classifier.to(self.device)
        classifier.eval()

        # Define parameters for generating fake images
        num_fake_images: int = 1000
        eval_batch_size: int = 100  # mini-batch size for evaluation
        num_batches: int = num_fake_images // eval_batch_size

        # Obtain the generator (assumed to be trained) and set it to evaluation mode
        generator = self.model.build_generator().to(self.device)
        generator.eval()

        all_probs: List[torch.Tensor] = []
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc="Generating fake images for quantitative eval"):
                noise: torch.Tensor = torch.randn(eval_batch_size, self.noise_dim, device=self.device)
                fake_images: torch.Tensor = generator(noise)  # shape: (batch_size, 784)
                # Pass generated images through classifier to obtain logits and then softmax probabilities
                logits: torch.Tensor = classifier(fake_images)
                probs: torch.Tensor = F.softmax(logits, dim=1)
                all_probs.append(probs)
        # Concatenate all probabilities into a tensor of shape (num_fake_images, 10)
        probs_all: torch.Tensor = torch.cat(all_probs, dim=0)

        # Compute the marginal probability distribution p(y)
        p_y: torch.Tensor = torch.mean(probs_all, dim=0, keepdim=True)  # shape: (1, 10)

        # Compute KL divergence for each generated image: KL(p(y|x) || p(y))
        kl_divs: torch.Tensor = probs_all * (torch.log(probs_all + 1e-10) - torch.log(p_y + 1e-10))
        sum_kl_divs: torch.Tensor = torch.sum(kl_divs, dim=1)  # shape: (num_fake_images,)
        mean_kl_div: torch.Tensor = torch.mean(sum_kl_divs)
        inception_score: float = torch.exp(mean_kl_div).item()

        self.logger.info("Quantitative evaluation (inception score) computed: %.4f", inception_score)
        return inception_score

    def _get_classifier(self) -> nn.Module:
        """
        Retrieves a pretrained MNIST classifier. If a saved classifier checkpoint exists,
        it is loaded. Otherwise, a new classifier is trained on the MNIST training data.

        Returns:
            nn.Module: A pretrained MNIST classifier.
        """
        classifier_path: str = "mnist_classifier.pth"
        classifier: nn.Module = SimpleClassifier().to(self.device)
        if os.path.exists(classifier_path):
            try:
                classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
                self.logger.info("Pretrained MNIST classifier loaded from %s", classifier_path)
                return classifier
            except Exception as e:
                self.logger.error("Error loading classifier from %s: %s", classifier_path, str(e))
        # If no pretrained classifier is available, train one using the provided training data
        self.logger.info("No pretrained classifier found. Training a new MNIST classifier.")
        classifier = self._train_classifier(classifier)
        try:
            torch.save(classifier.state_dict(), classifier_path)
            self.logger.info("Trained MNIST classifier saved to %s", classifier_path)
        except Exception as e:
            self.logger.error("Error saving trained classifier: %s", str(e))
        return classifier

    def _train_classifier(self, classifier: nn.Module) -> nn.Module:
        """
        Trains the provided classifier on the MNIST training data.

        Args:
            classifier (nn.Module): The classifier to be trained.

        Returns:
            nn.Module: The trained classifier.
        """
        num_epochs: int = 3  # Default number of epochs for classifier training
        learning_rate: float = 0.001  # Learning rate for classifier training
        criterion: nn.Module = nn.CrossEntropyLoss()
        optimizer: optim.Optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

        classifier.train()
        self.logger.info("Starting training of the MNIST classifier for %d epochs.", num_epochs)
        for epoch in range(num_epochs):
            epoch_loss: float = 0.0
            correct: int = 0
            total: int = 0
            for images, labels in self.train_loader:
                images = images.to(self.device)  # images shape: (batch_size, 784)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = classifier(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            avg_loss: float = epoch_loss / total
            accuracy: float = (correct / total) * 100.0
            self.logger.info("Epoch [%d/%d] - Loss: %.4f, Accuracy: %.2f%%",
                             epoch + 1, num_epochs, avg_loss, accuracy)
        classifier.eval()
        return classifier
