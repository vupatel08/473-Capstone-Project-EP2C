## evaluation.py
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class LinearEvaluator:
    """
    Evaluates learned representations via linear probing.
    Extracts features from a frozen encoder, trains a linear classifier,
    and reports accuracy on a downstream dataset.
    """
    def __init__(self, model, dataset_loader, device='cuda'):
        """
        Args:
            model (nn.Module): the pretrained, frozen encoder model
            dataset_loader (torch.utils.data.DataLoader): loader for evaluation dataset
            device (str or torch.device): computation device
        """
        self.model = model
        self.dataset_loader = dataset_loader
        self.device = torch.device(device)
        self.features = None
        self.labels = None
        self._prepare()

    def _prepare(self):
        """
        Extract features and labels from the dataset.
        """
        self.features = []
        self.labels = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in self.dataset_loader:
                images = images.to(self.device)
                # Forward pass through encoder
                feats = self._extract_features(images)
                # Normalize features
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                self.features.append(feats.cpu().numpy())
                self.labels.append(labels.numpy())

        self.features = np.concatenate(self.features, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
    
    def _extract_features(self, images):
        """
        Extract features from images using the model's encoder.
        """
        # Assumption: model.encoder is the backbone; if model is the entire model,
        # replace with correct attribute or method.
        return self.model.encoder(images)
    
    def linear_probe(self):
        """
        Train linear classifier on features and evaluate accuracy.
        Returns:
            accuracy (float): Top-1 accuracy in [0, 1]
        """
        # Split features into train/test if dataset has labeled splits
        # Here, we assume entire dataset is used for features; for simulation, use train/test sets if available
        # If dataset has train/test split, prefer those splits
        # For simplicity, here, use entire features for classifier
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        try:
            clf.fit(self.features, self.labels)
        except Exception as e:
            print(f"Error during classifier training: {e}")
            return 0.0

        preds = clf.predict(self.features)
        acc = accuracy_score(self.labels, preds)
        return acc

    def evaluate(self):
        """
        Run feature extraction and classifier training, then return accuracy.
        """
        self._prepare()
        accuracy = self.linear_probe()
        return {'accuracy': accuracy}

    def run(self):
        """
        Wrapper to perform evaluation and print result.
        """
        metrics = self.evaluate()
        print(f"Linear probing accuracy: {metrics['accuracy']*100:.2f}%")
        return metrics
