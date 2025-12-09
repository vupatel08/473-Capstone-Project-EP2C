## evaluation.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import Accuracy, ExpectationAligmentError
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple, Optional
import math
import os

class Evaluation:
    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 128,
        transformations: Optional[Dict] = None,
        inference_steps: int = 10000,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize the Evaluation object with configuration parameters.

        Args:
            dataset_name (str): Name identifier for datasets (e.g., 'ImageNet-1K', 'ImageNet-R')
            batch_size (int): Batch size for evaluation
            transformations (dict): Dict specifying scale, shift, resolution params
            inference_steps (int): Number of inference samples (not used here but placeholder)
            device (str): 'cpu' or 'cuda', defaults to CUDA if available
            seed (int): Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.transformations = transformations or {}
        self.inference_steps = inference_steps
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed

        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Initialize metrics
        self.accuracy_metric = Accuracy().to(self.device)
        # For ECE, initialize with number of bins 15 (as per paper)
        self.num_bins = 15
        self.ece_metric = ExpectationAligmentError(self.num_bins, n_bins=self.num_bins).to(self.device)

    def compute_accuracy(self, model, dataloader) -> float:
        """
        Compute top-1 accuracy over the dataset.

        Args:
            model: Model object with predict(inputs) method
            dataloader: DataLoader providing images and labels

        Returns:
            float: accuracy in percentage
        """
        total_correct = 0
        total_samples = 0
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = model.predict(images)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        accuracy = 100.0 * total_correct / total_samples
        return accuracy

    def compute_calibration(self, model, dataloader) -> Dict:
        """
        Compute ECE, reliability diagram data, and confidence histogram.

        Args:
            model: Model object
            dataloader: DataLoader

        Returns:
            dict: containing 'ece', 'confidence_bins', 'accuracy_bins', 'average_confidence'
        """
        all_confidences = []
        all_correct = []
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = model.predict(images)
            probs = model.get_probabilities(logits)
            max_prob, preds = probs.max(dim=1)
            all_confidences.extend(max_prob.detach().cpu().numpy())
            all_correct.extend((preds == labels).cpu().numpy())

        # Bin confidences for ECE
        confidences = np.array(all_confidences)
        correctness = np.array(all_correct)

        # For torchmetrics.Extand: use raw data for ECE
        # As torchmetrics's ExpectationAligmentError expects tensors
        self.ece_metric.reset()
        conf_tensor = torch.tensor(confidences).to(self.device)
        acc_tensor = torch.tensor(correctness).to(self.device)
        self.ece_metric.update(conf_tensor, acc_tensor)
        ece_value = self.ece_metric.compute().item()

        # Compute accuracy and confidence per bin for visualization
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_indices = np.digitize(confidences, bin_boundaries, right=True)

        average_confidence_per_bin = []
        accuracy_per_bin = []
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) /2
        for i in range(1, self.num_bins +1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                avg_conf = np.mean(confidences[mask])
                acc = np.mean(correctness[mask])
            else:
                avg_conf = 0
                acc = 0
            average_confidence_per_bin.append(avg_conf)
            accuracy_per_bin.append(acc)

        # Plot reliability diagram
        self._plot_reliability_diagram(accuracy_per_bin, average_confidence_per_bin, ece_value)

        # Plot confidence histogram
        self._plot_confidence_histogram(confidences)

        return {
            'ece': ece_value,
            'confidence_bin_centers': bin_centers,
            'accuracy_per_bin': accuracy_per_bin,
            'average_confidence_per_bin': average_confidence_per_bin
        }

    def mistake_analysis(self, model, dataloader, annotations: Optional[Dict]=None) -> Dict:
        """
        Analyze mistake types via confusion and factor errors.

        Args:
            model: Model object
            dataloader: DataLoader
            annotations: Optional; dict mapping image IDs to factors (pose, style, etc.)

        Returns:
            dict: with error ratios per factor, confusion matrix plot
        """
        preds_list = []
        labels_list = []
        image_ids = []  # Placeholder for IDs if needed for annotations
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = model.predict(images)
            preds = logits.argmax(dim=1)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

        preds_array = np.array(preds_list)
        labels_array = np.array(labels_list)

        # Compute overall accuracy
        overall_acc = np.mean(preds_array == labels_array)

        # Confusion matrix
        conf_mat = confusion_matrix(labels_array, preds_array)

        # Mistake factors: if annotations provided, compute error ratio per factor
        error_ratios = {}
        if annotations:
            # annotations: dict {image_id: dict{factor: value}}
            # Skip actual implementation: assuming annotations are aligned with dataset
            # For simplicity, placeholder
            factors = ['pose', 'style', 'texture', 'occlusion', 'background']
            for factor in factors:
                # Extract factor labels for images
                # For demo, assign dummy
                acc_factor = 0.5  # Placeholder for accuracy on factor
                error_ratio = (1 - acc_factor) / (1 - overall_acc) if (1 - overall_acc) > 0 else 0
                error_ratios[factor] = error_ratio

        # Plot confusion matrix heatmap
        self._plot_confusion_matrix(conf_mat)

        return {
            'overall_accuracy': overall_acc,
            'error_ratios': error_ratios
        }

    def bias_evaluation(self, model, dataloader_conflict) -> Dict:
        """
        Compute shape vs texture bias on cue-conflict dataset.

        Args:
            model: Model object
            dataloader_conflict: DataLoader for cue-conflict images

        Returns:
            dict: shape bias fraction, texture bias fraction
        """
        shape_decisions = []
        texture_decisions = []

        for images, labels in dataloader_conflict:
            images = images.to(self.device)
            logits = model.predict(images)
            probs = model.get_probabilities(logits)
            preds = probs.argmax(dim=1).cpu().numpy()

            # Placeholder: assuming annotations specify whether each image favors shape or texture cue
            # Here, simulate decision based on some heuristic or label
            # For demo, suppose even labels suggest shape, odd labels suggest texture
            for pred, label in zip(preds, labels.numpy()):
                # Dummy logic: in real case, use actual cue-categorization
                if label % 2 == 0:
                    shape_decisions.append(pred)
                else:
                    texture_decisions.append(pred)

        # Compute bias proportions
        total_decisions = len(shape_decisions) + len(texture_decisions)
        shape_bias_fraction = len(shape_decisions) / total_decisions if total_decisions > 0 else 0
        texture_bias_fraction = len(texture_decisions) / total_decisions if total_decisions > 0 else 0

        # Plot bias bars
        self._plot_bias_bars(shape_bias_fraction, texture_bias_fraction)

        return {
            'shape_bias_fraction': shape_bias_fraction,
            'texture_bias_fraction': texture_bias_fraction
        }

    def invariance_tests(self, model, dataset, test_type='scale'):
        """
        Evaluate invariance to scale, shift, or resolution transformations.

        Args:
            model: Model object
            dataset: Dataset object with original images
            test_type: one of 'scale', 'shift', 'resolution'

        Returns:
            dict: accuracy per transformation level
        """
        results = {}
        scale_factors = self.transformations.get('scale_factors', [1, 1.25, 1.5, 2, 3])
        shift_pixels = self.transformations.get('shift_pixels', [0, 10, 20])
        resolution_sizes = self.transformations.get('resolution_sizes', [112, 224, 336, 512, 640])

        if test_type == 'scale':
            for scale in scale_factors:
                transform = self._get_invariance_transform(scale=scale)
                dataloader = self._create_dataloader_from_dataset(dataset, transform)
                acc = self._compute_accuracy_batch(model, dataloader)
                results[scale] = acc
        elif test_type == 'shift':
            for shift in shift_pixels:
                transform = self._get_shift_transform(shift_pixels=shift)
                dataloader = self._create_dataloader_from_dataset(dataset, transform)
                acc = self._compute_accuracy_batch(model, dataloader)
                results[shift] = acc
        elif test_type == 'resolution':
            for size in resolution_sizes:
                transform = self._get_resolution_transform(size=size)
                dataloader = self._create_dataloader_from_dataset(dataset, transform)
                acc = self._compute_accuracy_batch(model, dataloader)
                results[size] = acc
        else:
            raise ValueError(f"Unknown invariance test type: {test_type}")

        # Plot or return results as needed
        self._plot_invariance(results, test_type)
        return results

    def transferability(self, model, datasets_dict: Dict[str, torch.utils.data.Dataset]) -> Dict:
        """
        Evaluate transferability on multiple datasets (VTAB subset).

        Args:
            model: Model object
            datasets_dict: dict of dataset_name: Dataset object

        Returns:
            dict: dataset_name -> accuracy, calibration (ECE)
        """
        results = {}
        for name, dataset in datasets_dict.items():
            dataloader = self._create_dataloader_from_dataset(dataset, self._get_transform(for_eval=True))
            acc = self._compute_accuracy_batch(model, dataloader)
            cal_data = self.compute_calibration(model, dataloader)
            results[name] = {
                'accuracy': acc,
                'ece': cal_data['ece']
            }
        return results

    def synthetic_data_evaluation(self, model, dataset) -> Dict:
        """
        Evaluate model on synthetic datasets like PUG-ImageNet.

        Args:
            model: Model object
            dataset: Synthetic dataset object

        Returns:
            Dict: overall accuracy and per-factor accuracies
        """
        dataloader = self._create_dataloader_from_dataset(dataset, self._get_transform(for_eval=True))
        total_correct = 0
        total_samples = 0
        factor_correct = {}  # per-factor accuracy
        total_per_factor = {}
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = model.predict(images)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            # For per-factor: count per factor if info available
            # Placeholder: assume factors are encoded in dataset
            # For demo, skip per-factor detailed accuracy
        overall_acc = total_correct / total_samples * 100
        return {'overall_accuracy': overall_acc}

    ########################## Internal Helper Methods ##############################

    def _plot_reliability_diagram(self, accuracy_bins, confidence_bins, ece_value):
        plt.figure()
        plt.plot([0,1], [0,1], linestyle='--', color='gray')
        plt.plot(confidence_bins, accuracy_bins, marker='.', linewidth=2, label='Model')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Reliability Diagram\nECE={ece_value:.3f}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _plot_confidence_histogram(self, confidences):
        plt.figure()
        plt.hist(confidences, bins=self.num_bins, range=(0,1))
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Histogram')
        plt.grid(True)
        plt.show()

    def _plot_confusion_matrix(self, conf_mat):
        plt.figure(figsize=(8,8))
        plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def _plot_bias_bars(self, shape_bias, texture_bias):
        plt.figure()
        labels = ['Shape Bias', 'Texture Bias']
        values = [shape_bias, texture_bias]
        plt.bar(labels, values)
        plt.ylabel('Fraction')
        plt.title('Shape vs Texture Bias')
        plt.ylim(0,1)
        plt.show()

    def _plot_invariance(self, results: Dict, test_type: str):
        plt.figure()
        levels = list(results.keys())
        accuracies = list(results.values())
        plt.plot(levels, accuracies, marker='o')
        plt.xlabel(f'{test_type.capitalize()} Magnitude')
        plt.ylabel('Accuracy')
        plt.title(f'Invariance to {test_type}')
        plt.grid(True)
        plt.show()

    def _create_dataloader_from_dataset(self, dataset, transform):
        dataset.transform = transform
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def _get_invariance_transform(self, scale: float=1.0):
        def transform(image):
            size = int(image.size(1) * scale)
            resize = (size, size)
            transform_list = [
                lambda img: img.resize(resize, Image.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ]
            for t in transform_list:
                image = t(image) if callable(t) else t(image)
            return image
        return transform

    def _get_shift_transform(self, shift_pixels: int=0):
        def transform(image):
            width, height = image.size
            shift_x = shift_pixels if width > shift_pixels else 0
            shift_y = shift_pixels if height > shift_pixels else 0
            left = shift_x
            upper = shift_y
            right = width - shift_x
            lower = height - shift_y
            cropped = image.crop((left, upper, right, lower))
            return cropped.resize((224, 224), Image.BILINEAR)
        return transform

    def _get_resolution_transform(self, size: int=224):
        def transform(image):
            resized = image.resize((size, size), Image.BILINEAR)
            cropped = resized
            if size != 224:
                # For models expecting 224, normalize accordingly
                pass
            return resized
        return transform

    def _compute_accuracy_batch(self, model, dataloader) -> float:
        total_correct = 0
        total_samples = 0
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = model.predict(images)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        return 100.0 * total_correct / total_samples
