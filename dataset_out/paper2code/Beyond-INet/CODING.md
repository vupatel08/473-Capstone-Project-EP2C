# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## app.py

```python
## app.py
import os
import yaml
import torch
import numpy as np
import json
from tqdm import tqdm

# Import custom modules
from dataset_loader import DatasetLoader
from model import Model
from evaluation import Evaluation
from visualization import (
    plot_reliability_diagram_and_histogram,
    plot_confusion_matrix,
    plot_bias_bars,
    plot_invariance_results,
)

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    seed = config.get('misc', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize DatasetLoader with dataset settings
    dataset_cfg = config.get('dataset', {})
    dataset_loader = DatasetLoader(dataset_cfg)

    # Load the main validation dataset (ImageNet-1K)
    val_dataloader = dataset_loader.load_data(split='validation')

    # Initialize models as per configuration
    models_config = config.get('models', {})
    model_instances = {}
    for key, m_cfg in models_config.items():
        arch = m_cfg.get('architecture')
        pretrained = m_cfg.get('pretrained', True)
        dataset_name = m_cfg.get('dataset')
        source = m_cfg.get('pretrained_source', None)

        print(f"Loading model: {key} - {arch }, pretrained: {pretrained}")
        model_instance = Model(
            architecture=arch,
            pretrained=pretrained,
            dataset=dataset_name,
            pretrained_source=source
        )
        model_instance.model.eval()
        model_instance.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model_instances[key] = model_instance

    # Initialize evaluation object
    eval_cfg = config.get('evaluation', {})
    evaluation = Evaluation(
        dataset_name=dataset_cfg.get('name', 'ImageNet-1K'),
        batch_size=eval_cfg.get('batch_size', 128),
        transformations=eval_cfg.get('transformations', {}),
        inference_steps=eval_cfg.get('inference_steps', 10000),
        device=device,
        seed=seed
    )

    # Results dictionary to store all metrics and diagnostics
    results = {}

    # ===================== 1. Standard Accuracy on ImageNet-1K =======================
    print("\n=== 1. Computing Top-1 Accuracy on ImageNet-1K Validation ====")
    top1_acc = evaluation.compute_accuracy(model_instances['model'], val_dataloader)
    results['accuracy'] = top1_acc
    print(f"ImageNet-1K Validation Top-1 Accuracy: {top1_acc:.2f}%\n")

    # ===================== 2. Mistake / Confusion Analysis ============================
    print("=== 2. Mistake and Confusion Analysis ===")
    mistake_stats = evaluation.compute_mistake_stats(model_instances['model'], val_dataloader)
    results['mistake_analysis'] = mistake_stats

    # ===================== 3. Calibration on ImageNet-1K and ImageNet-R ==================
    print("=== 3. Calibration Assessment on ImageNet-1K ===")
    calib_in = evaluation.compute_calibration(model_instances['model'], val_dataloader)
    results['calibration_in'] = calib_in

    print("=== Calibration on ImageNet-R (out-of-distribution) ===")
    imagenet_r_loader = dataset_loader.load_data(split='validation', dataset_name='ImageNet-R')
    calib_out = evaluation.compute_calibration(model_instances['model'], imagenet_r_loader)
    results['calibration_out'] = calib_out

    # Plot calibration reliability diagram and confidence histogram (IN)
    plot_reliability_diagram_and_histogram(
        calib_in,
        calib_in['confidence_bin_centers'],
        calib_in['ece']
    )

    # ===================== 4. Bias: Shape vs Texture Bias on Cue-Conflict Dataset =======
    print("=== 4. Shape vs Texture Bias on Cue-Conflict Dataset ===")
    cue_conflict_loader = dataset_loader.load_data(split='validation', dataset_name='CueConflict')
    bias_stats = evaluation.compute_bias(model_instances['model'], cue_conflict_loader)
    results['bias'] = bias_stats

    # Plot bias bar chart
    plot_bias_bars(bias_stats)

    # ===================== 5. Invariance Tests (Scale, Shift, Resolution) =============
    for test_type in ['scale', 'shift', 'resolution']:
        print(f"=== 5. Invariance test: {test_type} ===")
        # Load dataset; transformations will be applied inside evaluation
        dataset_instance = dataset_loader.load_data(split='validation')
        invariance_results = evaluation.invariance_tests(
            model_instances['model'], dataset_instance, test_type=test_type
        )
        results[f'invariance_{test_type}'] = invariance_results
        # Plot invariance results
        plot_invariance_results(invariance_results, test_type=test_type)

    # ===================== 6. Transferability Evaluation (VTAB or similar) ==============
    print("=== 6. Transferability Evaluation on VTAB-like Datasets ===")
    datasets_dict = _load_vtab_datasets()
    transfer_results = evaluation.evaluate_transferability(model_instances['model'], datasets_dict)
    results['transferability'] = transfer_results

    # ===================== 7. Synthetic Data Evaluation (PUG-ImageNet) =================
    print("=== 7. Synthetic Data Evaluation (PUG-ImageNet) ===")
    pug_dataset = _load_synthetic_dataset()
    synthetic_results = evaluation.synthetic_data_evaluation(model_instances['model'], pug_dataset)
    results['synthetic'] = synthetic_results

    # Save all results to JSON
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\n=== Evaluation complete. Results saved to 'evaluation_results.json' ===")


# Helper functions for dataset loading (replace with actual paths or logic)
def _load_vtab_datasets():
    # Placeholder: you need to replace with actual dataset loading code
    # For demonstration, return an empty dict
    return {}

def _load_synthetic_dataset():
    # Placeholder: replace with actual synthetic dataset loading
    class DummySyntheticDataset:
        def __len__(self):
            return 1000
        def __getitem__(self, idx):
            from PIL import Image
            import torch
            dummy_img = Image.new('RGB', (224, 224))
            return self._transform(dummy_img), np.random.randint(0, 1000)
        def _transform(self, img):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225])
            ])
            return transform(img)
    return DummySyntheticDataset()


if __name__ == "__main__":
    main()
```

## dataset_loader.py

```python
## dataset_loader.py
import os
from typing import Optional, Tuple, List, Dict, Union
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

# Optional: Import for custom datasets if needed (e.g., ImageNet-R, cue-conflict, synthetic)
# For demonstration, assume datasets are stored locally in specific directories.

class ImageNetValidationDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[torch.nn.Module] = None):
        """
        Loads ImageNet validation images from a directory structured as:
        root_dir/
            class_1/
                img1.jpg
                ...
            class_2/
                ...
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self._load_samples()

    def _load_samples(self):
        # Map class names to indices
        classes = sorted(entry.name for entry in os.scandir(self.root_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        # Gather all image paths and labels
        for cls_name in classes:
            class_dir = os.path.join(self.root_dir, cls_name)
            for img in os.listdir(class_dir):
                if img.endswith('.jpg') or img.endswith('.png'):
                    self.samples.append((os.path.join(class_dir, img), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class ImageNetRDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[torch.nn.Module] = None):
        """
        Loads ImageNet-R images from a directory.
        Expected directory structure similar to ImageNet validation set.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    label = self._extract_label_from_path(root, file)
                    self.samples.append((os.path.join(root, file), label))

    def _extract_label_from_path(self, root: str, filename: str) -> int:
        # Placeholder: Implement label extraction based on directory name or filename
        # For simplicity, assume folder name corresponds to class; assign dummy labels
        # Replace as needed with actual label mapping
        return 0  # Dummy label, since labels are often not annotated

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class SyntheticDataset(Dataset):
    def __init__(self, images_dir: str, labels_dict: Dict[str, int], transform: Optional[torch.nn.Module] = None):
        """
        Loads synthetic images generated with varying factors.
        - images_dir: path to the directory containing images.
        - labels_dict: mapping from filename to label based on factors.
        """
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []
        self._load_samples(labels_dict)

    def _load_samples(self, labels_dict):
        for filename, label in labels_dict.items():
            img_path = os.path.join(self.images_dir, filename)
            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class CueConflictDataset(Dataset):
    def __init__(self, images_dir: str, labels: List[int], transform: Optional[torch.nn.Module] = None):
        """
        Loads cue-conflict images for shape/texture bias evaluation.
        - images_dir: directory with images.
        - labels: list of labels for images indicating conflict type.
        """
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []
        for idx, filename in enumerate(os.listdir(images_dir)):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                self.samples.append((os.path.join(images_dir, filename), labels[idx]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class DatasetLoader:
    def __init__(self, config: dict):
        """
        Initializes dataset loader based on configuration.
        Assumes paths are relative or absolute as specified.
        """
        self.config = config
        self.dataset_cache = {}

        # Set dataset base directory paths based on dataset name
        self.base_dir = {
            'ImageNet-1K': '/path/to/imagenet/val',  # replace with actual path
            'ImageNet-R': '/path/to/imagenet_r',    # replace with actual path
            'PUG-ImageNet': '/path/to/pug_imagenet', # replace with actual synthetic dataset path
            'CueConflict': '/path/to/cue_conflict'   # replace with actual cue-conflict images path
        }

        # Load dataset once if needed
        self._dataset_cache = {}

    def load_data(self, split: str = 'validation', dataset_name: Optional[str] = None) -> DataLoader:
        """
        Loads dataset according to name and split.
        """
        dataset_name = dataset_name or self.config['name']
        dataset_name = dataset_name.strip()

        if hasattr(self, f'_{dataset_name}_loader'):
            # Call specific loader if implemented
            load_fn = getattr(self, f'_{dataset_name}_loader')
            return load_fn()

        # Else, fallback to common methods
        if dataset_name == 'ImageNet-1K':
            return self._load_imagenet_validation()
        elif dataset_name == 'ImageNet-R':
            return self._load_imagenet_r()
        elif dataset_name == 'PUG-ImageNet':
            return self._load_pug_imagenet()
        elif dataset_name == 'CueConflict':
            return self._load_cue_conflict()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _get_transform(self, for_eval: bool = True, custom_res: Optional[int] = None, **kwargs):
        """
        Defines image transformations based on evaluation or invariance tests.
        """
        image_size = self.config.get('image_size', 224)
        resize_size = custom_res if custom_res else image_size

        transform_list = [
            transforms.Resize(resize_size),
        ]

        if for_eval:
            transform_list += [
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        else:
            # For training or specific augmentation, customize as needed
            transform_list += [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        return transforms.Compose(transform_list)

    def _load_imagenet_validation(self) -> DataLoader:
        dataset_dir = self.base_dir['ImageNet-1K']
        transform = self._get_transform(for_eval=True)
        dataset = ImageNetValidationDataset(root_dir=dataset_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False,
                                num_workers=4, pin_memory=True)
        return dataloader

    def _load_imagenet_r(self) -> DataLoader:
        dataset_dir = self.base_dir['ImageNet-R']
        transform = self._get_transform(for_eval=True)
        dataset = ImageNetRDataset(root_dir=dataset_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False,
                                num_workers=4, pin_memory=True)
        return dataloader

    def _load_pug_imagenet(self) -> DataLoader:
        # Placeholder: Load synthetic dataset images and labels
        # Assume images are in a directory; labels dict provided or inferred
        images_dir = self.base_dir['PUG-ImageNet']
        # Here, you'd load the labels mapping from a file or define programmatically
        labels_dict = {}  # Replace with actual label mapping
        transform = self._get_transform(for_eval=True)
        dataset = SyntheticDataset(images_dir, labels_dict, transform)
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False,
                                num_workers=4, pin_memory=True)
        return dataloader

    def _load_cue_conflict(self) -> DataLoader:
        # Placeholder: Load cue-conflict images with specific labels
        images_dir = self.base_dir['CueConflict']
        labels = []  # Generate or load labels matching images
        # For example, label 0 for shape-bias images, 1 for texture-bias images
        # Or as per dataset annotation
        labels = [0] * len(os.listdir(images_dir))  # Dummy placeholder
        transform = self._get_transform(for_eval=True)
        dataset = CueConflictDataset(images_dir, labels, transform)
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False,
                                num_workers=4, pin_memory=True)
        return dataloader
```

## evaluation.py

```python
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
```

## model.py

```python
## model.py
import torch
import torch.nn.functional as F
from typing import Optional
import os

# Import HuggingFace transformers for ViT and CLIP, torchvision models for ConvNeXt
from torchvision import models as torchvision_models
from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_huge
from transformers import ViTForImageClassification, ViTFeatureExtractor
# For CLIP, use openclip package
import open_clip

class Model:
    def __init__(
        self,
        architecture: str,
        pretrained: bool = True,
        dataset: str = 'ImageNet-21K',
        pretrained_source: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Model object.
        :param architecture: Model type identifier, e.g., 'ConvNeXt-Huge', 'ViT-L/16', 'CLIP-Large'
        :param pretrained: Whether to load pretrained weights
        :param dataset: Training dataset name (e.g., 'ImageNet-21K', 'LAION-400M')
        :param pretrained_source: Source for pretrained weights ('OpenCLIP' or None for torchvision)
        :param device: 'cpu' or 'cuda'; default is CUDA if available
        """
        self.architecture = architecture
        self.pretrained = pretrained
        self.dataset = dataset
        self.pretrained_source = pretrained_source
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_extractor = None  # For CLIP
        # Load model upon initialization
        self.load_pretrained()

    def load_pretrained(self):
        """
        Load the pretrained model based on architecture.
        Raises errors if model not recognized or loading fails.
        """
        arch = self.architecture.lower()
        # ConvNeXt models
        if 'convnext' in arch:
            if 'tiny' in arch:
                self.model = convnext_tiny(pretrained=self.pretrained)
            elif 'small' in arch:
                self.model = convnext_small(pretrained=self.pretrained)
            elif 'base' in arch:
                self.model = convnext_base(pretrained=self.pretrained)
            elif 'large' in arch:
                self.model = convnext_large(pretrained=self.pretrained)
            elif 'huge' in arch:
                self.model = convnext_huge(pretrained=self.pretrained)
            else:
                raise ValueError(f"Unknown ConvNeXt size in architecture: {self.architecture}")
            self.model = self.model.to(self.device)
            self.model.eval()
            # Normalize parameters are standard in torchvision pretrained models
            self.input_size = 224  # for all ConvNeXt
        # Vision Transformer models
        elif 'vit' in arch:
            # Parse size, e.g., 'vit-s/16'
            # Expected format: 'vit-s/16', 'vit-l/16', 'vit-h/14'
            if 's/16' in arch:
                model_name = 'google/vit-base-patch16-224-in21k'
            elif 'l/16' in arch:
                model_name = 'google/vit-large-patch16-224-in21k'
            elif 'h/14' in arch:
                model_name = 'google/vit-huge-patch14-224-in21k'
            else:
                raise ValueError(f"Unknown ViT size in architecture: {self.architecture}")
            self.model = ViTForImageClassification.from_pretrained(model_name)
            self.model.eval()
            self.model = self.model.to(self.device)
            # Use default feature extractor (for normalization)
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
            self.input_size = 224
        # CLIP models
        elif 'clip' in arch:
            # For CLIP, use open_clip
            if self.pretrained_source != 'OpenCLIP':
                raise ValueError("For CLIP models, pretrained_source must be 'OpenCLIP'")
            if 'large' in arch:
                clip_type = 'ViT-B-32'  # default, check model size more specifically if needed
                # For larger models, you might use 'ViT-H-14' or 'RN50x4' as per open_clip
                # but based on description, select the matching:
                # For 'Large': use 'ViT-B/16' or 'ViT-B/32'
                if 'xl' in self.architecture.lower():
                    clip_type = 'ViT-L-14'  # example for XL
                elif 'huge' in self.architecture.lower():
                    clip_type = 'ViT-H-14'  # if available
                else:
                    clip_type = 'ViT-B-32'  # default
            elif 'xl' in self.architecture.lower() or 'huge' in self.architecture.lower():
                clip_type = 'ViT-L-14' if 'xl' in self.architecture.lower() else 'ViT-H-14'
            else:
                clip_type = 'ViT-B-32'  # fallback
            # Load with open_clip
            self.model, self.preprocess = open_clip.load(clip_type, device=self.device,
                                                         pretrained=True, source='openai' if self.pretrained else None)
            self.model.eval()
            self.input_size = 224
        else:
            raise ValueError(f"Model architecture '{self.architecture}' not recognized.")

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run inference on input images and output logits.
        :param images: tensor of shape (batch_size, 3, H, W), preprocessed
        :return: logits tensor of shape (batch_size, num_classes)
        """
        with torch.no_grad():
            images = images.to(self.device)
            # Handle CLIP differently if needed
            if 'clip' in self.architecture.lower():
                # CLIP's model: output similarity scores
                logits_per_image = self.model.encode_image(images)
                # Normalize to get cosine similarity as logits
                logits = logits_per_image / self.model.l2_norm
            else:
                logits = self.model(images).logits
        return logits

    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to class probabilities
        :param logits: tensor of shape (batch_size, num_classes)
        :return: probabilities tensor
        """
        return F.softmax(logits, dim=1)

    def get_confidence(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Compute maximum class probability for each sample.
        :param probabilities: tensor of shape (batch_size, num_classes)
        :return: tensor of shape (batch_size,)
        """
        return probabilities.max(dim=1).values

    def preprocess_image(self, image: 'PIL.Image') -> torch.Tensor:
        """
        Preprocess input image for model inference, normalize, resize.
        It should be used externally if applying per-image processing.
        """
        if hasattr(self, 'feature_extractor'):
            # For ViT with HuggingFace feature extractor
            inputs = self.feature_extractor(images=image, return_tensors='pt')
            return inputs['pixel_values'].squeeze(0)
        else:
            # For torchvision models
            transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            return transform(image)
```

## trainer.py

```python
## trainer.py
import torch
import numpy as np
import json
from typing import Dict, Optional
from tqdm import tqdm
from evaluation import Evaluation
from visualization import plot_reliability_diagram_and_histogram, plot_confusion_matrix, plot_bias_bars, plot_invariance_results

class Trainer:
    def __init__(
        self,
        model,
        dataset_loader,
        config: Dict,
        device: Optional[str] = None
    ):
        """
        Initializes the Trainer with model, dataset loader, configuration, and device.
        """
        self.model = model
        self.dataset_loader = dataset_loader
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Store evaluation results
        self.results = {}

        # Initialize the evaluation object
        self.evaluator = Evaluation(
            dataset_name=self.dataset_loader.config.get('name', 'Unknown'),
            batch_size=self.config.get('evaluation', {}).get('batch_size', 128),
            transformations=self.config.get('evaluation', {}).get('transformations', {}),
            inference_steps=self.config.get('evaluation', {}).get('inference_steps', 10000),
            device=self.device,
            seed=self.config.get('misc', {}).get('seed', 42)
        )

    def run(self):
        """
        Run the full suite of evaluations as per paper's methodology.
        """
        # Load datasets
        val_loader = self.dataset_loader.load_data(split='validation')

        # 1. Standard accuracy on ImageNet-1K validation set
        print("Evaluating Top-1 Accuracy on ImageNet-1K validation...")
        top1_acc = self.evaluator.compute_accuracy(self.model, val_loader)
        self.results['accuracy'] = top1_acc
        print(f"ImageNet-1K Validation Accuracy: {top1_acc:.2f}%")

        # 2. Mistake analysis / confusion
        print("Running mistake analysis...")
        mistake_stats = self.evaluator.compute_mistake_stats(self.model, val_loader)
        self.results['mistake_analysis'] = mistake_stats

        # 3. Calibration
        print("Evaluating calibration on ImageNet-1K...")
        calib_results_in = self.evaluator.compute_calibration(self.model, val_loader)
        self.results['calibration_in'] = calib_results_in

        # 4. Calibration on ImageNet-R
        print("Evaluating calibration on ImageNet-R...")
        imagenet_r_loader = self.dataset_loader.load_data(split='validation', dataset_name='ImageNet-R')
        calib_results_r = self.evaluator.compute_calibration(self.model, imagenet_r_loader)
        self.results['calibration_out'] = calib_results_r

        # Visualize calibration
        plot_reliability_diagram_and_histogram(
            calib_results_in, calib_results_in['confidence_bin_centers'], calib_results_in['ece']
        )

        # 5. Bias analysis: shape vs texture bias (cue-conflict images)
        print("Conducting shape vs texture bias analysis...")
        cue_conflict_loader = self.dataset_loader.load_data(split='validation', dataset_name='CueConflict')
        bias_stats = self.evaluator.compute_bias(self.model, cue_conflict_loader)
        self.results['bias'] = bias_stats

        # Visualize bias
        plot_bias_bars(bias_stats)

        # 6. Invariance tests: scale, shift, resolution
        print("Performing invariance tests (scale, shift, resolution)...")
        for test_type in ['scale', 'shift', 'resolution']:
            dataset = self.dataset_loader.load_data(split='validation')  # Use same dataset but with transformed images
            invariance_results = self.evaluator.invariance_tests(self.model, dataset, test_type=test_type)
            self.results[f'invariance_{test_type}'] = invariance_results
            # Plot invariance
            plot_invariance_results(invariance_results, test_type=test_type)

        # 7. Transferability on VTAB or similar datasets
        print("Evaluating transferability...")
        # Here you should load VTAB datasets; assuming they're preloaded or placeholder
        datasets_dict = self._load_vtab_datasets()
        transfer_results = self.evaluator.evaluate_transferability(self.model, datasets_dict)
        self.results['transferability'] = transfer_results

        # 8. Synthetic data evaluation (e.g., PUG-ImageNet)
        print("Evaluating on synthetic PUG-ImageNet data...")
        pug_dataset = self._load_synthetic_dataset()
        synthetic_results = self.evaluator.synthetic_data_evaluation(self.model, pug_dataset)
        self.results['synthetic'] = synthetic_results

        # Save results dict to JSON for record
        with open('evaluation_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)

        print("Evaluation complete. Results saved to 'evaluation_results.json'.")

    def _load_vtab_datasets(self):
        """
        Placeholder to load VTAB datasets or similar for transferability.
        Replace with actual dataset loading logic.
        """
        # Here, load actual datasets e.g., via dataset_loader or other sources
        # For demonstration, return empty dict or sample datasets
        return {}

    def _load_synthetic_dataset(self):
        """
        Placeholder for loading synthetic dataset (PUG-ImageNet).
        Replace with actual data loading.
        """
        # For demo, use the same dataset loader with a different split or mock data
        # Assume synthetic dataset is prepared similarly
        class DummySyntheticDataset:
            def __len__(self):
                return 1000
            def __getitem__(self, idx):
                # Return dummy image tensor and label
                # Replace with actual synthetic images loading process
                from PIL import Image
                import torch
                dummy_img = Image.new('RGB', (224, 224))
                dummy_tensor = self._transform(dummy_img)
                label = np.random.randint(0, 1000)
                return dummy_tensor, label
            def _transform(self, img):
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                ])
                return transform(img)
        return DummySyntheticDataset()
```

## visualization.py

```python
## visualization.py
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.ticker as ticker

def plot_reliability_diagram_and_histogram(
    rel_diag: Dict[str, np.ndarray],
    conf_hist: np.ndarray,
    ece_value: float,
    save_path: Optional[str] = None
) -> None:
    """
    Plots the reliability diagram with calibration bins, overlayed with the
    model's confidence accuracy curve, and the confidence histogram.

    Args:
        rel_diag (Dict[str, np.ndarray]): Dictionary containing 'bin_centers' and 'accuracy' arrays.
        conf_hist (np.ndarray): Histogram counts of model confidences over bins.
        ece_value (float): Calculated Expected Calibration Error.
        save_path (Optional[str]): Path to save the figure. If None, display instead.
    """
    # Extract bin data
    bin_centers = rel_diag.get('bin_centers')
    accuracy = rel_diag.get('accuracy')
    
    plt.figure(figsize=(12, 8))
    # Reliability diagram
    plt.subplot(2, 1, 1)
    plt.plot(bin_centers, accuracy, marker='.', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Diagram (ECE={ece_value:.3f})')
    plt.legend()
    plt.grid(True)

    # Confidence Histogram
    plt.subplot(2, 1, 2)
    bins = np.linspace(0, 1, len(conf_hist)+1)
    bin_centers_hist = (bins[:-1] + bins[1:]) / 2
    plt.bar(bin_centers_hist, conf_hist, width=bins[1]-bins[0], edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Number of Predictions')
    plt.title('Confidence Histogram')
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(
    conf_mat: np.ndarray,
    class_names: Optional[list] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plots a confusion matrix heatmap.

    Args:
        conf_mat (np.ndarray): 2D array (N x N) confusion matrix.
        class_names (list, optional): List of class labels for axes.
        save_path (Optional[str]): Path to save the plot. If None, display.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if class_names:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
    else:
        plt.xticks(ticks=np.arange(conf_mat.shape[1]))
        plt.yticks(ticks=np.arange(conf_mat.shape[0]))
    plt.grid(False)

    # Annotate cells with counts
    thresh = conf_mat.max() / 2
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, str(conf_mat[i, j]),
                     ha='center', va='center',
                     color='white' if conf_mat[i, j] > thresh else 'black')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_bias_bars(
    bias_dict: Dict[str, float],
    save_path: Optional[str] = None
) -> None:
    """
    Plots grouped bar chart comparing shape and texture bias fractions.

    Args:
        bias_dict (Dict[str, float]): Dictionary with keys 'shape_bias_fraction',
            'texture_bias_fraction'.
        save_path (Optional[str]): Path to save. If None, display.
    """
    labels = ['Shape Bias', 'Texture Bias']
    values = [
        bias_dict.get('shape_bias_fraction', 0),
        bias_dict.get('texture_bias_fraction', 0)
    ]
    colors = ['#1f77b4', '#ff7f0e']
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=colors)
    plt.ylim(0,1)
    plt.ylabel('Fraction')
    plt.title('Shape vs. Texture Bias')
    for idx, val in enumerate(values):
        plt.text(idx, val + 0.02, f"{val:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_invariance_results(
    results: Dict[str, float],
    test_type: str,
    save_path: Optional[str] = None
) -> None:
    """
    Plots accuracy over transformation levels for invariance testing.

    Args:
        results (Dict[str, float]): Dict mapping transformation levels to accuracies.
        test_type (str): 'scale', 'shift', or 'resolution'.
        save_path (Optional[str]): Path to save. If None, display.
    """
    levels = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(8, 6))
    plt.plot(levels, accuracies, marker='o', linewidth=2)
    plt.xlabel(f'{test_type.capitalize()} level')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Model Performance vs {test_type.capitalize()} Transformation')
    plt.grid(True)
    plt.ylim(0, 100)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\Beyond-INet\Beyond-INet_repo`
