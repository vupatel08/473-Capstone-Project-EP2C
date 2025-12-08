# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py

import os
import random
from typing import List, Tuple, Dict, Optional
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets

# We assume that the datasets are available locally or can be downloaded.
# For datasets not supported by torchvision, custom loaders will be implemented here.

class GenericImageFolder(Dataset):
    """
    Generic dataset loader based on directory structure.
    Assumes images are stored in:
        root/class_x/xxx.png
        root/class_x/xxy.png
        ...
    """
    def __init__(self, root_dir: str, classes: Optional[List[str]] = None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List of (image_path, label)
        self.class_name_to_idx = {}
        self.idx_to_class_name = {}
        self.classes = classes  # Optional: subset of classes to use

        # Load class directories
        all_classes = sorted(os.listdir(root_dir))
        if classes is not None:
            all_classes = [c for c in all_classes if c in classes]
        self.class_name_to_idx = {c: i for i, c in enumerate(all_classes)}
        self.idx_to_class_name = {i: c for c, i in self.class_name_to_idx.items()}

        for class_name in all_classes:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append((os.path.join(class_dir, fname), self.class_name_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class CustomPaintingsDataset(Dataset):
    """
    Placeholder for a custom paintings dataset loader.
    Assuming dataset in form of a CSV or annotations with image paths and labels.
    """
    def __init__(self, data_dir: str, split: str, transform=None):
        # Placeholder: Implement actual loading logic with annotations or directory structure
        # For now, assume images are in data_dir/split/class_name/*.jpg
        self.samples = []
        self.class_name_to_idx = {}
        self.idx_to_class_name = {}
        self.transform = transform

        split_dir = os.path.join(data_dir, split)
        all_classes = sorted(os.listdir(split_dir))
        self.class_name_to_idx = {c: i for i, c in enumerate(all_classes)}
        self.idx_to_class_name = {i: c for c, i in self.class_name_to_idx.items()}

        for class_name in all_classes:
            class_dir = os.path.join(split_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append((os.path.join(class_dir, fname), self.class_name_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class ChestXRayDataset(Dataset):
    """
    Placeholder for medical ChestX-ray dataset loader.
    Assumes images and labels are stored similarly.
    """
    def __init__(self, data_dir: str, split: str, transform=None):
        # Implement actual dataset loading here
        # For now, mimic similar structure to above
        self.samples = []
        self.class_name_to_idx = {}
        self.idx_to_class_name = {}
        self.transform = transform

        split_dir = os.path.join(data_dir, split)
        all_classes = sorted(os.listdir(split_dir))
        self.class_name_to_idx = {c: i for i, c in enumerate(all_classes)}
        self.idx_to_class_name = {i: c for c, i in self.class_name_to_idx.items()}

        for class_name in all_classes:
            class_dir = os.path.join(split_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, fname), self.class_name_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class DatasetWrapper:
    """
    Wrapper for datasets to handle dataset-specific loading and sampling.
    """
    def __init__(self, name: str, split: str, base_path: str, transform):
        """
        Initialize dataset based on name.
        """
        self.name = name
        self.split = split
        self.base_path = base_path
        self.transform = transform
        self.dataset_obj = None
        self.class_to_indices = {}  # class_idx: list of indices in dataset

        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset according to its name.
        """
        if self.name.lower() in ['mini-imagenet', 'tiered-imagenet', 'imagenet', 'cifar-fs', 'cifar10', 'cifar100', 'pascal voc', 'paintings', 'cub', 'aircraft', 'chestx']:
            # Use generic loader by directory structure
            self.dataset_obj = GenericImageFolder(self.base_path, transform=self.transform)
        else:
            # For unsupported datasets, raise error for now
            raise ValueError(f"Dataset {self.name} not supported for automatic loading.")
        # Build class to indices mapping
        self._build_class_to_indices()

    def _build_class_to_indices(self):
        """
        For the loaded dataset, build map from class labels to list of indices.
        """
        self.class_to_indices = {}
        for idx in range(len(self.dataset_obj)):
            _, label = self.dataset_obj[idx]
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

    def get_available_classes(self) -> List[int]:
        """
        Return list of class labels available in this dataset.
        """
        return list(self.class_to_indices.keys())

    def sample_classes(self, num_classes: int) -> List[int]:
        """
        Randomly sample 'num_classes' classes from available classes.
        """
        available = self.get_available_classes()
        assert len(available) >= num_classes, \
            f"Not enough classes to sample {num_classes} classes, only {len(available)} available."
        selected = random.sample(available, num_classes)
        return selected

    def sample_images_from_class(self, class_label: int, num_images: int) -> List[Tuple[torch.Tensor, int]]:
        """
        Sample 'num_images' images from the specified class.
        Returns list of (image_tensor, label).
        """
        indices = self.class_to_indices[class_label]
        assert len(indices) >= num_images, \
            f"Not enough images in class {class_label}; requested {num_images}, available {len(indices)}."
        selected_indices = random.sample(indices, num_images)
        images_list = []
        for idx in selected_indices:
            img, lbl = self.dataset_obj[idx]
            images_list.append((img, lbl))
        return images_list

    def get_dataset(self):
        """
        Return the underlying dataset object (for potential data loader).
        """
        return self.dataset_obj


class DatasetLoader:
    """
    Core class to manage loading of multiple datasets and episodic sampling.
    """
    def __init__(self, config: dict):
        """
        Initialize dataset loader based on configuration.
        """
        self.datasets_config = config['dataset']['datasets']
        self.transform = self._build_transform()
        self.datasets: List[DatasetWrapper] = []

        # Set seed for reproducibility if needed
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize dataset objects
        for ds_conf in self.datasets_config:
            name = ds_conf['name']
            split = ds_conf['split']
            base_path = ds_conf.get('path', './')  # Default path if not specified
            # For demo purposes, treat all datasets as directory-based
            dataset_obj = DatasetWrapper(name, split, base_path, self.transform)
            self.datasets.append(dataset_obj)

    def _build_transform(self):
        """
        Build image transforms according to CLIP model's preprocessing.
        """
        # CLIP requires resize to 224, center crop, normalize
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def load_data(self):
        """
        Placeholder for compatibility; datasets are loaded in init,
        so nothing more is needed here.
        """
        pass

    def sample_task(self, way: int, shot: int) -> Dict:
        """
        Sample one task: select one dataset and sample 'way' classes,
        with 'shot' support images per class, and query images from remaining.
        Returns a dict with support images/labels and query images/labels.
        """
        # Select a dataset at random
        dataset = random.choice(self.datasets)

        available_classes = dataset.get_available_classes()
        # Ensure enough classes
        assert len(available_classes) >= way, \
            f"Not enough classes in dataset {dataset.name} to sample {way} classes."

        # Sample classes
        selected_classes = random.sample(available_classes, way)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for class_idx, class_label in enumerate(selected_classes):
            # Sample support images
            support_samples = dataset.sample_images_from_class(class_label, shot)
            for img, lbl in support_samples:
                support_images.append(img)
                support_labels.append(class_idx)  # relabel classes 0..way-1

            # For query, sample a fixed number of images, e.g., 15
            # Here, for simplicity, use same number as support
            query_samples = dataset.sample_images_from_class(class_label, max(1, 15))
            # Remove support images to avoid duplication
            # But since we sample randomly, duplicates are unlikely; otherwise, handle explicitly
            for img, lbl in query_samples:
                query_images.append(img)
                query_labels.append(class_idx)

        # Convert lists to tensors
        support_images = support_images
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_images = query_images
        query_labels = torch.tensor(query_labels, dtype=torch.long)

        # Return as a dict
        return {
            'support_images': support_images,
            'support_labels': support_labels,
            'query_images': query_images,
            'query_labels': query_labels,
            'dataset_name': dataset.name,
            'class_mapping': selected_classes
        }
```

## evaluation.py

```python
## evaluation.py
import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm

class Evaluation:
    """
    Class Purpose:
        Evaluate the trained CAML model's zero-shot / few-shot performance across various datasets
        in the universal meta-learning setting. Performs standard accuracy metrics and tests permutation invariance.
    """
    def __init__(self, model: "Model", dataset_loader: "DatasetLoader", config: Dict[str, Any]):
        """
        Initialize Evaluation with model, dataset_loader, and config.
        Args:
            model (Model): The trained model with frozen backbone, label embeddings, transformer.
            dataset_loader (DatasetLoader): Loader providing episodic sampling functions.
            config (dict): YAML parsed configurations with evaluation parameters.
        """
        self.model = model
        self.dataset_loader = dataset_loader
        self.config = config
        # Evaluation parameters
        self.eval_episodes = self.config.get('evaluation', {}).get('episodes', 1000)
        self.support_shot = self.config.get('evaluation', {}).get('support_shot', 5)
        self.way = self.config.get('evaluation', {}).get('way', 5)
        self.permutation_test_episodes = self.config.get('evaluation', {}).get('permutation_test_episodes', 1000)
        self.datasets_list = self.config.get('evaluation', {}).get('datasets', [])
        # For reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # To store metrics
        self.accuracy_per_dataset = {}  # {dataset_name: {'mean': ..., 'std': ...}}
        self.permutation_invariance_stats = {}  # {dataset_name: {...}}

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate_on_dataset(self):
        """
        Main method to perform evaluation across all datasets listed in config.
        Returns:
            metrics: Dictionary with accuracies per dataset.
        """
        overall_results = {}
        print("Starting Dataset Evaluation...")
        for dataset_conf in self.datasets_list:
            dataset_name = dataset_conf['name']
            print(f"Evaluating dataset: {dataset_name}")
            accuracies = []
            for episode_idx in tqdm(range(self.eval_episodes), desc=f"Eval {dataset_name}"):
                # Sample episodic task
                task = self.dataset_loader.sample_task(self.way, self.support_shot)
                support_images = task['support_images']
                support_labels = task['support_labels']
                query_images = task['query_images']
                query_labels = task['query_labels']
                # Perform prediction
                pred_class_idx = self._predict_support_query(support_images, support_labels, query_images[0])
                true_class_idx = query_labels[0].item()  # assuming 1 query per episode
                correctness = (pred_class_idx == true_class_idx)
                accuracies.append(correctness)
            mean_acc = np.mean(accuracies) * 100.0 # percentage
            std_acc = np.std(accuracies) * 100.0
            overall_results[dataset_name] = {'accuracy': mean_acc, 'std': std_acc}
            print(f"{dataset_name} Accuracy: {mean_acc:.2f} ± {std_acc:.2f}")
            self.accuracy_per_dataset[dataset_name] = overall_results[dataset_name]
        return overall_results

    def _predict_support_query(self, support_images: List, support_labels: torch.Tensor, query_image):
        """
        Performs support-query prediction.
        Args:
            support_images: list of image tensors
            support_labels: tensor of class indices (relabeled support set)
            query_image: single image tensor
        Returns:
            predicted class index (int)
        """
        # Support labels are relabeled 0..way-1
        pred_idx = self.model.forward(support_images, support_labels, query_image)
        return pred_idx

    def test_permutation_invariance(self, support_images: List, support_labels: torch.Tensor, query_image):
        """
        Test if the model's prediction is invariant to permutations of support set order.
        Perform multiple permutations, record class predictions.
        Args:
            support_images: list of support set images
            support_labels: tensor of support labels
            query_image: tensor of query image
        Returns:
            permutation_results: dict containing distribution and stability metrics
        """
        num_permutations = self.permutation_test_episodes
        pred_classes = []
        class_prob_distributions = []

        support_indices = list(range(len(support_images)))
        # Store min-max class probability for stability measurement
        class_probs_list = []

        for _ in range(num_permutations):
            perm = random.sample(support_indices, len(support_indices))
            perm_support_images = [support_images[i] for i in perm]
            perm_support_labels = support_labels[perm]
            # Predict
            pred_idx = self.model.forward(perm_support_images, perm_support_labels, query_image)
            pred_classes.append(pred_idx)

        # Count most common predictions
        from collections import Counter
        class_counts = Counter(pred_classes)
        most_common_class, count = class_counts.most_common(1)[0]
        # Compute consistency
        consistency_ratio = count / num_permutations

        # Optional: compute standard deviation of predicted class probabilities if model supports it
        # here, we only have class predictions; for probability stability, need per-permutation probs
        # For demonstration, assume majority-vote suffices

        # Histogram or distribution info
        hist = dict(class_counts)
        # For visual similarity to Figure 5 (left), we can prepare histogram data
        permutation_results = {
            'distribution': hist,
            'most_common_class': most_common_class,
            'consistency_ratio': consistency_ratio
        }
        return permutation_results

    def run_full_evaluation(self):
        """
        Run evaluation and permutation invariance test, print or return detailed results.
        """
        results = {}
        for dataset_conf in self.datasets_list:
            dataset_name = dataset_conf['name']
            print(f"Evaluation for dataset: {dataset_name}")
            # Sample a typical task for permutation test
            task = self.dataset_loader.sample_task(self.way, self.support_shot)
            support_images = task['support_images']
            support_labels = task['support_labels']
            query_image = task['query_images'][0]
            # Run baseline accuracy
            _ = self.evaluate_on_dataset()
            # Run permutation invariance test
            perm_results = self.test_permutation_invariance(support_images, support_labels, query_image)
            self.permutation_invariance_stats[dataset_name] = perm_results
            print(f"Permutation invariance distribution: {perm_results['distribution']}")
            print(f"Most consistent class: {perm_results['most_common_class']} with {perm_results['consistency_ratio']*100:.2f}% consistency")
        return self.permutation_invariance_stats

```

## main.py

```python
# main.py
import os
import yaml
import torch
import random
import numpy as np

from dataset_loader import DatasetLoader
from model import CAMLModel
from trainer import Trainer
from evaluation import Evaluation

def main():
    # 1. Load configuration from 'config.yaml'
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. Fix random seeds for reproducibility
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)

    # 4. Initialize DatasetLoader
    dataset_loader = DatasetLoader(config)
    dataset_loader.load_data()

    # 5. Instantiate Model with config
    model = CAMLModel(config['model'])
    model.to(device)

    # 6. Prepare optimizer and scheduler
    # Only train parameters that require gradients (transformer + label embeddings)
    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(train_params, lr=config['training']['learning_rate'])
    
    total_steps = config['training']['total_steps']
    warmup_steps = config['training']['warmup_steps']
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return max(cosine_decay, 1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 7. Initialize Trainer
    trainer = Trainer(model, dataset_loader, config)
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler

    # 8. Pretrain large-scale on datasets
    print("Starting large-scale pretraining...")
    trainer.pretrain()
    print("Pretraining completed.")

    # 9. Initialize Evaluation
    evaluator = Evaluation(model, dataset_loader, config)

    # 10. Run evaluation on datasets
    print("Starting evaluation across datasets...")
    eval_results = evaluator.evaluate_on_dataset()
    print("Evaluation results:")
    for ds_name, metrics in eval_results.items():
        print(f"{ds_name}: Accuracy = {metrics['accuracy']:.2f} ± {metrics['std']:.2f}")

    # 11. Perform permutation invariance tests for a representative task from each dataset
    print("\nTesting permutation invariance...")
    perm_stats = evaluator.run_full_evaluation()
    for ds_name, stats in perm_stats.items():
        print(f"{ds_name} permutation test most common class: {stats['most_common_class']} "
              f"with {stats['consistency_ratio']*100:.2f}% consistency")
        print(f"Distribution: {stats['distribution']}\n")

    # 12. Optionally, save the trained model
    # torch.save(model.state_dict(), 'caml_trained_model.pth')

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, ViTModel, ViTConfig
import timm

class CAMLModel(nn.Module):
    def __init__(self, config: dict):
        """
        Initializes the CAML model components:
        - Frozen CLIP image encoder
        - Learnable label (ELMES) embeddings
        - Non-causal transformer sequence model
        """
        super().__init__()
        # Load and freeze CLIP encoder
        self.clip_model = CLIPModel.from_pretrained(config['model']['image_encoder'])
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # CLIP's image encoder outputs last_hidden_state or pooler_output
        # Use pooled output or mean pooling
        # The embedding size (hidden_dim) depends on the CLIP version
        self.clip_embedding_dim = self.clip_model.config.hidden_size  # e.g., 512 or 768

        # Label (ELMES) embeddings: initialize as trainable parameters
        self.label_embedding_dim = config['model']['label_embedding_dim']
        # For maximum class number during training
        self.max_classes = 100  # Can be adjusted or set dynamically
        self.label_embeddings = nn.Embedding(self.max_classes, self.label_embedding_dim)

        # Initialize label embeddings with random uniform
        nn.init.uniform_(self.label_embeddings.weight, -0.1, 0.1)

        # Build the transformer encoder for sequence modeling
        # Using vision transformer as backbone - huggingface ViT or timm
        transformer_name = config['model']['transformer_model_name']
        transformer_params = config['model']['transformer_params']
        # For flexibility, use timm's ViT implementation
        # Note: Adjust input embedding size to match combined support image + label embedding
        self.transformer = timm.create_model(
            transformer_name,
            pretrained=True,
            num_classes=0,
            global_pool='mean'
        )
        # Replace patch embedding or modify as needed to accept sequence of token embeddings
        # Here, since encoding support/query as sequence of feature vectors, better to implement custom transformer
        # For simplicity, define a standard nn.TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_params['hidden_dim'],
            nhead=transformer_params['num_heads'],
            dropout=transformer_params['dropout']
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_params['num_layers']
        )

        # Positional embeddings: learned parameters
        # Max sequence length: support set size + 1 (query)
        self.max_seq_len = 64  # Can be adjusted based on max support size
        self.positional_embeddings = nn.Parameter(
            torch.randn(self.max_seq_len, transformer_params['hidden_dim'])
        )

        # Hidden dimension matches the transformer's d_model
        self.hidden_dim = transformer_params['hidden_dim']
        # A final MLP to produce output if needed (optional)
        # Not necessary unless doing further classification step
        # For CAML: use inner product similarity directly

        # Initialize a special "unknown" label embedding for unknown class
        self.unknown_label_embedding = nn.Parameter(torch.randn(self.label_embedding_dim))

    def encode_image(self, images):
        """
        Encode images using frozen CLIP encoder.
        Args:
            images: list or tensor of images (PIL or tensors)
        Returns:
            Tensor of shape [batch_size, clip_embedding_dim]
        """
        # Expect images as tensor or list of PIL Images
        # Convert to feature vectors via CLIP
        with torch.no_grad():
            # If images are PIL Images, convert to tensor and normalize
            # Here, assume images are already preprocessed tensors
            # If not, preprocessing should be added outside
            outputs = self.clip_model.get_image_features(images)
        # outputs shape: [batch_size, clip_embedding_dim]
        return outputs

    def prepare_sequence(self, support_images, support_labels, query_image, support_labels_in_training=None):
        """
        Construct the input sequence for transformer.
        Support images and labels are converted to tokens and concatenated with query.
        Args:
            support_images: list or tensor of support images
            support_labels: tensor of shape [support_size], class indices
            query_image: single image tensor
        Returns:
            sequence: tensor of shape [sequence_length, embed_dim]
            support_label_indices: tensor of shape [support_size]
            support_class_indices: list of class indices in support set (for mapping)
        """
        # Encode support images
        support_embeddings = self.encode_image(support_images)  # shape: [support_size, clip_dim]
        query_embedding = self.encode_image(query_image.unsqueeze(0)).squeeze(0)  # shape: [clip_dim]

        support_size = support_embeddings.shape[0]
        support_class_indices = support_labels  # used to index label embeddings

        # Map support labels to label embeddings (support support_labels help index)
        # support_labels are class indices in support set, map to our label embeddings
        support_label_embeddings = self.label_embeddings(support_class_indices)  # [support_size, label_dim]

        # Concatenate support image embedding + label embedding for each support
        support_tokens = torch.cat([support_embeddings, support_label_embeddings], dim=1)  # [support_size, support_dim]
        # For the support tokens, project to the sequence feature size if needed (assumed compatible)
        # Assume clip_dim and label_dim are the same or adjust
        support_tokens_seq = support_tokens  # shape: [support_size, embed_dim]

        # Create query token (just image embedding; label is unknown)
        # For the query, initialize a support_label_embedding placeholder (could use unknown token)
        # For current, we do not have label info; use unknown token if desired:
        query_token = torch.cat([query_embedding, self.unknown_label_embedding], dim=0)  # shape: [embed_dim]

        # Build sequence
        sequence = torch.cat([support_tokens_seq, query_token.unsqueeze(0)], dim=0)  # shape: [support_size+1, embed_dim]

        # Add positional embeddings
        seq_len = sequence.shape[0]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum supported {self.max_seq_len}")
        positional = self.positional_embeddings[:seq_len, :]  # [seq_len, embed_dim]
        sequence = sequence + positional

        return sequence, support_class_indices

    def forward(self, support_images, support_labels, query_image):
        """
        Complete forward pass:
        - build sequence
        - pass through transformer
        - extract query output
        - compute similarities with label embeddings
        - output predicted class index
        """
        # Prepare sequence
        sequence, support_class_indices = self.prepare_sequence(support_images, support_labels, query_image)

        # Transformer expects [sequence_len, batch=1, embed_dim]
        sequence = sequence.unsqueeze(1)  # [seq_len, 1, embed_dim]

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(sequence)  # [seq_len, 1, embed_dim]

        # Extract last token (query output)
        query_output = transformer_output[-1, 0, :]  # [embed_dim]

        # Compare query output to label embeddings
        # Get all class label embeddings
        # During inference/support, number of classes may be less than max_classes
        # So, for predicted, compute similarity with all label embeddings
        class_embeds = self.label_embeddings.weight[:len(support_class_indices)]  # [support_size, label_dim]
        # Compute inner products
        similarities = torch.matmul(class_embeds, query_output)  # [support_size]
        # For classification, aggregate over support set or pick maximum
        # Here, assume the support set labels are for classes; but in practice, classify among all classes
        # For generalization, compare query to all label embeddings
        # So, extend class_embeds to total number of classes (if known), or use a fixed large number
        # For simplicity, assume only support classes are in consideration
        probs = torch.softmax(similarities, dim=0)
        pred_class_idx = torch.argmax(probs).item()
        return pred_class_idx
```


## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
from typing import Dict, Any
from tqdm import tqdm
from itertools import islice

class Trainer:
    def __init__(self, model: nn.Module, dataset_loader: 'DatasetLoader', config: Dict[str, Any]):
        """
        Initialize the Trainer with model, dataset_loader, and training configuration.
        Args:
            model (nn.Module): The CAML model instance.
            dataset_loader (DatasetLoader): The dataset loader instance.
            config (dict): Configuration dictionary loaded from YAML.
        """
        self.model = model
        self.dataset_loader = dataset_loader
        self.config = config

        # Hyperparameters from config with defaults
        self.learning_rate = self.config.get('training', {}).get('learning_rate', 1e-5)
        self.warmup_steps = self.config.get('training', {}).get('warmup_steps', 9600)
        self.total_steps = self.config.get('training', {}).get('total_steps', 400000)
        self.support_shot = self.config.get('training', {}).get('support_shot', 5)
        self.way = self.config.get('training', {}).get('way', 5)
        self.batch_size = self.config.get('training', {}).get('batch_size', 525)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Prepare optimizer
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        # Learning rate scheduler with warmup and cosine decay
        self.scheduler = self._build_scheduler()

        # Initialize training state
        self.global_step = 0
        self.epoch = 0

        # Move model to device
        self.model.to(self.device)

    def _build_scheduler(self):
        """
        Builds a learning rate scheduler with linear warmup and cosine decay.
        """
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(cosine_decay, 1e-6 / self.learning_rate)  # prevent LR from going to zero
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def pretrain(self):
        """
        Pre-train the model over large datasets, across multiple datasets, in an episodic manner.
        """
        progress_bar = tqdm(total=self.total_steps, desc='Pretraining', unit='step')
        for step in range(1, self.total_steps + 1):
            self.global_step = step
            # Update learning rate
            self.scheduler.step()

            # Sample a dataset and task
            dataset_obj = random.choice(self.dataset_loader.datasets)
            task = dataset_obj.get_dataset().sample_task(self.way, self.support_shot)
            # task is a dict: support_images, support_labels, query_images, query_labels, dataset_name, class_mapping

            loss, acc = self.train_episode(task)

            # Optional: print logs every N steps
            if step % 100 == 0:
                progress_bar.set_postfix(loss=loss.item(), accuracy=acc, lr=self.scheduler.get_last_lr()[0])
                progress_bar.update(100)
            else:
                progress_bar.update(1)

            # Break early if needed
            if step >= self.total_steps:
                break
        progress_bar.close()

    def train_episode(self, task: Dict[str, Any]):
        """
        For a single episodic task:
            - construct support sequence
            - encode support/images
            - encode query image
            - form sequence input
            - forward pass through model
            - compute loss
            - backpropagate and update
        """
        self.model.train()

        # Support set data
        support_images = task['support_images']
        support_labels = task['support_labels']
        query_images = task['query_images']
        query_labels = task['query_labels']

        # Move support images to device
        support_images = [img.to(self.model.device) if hasattr(img, 'to') else img for img in support_images]
        query_images = [img.to(self.model.device) if hasattr(img, 'to') else img for img in query_images]
        support_labels = support_labels.to(self.model.device)
        query_labels = query_labels.to(self.model.device)

        # Encode support images
        support_embeddings = self.model.encode_image(support_images)  # shape: [support_size, embed_dim]
        # Encode query image
        query_embedding = self.model.encode_image(query_images)  # shape: [1, embed_dim]
        query_embedding = query_embedding.squeeze(0)  # shape: [embed_dim]

        # Map support labels to label embeddings (support_labels are class indices)
        support_label_embeddings = self.model.label_embeddings(support_labels)  # shape: [support_size, label_dim]

        support_size = support_embeddings.shape[0]
        # Support sequence: support image embedding + label embedding per example
        support_sequence = torch.cat([support_embeddings, support_label_embeddings], dim=1)  # shape: [support_size, combined_dim]
        # For simplicity, assume support_image embedding dims match label dims, or perform projection if needed
        # Here, we assume they are compatible or the model handles it internally

        # Construct sequence: all support tokens + query token
        # For the query, use support label embedding placeholder (unknown token)
        query_token = torch.cat([query_embedding, self.model.unknown_label_embedding], dim=0)  # shape: [embed_dim]
        # In practice, to match dimensions, support tokens may need projection
        # For now, assume support_token shape: [support_size, embed_dim], append query as last token
        sequence = torch.cat([support_sequence, query_token.unsqueeze(0)], dim=0)  # shape: [support_size+1, embed_dim]

        # Add positional encodings
        seq_len = sequence.shape[0]
        if seq_len > self.model.max_seq_len:
            # For extremely long sequences (unlikely in support+query), truncate or error
            sequence = sequence[:self.model.max_seq_len, :]
            seq_len = self.model.max_seq_len
        positional = self.model.positional_embeddings[:seq_len, :]
        sequence = sequence + positional

        # Pass through transformer
        # Expect shape: [sequence_length, batch_size=1, embed_dim]
        sequence_input = sequence.unsqueeze(1)  # add batch dimension
        transformer_output = self.model.transformer_encoder(sequence_input)  # shape: [seq_len, 1, embed_dim]

        # Extract query token output: last token
        query_output = transformer_output[-1, 0, :]  # shape: [embed_dim]

        # Compute similarities with label embeddings
        class_embeds = self.model.label_embeddings.weight[:self.way]  # only support classes
        similarities = torch.matmul(class_embeds, query_output)  # shape: [way]

        # Convert similarities to predicted class probabilities
        logits = similarities / 1.0  # optional temperature scaling
        probs = torch.softmax(logits, dim=0)

        # Loss: cross-entropy between probs and true query label
        loss_fn = nn.CrossEntropyLoss()
        # True label is the index in support_labels - the class index assigned in support set
        # Assume support_labels are relabeled 0..way-1
        target = torch.tensor([query_labels[0]]).to(self.model.device)  # Shape: [1]
        # But in the support_labels, class indices are already relabeled
        # For multi-query setting, more complex; here, assume 1 query per episode
        loss = loss_fn(logits.unsqueeze(0), target)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute accuracy
        pred_idx = torch.argmax(probs).item()
        correct = (pred_idx == query_labels[0].item())

        return loss.item(), correct
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\CAML\CAML_repo`
