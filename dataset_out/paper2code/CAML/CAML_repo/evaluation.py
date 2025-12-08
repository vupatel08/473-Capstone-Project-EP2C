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

