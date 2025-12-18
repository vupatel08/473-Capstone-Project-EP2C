# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import random
import numpy as np
from typing import Tuple, List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

class PoisonedDataset(Dataset):
    """
    Custom Dataset class that wraps existing datasets, applies poisoning,
    and keeps track of poisoned/noisy labels for evaluation.
    """
    def __init__(
        self,
        images: List[Image.Image],
        labels: List[int],
        poisoned_indices: Optional[set] = None,
        noisy_indices: Optional[set] = None,
        true_labels: Optional[List[int]] = None,
        dataset_name: str = ""
    ):
        self.images = images
        self.labels = labels
        self.poisoned_indices = poisoned_indices or set()
        self.noisy_indices = noisy_indices or set()
        self.true_labels = true_labels  # for noisy labels, ground truth for evaluation
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], idx

class DatasetLoader:
    """
    Responsible for loading datasets, applying poisoning or noisy labels,
    and providing train/test splits.
    """
    def __init__(self, config: dict):
        self.config = config
        self.dataset_name = None
        self.train_data = None
        self.test_data = None
        self.poisoned_indices = set()
        self.noisy_indices = set()

        # Set seed for reproducibility
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Map dataset name to handler
        self._dataset_map = {
            'cifar10': self._load_cifar10,
            'imagenet100': self._load_imagenet100,
            'imagenet_dog': self._load_imagenet_dog
        }

    def load_data(self, dataset_key: str) -> Tuple[Dataset, Dataset]:
        """
        Load dataset according to key, apply poisoning/noise, and return train/test datasets.
        """
        if dataset_key not in self._dataset_map:
            raise ValueError(f"Dataset {dataset_key} not supported.")

        self.dataset_name = dataset_key
        load_func = self._dataset_map[dataset_key]
        train_dataset, test_dataset = load_func()

        # Apply poisoning if requested
        if self.config['attack'].get('poisoned', False):
            train_dataset, poisoned_idx = self._apply_poisoning(train_dataset)
            self.poisoned_indices.update(poisoned_idx)

        # Apply noisy labels if requested
        if self.config['attack'].get('noisy_labels', {}).get('ratio', 0) > 0:
            noise_type = self.config['attack']['noisy_labels'].get('type', 'symmetric')
            noise_ratio = self.config['attack']['noisy_labels'].get('ratio', 0)
            train_dataset, noisy_idx, true_labels = self._apply_noisy_labels(
                train_dataset, noise_type, noise_ratio
            )
            self.noisy_indices.update(noisy_idx)
            # Store true labels for evaluation if needed
            train_dataset.true_labels = true_labels

        return train_dataset, test_dataset

    def _load_cifar10(self) -> Tuple[Dataset, Dataset]:
        """Load CIFAR-10 dataset with transformations."""
        image_size = self.config['dataset']['cifar10']['image_size']
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # Convert to list of images and labels for convenience
        train_images = [Image.fromarray(train_set.data[i]) for i in range(len(train_set))]
        train_labels = [train_set.targets[i] for i in range(len(train_set))]

        test_images = [Image.fromarray(test_set.data[i]) for i in range(len(test_set))]
        test_labels = [test_set.targets[i] for i in range(len(test_set))]

        return (train_images, train_labels), (test_images, test_labels)

    def _load_imagenet100(self) -> Tuple[Dataset, Dataset]:
        """
        Load ImageNet-100 dataset.
        Assumes dataset is stored in a folder structure:
        root/
            class1/
                img1.jpg, img2.jpg, ...
            class2/
                ...
        """
        root_dir = self.config.get('dataset_path', './imagenet100/')
        image_size = self.config['dataset']['imagenet100']['image_size']
        dataset_dir = root_dir
        class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        class_dirs.sort()
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        images = []
        labels = []

        for cls_name in class_dirs:
            cls_dir = os.path.join(dataset_dir, cls_name)
            img_files = [f for f in os.listdir(cls_dir) if f.endswith('.jpg') or f.endswith('.png')]
            for f in img_files:
                img_path = os.path.join(cls_dir, f)
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                labels.append(class_to_idx[cls_name])

        # Resize images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        images = [transform(img) for img in images]

        # Split into train/test based on ratio
        total_samples = len(images)
        train_ratio = 0.8
        indices = list(range(total_samples))
        random.shuffle(indices)
        split_idx = int(total_samples * train_ratio)

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_images = [images[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_images = [images[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]

        return (train_images, train_labels), (test_images, test_labels)

    def _load_imagenet_dog(self) -> Tuple[Dataset, Dataset]:
        """
        Load ImageNet-Dog dataset.
        Assumes a similar folder structure as above with dog classes.
        """
        root_dir = self.config.get('dataset_path', './imagenet_dog/')
        image_size = self.config['dataset']['imagenet_dog']['image_size']
        class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        class_dirs.sort()
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        images = []
        labels = []

        for cls_name in class_dirs:
            cls_dir = os.path.join(root_dir, cls_name)
            img_files = [f for f in os.listdir(cls_dir) if f.endswith('.jpg') or f.endswith('.png')]
            for f in img_files:
                img_path = os.path.join(cls_dir, f)
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                labels.append(class_to_idx[cls_name])

        # Resize images
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        images = [transform(img) for img in images]

        # Split into train/test
        total_samples = len(images)
        train_ratio = 0.8
        indices = list(range(total_samples))
        random.shuffle(indices)
        split_idx = int(total_samples * train_ratio)

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_images = [images[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_images = [images[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]

        return (train_images, train_labels), (test_images, test_labels)

    def _apply_poisoning(self, dataset: Tuple[Tuple[List[Image.Image], List[int]]]) -> Tuple:
        """
        Apply triggers embedding to a subset of dataset based on ratio.
        Change labels to target label.
        """
        images, labels = dataset
        total_samples = len(images)
        poison_ratio = self.config['attack'].get('poison_ratio', 0.1)
        target_label = self.config['attack'].get('target_label', 0)
        num_poison = int(total_samples * poison_ratio)
        all_indices = list(range(total_samples))
        random.shuffle(all_indices)
        poison_indices = set(all_indices[:num_poison])

        poisoned_images = []
        poisoned_labels = []
        for idx in range(total_samples):
            img = images[idx]
            lbl = labels[idx]
            if idx in poison_indices:
                # Embed trigger
                img_poisoned = self._embed_trigger(img, method=self.config['attack'].get('embed_method'))
                poisoned_images.append(img_poisoned)
                # Change label to target
                poisoned_labels.append(target_label)
            else:
                poisoned_images.append(img)
                poisoned_labels.append(lbl)

        poisoned_dataset = (poisoned_images, poisoned_labels)
        return poisoned_dataset, poison_indices

    def _embed_trigger(self, image: Image.Image, method: str = "overlay") -> Image.Image:
        """
        Embed trigger into the image based on specified method.
        For simplicity, implement overlay patch at bottom right corner.
        """
        img = image.copy()
        trigger_size = 4  # in pixels, can be configured
        overlay_color = (255, 255, 255)  # white patch
        # For more complex methods, extend here
        # Example: overlay white square at bottom right
        width, height = img.size
        for i in range(width - trigger_size, width):
            for j in range(height - trigger_size, height):
                img.putpixel((i, j), overlay_color)
        return img

    def _apply_noisy_labels(self, dataset: Tuple, noise_type: str, noise_ratio: float) -> Tuple:
        """
        Corrupt labels according to specified noise type and ratio.
        Return dataset with possibly noisy labels and ground truth labels for evaluation.
        """
        images, labels = dataset
        total_samples = len(labels)
        num_noisy = int(total_samples * noise_ratio)
        all_indices = list(range(total_samples))
        random.shuffle(all_indices)
        noisy_indices = set(all_indices[:num_noisy])
        true_labels = labels.copy()

        new_labels = labels.copy()
        num_classes = self._get_num_classes()
        for idx in noisy_indices:
            true_label = labels[idx]
            if noise_type == 'symmetric':
                # Flip to random label different from original
                new_label = true_label
                while new_label == true_label:
                    new_label = random.randint(0, num_classes - 1)
                new_labels[idx] = new_label
            elif noise_type == 'asymmetric':
                # Flip to next class
                new_label = (true_label + 1) % num_classes
                new_labels[idx] = new_label
            # else, can add other strategies
        corrupted_dataset = (images, new_labels)
        return corrupted_dataset, noisy_indices, true_labels

    def _get_num_classes(self) -> int:
        """Return number of classes based on dataset."""
        if self.dataset_name == 'cifar10':
            return 10
        elif self.dataset_name == 'imagenet100':
            return 100
        elif self.dataset_name == 'imagenet_dog':
            return 10
        else:
            return 0
```

## evaluation.py

```python
## evaluation.py
from typing import List, Tuple
import numpy as np

class Evaluator:
    """
    Implements the evaluation, scoring, and detection logic for the VDC framework.
    It aggregates question-level correctness indicators per sample, applies threshold-based classification,
    and computes detection metrics such as TPR and FPR.
    """

    def __init__(self, config: dict):
        """
        Initializes the Evaluator with configuration parameters.
        Args:
            config (dict): Configuration dictionary, expecting at least:
                - 'detection': {
                    'threshold': float, detection threshold alpha
                }
                - 'use_similarity_score': bool, whether to incorporate similarity scores
                - 'similarity_weight': float, weight of similarity in combined score if used
                - optional: other scoring hyperparameters
        """
        self.threshold: float = config.get("detection", {}).get("threshold", 0.2)
        # Decide whether to incorporate similarity score
        self.use_similarity_score: bool = config.get("use_similarity_score", True)
        # Weight for similarity component if used
        self.similarity_weight: float = config.get("similarity_weight", 0.5)

    def compute_sample_score(
        self,
        correctness_indicators: List[bool],
        similarity_score: float = 0.0
    ) -> float:
        """
        Compute the aggregated sample score based on question correctness and optional similarity.
        Args:
            correctness_indicators (List[bool]): List of per-question correctness flags.
            similarity_score (float): Similarity score between image and label (if used).
        Returns:
            float: final score for the sample (higher indicates more likely clean).
        """
        correct_count = sum(correctness_indicators)
        total_q = len(correctness_indicators)
        correctness_ratio = correct_count / total_q if total_q > 0 else 0.0

        if self.use_similarity_score:
            # Combine correctness ratio and similarity with weights
            combined_score = (
                (1 - self.similarity_weight) * correctness_ratio
                + self.similarity_weight * similarity_score
            )
            return combined_score
        else:
            return correctness_ratio

    def classify_sample(self, sample_score: float) -> bool:
        """
        Determine if the sample is dirty based on the score and threshold.
        Args:
            sample_score (float): Aggregated score for the sample.
        Returns:
            bool: True if sample is classified as dirty, False otherwise.
        """
        return sample_score < self.threshold

    def evaluate_samples(
        self,
        correctness_list: List[List[bool]],
        similarity_list: List[float],
        ground_truth_dirty_flags: List[bool]
    ) -> dict:
        """
        Evaluate the dataset: compute per-sample scores, classify, and compute metrics.
        Args:
            correctness_list (List[List[bool]]): List of correctness flags per sample.
            similarity_list (List[float]): List of similarity scores per sample.
            ground_truth_dirty_flags (List[bool]): List indicating true dirty samples.
        Returns:
            dict: metrics and per-sample results containing:
                - 'scores': list of float, sample scores
                - 'pred_labels': list of bool, predicted dirty (True) or clean (False)
                - 'TPR': float, true positive rate
                - 'FPR': float, false positive rate
        """
        num_samples = len(correctness_list)
        scores = []
        pred_labels = []

        # Thresholding and detection
        for i in range(num_samples):
            correctness_indicators = correctness_list[i]
            sim_score = similarity_list[i]
            sample_score = self.compute_sample_score(correctness_indicators, sim_score)
            scores.append(sample_score)
            is_dirty_pred = self.classify_sample(sample_score)
            pred_labels.append(is_dirty_pred)

        # Compute TPR and FPR
        true_positives = sum(p and gt for p, gt in zip(pred_labels, ground_truth_dirty_flags))
        false_positives = sum(p and not gt for p, gt in zip(pred_labels, ground_truth_dirty_flags))
        total_dirty = sum(ground_truth_dirty_flags)
        total_clean = num_samples - total_dirty

        TPR = true_positives / total_dirty if total_dirty > 0 else 0.0
        FPR = false_positives / total_clean if total_clean > 0 else 0.0

        return {
            'scores': scores,
            'pred_labels': pred_labels,
            'TPR': TPR,
            'FPR': FPR
        }
```

## main.py

```python
## main.py
import yaml
import os
import time
import logging
from tqdm import tqdm

# Import custom modules as per project structure
import utils
from dataset_loader import DatasetLoader
from prompt_generator import PromptGenerator
from model_inference import ModelInference
from evaluation import Evaluator

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. Load configuration
    config_path = 'config.yaml'
    config = utils.load_config(config_path)

    # 2. Initialize components
    dataset_loader = DatasetLoader(config)
    prompt_gen = PromptGenerator()
    model_inf = ModelInference(config)
    evaluator = Evaluator(config)

    # 3. Load dataset (supporting CIFAR-10, ImageNet-100, ImageNet-Dog)
    dataset_name = list(config['dataset'].keys())[0]  # Assume single dataset in config
    dataset_path = config['dataset'].get('dataset_path', './')
    # Explicitly set dataset_path if needed in config
    dataset_loader.config['dataset_path'] = dataset_path

    logger.info(f"Loading dataset: {dataset_name}")
    train_data, test_data = dataset_loader.load_data(dataset_name)
    # Unpack train/test
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    # 4. Generate questions per sample
    questions_per_sample = config.get('detection', {}).get('questions_per_sample', 4)
    label_question_type = config.get('detection', {}).get('question_type', 'general and label-specific')
    detection_threshold = config.get('evaluation', {}).get('detection_threshold', 0.2)

    total_samples = len(train_images)
    sample_scores = []  # store final score for each sample
    sample_predictions = []  # store dirty (True) or clean (False)

    correctness_list_per_sample = []
    similarity_list_per_sample = []

    logger.info("Starting question answering and evaluation for each sample.")

    # Progress bar for large datasets
    for idx in tqdm(range(total_samples)):
        image = train_images[idx]
        label = train_labels[idx]

        # Generate questions
        questions = []
        # General questions
        questions.extend(prompt_gen.generate_general_questions(label, questions_per_sample))
        # Label-specific questions
        questions.extend(prompt_gen.generate_label_specific_questions(label, questions_per_sample))
        
        # 5. Answer questions using MLLM
        try:
            answers = model_inf.answer_questions('', questions)  # image path empty, handle internally
        except NotImplementedError:
            # For this example, if answer_questions is not implemented, skip detection
            logger.warning("ModelInference.answer_questions is not implemented. Exiting.")
            return
        except Exception as e:
            logger.warning(f"API error at sample {idx}: {e}")
            answers = [""] * len(questions)  # fallback empty answers

        correctness_list = []
        similarity_score = 0.0

        # 6. Evaluate correctness of each answer
        for q_idx, question in enumerate(questions):
            answer = answers[q_idx]
            try:
                is_correct, sim_score = model_inf.evaluate_response(question, answer, label)
            except Exception:
                # On failure, fallback to heuristic
                is_correct = answer.lower().find("yes") != -1
                sim_score = 0.0
            correctness_list.append(is_correct)
            # For simplicity, take the mean similarity of all questions or last one
            similarity_score = sim_score  # could be averaged over questions

        # 7. Aggregate score
        sample_score = evaluator.compute_sample_score(correctness_list, similarity_score)
        sample_scores.append(sample_score)

        # 8. Classify sample as dirty or clean
        is_dirty = evaluator.classify_sample(sample_score, threshold=detection_threshold)
        sample_predictions.append(is_dirty)

        # Save correctness and similarity for metrics
        correctness_list_per_sample.append(correctness_list)
        similarity_list_per_sample.append(similarity_score)

    # 9. Compute detection performance metrics
    true_dirty_flags = [False] * total_samples  # Placeholder: in real experiment, you'd have true labels for dirty/noisy
    # Assuming we might have known true labels for evaluation:
    # For the purpose of this code, we mark the poisoned/noisy samples as 'True' if identified.
    metrics = evaluator.evaluate_samples(correctness_list_per_sample, similarity_list_per_sample, true_dirty_flags)

    # 10. Log detection results
    print("Detection Results:")
    print(f"TPR: {metrics['TPR']*100:.2f}%, FPR: {metrics['FPR']*100:.2f}%")

    # 11. Filter dataset based on detection for retraining
    clean_indices = [i for i, pred in enumerate(sample_predictions) if not pred]
    clean_images = [train_images[i] for i in clean_indices]
    clean_labels = [train_labels[i] for i in clean_indices]

    # 12. Optional: retrain classifier on purified dataset
    retrain_model = True  # Set based on config
    if retrain_model:
        # Load classifier (ResNet-18)
        from torchvision import models
        import torch.optim as optim
        import torch.nn as nn

        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(set(train_labels)))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Prepare DataLoader
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((config['dataset'].get(f'imagenet_{dataset_name}', {}).get('image_size', 32))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
        # Create dataset
        class PurifiedDataset(torch.utils.data.Dataset):
            def __init__(self, imgs, lbls):
                self.imgs = imgs
                self.lbls = lbls
            def __len__(self):
                return len(self.lbls)
            def __getitem__(self, idx):
                return transform(self.imgs[idx]), self.lbls[idx]
        purified_dataset = PurifiedDataset(clean_images, clean_labels)
        train_loader = torch.utils.data.DataLoader(purified_dataset, batch_size=64, shuffle=True)

        # Train
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        epochs = config['training'].get('epochs', 40)
        for epoch in range(epochs):
            model.train()
            for batch_imgs, batch_lbls in train_loader:
                batch_imgs = batch_imgs.to(device)
                batch_lbls = batch_lbls.to(device)
                optimizer.zero_grad()
                outputs = model(batch_imgs)
                loss = criterion(outputs, batch_lbls)
                loss.backward()
                optimizer.step()
            # Optional: decay LR at epochs
            if epoch == epochs // 2:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
        # Evaluate on test data
        # ... (similar code for test set)
        print("Retraining completed.")

    # 13. Save detection results, cleaned dataset, or models as needed
    # Example:
    results = {
        'detection_threshold': detection_threshold,
        'TPR': metrics['TPR'],
        'FPR': metrics['FPR']
        # Add more info if needed
    }
    with open('detection_results.json', 'w') as f:
        import json
        json.dump(results, f)

if __name__ == '__main__':
    main()
```

## model_inference.py

```python
## model_inference.py
import requests
import time
import logging
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
from io import BytesIO
from math import cos, sin
from transformers import CLIPProcessor, CLIPModel

# For API calls to OpenAI (GPT-4 with vision capabilities)
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, config: dict):
        """
        Initialize the ModelInference with configuration.
        Args:
            config (dict): Configuration dictionary with necessary API keys and parameters.
        """
        # API keys for OpenAI; should be set in config
        self.api_key = config.get("api_key", "")
        # Api models for GPT family
        self.gpt_model = config.get("gpt_model", "gpt-4-vision")  # Use GPT-4 with vision if available
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 512)

        # Initialize CLIP model and processor for similarity scores
        self.clip_model_name = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)

        # Optional: cache for similarities to avoid repetitive processing
        self.similarity_cache = {}

    def _call_openai_api(self, messages: List[dict], max_retry: int = 3) -> str:
        """
        Helper function to call OpenAI API with retries.
        Args:
            messages (list): List of message dicts for chat completion.
            max_retry (int): Maximum number of retries.
        Returns:
            str: The textual response from the API.
        """
        for attempt in range(max_retry):
            try:
                response = openai.ChatCompletion.create(
                    model=self.gpt_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                answer = response.choices[0].message['content'].strip()
                return answer
            except Exception as e:
                logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
                time.sleep(2 ** attempt)  # exponential backoff
        logger.error("API call failed after retries.")
        return ""

    def answer_questions(self, image_path: str, questions: List[str]) -> List[str]:
        """
        Given an image path and questions, generate answers using GPT-4 (vision).
        Args:
            image_path (str): Path to the image file.
            questions (List[str]): List of questions.
        Returns:
            List[str]: Answers corresponding to questions.
        """
        answers = []
        # Read and possibly encode image
        image = Image.open(image_path).convert('RGB')
        # Prepare a prompt that includes the image and questions
        # Note: Implementation depends on API capabilities
        # Placeholder: Assuming API accepts image path or image encoded as base64
        # For GPT-4 with vision, the exact API call supports image input directly.
        # However, in typical implementations, you pass image via API parameters.
        # Here, we simulate with a placeholder function.
        for question in questions:
            prompt = f"Answer the following question based on the provided image:\nQuestion: {question}"
            messages = [
                {"role": "system", "content": "You are a multimodal assistant that answers questions based on images."},
                {"role": "user", "content": prompt}
            ]
            answer = self._call_openai_api(messages)
            answers.append(answer)
        return answers

    def evaluate_response(self, question: str, answer: str, label: str) -> Tuple[bool, float]:
        """
        Evaluate if the answer correctly describes the label, using GPT prompt.
        For yes/no questions, parse response.
        Args:
            question (str): The question asked.
            answer (str): Model's answer.
            label (str): Ground truth label.
        Returns:
            Tuple[bool, float]: (is_correct, similarity_score)
        """
        # For label-specific deterministic questions, apply string matching
        q_lower = question.lower()
        a_lower = answer.lower()

        # Determine if question is about presence/attribute (heuristic)
        if "?" in question:
            # Use GPT to evaluate correctness for general questions
            prompt = (
                f"Assume you are an helpful assistant. "
                f"Question: \"{question}\" "
                f"Answer: \"{answer}\" "
                f"Label: \"{label}\" "
                f"Please decide whether the answer correctly describes the label. "
                f"Respond with 'yes' or 'no' only."
            )
            response = self._call_openai_api([
                {"role": "system", "content": "You are a helpful and precise assistant."},
                {"role": "user", "content": prompt}
            ])
            is_correct = response.strip().lower().startswith("yes")
            # For similarity: use CLIP cosine similarity as an auxiliary score
            similarity = self.get_semantic_similarity(image_path=None, label=label)
            return is_correct, similarity
        else:
            # For deterministic questions, use string matching
            if "yes" in a_lower:
                is_correct = True
            elif "no" in a_lower:
                is_correct = False
            else:
                # fallback: treat as incorrect
                is_correct = False
            similarity = self.get_semantic_similarity(image_path=None, label=label)
            return is_correct, similarity

    def get_semantic_similarity(self, image_path: Optional[str], label: str) -> float:
        """
        Compute cosine similarity between image and label text via CLIP.
        Args:
            image_path (Optional[str]): Path to image; if None, use cached or simulate.
            label (str): Label text.
        Returns:
            float: similarity score in [-1, 1]
        """
        cache_key = (image_path, label)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        # Encode label text
        text_inputs = self.clip_processor(text=label, return_tensors="pt", padding=True)
        text_embeddings = self.clip_model.get_text_features(**text_inputs)
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        if image_path is not None:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            image_inputs = self.clip_processor(images=image, return_tensors="pt")
            image_embeddings = self.clip_model.get_image_features(**image_inputs)
            image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                image_embeddings, text_embeddings
            ).item()
        else:
            # If image_path is None, fallback or simulate similarity
            # Due to the context, default to 0 (neutral)
            similarity = 0.0

        self.similarity_cache[cache_key] = similarity
        return similarity
```

## prompt_generator.py

```python
## prompt_generator.py
import random
from typing import List

class PromptGenerator:
    """
    Implements question prompt generation for the VDC pipeline's question modules.
    Generates both general and label-specific questions using structured templates.
    """

    def __init__(self, dataset_name: str = 'generic'):
        """
        Initialize the PromptGenerator with dataset context if necessary.
        Supports custom prompts per dataset type.
        Args:
            dataset_name (str): Optional dataset identifier for customization.
        """
        self.dataset_name = dataset_name
        # Predefine set of templates for general questions (as per Appendix E1)
        self.general_templates: List[str] = [
            "Describe the image in detail.",
            "Describe the image briefly.",
            "How would you summarize the content of the image in a few words?",
            "Provide a detailed description of the given image.",
            "Describe the image concisely.",
            "Provide a brief description of the given image.",
            "Offer a succinct explanation of the picture presented.",
            "Summarize the visual content of the image.",
            "Give a short and clear explanation of the given image.",
            "Share a concise interpretation of the image provided.",
            "Present a compact description of the photo’s key features.",
            "Relay a brief, clear account of the picture shown.",
            "Render a clear and concise summary of the photo.",
            "Write a terse but informative summary of the picture.",
            "Create a compact narrative representing the image presented."
        ]

        # Template for label-specific questions (can be dataset-specific)
        # For extensibility, different prompts can be added per dataset.
        self.label_question_template = (
            "Generate questions to verify if the object in the image corresponds to the label '{label}'. "
            "The questions should be answerable with 'yes' or 'no'. Focus on attributes, features, or functions "
            "that are characteristic of the label '{label}'."
        )

    def generate_general_questions(self, label: str, question_count: int = 2) -> List[str]:
        """
        Generate a list of general questions about the image.
        Args:
            label (str): The label associated with the image.
            question_count (int): Number of questions to generate.
        Returns:
            List[str]: List of generated questions.
        """
        questions = []
        # Randomly select templates for diversity
        for _ in range(question_count):
            template = random.choice(self.general_templates)
            questions.append(template)
        return questions

    def generate_label_specific_questions(self, label: str, question_count: int = 4) -> List[str]:
        """
        Generate label-specific questions based on the label, using prompt templates.
        Args:
            label (str): The label/class name for which questions are generated.
            question_count (int): Number of questions to generate.
        Returns:
            List[str]: List of generated label-specific questions.
        """
        questions = []
        # For each label, generate questions using the dataset-specific prompt template
        for _ in range(question_count):
            prompt = self.label_question_template.format(label=label)
            questions.append(prompt)
        return questions

    def generate_questions_for_label(
        self, label: str, num_general: int = 2, num_label_specific: int = 4
    ) -> List[str]:
        """
        Generate combined questions: general + label-specific.
        Args:
            label (str): Label/class name.
            num_general (int): Number of general questions.
            num_label_specific (int): Number of label-specific questions.
        Returns:
            List[str]: Combined list of questions ready for inference.
        """
        questions: List[str] = []
        questions.extend(self.generate_general_questions(label, num_general))
        questions.extend(self.generate_label_specific_questions(label, num_label_specific))
        return questions
```


## utils.py

```python
## utils.py
import os
import yaml
import json
import time
import logging
import random
import numpy as np
import torch
from torch import nn
from PIL import Image
from typing import Tuple, List, Dict, Optional

import requests
import base64
from io import BytesIO

# Optional: import for CLIP features
from transformers import CLIPProcessor, CLIPModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration globally for reuse
_CONFIG = None

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load YAML configuration once and cache."""
    global _CONFIG
    if _CONFIG is None:
        with open(config_path, 'r') as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG

# --------------------- Image Handling & Preprocessing ---------------------
def load_image(image_path: str) -> Image.Image:
    """Load an image from disk."""
    return Image.open(image_path).convert('RGB')

def preprocess_image(image: Image.Image, size: int, mean: List[float], std: List[float]) -> torch.Tensor:
    """Resize and normalize an image for model input."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform(image)

def image_to_tensor(image: Image.Image, size: int, mean: List[float], std: List[float]) -> torch.Tensor:
    return preprocess_image(image, size, mean, std)

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor back to PIL image."""
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    return to_pil(tensor.cpu())

# --------------------- Poisoning & Noise augmentation ---------------------
def overlay_trigger(image: Image.Image, trigger_size: int = 4, color: Tuple[int, int, int]=(255,255,255))
    """Overlay a white square patch at bottom-right corner."""
    img = image.copy()
    width, height = img.size
    for i in range(width - trigger_size, width):
        for j in range(height - trigger_size, height):
            img.putpixel((i, j), color)
    return img

def flip_labels(labels: List[int], method: str, ratio: float, num_classes: int) -> Tuple[List[int], set, List[int]]:
    """Apply noisy label flipping (symmetric/asymmetric). Returns new labels, indices, true labels."""
    corrupted_labels = labels.copy()
    total = len(labels)
    num_noisy = int(total * ratio)
    indices = list(range(total))
    random.shuffle(indices)
    noisy_idx = set(indices[:num_noisy])
    true_labels = labels.copy()

    for idx in noisy_idx:
        true_lbl = labels[idx]
        if method == 'symmetric':
            new_lbl = true_lbl
            while new_lbl == true_lbl:
                new_lbl = random.randint(0, num_classes -1)
            corrupted_labels[idx] = new_lbl
        elif method == 'asymmetric':
            new_lbl = (true_lbl + 1) % num_classes
            corrupted_labels[idx] = new_lbl
        # Extend if needed with other strategies
    return corrupted_labels, noisy_idx, true_labels

# --------------------- API Wrappers for LLM/MLLM ---------------------
def call_llm(prompt: str, api_key: str, model: str='gpt-4', temperature: float=0.2, max_tokens: int=512) -> str:
    """
    Call OpenAI API with retries.
    """
    retry = 3
    for attempt in range(retry):
        try:
            import openai
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            answer = response.choices[0].message['content'].strip()
            return answer
        except Exception as e:
            logger.warning(f"API call attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    logger.error("LLM API call failed after retries.")
    return ""

def answer_question_with_ml(image: Image.Image, question: str, ml_model, ml_processor, device='cpu') -> str:
    """
    Use local or API multimodal model to answer a question about an image.
    Args:
        image: PIL Image
        question: str
        ml_model: the loaded multimodal model (e.g., BLIP2, Otter)
        ml_processor: corresponding processor
        device: computation device
    Return:
        str: answer text
    """
    # Example implementation for models like OFA, BLIP2, Otter, etc.
    # This function needs to be adapted per specific model.
    # Here, we assume the model has a method 'generate_answer'
    # For illustration, using BLIP2 pipeline if available:
    # from transformers import pipeline
    # qa_pipeline = pipeline('image-question-answering', model=ml_model, tokenizer=ml_processor)
    # answer = qa_pipeline(image, question)
    # return answer['answer']
    # Since the design supports plug-and-play, we just raise NotImplementedError
    raise NotImplementedError("Implement model inference according to specific multimodal model used.")

# Placeholder: For openai's GPT-4 with vision, actual API call is required, which isn't publicly available yet.
# Here, just as a framework placeholder:
def answer_question_with_openai(image_path: str, question: str, api_key: str, model: str='gpt-4-vision') -> str:
    """
    Actual GPT-4 with vision API call (if accessible). Placeholder.
    """
    # TODO: Implement with appropriate API call as per provider's docs
    return ""  # Return empty string or placeholder

# --------------------- Prompt Construction ---------------------
def generate_general_question_prompts() -> List[str]:
    """Generate fixed set of general questions (from Appendix E1)."""
    return [
        "Describe the image in detail.",
        "Describe the image briefly.",
        "How would you summarize the content of the image in a few words?",
        "Provide a detailed description of the given image.",
        "Describe the image concisely.",
        "Provide a brief description of the given image.",
        "Offer a succinct explanation of the picture presented.",
        "Summarize the visual content of the image.",
        "Give a short and clear explanation of the given image.",
        "Share a concise interpretation of the image provided.",
        "Present a compact description of the photo’s key features.",
        "Relay a brief, clear account of the picture shown.",
        "Render a clear and concise summary of the photo.",
        "Write a terse but informative summary of the picture.",
        "Create a compact narrative representing the image presented."
    ]

def generate_label_specific_prompt(label: str) -> str:
    """Create prompt template for label-specific questions."""
    return (
        "Generate questions to verify if the object in the image corresponds to the label '{label}'. "
        "The questions should be answerable with 'yes' or 'no'. Focus on attributes, features, "
        "or functions characteristic of the label '{label}'."
    ).format(label=label)

def generate_evaluate_prompt(response: str, label: str) -> str:
    """Create prompt for GPT to evaluate correctness of answer."""
    prompt = (
        f"Assume you are a helpful and precise assistant. "
        f"Question: 'Does the answer \"{response}\" correctly describe the label \"{label}\"?' "
        f"Respond with 'yes' or 'no' only."
    )
    return prompt

# --------------------- Response Parsing & Scoring ---------------------
def parse_and_evaluate_response(question: str, answer: str, label: str) -> Tuple[bool, float]:
    """
    Determine correctness based on answer text and generate semantic similarity.
    For yes/no, parse answer string.
    """
    q_lower = question.lower()
    a_lower = answer.lower()

    # Use heuristics: if question is likely deterministic
    if "?" in question:
        # For open-ended, use GPT prompt (call API)
        prompt = generate_evaluate_prompt(answer, label)
        # Do API call (not implemented here)
        # is_correct = call_llm(prompt, api_key, ...)
        # Placeholder:
        is_correct = a_lower.startswith("yes")
        # For similarity, could compute via CLIP
        similarity = 0.0  # Placeholder
        return (is_correct, similarity)
    else:
        # deterministic: check string presence
        if "yes" in a_lower:
            return (True, 0.0)
        elif "no" in a_lower:
            return (False, 0.0)
        else:
            return (False, 0.0)

def compute_similarity_score(image: Image.Image, label: str, clip_model, clip_processor, device='cpu') -> float:
    """
    Compute cosine similarity between image and label text using CLIP.
    """
    # Encode label text
    text_inputs = clip_processor(text=label, return_tensors="pt", padding=True).to(device)
    text_emb = clip_model.get_text_features(**text_inputs)
    text_emb = nn.functional.normalize(text_emb, p=2, dim=-1)

    # Encode image
    image_input = clip_processor(images=image, return_tensors="pt").to(device)
    image_emb = clip_model.get_image_features(**image_input)
    image_emb = nn.functional.normalize(image_emb, p=2, dim=-1)

    similarity = torch.nn.functional.cosine_similarity(image_emb, text_emb).item()
    return similarity

# --------------------- Scoring and Detection ---------------------
def aggregate_sample_score(
    correctness_list: List[bool],
    similarity_score: float,
    use_similarity: bool = True,
    similarity_weight: float =0.5
) -> float:
    """Calculate sample's overall score based on correctness and similarity."""
    correct_count = sum(correctness_list)
    total_q = len(correctness_list)
    correctness_ratio = correct_count / total_q if total_q > 0 else 0.0
    if use_similarity:
        combined_score = (1 - similarity_weight) * correctness_ratio + similarity_weight * similarity_score
        return combined_score
    else:
        return correctness_ratio

def classify_sample(score: float, threshold: float=0.2) -> bool:
    """Classify sample as dirty if score below threshold."""
    return score < threshold

# --------------------- Performance Metrics Computation ---------------------
def compute_metrics(
    correctness_per_sample: List[List[bool]],
    similarity_scores: List[float],
    true_dirty_flags: List[bool]
) -> Dict:
    """Compute TPR, FPR given correctness, similarity, and ground truth."""
    scores = []
    pred_dirty = []
    for correctness_list, sim_score in zip(correctness_per_sample, similarity_scores):
        score = aggregate_sample_score(correctness_list, sim_score)
        scores.append(score)
        pred_dirty.append(score < 0.2)
    # Calculate TPR and FPR
    TP = sum(p and gt for p, gt in zip(pred_dirty, true_dirty_flags))
    FP = sum(p and not gt for p, gt in zip(pred_dirty, true_dirty_flags))
    total_dirty = sum(true_dirty_flags)
    total_clean = len(true_dirty_flags) - total_dirty
    TPR = TP / total_dirty if total_dirty > 0 else 0.0
    FPR = FP / total_clean if total_clean > 0 else 0.0
    return {
        'scores': scores,
        'pred_labels': pred_dirty,
        'TPR': TPR,
        'FPR': FPR
    }
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\vdc\vdc_repo`
