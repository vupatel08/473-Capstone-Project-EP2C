# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## attention_calibrator.py

```python
## attention_calibrator.py
import torch
from typing import List, Tuple, Dict, Optional

class AttentionCalibrator:
    """
    Implements attention sink detection and suppression to calibrate attention maps
    during inference, inspired by the paper's methodology. Designed for input-adaptive,
    training-free adjustment of attention distributions to improve model performance.
    """
    def __init__(
        self,
        alpha: float = 5.0,
        suppress_factor: float = 0.4,
        subset_percent: float = 0.4
    ):
        """
        Initialize the AttentionCalibrator with default hyperparameters.
        Args:
            alpha (float): Threshold multiplier to identify high-attention tokens as sinks.
            suppress_factor (float): Factor (\(\beta\)) to reduce attention weights of sinks.
            subset_percent (float): Percentage (e.g., 0.4) of top tokens considered sinks per layer-head.
        """
        self.alpha = alpha
        self.suppress_factor = suppress_factor
        self.subset_percent = subset_percent

    def detect_sinks(
        self,
        attention_maps: List[Dict[str, torch.Tensor]],
        input_token_mask: Optional[torch.Tensor] = None
    ) -> List[Tuple[int, int, int, float]]:
        """
        Detects attention sinks in the provided attention maps.
        Args:
            attention_maps (List[Dict[str, torch.Tensor]]): List of attention matrices per layer.
                Each element: dict with key 'attentions': Tensor shape (batch_size, num_heads, seq_len, seq_len)
            input_token_mask (Optional[Tensor]): Boolean tensor indicating valid tokens (for skipping padding).
        Returns:
            sinks (List of Tuples): List of detected sinks with (layer_idx, head_idx, token_idx, attention_score).
        """
        sinks = []
        for layer_idx, layer_attention in enumerate(attention_maps):
            attn_tensor = layer_attention['attentions']  # shape: (batch_size, heads, seq_len, seq_len)
            # Sum attention over source tokens to get attention received per token
            # sum over axis=2: source tokens
            token_attn_sum = attn_tensor.sum(dim=2)  # shape: (batch_size, num_heads, seq_len)
            # Average across batch
            token_attn_avg = token_attn_sum.mean(dim=0)  # shape: (num_heads, seq_len)

            num_heads = token_attn_avg.shape[0]
            seq_len = token_attn_avg.shape[1]

            for h in range(num_heads):
                attn_scores = token_attn_avg[h]  # shape: (seq_len,)
                # Optional: mask out padding tokens
                if input_token_mask is not None:
                    mask = input_token_mask[:seq_len]
                    valid_scores = attn_scores[mask]
                else:
                    valid_scores = attn_scores

                mean_score = valid_scores.mean()
                # Threshold for high attention tokens: tokens with scores > alpha * mean
                threshold = self.alpha * mean_score
                # Select top subset_percent of tokens
                num_sinks = max(1, int(self.subset_percent * seq_len))
                topk_scores, topk_indices = torch.topk(attn_scores, k=num_sinks)
                for token_idx, score in zip(topk_indices.tolist(), topk_scores.tolist()):
                    if input_token_mask is not None and not mask[token_idx]:
                        continue
                    if score > threshold:
                        sinks.append((layer_idx, h, token_idx, score))
        return sinks

    def apply_suppression(
        self,
        attention_maps: List[Dict[str, torch.Tensor]],
        sinks: List[Tuple[int, int, int, float]],
        batch_size: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Apply suppression to attention maps by reducing sink token attention weights.
        Args:
            attention_maps (List[Dict]): Original attention maps.
            sinks (List[Tuple]): Detected sink tokens with (layer, head, token, score).
            batch_size (int): Size of the current batch (for tensor operations).
        Returns:
            calibrated_attention_maps (List[Dict]): Attention maps with sinks suppressed.
        """
        # Prepare a new list for calibrated attention maps
        calibrated_maps = []
        for layer_idx, layer_attention in enumerate(attention_maps):
            attn_tensor = layer_attention['attentions'].clone()  # shape: (batch, heads, seq_len, seq_len)
            # For each sink in this layer, head, modify attention weights
            for sink in sinks:
                l, h, t_idx, _ = sink
                if l != layer_idx:
                    continue
                # Create mask for sink tokens in head h of layer l
                sink_mask = torch.zeros(attn_tensor.shape[0], attn_tensor.shape[2], device=attn_tensor.device)
                sink_mask[:, t_idx] = 1.0  # shape: (batch, seq_len)
                # Reduce attention weights pointing to sink tokens
                # Multiply target sink column by suppression factor
                # attention shape: (batch, heads, seq_len, seq_len)
                for b in range(batch_size):
                    # Apply only to the relevant layer-head
                    attn_b = attn_tensor[b, h]
                    # Multiply sink column
                    attn_b[:, t_idx] = attn_b[:, t_idx] * self.suppress_factor
                    # Optional: zero out sink row if you want to prevent source attention
                    # For this implementation, only suppress sink column
                    # Renormalize rows to sum to 1
                    row_sums = attn_b.sum(dim=1, keepdim=True)
                    # Avoid division by zero
                    row_sums = torch.clamp(row_sums, min=1e-8)
                    attn_b = attn_b / row_sums
                    attn_tensor[b, h] = attn_b
            # Save adjusted attention map for this layer
            calibrated_maps.append({'attentions': attn_tensor})
        return calibrated_maps

    def calibrate_attention(
        self,
        attention_maps: List[Dict[str, torch.Tensor]],
        input_token_mask: Optional[torch.Tensor] = None,
        batch_size: int = 1
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Orchestrate detection and suppression on the provided attention maps.
        Args:
            attention_maps (List[Dict]): Raw attention maps during inference.
            input_token_mask (Optional[Tensor]): Valid tokens mask (for ignoring padding).
            batch_size (int): Batch size of the current input.
        Returns:
            calibrated_maps (List[Dict]): Attention maps after calibration.
        """
        # Detect high-attention sink tokens
        sinks = self.detect_sinks(attention_maps, input_token_mask)
        # Apply suppression to reduce attention on sinks
        calibrated_maps = self.apply_suppression(attention_maps, sinks, batch_size)
        return calibrated_maps
```

## dataset_loader.py

```python
## dataset_loader.py

import os
import json
import csv
from typing import List, Dict, Optional, Union
from datasets import load_dataset

class DatasetLoader:
    """
    Responsible for loading datasets for various NLP tasks, formatting prompts,
    and returning data samples suitable for inference and evaluation.
    """

    def __init__(
        self,
        dataset_configs: Dict[str, Dict],
        prompts: Dict[str, str],
        split: str = 'test',
        max_input_length: int = 1024
    ):
        """
        Initialize DatasetLoader.

        Args:
            dataset_configs (Dict[str, Dict]]): Dictionary mapping dataset name to configuration, e.g.,
                {
                    "Hellaswag": {"path": "path/to/file.json", "type": "json"},
                    "SQuAD": {"name": "squad", "split": "validation"},
                    ...
                }
            prompts (Dict[str, str]): Dictionary of prompt templates for each dataset/task.
            split (str): Dataset split to load, default 'test'.
            max_input_length (int): Max token length for inputs, truncates if exceeded.
        """
        self.dataset_configs = dataset_configs
        self.prompts = prompts
        self.split = split
        self.max_input_length = max_input_length

    def load_data(self) -> List[Dict]:
        """
        Load all datasets as per configuration, format prompts, and prepare samples.

        Returns:
            List of data samples, each as dict with keys:
            - 'prompt': formatted input string for model
            - 'label': ground truth answer/label (optional)
            - 'metadata': additional info (original sample, dataset name)
        """
        all_samples = []

        for dataset_name, config in self.dataset_configs.items():
            source_path = config.get('path', None)
            dataset_type = config.get('type', 'local')  # 'json', 'csv', or 'hf'
            hf_name = config.get('name', None)  # For HuggingFace datasets
            split = config.get('split', self.split)

            # Load dataset based on type
            if dataset_type == 'hf' and hf_name:
                dataset = load_dataset(hf_name, split=split)
                dataset_samples = [dict(row) for row in dataset]
            elif source_path:
                ext = os.path.splitext(source_path)[1].lower()
                if ext == '.json':
                    with open(source_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            dataset_samples = data
                        else:
                            dataset_samples = data.get(split, [])
                elif ext == '.csv':
                    with open(source_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        dataset_samples = list(reader)
                else:
                    # Default: assuming a JSON Lines file
                    dataset_samples = []
                    with open(source_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            dataset_samples.append(json.loads(line))
            else:
                raise ValueError(f"Dataset {dataset_name} has no valid source path or name.")

            # For each sample, generate prompt and extract label
            for sample in dataset_samples:
                prompt = self._format_prompt(sample, dataset_name)
                label = self._extract_label(sample, dataset_name)
                metadata = {
                    'original_sample': sample,
                    'dataset_name': dataset_name
                }
                all_samples.append({
                    'prompt': prompt,
                    'label': label,
                    'metadata': metadata
                })

        # Optional: truncate inputs if exceeding max_input_length
        # Here, for simplicity, assume inputs are tokenized later with truncation
        return all_samples

    def _format_prompt(self, sample: Dict, dataset_name: str) -> str:
        """
        Apply the dataset-specific prompt template to create the input prompt.

        Args:
            sample (Dict): raw data sample from dataset
            dataset_name (str): name of the dataset/task

        Returns:
            str: formatted prompt string
        """
        # Determine prompt template based on dataset/task
        if dataset_name.lower() in ['hellaswag', 'arce', 'piqa', 'ob', 'arcc', 'copa', 'cqa']:
            template = self.prompts.get('multiple_choice', None)
            # Expect sample to have 'sentence', 'choices', 'answer'
            question_text = sample.get('sentence', '')
            choices = sample.get('choices', [])
            answer = sample.get('answer', None)
            choices_str = ' '.join([f"<choice {i+1}> {choice} " for i, choice in enumerate(choices)])
            prompt = f"Complete the following sentence with an appropriate ending. {question_text} {choices_str} Answer:"
            return prompt

        elif dataset_name.lower() == 'mmlu':
            subject = sample.get('subject', 'general')
            template = self.prompts.get('mmlu', None)
            question = sample.get('question', '')
            choices = sample.get('choices', [])
            answer = sample.get('answer', None)
            choices_str = ' '.join([f"<choice {i+1}> {choice} " for i, choice in enumerate(choices)])
            prompt = f"The following are multiple choice questions (with answers) about {subject}.\n{question} {choices_str} Answer:"
            return prompt

        elif dataset_name.lower() in ['sst2', 'sst5', 'mr', 'agnews', 'trec', 'cb', 'boolq']:
            # Text classification datasets
            sentence = sample.get('sentence') or sample.get('paragraph') or ''
            if dataset_name.lower() == 'sst2':
                prompt = f"Classify the sentiment of the user\u2019s message into one of the following categories:\'positive\' or \'negative\'. Sentence: {sentence} Sentiment:"
            elif dataset_name.lower() == 'sst5':
                prompt = f"Classify the sentiment of the user\u2019s message into one of the following categories:\'terrible\', \'negative\', \'neutral\', \'positive\', or \'great\'. Sentence: {sentence} Sentiment:"
            elif dataset_name.lower() == 'mr':
                prompt = f"Classify the sentiment of the movie\u2019s review into one of the following categories:\'positive\' or \'negative\'. - Review: {sentence} Sentiment:"
            elif dataset_name.lower() == 'agnews':
                prompt = f"Classify the news articles into the categories of \'World\', \'Sports\', \'Business\', or \'Technology\'. Article: {sentence} Category:"
            elif dataset_name.lower() == 'trec':
                prompt = f"Classify the given questions into the following categories of \'Description\', \'Entity\', \'Expression\', \'Person\', \'Number\', or \'Location\'. Question: {sentence} Type:"
            elif dataset_name.lower() == 'cb':
                premise = sample.get('premise', '')
                hypothesis = sample.get('hypothesis', '')
                prompt = f"Read the following paragraph and determine if the hypothesis is true. Premise: {premise} Hypothesis: {hypothesis}. Answer:"
            elif dataset_name.lower() == 'boolq':
                passage = sample.get('passage', '')
                question_text = sample.get('question', '')
                prompt = f"Read the text and answer the question by True or False. Text: {passage} Question: {question_text} ? Answer:"
            else:
                prompt = ''
            return prompt

        elif dataset_name.lower() in ['squad', 'squadv1', 'squadv2']:
            # QA datasets
            title = sample.get('title', '')
            context = sample.get('context', '')
            question_text = sample.get('question', '')
            prompt = (
                f"Answer question using information in the preceding background paragraph. "
                f"Title: {title}\nBackground: {context}\nQ: {question_text}\nA:"
            )
            return prompt

        else:
            # Default fallback prompt
            # use available text fields
            default_text = ''
            for key in ['sentence', 'question', 'paragraph', 'context']:
                if key in sample:
                    default_text = sample[key]
                    break
            prompt = f"{default_text}\nAnswer:"
            return prompt

    def _extract_label(self, sample: Dict, dataset_name: str) -> Optional[Union[str, int]]:
        """
        Extract the ground truth label or answer from the raw sample for evaluation.

        Args:
            sample (Dict): raw dataset sample
            dataset_name (str): dataset/task name

        Returns:
            label (str or int) or None if not available
        """
        if dataset_name.lower() in ['hellaswag', 'arce', 'piqa', 'ob', 'arcc', 'copa', 'cqa']:
            # For multiple-choice, answer is index or value
            return sample.get('answer', None)
        elif dataset_name.lower() == 'mmlu':
            return sample.get('answer', None)
        elif dataset_name.lower() in ['sst2', 'sst5', 'mr', 'agnews', 'trec', 'cb', 'boolq']:
            return sample.get('label', None)
        elif dataset_name.lower() in ['squad', 'squadv1', 'squadv2']:
            return sample.get('answer', None)
        else:
            # For other datasets, attempt to retrieve answer label
            return sample.get('label', None)
```

## evaluation.py

```python
## evaluation.py
import torch
from typing import List, Dict, Union, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from transformers import AutoTokenizer
from dataset_loader import DatasetLoader
from model_wrapper import ModelWrapper
from attention_calibrator import AttentionCalibrator

class Evaluation:
    """
    Handles model evaluation on multiple datasets, with optional attention calibration (ACT).
    Supports computing metrics like accuracy, EM, F1, and detailed result aggregation.
    """

    def __init__(
        self,
        model: ModelWrapper,
        dataset_configs: Dict[str, Dict],
        prompts: Dict[str, str],
        config: Dict,
        device: str = "cuda"
    ):
        """
        Initialize Evaluation class.
        Args:
            model (ModelWrapper): The pretrained model wrapper.
            dataset_configs (Dict): Dataset configuration dict as per 'config.yaml'.
            prompts (Dict): Dictionary of prompt templates per task.
            config (Dict): Full configuration dictionary from 'config.yaml'.
            device (str): Device string.
        """
        self.model = model
        self.device = device

        # Load datasets using DatasetLoader
        self.dataset_loader = DatasetLoader(dataset_configs, prompts)
        self.datasets = self.dataset_loader.load_data()  # list of samples with prompt and label

        # Save hyperparameters
        self.alpha = config['attention_calibration'].get('alpha', 5.0)
        self.beta = config['attention_calibration'].get('suppress_factor', 0.4)
        self.subset_percent = config['attention_calibration'].get('subset_percent', 0.4)
        self.calibrate_layers = config['attention_calibration'].get('calibrate_layers', None)
        self.calibrate_heads = config['attention_calibration'].get('calibrate_heads', None)
        # Whether to perform calibration (ACT) or vanilla
        self.use_act = True

        # Initialize the AttentionCalibrator
        self.calibrator = AttentionCalibrator(
            alpha=self.alpha,
            suppress_factor=self.beta,
            subset_percent=self.subset_percent
        )

        # Prepare the tokenizer
        self.tokenizer = self.model.tokenizer

    def evaluate_dataset(self, samples: List[Dict], task_type: str, metric_name: str) -> Dict:
        """
        Evaluate a dataset given list of samples.
        Args:
            samples (List[Dict]): List of dict with 'prompt', 'label'.
            task_type (str): 'classification', 'qa', 'multi-turn', etc.
            metric_name (str): Metric to compute: 'accuracy', 'EM', 'F1'
        Returns:
            Dict with keys: 'preds', 'labels', 'score'
        """
        preds = []
        labels = []

        for sample in samples:
            input_text = sample['prompt']
            label = sample['label']
            labels.append(label)

            # Tokenize input
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

            # Clear previous attention maps
            self.model.clear_attention_maps()

            # Register hooks if ACT is enabled
            if self.use_act:
                # Register hooks to extract attention
                self.model.register_attention_hooks()

            # Run inference with optional attention calibration
            if self.use_act:
                # Extract attention maps
                attention_maps = self.model.compute_attention()

                # Determine if attention maps are available
                if attention_maps:
                    # For current input, perform detection and suppression
                    # Here, attention_maps could be list across layers; you may extract as needed
                    cal_attention_maps = self.calibrator.calibrate_attention(
                        attention_maps=attention_maps,
                        input_token_mask=self._get_token_mask(input_ids),
                        batch_size=1
                    )
                    # Generate output using calibrated attention
                    output_text = self.model.generate_output(
                        input_ids=input_ids,
                        attention_maps=cal_attention_maps,
                        max_new_tokens=50
                    )
                else:
                    # No attention maps; fallback to vanilla
                    output_text = self.model.generate_output(input_ids=input_ids, max_new_tokens=50)
            else:
                # Vanilla inference
                output_text = self.model.generate_output(input_ids=input_ids, max_new_tokens=50)

            # Store prediction logic based on dataset/task
            pred = self._extract_prediction(output_text, task_type)
            preds.append(pred)

        # Compute metric
        score = self._compute_metric(preds, labels, metric_name)

        return {'preds': preds, 'labels': labels, 'score': score}

    def run(self):
        """
        Run evaluation on all datasets and compile results.
        """
        results = {}
        for task_name, dataset in self.datasets.items():
            # Determine task type and metric from config
            task_type = self._determine_task_type(task_name)
            metric_name = self._get_metric_name(task_type)

            # Evaluate dataset
            res = self.evaluate_dataset(dataset, task_type, metric_name)
            results[task_name] = res

        # Aggregate overall and per-dataset metrics
        return results

    def _get_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create mask for valid tokens (non-padding).
        """
        return (input_ids != self.tokenizer.pad_token_id).squeeze(0)

    def _extract_prediction(self, output_text: str, task_type: str) -> Union[str, int]:
        """
        Extract predicted label/class from model output depending on task type.
        """
        output_text = output_text.strip()

        # Implementation depends on the dataset/task
        # For classification and multiple-choice, map output tokens to labels
        # For QA/general, output is generated answer text
        # This function should be customized per dataset

        # For simplicity, assume the first token or string is the label
        # Example for 4 options: pick the option with highest similarity
        # Here, implementing a naive approach: return the complete output or first token
        return output_text  # Further processing can be added for specific datasets

    def _compute_metric(self, preds: List, labels: List, metric_name: str) -> float:
        """
        Compute specified metric based on task.
        """
        if metric_name == 'accuracy':
            return accuracy_score(labels, preds)
        elif metric_name == 'F1':
            return f1_score(labels, preds, average='macro')
        elif metric_name == 'EM':
            # Exact Match for QA
            matches = [pred.strip() == label.strip() for pred, label in zip(preds, labels)]
            return sum(matches) / len(matches)
        elif metric_name == 'score':
            # General score placeholder
            return sum([float(p == l) for p, l in zip(preds, labels)]) / len(labels)
        else:
            # Default fallback
            return 0.0

    def _determine_task_type(self, dataset_name: str) -> str:
        """
        Derive task type ('classification', 'qa', etc.) based on dataset name or config.
        """
        name = dataset_name.lower()
        if name in ['sst2', 'sst5', 'agnews', 'mr', 'trec', 'cb', 'boolq']:
            return 'classification'
        elif name in ['squad', 'squadv1', 'squadv2']:
            return 'qa'
        elif name == 'mt-bench':
            return 'multi-turn'
        else:
            return 'classification'  # default assumption

    def _get_metric_name(self, task_type: str) -> str:
        """
        Map task type to metric name(s).
        """
        if task_type == 'classification':
            return 'accuracy'
        elif task_type == 'qa':
            return 'EM'
        elif task_type == 'multi-turn':
            return 'score'
        else:
            return 'accuracy'  # fallback

```

## main.py

```python
## main.py
import yaml
import torch
import os
from datasets import load_dataset
import numpy as np

from dataset_loader import DatasetLoader
from model_wrapper import ModelWrapper
from attention_calibrator import AttentionCalibrator
from evaluation import Evaluation

def main():
    # 1. Load configuration from YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Extract model config
    model_name = config['model'].get('name', 'Llama-7b')
    device_str = config['model'].get('device', 'cuda')
    model_path = config['model'].get('model_path', None)

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # 2. Load datasets
    dataset_configs = config['training']['datasets']  # Expect dict of dataset configs
    prompts = {}  # Could define prompts based on dataset; here skipped for brevity or use defaults
    dataset_loader = DatasetLoader(dataset_configs, prompts)
    datasets = dataset_loader.load_data()

    # 3. Initialize model
    model = ModelWrapper(model_name, device=device_str, model_path=model_path)
    # Register hooks to enable attention map extraction
    model.register_attention_hooks()

    # 4. Set up AttentionCalibrator with hyperparameters from config
    attn_config = config.get('attention_calibration', {})
    alpha = attn_config.get('alpha', 5.0)
    suppress_factor = attn_config.get('suppress_factor', 0.4)
    subset_percent = attn_config.get('subset_percent', 0.4)

    calibrator = AttentionCalibrator(
        alpha=alpha,
        suppress_factor=suppress_factor,
        subset_percent=subset_percent
    )

    # 5. Loop over each dataset/task for evaluation
    results = {}
    for dataset_name, dataset_samples in datasets.items():
        task_type = determine_task_type(dataset_name)
        metric_name = get_metric_name(task_type)

        preds = []
        labels = []

        for sample in dataset_samples:
            # Format prompt based on dataset
            prompt_text = sample['prompt']
            label = sample['label']
            labels.append(label)

            # Tokenize input prompt
            input_ids = model.tokenizer.encode(prompt_text, return_tensors='pt').to(device)

            # Clear previous attention maps
            model.clear_attention_maps()

            # Run inference with hooks registered
            with torch.no_grad():
                # Extract attention maps (stored during forward due to hooks)
                attention_maps = model.compute_attention()

                # If attention maps are present, perform ACT
                if attention_maps:
                    # Detect sinks
                    # Here, for more accurate detection, pass input mask
                    input_mask = (input_ids != model.tokenizer.pad_token_id).squeeze(0)
                    calibrated_maps = calibrator.calibrate_attention(
                        attention_maps,
                        input_token_mask=input_mask,
                        batch_size=1
                    )
                    # Generate output using calibrated attention
                    output_text = model.generate_output(
                        input_ids=input_ids,
                        attention_maps=calibrated_maps,
                        max_new_tokens=50
                    )
                else:
                    # No attention maps captured; fallback to standard generation
                    output_text = model.generate_output(
                        input_ids=input_ids,
                        max_new_tokens=50
                    )

            # Extract prediction string
            pred = extract_prediction(output_text, task_type)
            preds.append(pred)

        # Compute metric for current dataset
        score = compute_metric(preds, labels, metric_name)
        results[dataset_name] = {
            'preds': preds,
            'labels': labels,
            'score': score
        }

    # 6. Print and Save Results
    for dataset_name, res in results.items():
        print(f"Dataset: {dataset_name} | {res['score']:.4f}")

def determine_task_type(dataset_name: str) -> str:
    name = dataset_name.lower()
    if name in ['sst2', 'sst5', 'agnews', 'mr', 'trec', 'cb', 'boolq']:
        return 'classification'
    elif name in ['squad', 'squadv1', 'squadv2']:
        return 'qa'
    elif name == 'mt-bench':
        return 'multi-turn'
    else:
        return 'classification'  # default fallback

def get_metric_name(task_type: str) -> str:
    if task_type == 'classification':
        return 'accuracy'
    elif task_type == 'qa':
        return 'EM'
    elif task_type == 'multi-turn':
        return 'score'
    else:
        return 'accuracy'

def extract_prediction(output_text: str, task_type: str) -> str:
    # For simplicity, just strip and return predicted string
    # Can be extended with more nuanced extraction based on task
    return output_text.strip()

def compute_metric(preds: list, labels: list, metric_name: str) -> float:
    from sklearn.metrics import accuracy_score, f1_score
    preds_str = [str(p) for p in preds]
    labels_str = [str(l) for l in labels]

    if metric_name == 'accuracy':
        return accuracy_score(labels_str, preds_str)
    elif metric_name == 'F1':
        return f1_score(labels_str, preds_str, average='macro')
    elif metric_name == 'EM':
        matches = [p.strip().lower() == l.strip().lower() for p, l in zip(preds_str, labels_str)]
        return sum(matches) / len(matches)
    elif metric_name == 'score':
        # Placeholder for other scoring
        score = sum([float(p == l) for p, l in zip(preds_str, labels_str)]) / len(labels_str)
        return score
    else:
        return 0.0

if __name__ == '__main__':
    main()
```

## model_wrapper.py

```python
## model_wrapper.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional


class ModelWrapper:
    """
    Wraps a HuggingFace transformer language model to:
    - Load model and tokenizer
    - Register hooks to extract attention weights during inference
    - Provide method to retrieve attention maps
    - Generate outputs with optional modified attention
    """

    def __init__(self, model_name: str = "Llama-7b", device: str = "cuda", model_path: Optional[str] = None):
        """
        Initialize the model wrapper.
        Args:
            model_name (str): Identifier of the pretrained model.
            device (str): Device to run inference ("cpu" or "cuda").
            model_path (str, optional): Path to local model or HF model id.
        """
        import transformers  # import locally to delay loading if needed
        # Load model and tokenizer
        if model_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Prepare storage for attention maps
        # Structure: dict[layer_idx] -> dict['attentions': List[Tensor]]
        self.attention_maps: Dict[int, Dict[str, List[torch.Tensor]]] = {}
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Register hooks on transformer layers
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        """
        Register hooks on all attention layers to capture attention weights during forward pass.
        Assumes model has attribute `model.transformer.h`, typical for HuggingFace models like GPT.
        """
        # Clear previous hooks if any
        self._remove_attention_hooks()

        # Find all attention modules
        # The structure depends on model origin; common attribute: model.transformer.h (list of layers)
        for layer_idx, layer in enumerate(self.model.transformer.h):
            # Some models use 'attn' attribute for attention modules
            # For GPT-like models, layer.attn is typical
            if hasattr(layer, 'attn'):
                attn_module = layer.attn

                def get_attention_hook(layer_index: int):
                    # Closure to retain layer index
                    def hook(module, input, output):
                        # output: tuple (attn_probs, attn_mask, attn_output, new_attn_weights)
                        # For most models, the first element is attention probabilities
                        # Capture attention probs: shape (batch, num_heads, seq_len, seq_len)
                        if isinstance(output, tuple):
                            attn_probs = output[0]
                        else:
                            attn_probs = output
                        # Store a clone to avoid mutation
                        self.attention_maps[layer_index] = {
                            'attentions': [attn_probs.detach().cpu()]
                        }
                    return hook

                handle = attn_module.register_forward_hook(get_attention_hook(layer_idx))
                self.hook_handles.append(handle)

    def _remove_attention_hooks(self):
        """
        Remove all registered hooks to avoid duplication.
        """
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def register_attention_hooks(self):
        """
        Public method to register hooks during inference.
        """
        self._register_attention_hooks()

    def clear_attention_maps(self):
        """
        Clear stored attention maps.
        """
        self.attention_maps.clear()

    def compute_attention(self) -> Dict[int, Dict[str, List[torch.Tensor]]]:
        """
        Return the latest stored attention maps.
        """
        return self.attention_maps

    def generate_output(
        self,
        input_ids: torch.Tensor,
        attention_maps: Optional[Dict[int, Dict[str, List[torch.Tensor]]]] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """
        Generate output text with optional attention modifications.
        Args:
            input_ids (torch.Tensor): Tokenized input IDs, shape (1, seq_len).
            attention_maps: Optional attention maps to modify during generation.
            max_new_tokens (int): Max tokens to generate.
            temperature (float): Sampling temperature.
            do_sample (bool): Whether to sample or greedy decoding.
        Returns:
            Generated text string.
        """
        # If attention_maps provided, replace stored maps before generation
        if attention_maps is not None:
            self._apply_attention_modifications(attention_maps)

        # Generate output
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids.to(self.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs
            )
        # Decode output
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

    def _apply_attention_modifications(self, attention_maps: Dict[int, Dict[str, List[torch.Tensor]]]):
        """
        Inject modified attention maps into the model by overriding attention in transformer layers.
        Note: HuggingFace transformers do not support direct input of attention weights.
        As a workaround, we can patch the attention probabilities after the hook, but this
        requires model internals modification or a custom forward hook.
        
        Since the task demands, here we will assume that the model's transformer layers have
        a method or attribute to override attentions or that we replace the internal attention matrices.
        For most models, this is non-trivial; for demonstration, we provide a patch approach.
        """
        # WARNING: Overriding internal attention probabilities requires model modification,
        # which is complex. For this implementation, we assume attention modifications are
        # feasible via hooks or a suitable interface.
        # The following is a placeholder: in practice, need model-specific hooks or code.

        # Example (hypothetical): Patch attention logits/weights if model allows
        pass

    def unload(self):
        """
        Cleanup: unregister hooks and free resources.
        """
        self._remove_attention_hooks()
        # Additional cleanup if needed
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\ACT\ACT_repo`
