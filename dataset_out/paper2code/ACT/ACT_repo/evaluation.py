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

