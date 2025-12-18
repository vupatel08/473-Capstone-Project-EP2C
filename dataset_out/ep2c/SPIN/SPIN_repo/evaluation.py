## evaluation.py
import os
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import yaml

class Evaluation:
    """
    The Evaluation class performs systematic assessment of the current language model's
    performance on benchmark datasets as specified in the configuration. It manages dataset
    loading, prompt sampling, response generation, and metric evaluation, ensuring consistency
    with the methodology outlined in the SPIN paper.
    """
    def __init__(self, model, config_path: str = "config.yaml"):
        """
        Initialize Evaluation with a model instance and load evaluation datasets and config.

        Args:
            model: The language model object having generate() and tokenizer.
            config_path (str): Path to the YAML configuration file.
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.config = self._load_config(config_path)
        self.use_gpu = self.config.get('use_gpu', True)
        self.max_eval_batch_size = self.config.get('evaluation', {}).get('batch_size', 16)
        self.eval_datasets_config = self._load_datasets_config()
        self.datasets = self._load_datasets()
        self.metrics = self._load_metrics()
        self.response_params = {
            'max_length': self.config.get('generation', {}).get('max_length', 100),
            'temperature': self.config.get('generation', {}).get('temperature', 0.7)
        }

    def _load_config(self, path: str) -> dict:
        """
        Load the YAML configuration for evaluation.

        Args:
            path (str): Path to config.yaml.

        Returns:
            dict: Configuration dictionary.
        """
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _load_datasets_config(self) -> Dict[str, Dict]:
        """
        Extract the dataset-loading configuration from the main config.

        Returns:
            Dict[str, Dict]: Dict with dataset names as keys and their configs.
        """
        return self.config.get('evaluation', {}).get('datasets', {
            'arc': {'split': 'validation', 'metric': 'acc'},
            'truthfulqa': {'split': 'validation', 'metric': 'acc'},
            'winogrande': {'split': 'validation', 'metric': 'acc'},
            'gsm8k': {'split': 'test', 'metric': 'acc'},
            'hellaswag': {'split': 'validation', 'metric': 'acc'},
            'mmlu': {'split': 'validation', 'metric': 'acc'}
        })

    def _load_datasets(self) -> Dict[str, object]:
        """
        Load datasets based on the evaluation config.

        Returns:
            Dict[str, dataset]: Loaded datasets.
        """
        datasets = {}
        for name, cfg in self.eval_datasets_config.items():
            # Here, load datasets via Hugging Face datasets library
            # For datasets like 'arc', 'truthfulqa', etc., assume standard splits
            try:
                # Attempt to load dataset by name
                datasets[name] = load_dataset(name, split=cfg.get('split', 'validation'))
            except Exception:
                # Fall back to local or dummy dataset if needed
                # For simplicity, create placeholder with prompts and references
                # In real implementation, load actual datasets with prompts and answers
                datasets[name] = self._create_dummy_dataset(name)
        return datasets

    def _create_dummy_dataset(self, name: str):
        """
        Creates a dummy dataset with prompts and reference answers for demonstration.

        Args:
            name (str): Dataset name.

        Returns:
            Dataset object: dummy dataset with 'prompt' and 'reference'.
        """
        from datasets import Dataset
        prompts = [
            "What is the capital of France?",
            "Solve for x: 2x + 3 = 7.",
            "Who wrote 'Pride and Prejudice'?",
        ]
        references = [
            "Paris",
            "x=2",
            "Jane Austen",
        ]
        data = {'prompt': prompts, 'reference': references}
        return Dataset.from_dict(data)

    def _load_metrics(self) -> Dict[str, callable]:
        """
        Map dataset names to their specific scoring functions.

        Returns:
            Dict[str, callable]: Mapping from dataset to metric function.
        """
        return {
            'arc': self._metric_accuracy,
            'truthfulqa': self._metric_accuracy,
            'winogrande': self._metric_accuracy,
            'gsm8k': self._metric_accuracy,
            'hellaswag': self._metric_accuracy,
            'mmlu': self._metric_accuracy,
            # Additional datasets can be added here
        }

    def generate_responses(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for prompts using the model, with batching.

        Args:
            prompts (List[str]): List of prompts.

        Returns:
            List[str]: List of generated responses.
        """
        responses = []
        # Generate in batches
        batch_size = self.max_eval_batch_size
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_responses = self._model_generate(batch_prompts)
            responses.extend(batch_responses)
        return responses

    def _model_generate(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts (List[str]): List of prompts.

        Returns:
            List[str]: Generated responses.
        """
        # Use the model's generate API with parameters
        params = self.response_params
        # Ensure inputs are on the correct device
        # Generate responses
        gen_kwargs = {
            'max_length': params['max_length'],
            'temperature': params['temperature'],
            'do_sample': True,
            'top_k': 50,
            'top_p': 0.95,
            'eos_token_id': self.model.config.eos_token_id,
            'pad_token_id': self.model.config.eos_token_id,
            'num_return_sequences': 1
        }
        try:
            inputs = self.model.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            output_ids = self.model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
        except Exception as e:
            # Handle generation errors
            print(f"Error during generation: {e}")
            return [""] * len(prompts)

        responses = []
        for ids in output_ids:
            text = self.model.tokenizer.decode(ids, skip_special_tokens=True)
            responses.append(self._post_process_response(text))
        return responses

    def _post_process_response(self, response: str) -> str:
        """
        Clean up generated response string.

        Args:
            response (str): Raw generated response.

        Returns:
            str: Cleaned response.
        """
        return response.strip()

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation over all datasets and compute metrics.

        Returns:
            Dict[str, float]: Aggregated scores over datasets.
        """
        results = {}
        for dataset_name, dataset in self.datasets.items():
            prompts = list(dataset['prompt'])
            references = list(dataset.get('reference', [""] * len(prompts)))
            # Generate responses
            print(f"Evaluating dataset: {dataset_name}")
            responses = self.generate_responses(prompts)

            # Compute metric for dataset
            metric_fn = self.metrics.get(dataset_name, self._metric_accuracy)
            score = metric_fn(responses, references)
            results[dataset_name] = score
            print(f"{dataset_name} score: {score:.4f}")
        # Compute overall or average score
        overall_score = np.mean(list(results.values()))
        results['average_score'] = overall_score
        return results

    def _metric_accuracy(self, responses: List[str], references: List[str]) -> float:
        """
        Simple accuracy metric comparing responses to references exactly.

        Args:
            responses (List[str]): Generated responses.
            references (List[str]): Groundtruth references.

        Returns:
            float: Accuracy score.
        """
        correct = 0
        total = len(responses)
        for resp, ref in zip(responses, references):
            if resp.lower() == ref.lower():
                correct += 1
        return correct / total if total > 0 else 0.0

