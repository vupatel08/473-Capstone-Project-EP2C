# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py
import os
import json
import asyncio
import aiohttp
from typing import List, Optional
import numpy as np

# Define the ComparisonPair data structure
class ComparisonPair:
    def __init__(
        self,
        prompt_response_a: Optional[str],
        response_b: Optional[str],
        preference: int,
        label_objective: str = 'undefined',
        z: Optional[float] = None  # Hidden context variable
    ):
        self.prompt_response_a = prompt_response_a
        self.response_b = response_b
        self.preference = preference  # 1 if A preferred, 0 if B preferred
        self.label_objective = label_objective  # e.g., 'helpful', 'harmless'
        self.z = z  # latent hidden context, optional

class Dataset:
    def __init__(self, pairs: List[ComparisonPair]):
        self.pairs = pairs

class DatasetLoader:
    def __init__(self, dataset_type: str, relabel: bool):
        """
        Initialize with dataset type (e.g., 'synthetic', 'relabeled_hh_rlhf', 'real')
        and whether to relabel (for simulating divergence).
        """
        self.dataset_type = dataset_type
        self.relabel = relabel
        self.data: List[ComparisonPair] = []

        # Paths for preloaded datasets (placeholders)
        self.preloaded_data_path = 'data/preloaded_dataset.json'
        # For relabeling, cache responses
        self._gpt_cache = {}
        # GPT API details (replace with actual endpoint and API key)
        self.api_url = 'https://api.openai.com/v1/chat/completions'
        self.api_key = os.getenv('OPENAI_API_KEY', '')

    def load_data(self) -> Dataset:
        if self.dataset_type == 'synthetic':
            self.data = self.generate_synthetic_data()
        elif self.dataset_type == 'relabeled_hh_rlhf':
            # Load existing dataset
            self.data = self.load_preloaded_dataset()
            if self.relabel:
                # Run relabel asynchronously
                self.data = asyncio.run(self.relabel_data(self.data))
        elif self.dataset_type == 'real':
            # Load real dataset from file (placeholder)
            self.data = self.load_real_dataset()
            if self.relabel:
                self.data = asyncio.run(self.relabel_data(self.data))
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")
        return Dataset(self.data)

    def load_preloaded_dataset(self) -> List[ComparisonPair]:
        """
        Load dataset from a JSON file.
        Assumes a list of dicts with keys: 'prompt_response_a', 'response_b', 'preference', 'objective' (optional), 'z' (if stored)
        """
        if not os.path.exists(self.preloaded_data_path):
            raise FileNotFoundError(f"Preloaded dataset not found at {self.preloaded_data_path}")
        with open(self.preloaded_data_path, 'r') as f:
            data_json = json.load(f)
        pairs = []
        for entry in data_json:
            pairs.append(
                ComparisonPair(
                    prompt_response_a=entry.get('prompt_response_a'),
                    response_b=entry.get('response_b'),
                    preference=entry.get('preference'),
                    label_objective=entry.get('objective', 'unknown'),
                    z=entry.get('z', None)
                )
            )
        return pairs

    def load_real_dataset(self) -> List[ComparisonPair]:
        """
        Placeholder: Should load actual real dataset of preferences.
        For demonstration, load similar to preloaded and adapt.
        """
        # Replace with actual dataset loading code as needed
        # For now, simulate with empty or minimal data
        # Alternatively, load from CSV, JSONL, or other formats
        # Here, just raise exception to indicate placeholder
        raise NotImplementedError("Implement actual real dataset loading here.")

    def generate_synthetic_data(self) -> List[ComparisonPair]:
        """
        Generate synthetic alternatives and comparison pairs with known hidden context.
        """
        # Generate alternatives
        alternatives = np.linspace(0, 1, num=100, dtype=float)
        pairs = []
        num_pairs = 10000  # As per config or fixed
        for _ in range(num_pairs):
            a, b = np.random.choice(alternatives, size=2, replace=False)
            z = np.random.binomial(1, 0.5)  # hidden context
            u_a = self.true_utility(a, z)
            u_b = self.true_utility(b, z)
            preference = int(u_a > u_b)
            # Optional: generate prompt text or keep None
            prompt_text = None
            pairs.append(
                ComparisonPair(
                    prompt_response_a=prompt_text,
                    response_b=None,
                    preference=preference,
                    label_objective='synthetic',
                    z=z
                )
            )
        return pairs

    def true_utility(self, a: float, z: int) -> float:
        """
        True utility function: u(a,z)=a if a<0.8; else 2*a*z
        """
        if a < 0.8:
            return a
        else:
            return 2 * a * z

    async def relabel_data(self, dataset: List[ComparisonPair]) -> List[ComparisonPair]:
        """
        Use GPT API to re-annotate pairs according to a specified objective.
        For each pair, send conversational prompt to GPT and parse the response.
        """
        relabeled_pairs = []
        tasks = []
        for comp in dataset:
            prompt_text = self.build_relabel_prompt(comp, objective=comp.label_objective)
            tasks.append(self._call_gpt_comp(prompt_text))
        # Execute in parallel
        responses = await asyncio.gather(*tasks)
        for comp, resp_text in zip(dataset, responses):
            preference = self.parse_gpt_response(resp_text, comp)
            # For harmfulness relabeling, may invert preference
            if comp.label_objective == 'harmful':
                preference = 1 - preference
            relabeled_pairs.append(
                ComparisonPair(
                    prompt_response_a=comp.prompt_response_a,
                    response_b=comp.response_b,
                    preference=preference,
                    label_objective=comp.label_objective,
                    z=comp.z
                )
            )
        return relabeled_pairs

    def build_relabel_prompt(self, comp: ComparisonPair, objective: str) -> str:
        """
        Construct a prompt string to send to GPT for relabeling.
        Insert the comparison info into the template.
        """
        # Example prompt template (can be customized)
        if objective == 'helpful':
            prompt_template = (
                "In this task, you will be presented with a prompt and two responses from an AI assistant. "
                "Your task is to determine which response is more *helpful* and *honest*. "
                "Please consider carefully and select 'A' if the first response is more helpful/honest, or 'B' if the second is. "
                "Respond only with 'A' or 'B'.\n"
                "Prompt: <PROMPT_PLACEHOLDER>\n"
                "Response A: <RESPONSE_A_PLACEHOLDER>\n"
                "Response B: <RESPONSE_B_PLACEHOLDER>\n"
                "Which is more helpful and honest? (A/B):"
            )
        elif objective == 'harmful':
            prompt_template = (
                "In this task, you will be presented with a prompt and two responses from an AI assistant. "
                "Your task is to determine which response is *more harmful*. "
                "Please select 'A' or 'B' based on which is more harmful. Respond only with 'A' or 'B'.\n"
                "Prompt: <PROMPT_PLACEHOLDER>\n"
                "Response A: <RESPONSE_A_PLACEHOLDER>\n"
                "Response B: <RESPONSE_B_PLACEHOLDER>\n"
                "Which is more harmful? (A/B):"
            )
        else:
            prompt_template = "Unknown objective."

        prompt_text = prompt_template.replace('<PROMPT_PLACEHOLDER>', comp.prompt_response_a or '')
        prompt_text = prompt_text.replace('<RESPONSE_A_PLACEHOLDER>', comp.response_b or '')
        prompt_text = prompt_text.replace('<RESPONSE_B_PLACEHOLDER>', comp.response_b or '')
        return prompt_text

    async def _call_gpt_comp(self, prompt: str) -> str:
        """
        Call the GPT API asynchronously with caching.
        """
        # Check cache first
        if prompt in self._gpt_cache:
            return self._gpt_cache[prompt]
        # Compose payload
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'system', 'content': 'You are an AI assistant.'},
                         {'role': 'user', 'content': prompt}],
            'max_tokens': 1,
            'temperature': 0
        }
        # Send request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as resp:
                    resp_json = await resp.json()
                    message = resp_json['choices'][0]['message']['content']
        except Exception as e:
            # fallback or error handling
            message = 'A'  # default fallback
        self._gpt_cache[prompt] = message.strip()
        return message.strip()

    def parse_gpt_response(self, response_text: str, comp: ComparisonPair) -> int:
        """
        Parse GPT reply ('A' or 'B') and convert to preference label (1 or 0).
        """
        resp = response_text.lower()
        if 'a' in resp:
            return 1
        elif 'b' in resp:
            return 0
        else:
            # fallback to majority or default
            return 1  # default preference if ambiguous
```

## evaluation.py

```python
## evaluation.py
"""
Evaluation module for preference models trained under various settings, focusing on:
- Calculating preference accuracy against ground truth utilities or preferences.
- Computing rank correlation metrics (Spearman's ρ, Kendall's τ) between model predictions and true utilities.
- Deriving Borda counts from model pairwise preference probabilities and comparing their orderings.
- Detecting the influence of hidden context via analysis of distributional outputs (mean, variance, or categorical probs).
Provides flexibility for synthetic and real datasets, distributional and scalar models.
"""

import numpy as np
from scipy.stats import spearmanr, kendalltau

class Evaluation:
    def __init__(self, model, dataset, ground_truth_utils=None, model_output_type='scalar'):
        """
        Initialize the evaluator.
        
        Args:
            model: The preference model instance, with method predict() that returns model outputs.
            dataset: Dataset object containing ComparisonPair instances.
            ground_truth_utils: Optional dict mapping alternative to true utility (for synthetic data).
            model_output_type: str, one of {'scalar', 'mean_var', 'categorical'} indicating model's output form.
        """
        self.model = model
        self.dataset = dataset
        self.ground_truth_utils = ground_truth_utils
        self.model_output_type = model_output_type
        # Parse dataset into structured form for evaluation
        self._prepare_data()

    def _prepare_data(self):
        """
        Prepare data arrays for evaluation:
        - All alternatives involved
        - Ground truth utilities if available
        """
        # Collect all unique alternatives
        self.alternatives = set()
        for pair in self.dataset.pairs:
            self.alternatives.add(pair.a)
            self.alternatives.add(pair.b)
        self.alternatives = sorted(list(self.alternatives))
        self.idx_map = {a: i for i, a in enumerate(self.alternatives)}
        # Store ground truth utilities if provided
        if self.ground_truth_utils:
            self.true_utils = np.array([self.ground_truth_utils.get(a, 0.0) for a in self.alternatives])
        else:
            self.true_utils = None

        # Initialize container for model predictions
        self.predicted_utils = np.zeros(len(self.alternatives))
        # To hold distributional info if needed
        if self.model_output_type != 'scalar':
            self.dist_params = [None] * len(self.alternatives)
        else:
            self.dist_params = None
        # Collect pairwise preferences for preference accuracy
        self.preference_labels = []

    def evaluate(self):
        """
        Compute evaluation metrics:
        - Preference accuracy
        - Rank correlation (Spearman, Kendall)
        - Borda counts and their comparison
        - Hidden context detection metrics
        Returns a dictionary of metrics.
        """
        preference_acc = self._compute_preference_accuracy()
        spearman_corr, _ = spearmanr(self.predicted_utils, self.true_utils) if self.true_utils is not None else (None, None)
        kendall_tau, _ = kendalltau(self.predicted_utils, self.true_utils) if self.true_utils is not None else (None, None)
        borda_scores = self.compute_borda_counts()

        detection_metrics = self.detect_hidden_context()

        metrics = {
            'preference_accuracy': preference_acc,
            'spearman_correlation': spearman_corr,
            'kendall_tau': kendall_tau,
            'borda_counts': borda_scores,
            'hidden_context_detection': detection_metrics
        }
        return metrics

    def _compute_preference_accuracy(self):
        """
        Compute the fraction of preference pairs correctly predicted by the model.
        """
        correct = 0
        total = 0
        for pair in self.dataset.pairs:
            a_idx = self.idx_map[pair.a]
            b_idx = self.idx_map[pair.b]

            pred = self._predict_preference_for_pair(pair.a, pair.b)
            if pred is None:
                continue  # Skip if prediction not computable
            # ground truth preference
            gt_pref = pair.preference
            pred_pref = 1 if pred > 0.5 else 0
            if pred_pref == gt_pref:
                correct += 1
            total += 1
        return correct / total if total > 0 else None

    def _predict_preference_for_pair(self, a, b):
        """
        Estimate preference probability that a > b based on model outputs.
        """
        a_idx = self.idx_map[a]
        b_idx = self.idx_map[b]

        if self.model_output_type == 'scalar':
            ua = self._get_utility_value(a_idx)
            ub = self._get_utility_value(b_idx)
        elif self.model_output_type == 'mean_var':
            ua = self._get_mean_variance(a_idx)[0]
            ub = self._get_mean_variance(b_idx)[0]
        elif self.model_output_type == 'categorical':
            ua = self._get_expected_utility_categorical(a_idx)
            ub = self._get_expected_utility_categorical(b_idx)
        else:
            raise ValueError(f"Unknown model output type: {self.model_output_type}")

        diff = ua - ub
        pred_prob = 1 / (1 + np.exp(-diff))
        return pred_prob

    def _get_utility_value(self, idx):
        """
        Get scalar utility value for alternative at index, from model.
        """
        # Obtain model prediction
        a_str = self.dataset.pairs[0].prompt_response_a  # Placeholder if needed
        # For batch evaluation, generate a batch of prompts as needed
        # For simplicity, assuming model.predict(a) which returns scalar
        try:
            pred_output = self.model.predict(self.dataset.pairs[idx].prompt_response_a, None)
            return pred_output['utility'].item()
        except Exception:
            # fallback or default
            return 0.0

    def _get_mean_variance(self, idx):
        """
        Retrieve mean and variance from distributional model output.
        """
        try:
            pred_output = self.model.predict(self.dataset.pairs[idx].prompt_response_a, None)
            mean = pred_output['mean'].item()
            var = pred_output['variance'].item()
            return mean, var
        except Exception:
            return 0.0, 0.0

    def _get_expected_utility_categorical(self, idx):
        """
        Compute expected utility from categorical distribution.
        """
        try:
            pred_output = self.model.predict(self.dataset.pairs[idx].prompt_response_a, None)
            probs = pred_output['probs'].detach().cpu().numpy()
            u_bins = np.linspace(0, 1, self.model.num_outputs)
            expected_util = np.sum(probs * u_bins)
            return expected_util
        except Exception:
            return 0.0

    def compute_borda_counts(self):
        """
        Calculate Borda counts for each alternative based on predicted pairwise preference probabilities.
        """
        n = len(self.alternatives)
        BC = np.zeros(n)
        for i, a in enumerate(self.alternatives):
            sum_probs = 0.0
            for j, b in enumerate(self.alternatives):
                if i == j:
                    continue
                prob_ab = self._predict_pairwise_probability(a, b)
                sum_probs += prob_ab
            BC[i] = sum_probs / (n - 1)
        # Store or output BC scores
        return {a: BC[i] for i, a in enumerate(self.alternatives)}

    def _predict_pairwise_probability(self, a: float, b: float):
        """
        Predict probability that alternative a is preferred over b.
        """
        a_idx = self.idx_map[a]
        b_idx = self.idx_map[b]
        if self.model_output_type == 'scalar':
            ua = self._get_utility_value(a_idx)
            ub = self._get_utility_value(b_idx)
        elif self.model_output_type == 'mean_var':
            ua = self._get_mean_variance(a_idx)[0]
            ub = self._get_mean_variance(b_idx)[0]
        elif self.model_output_type == 'categorical':
            ua = self._get_expected_utility_categorical(a_idx)
            ub = self._get_expected_utility_categorical(b_idx)
        else:
            raise ValueError(f"Unknown model output type: {self.model_output_type}")

        # Logistic probability
        prob = 1 / (1 + np.exp(-(ua - ub)))
        return prob

    def detect_hidden_context(self):
        """
        Analyze distributional parameters or variance to identify signals of hidden context influence.
        For models outputting variance, high variance may suggest high hidden context.
        For models with explained variance (e.g., r^2), compare residual vs. total variance.
        """
        high_variance_alternatives = []
        explained_variances = []
        for idx in range(len(self.alternatives)):
            if self.model_output_type != 'scalar' and hasattr(self.model, 'dist_params') and self.model.dist_params:
                # For distributional outputs
                params = self.model.dist_params[idx]
                if params is None:
                    continue
                if 'variance' in params:
                    var = params['variance']
                    # Threshold arbitrarily set at some value (e.g., >0.1)
                    if var > 0.1:
                        high_variance_alternatives.append(self.alternatives[idx])
                elif 'probs' in params:
                    # For categorical, variance can be derived
                    probs = params['probs']
                    expected_util = self._get_expected_utility_categorical(idx)
                    variance = np.sum(probs.numpy() * (np.linspace(0,1,self.model.num_outputs) - expected_util)**2)
                    if variance > 0.1:
                        high_variance_alternatives.append(self.alternatives[idx])
        # Optionally, compute aggregate metrics
        return {
            'high_variance_alternatives': high_variance_alternatives
        }
```

## main.py

```python
## main.py
import yaml
import os
import torch
import numpy as np
from tqdm import tqdm

from dataset_loader import DatasetLoader
from model import PreferenceModel
from trainer import Trainer
from evaluation import Evaluation

def main():
    # 1. Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Parse training parameters
    train_cfg = config.get('training', {})
    learning_rate = train_cfg.get('learning_rate', 3e-6)
    min_lr = train_cfg.get('min_learning_rate', 3e-7)
    batch_size = train_cfg.get('batch_size', 2)
    epochs = train_cfg.get('epochs', 2)
    lambda_reg = train_cfg.get('lambda_reg', 0.0001)
    reg_type = train_cfg.get('regularization_type', 'l2')
    use_regularization = train_cfg.get('use_regularization', True)
    optimizer_name = train_cfg.get('optimizer', 'AdamW')
    lr_schedule = train_cfg.get('lr_schedule', 'cosine')
    # total_steps can be computed later based on dataset size and epochs

    # Parse model parameters
    model_cfg = config.get('model', {})
    base_model_name = model_cfg.get('base_model', 'llama-2-7b-hf')
    head_type = model_cfg.get('head_type', 'scalar')
    num_outputs = model_cfg.get('num_outputs', 1)
    lora_rank = model_cfg.get('lora_rank', 8)

    # Parse dataset setup
    dataset_cfg = config.get('dataset', {})
    dataset_type = dataset_cfg.get('dataset_type', 'synthetic')
    synthetic_size = dataset_cfg.get('synthetic_size', 10000)
    relabel = dataset_cfg.get('relabel', False)

    # Save directory for checkpoints
    save_dir = 'saved_model'
    os.makedirs(save_dir, exist_ok=True)

    # 2. Load or generate dataset
    print("Loading dataset...")
    data_loader_obj = DatasetLoader(dataset_type, relabel)

    if dataset_type == 'synthetic':
        # Generate synthetic dataset with known hidden context effects
        alternatives, pairs = data_loader_obj.generate_synthetic_data({'synthetic_size': synthetic_size})
        # For evaluation, store true utilities if needed
        true_utilities = {}
        for a in alternatives:
            # example: true U(a,z) with z ~ Bernoulli(0.5)
            # But in code, actual true utility is defined in synthetic_data, so optional
            pass
        dataset = data_loader_obj.load_data()
    elif dataset_type == 'relabeled_hh_rlhf':
        # Load existing dataset, then relabel if needed
        dataset = data_loader_obj.load_data()
    else:
        # For real datasets, implement actual loading here
        dataset = data_loader_obj.load_data()

    # 3. Initialize model
    print("Initializing model...")
    model_config = {
        'base_model': base_model_name,
        'head_type': head_type,
        'num_outputs': num_outputs,
        'lora_rank': lora_rank
    }
    model = PreferenceModel(model_config)

    # 4. Setup Trainer
    print("Setting up training...")
    total_dataset_size = len(dataset.pairs)
    steps_per_epoch = (total_dataset_size + batch_size - 1) // batch_size
    total_training_steps = steps_per_epoch * epochs

    trainer_args = {
        'batch_size': batch_size,
        'epochs': epochs,
        'lambda_reg': lambda_reg,
        'regularization_type': reg_type,
        'use_regularization': use_regularization,
        'learning_rate': learning_rate,
        'min_learning_rate': min_lr,
        'optimizer': optimizer_name,
        'scheduler': lr_schedule,
        'total_steps': total_training_steps
    }

    trainer = Trainer(model, dataset, trainer_args)

    # 5. Train the model
    print("Starting training...")
    trainer.train()

    # Save final model
    final_path = os.path.join(save_dir, 'final_model')
    print(f"Saving final model to {final_path}")
    model.transformer.save_pretrained(final_path)

    # 6. Evaluation
    print("Evaluating model...")
    # For synthetic data, true utilities are known (if stored); for real, may be absent
    # Here, just demonstrate evaluation
    eval_obj = Evaluation(model, dataset)
    metrics = eval_obj.evaluate()

    # Print evaluation metrics
    print("\n=== Evaluation Results ===")
    print(f"Preference accuracy: {metrics.get('preference_accuracy', 'N/A'):.4f}")
    spearman = metrics.get('spearman_correlation', None)
    if spearman is not None:
        print(f"Spearman correlation to true utilities: {spearman:.4f}")
    kendall = metrics.get('kendall_tau', None)
    if kendall is not None:
        print(f"Kendall tau: {kendall:.4f}")
    print("Borda counts:")
    for alt, score in metrics.get('borda_counts', {}).items():
        print(f"  Alt {alt:.3f}: {score:.3f}")
    print("Hidden context detection (variance):")
    high_var_alts = metrics.get('hidden_context_detection', {}).get('high_variance_alternatives', [])
    if high_var_alts:
        print("Alternatives with high variance indicating hidden context influence:")
        for alt in high_var_alts:
            print(f"  {alt}")
    else:
        print("No high variance signals detected.")

    # Optional: Save metrics to file
    metrics_path = os.path.join(save_dir, 'evaluation_metrics.yaml')
    with open(metrics_path, 'w') as f:
        yaml.safe_dump(metrics, f)

    print("Experiment complete.")

if __name__ == "__main__":
    main()
```

## model.py

```python
# model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, Tuple

# Import LoRA modules
try:
    from peft import get_peft_model, LoraConfig
except ImportError:
    raise ImportError("Please install 'peft' library for LoRA support: pip install peft")


class PreferenceModel(nn.Module):
    """
    This class defines a flexible preference model built upon a pre-trained LLAMA-2-7B.
    It supports three output head configurations:
      - 'scalar': outputs a single utility score.
      - 'mean_var': outputs mean and variance of utility distribution.
      - 'categorical': outputs logits over discretized utility bins.
    It integrates LoRA with a specified rank and ensures compatibility with training routines.
    """
    def __init__(
        self,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the PreferenceModel.
        
        Args:
            config (Dict): configuration dictionary containing:
                - 'base_model': str, name of pre-trained model, e.g. 'llama-2-7b-hf'
                - 'head_type': str, one of {'scalar', 'mean_var', 'categorical'}
                - 'num_outputs': int, number of outputs (for 'categorical', e.g., 10)
                - 'lora_rank': int, rank for LoRA adaptation
            device (Optional[torch.device]): computation device, default=None (auto)
        """
        super().__init__()
        # Load pre-trained LLAMA-2-7B model with causal language modeling head
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model_name = config.get('base_model', 'llama-2-7b-hf')
        self.head_type = config.get('head_type', 'scalar')
        self.num_outputs = config.get('num_outputs', 1)
        self.lora_rank = config.get('lora_rank', 8)
        
        # Load model
        self.transformer = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            load_in_8bit=False
        ).to(self.device)
        
        # Optional: Initialize tokenizer if needed elsewhere
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Initialize LoRA configuration
        self._apply_lora()

        # Build output head based on head_type
        hidden_size = self.transformer.config.hidden_size
        if self.head_type == 'scalar':
            self.head = nn.Linear(hidden_size, 1)
        elif self.head_type == 'mean_var':
            # Two heads: one for mean, one for log-variance
            self.head_mean = nn.Linear(hidden_size, 1)
            self.head_log_var = nn.Linear(hidden_size, 1)
        elif self.head_type == 'categorical':
            self.head = nn.Linear(hidden_size, self.num_outputs)
        else:
            raise ValueError(f"Invalid head_type: {self.head_type}")

    def _apply_lora(self):
        """
        Wrap the transformer model with LoRA using PEFT library.
        """
        peft_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            target_modules=["\u03A3" if hasattr(self.transformer, 'model') else 'mlp', 'q_proj', 'k_proj', 'v_proj', 'o_proj'],
            # For LLAMA, typical target modules include attention projection matrices
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.transformer = get_peft_model(self.transformer, peft_config)

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        Args:
            input_ids (torch.LongTensor): tokenized input of shape (batch_size, seq_len)
            attention_mask (Optional[torch.LongTensor]): attention mask
        
        Returns:
            Dict[str, torch.Tensor]: outputs with key 'utility', 'mean', 'variance', or 'logits'
        """
        # Pass through transformers
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Pooling: use the last token (e.g., for causal LM, the last token output)
        # Alternatively, can pool over tokens or use CLS token if available
        pooled_output = last_hidden_state[:, -1, :]  # (batch_size, hidden_size)
        
        # Generate the utility estimate based on head_type
        if self.head_type == 'scalar':
            utility = self.head(pooled_output).squeeze(-1)  # (batch_size,)
            return {'utility': utility}
        elif self.head_type == 'mean_var':
            mean = self.head_mean(pooled_output).squeeze(-1)  # (batch_size,)
            log_var = self.head_log_var(pooled_output).squeeze(-1)  # (batch_size,)
            variance = torch.exp(log_var)
            return {'mean': mean, 'variance': variance}
        elif self.head_type == 'categorical':
            logits = self.head(pooled_output)  # (batch_size, num_outputs)
            probs = torch.softmax(logits, dim=-1)
            return {'logits': logits, 'probs': probs}
        else:
            raise ValueError(f"Invalid head_type: {self.head_type}")

    def save(self, save_path: str):
        """
        Save model checkpoint.
        """
        self.transformer.save_pretrained(save_path)
        # Save additional config if necessary
        torch.save({
            'head_type': self.head_type,
            'num_outputs': self.num_outputs,
            'lora_rank': self.lora_rank
        }, f"{save_path}/config.pt")

    def load(self, load_path: str):
        """
        Load model checkpoint.
        """
        self.transformer = AutoModelForCausalLM.from_pretrained(load_path).to(self.device)
        # Load config if needed
        # Possibly reload LoRA weights if stored separately
        #For simplicity, assume model is saved with PEFT
        # The model's config (head_type, etc.) should be reloaded as needed
        pass

    def to(self, device: torch.device):
        """
        Move model to specified device.
        """
        self.device = device
        self.transformer.to(device)
        return self
```

## synthetic_data.py

```python
## synthetic_data.py

import numpy as np
from scipy.stats import bernoulli

# Configuration defaults (can be overridden externally)
DEFAULT_N_ALTERNATIVES = 100  # number of alternatives
DEFAULT_NOISE_LEVEL = 0.0     # 0 for noiseless preference, >0 for stochastic
DEFAULT_Z_DISTRIBUTION = 'bernoulli'  # 'bernoulli' or 'uniform'
DEFAULT_P_Z = 0.5             # Bernoulli parameter p
DEFAULT_Z_RANGE = (0, 1)      # uniform range for z if used

# Define the data structure for comparison pairs
class ComparisonPair:
    def __init__(self, a, b, preference, z=None):
        self.a = a
        self.b = b
        self.preference = preference  # 1 if a preferred, 0 if b preferred
        self.z = z  # latent context sample, optional

def generate_alternatives(n):
    """
    Generate n alternatives evenly spaced in [0, 1].
    """
    return np.linspace(0, 1, n)

def sample_hidden_context(size, distribution='bernoulli', p=0.5, z_range=(0, 1)):
    """
    Sample hidden context variable z for each comparison.
    Supports Bernoulli or uniform distributions.
    """
    if distribution == 'bernoulli':
        # Bernoulli with parameter p
        return np.random.binomial(1, p, size)
    elif distribution == 'uniform':
        low, high = z_range
        return np.random.uniform(low, high, size)
    else:
        raise ValueError(f"Unsupported Z distribution: {distribution}")

def true_utility(a, z):
    """
    True utility function u(a, z):
    - For a < 0.8: utility = a
    - For a >= 0.8: utility = 2 * a * z
    """
    if np.isscalar(a):
        a = float(a)
        if a < 0.8:
            return a
        else:
            return 2 * a * z
    else:
        # a is array
        util = np.zeros_like(a)
        mask = a >= 0.8
        util[~mask] = a[~mask]
        util[mask] = 2 * a[mask] * z
        return util

def preference_outcome(a, b, z, noise=False):
    """
    Simulate preference between a and b given hidden context z.
    Uses probabilistic Bradley-Terry model.
    """
    util_a = true_utility(a, z)
    util_b = true_utility(b, z)
    prob_a_pref = np.exp(util_a) / (np.exp(util_a) + np.exp(util_b))
    if noise:
        # Add stochasticity: prefer a with probability p
        return np.random.rand() < prob_a_pref
    else:
        # Deterministic: highest utility preferred
        return util_a > util_b

def generate_comparison_pair(alternatives):
    """
    Generate a single pair of alternatives with associated preference outcome.
    """
    a, b = np.random.choice(alternatives, size=2, replace=False)
    # Sample hidden context z for the comparison
    z_samples = sample_hidden_context(1)
    z = z_samples[0]
    # Generate preference outcome (no noise for ground truth)
    preference = int(preference_outcome(a, b, z, noise=False))
    return ComparisonPair(a=a, b=b, preference=preference, z=z)

def generate_dataset(alternatives, num_pairs, dist='bernoulli', p=0.5, z_range=(0,1)):
    """
    Generate a dataset of comparison pairs with hidden context.
    """
    dataset = []
    for _ in range(num_pairs):
        pair = generate_comparison_pair(alternatives)
        dataset.append(pair)
    return dataset

# Additional helper: generate synthetic dataset with known true utilities and optional noise
def generate_synthetic_data(config):
    """
    Generate synthetic preference dataset based on config parameters.
    """
    n = config.get('synthetic_size', DEFAULT_N_ALTERNATIVES)
    alternatives = generate_alternatives(n)
    num_pairs = n * (n - 1) // 2  # all pairs or can be adjusted
    dist = config.get('z_distribution', DEFAULT_Z_DISTRIBUTION)
    p_z = config.get('p_z', DEFAULT_P_Z)
    z_range = config.get('z_range', DEFAULT_Z_RANGE)
    dataset = generate_dataset(alternatives, num_pairs, dist, p_z, z_range)
    return alternatives, dataset

# Example usage (can be removed or commented out in production)
if __name__ == "__main__":
    # Generate synthetic data with default parameters
    alternatives, dataset = generate_synthetic_data({'synthetic_size': 1000})
    # Print some sample data
    for idx, pair in enumerate(dataset[:5]):
        print(f"Pair {idx}: a={pair.a:.3f}, b={pair.b:.3f}, preference={pair.preference}, z={pair.z}")
```

## trainer.py

```python
# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from typing import Optional
from transformers import get_scheduler
from tqdm import tqdm
from dataset_loader import Dataset, ComparisonPair
from model import PreferenceModel

class Trainer:
    def __init__(
        self,
        model: PreferenceModel,
        dataset: Dataset,
        config: dict,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the Trainer with model, dataset, and configuration parameters.
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        # Dataset and DataLoader
        self.dataset = dataset
        # We'll create a DataLoader with a batch of comparison pairs
        self.batch_size = config.get('batch_size', 2)
        self.epochs = config.get('epochs', 2)
        self.lambda_reg = config.get('lambda_reg', 0.0001)
        self.regularization_type = config.get('regularization_type', 'l2')
        self.use_regularization = config.get('use_regularization', True)
        self.learning_rate = config.get('learning_rate', 3e-6)
        self.min_learning_rate = config.get('min_learning_rate', 3e-7)
        self.total_steps = None  # Will be calculated
        self.optimizer_name = config.get('optimizer', 'AdamW')
        self.lr_scheduler_type = config.get('scheduler', 'cosine')
        self._setup_dataloader()

        # Initialize optimizer
        if self.optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        else:
            raise RuntimeError(f"Unsupported optimizer: {self.optimizer_name}")

        # Calculate total training steps
        dataset_size = len(self.dataset.pairs)
        steps_per_epoch = math.ceil(dataset_size / self.batch_size)
        self.total_steps = steps_per_epoch * self.epochs

        # Setup scheduler
        self.lr_scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.total_steps
        )

        # Save path for checkpoints
        self.checkpoint_path = 'checkpoint'
        self.best_metric = None
        self.best_model_state = None

    def _setup_dataloader(self):
        """
        Create a PyTorch DataLoader for batch processing.
        """
        # For simplicity, create a list of data point tuples for batching
        # Since dataset.pairs is a list of ComparisonPair, wrap accordingly
        # We'll directly convert pairs into a DataLoader with collate_fn
        self.data_loader = DataLoader(
            self.dataset.pairs,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        """
        Collate function to process list of ComparisonPair into tensors.
        """
        # Batch is a list of ComparisonPair objects
        a_list = [pair.prompt_response_a for pair in batch]
        b_list = [pair.response_b for pair in batch]
        preference_labels = torch.tensor([pair.preference for pair in batch], dtype=torch.float32).to(self.device)
        # For easy processing, store in dict
        batch_dict = {
            'a_strs': a_list,
            'b_strs': b_list,
            'preference_labels': preference_labels
        }
        return batch_dict

    def train(self):
        """
        Run the training loop over epochs, steps, compute losses, update model.
        """
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            progress_bar = tqdm(self.data_loader, desc=f"Epoch {epoch}/{self.epochs}")
            for batch in progress_bar:
                # Tokenize inputs
                a_inputs = self._tokenize_batch(batch['a_strs'])
                b_inputs = self._tokenize_batch(batch['b_strs'])

                # Forward pass for both alternatives
                self.model.train()
                self.optimizer.zero_grad()

                out_a = self.model(**a_inputs)
                out_b = self.model(**b_inputs)

                # Compute preference loss
                loss_pref = self._compute_preference_loss(out_a, out_b, batch['preference_labels'])

                # Optional: add regularization
                loss_reg = torch.tensor(0.0, device=self.device)
                if self.use_regularization:
                    loss_reg = self._compute_regularization(out_a, out_b)

                total_loss = loss_pref + self.lambda_reg * loss_reg

                total_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                epoch_loss += total_loss.item()
                progress_bar.set_postfix(loss=total_loss.item(), lr=self.optimizer.param_groups[0]['lr'])

            # Save checkpoint at epoch end
            if epoch == self.epochs or epoch % 1 == 0:
                self._save_checkpoint(epoch)

            # Optionally, evaluate metrics on validation set here if available

    def _tokenize_batch(self, texts):
        """
        Tokenize list of strings into model inputs.
        Assumes model has a tokenizer.
        """
        # Access model's tokenizer (assumed stored)
        tokenizer = self.model.tokenizer
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        return encoded

    def _compute_preference_loss(self, out_a, out_b, preference_labels):
        """
        Compute the pairwise preference logistic loss.
        """
        if self.model.head_type == 'scalar':
            # Extract utility scores
            u_a = out_a['utility']
            u_b = out_b['utility']
            diff = u_a - u_b
        elif self.model.head_type == 'mean_var':
            # Use mean
            u_a = out_a['mean']
            u_b = out_b['mean']
            diff = u_a - u_b
        elif self.model.head_type == 'categorical':
            # Use expected utility: sum over probs * utility bins
            probs_a = out_a['probs']
            probs_b = out_b['probs']
            # For estimation, approximate expected utility
            u_bins = torch.linspace(0, 1, self.model.num_outputs, device=self.device)
            util_a = (probs_a * u_bins).sum(dim=-1)
            util_b = (probs_b * u_bins).sum(dim=-1)
            diff = util_a - util_b
        else:
            raise ValueError(f"Invalid head_type: {self.model.head_type}")

        # Logistic sigmoid
        pred_probs = torch.sigmoid(diff)
        # Binary cross-entropy loss
        loss = nn.BCELoss()(pred_probs, preference_labels)
        return loss

    def _compute_regularization(self, out_a, out_b):
        """
        Compute regularization penalty on model outputs.
        For simplicity, apply to scalar/utilities or logits.
        """
        reg_loss = torch.tensor(0.0, device=self.device)
        if self.model.head_type == 'scalar':
            # L2 on utility outputs
            # Assume model.head weights and biases as regularization
            for param in self.model.parameters():
                if param.ndim > 1:
                    reg_loss += torch.norm(param, p=2)
        elif self.model.head_type == 'mean_var':
            # L2 on network weights
            for param in self.model.parameters():
                if param.ndim > 1:
                    reg_loss += torch.norm(param, p=2)
        elif self.model.head_type == 'categorical':
            # L2 on logits
            for param in self.model.parameters():
                if param.ndim > 1:
                    reg_loss += torch.norm(param, p=2)
        return reg_loss

    def _save_checkpoint(self, epoch):
        """
        Save model checkpoint.
        """
        save_path = f"{self.checkpoint_path}_epoch{epoch}"
        self.model.transformer.save_pretrained(save_path)
        # Save additional configs if necessary
        # Could save optimizer state, scheduler state, hyperparameters etc.
        print(f"Saved checkpoint at {save_path}")

    def _load_checkpoint(self, load_path):
        """
        Load model checkpoint.
        """
        self.model.transformer = self.model.transformer.from_pretrained(load_path).to(self.device)

```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..dataset_out\paper2code\hidden-context\hidden-context_repo`
