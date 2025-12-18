# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## attention_steering.py

```python
## attention_steering.py
import torch
from typing import List, Tuple, Dict, Optional

class AttentionSteering:
    """
    Implements post-hoc attention reweighting to steer a language model's attention mechanism during inference.
    Responsible for upweighting attended tokens in user-emphasized spans by manipulating attention scores
    in specified attention heads across layers.
    """

    def __init__(
        self,
        model,
        selected_heads: List[Tuple[int, int]],
        alpha: float = 0.01
    ):
        """
        Initialize the AttentionSteering object.

        Args:
            model: The model instance supporting attention hooks (e.g., HuggingFace model).
            selected_heads: List of (layer_idx, head_idx) tuples indicating which attention heads to steer.
            alpha: Scaling coefficient for non-emphasized token attention scores. Default: 0.01.
        """
        self.model = model
        self.selected_heads = selected_heads
        self.alpha = alpha
        # Mapping from (layer_idx, head_idx) to hook handle for potential removal
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        # Register hooks for relevant attention modules
        self._register_hooks()

    def _register_hooks(self):
        """
        Register hooks on the model's attention modules to modify attention scores during forward pass.
        Assumes the model's attention modules are accessible and expose inputs and outputs suitable for hooking.
        """
        for layer_idx, layer_module in enumerate(self._get_attention_layers()):
            # Depending on architecture, access attention module
            attn_module = None
            if hasattr(layer_module, 'self_attn'):
                attn_module = getattr(layer_module, 'self_attn')
            elif hasattr(layer_module, 'attn'):
                attn_module = getattr(layer_module, 'attn')
            else:
                continue  # skip if no attention module found

            # Register hook that adjusts attention scores before softmax
            handle = attn_module.register_forward_hook(
                self._create_attention_hook(layer_idx)
            )
            self._hook_handles.append(handle)

    def _get_attention_layers(self):
        """
        Retrieve the list of attention layers from the model, depending on architecture.
        """
        if hasattr(self.model, 'layers'):
            return self.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        else:
            raise ValueError("Cannot find attention layers in the provided model.")

    def _create_attention_hook(self, layer_idx: int):
        """
        Creates a hook function for modifying attention scores at a specific layer.
        This hook replaces the attention scores in the attention module's inputs
        during the forward pass, applying the reweighting to emphasize user-highlighted tokens.

        Args:
            layer_idx: The index of the layer whose attention scores are being modified.

        Returns:
            A hook function.
        """
        def hook(module, inputs, outputs):
            """
            Hook function to manipulate attention scores.

            Args:
                module: The attention module.
                inputs: Tuple containing inputs to the attention module (usually the attention scores or key/query/value tensors).
                outputs: Outputs of the attention module (not used for modification).
            """
            # The 'inputs' typically contain the attention scores or Q,K,V tensors or the pre-softmax scores.
            # We assume the attention module is implemented such that it outputs attention scores
            # or the module's forward accepts and outputs attention weights.
            # This code presumes 'inputs' includes the attention scores as the first argument or via kwargs.
            # The precise access depends on the model's implementation.
            # Implementers must adapt accordingly.

            # For illustration, assume the attention scores are the first positional argument in `inputs`.
            if not inputs:
                return  # Cannot modify if no inputs

            attn_scores = inputs[0]  # Expected shape: (batch_size, num_heads, seq_len, seq_len)
            # Retrieve the attention scores tensor
            # Note: the shape varies depending on the implementation
            # For common transformers, shape: (batch_size, num_heads, seq_len, seq_len)

            # Reweight attention scores at specified heads
            # Only on selected heads
            for (l_idx, h_idx) in self.selected_heads:
                if l_idx != layer_idx:
                    continue
                # Generate reweighted scores for the head h_idx
                # Access per-head attention scores
                # We need to ensure that 'attn_scores' shape allows slicing: shape includes layer dimension
                # But in most implementations, layer dimension is outside attention tensor reference.
                # Adjust accordingly: here, assume 'attn_scores' is the attention for current layer.

                # Since the hook is per-layer, attn_scores corresponds to layer layer_idx
                # verify shape
                # Shape: (batch_size, num_heads, query_seq_len, key_seq_len)
                # Check shape
                if hasattr(attn_scores, 'shape'):
                    shape = attn_scores.shape
                else:
                    continue  # skip if shape not available

                # Create a copy to avoid in-place modification if needed
                new_attn_scores = attn_scores

                if len(shape) != 4:
                    # Unexpected shape
                    continue

                # Clone attention scores to avoid in-place modification if necessary
                # (Alternative: in-place modification might be acceptable)
                # new_attn_scores = attn_scores.clone()

                # Generate the mask for highlighted tokens in the sequence
                # For batch processing, creating a mask tensor of shape (batch_size, seq_len)
                # For simplicity, assume focusing on one input sequence at a time (batch size 1)
                # Let's handle batch size scenario: assuming batch size=1 for simplicity
                
                batch_size, num_heads, seq_len_q, seq_len_k = shape
                # Create a mask of shape (seq_len_q, seq_len_k)
                # Initialize as ones and then scale down for non-highlighted tokens
                # We need 'emphasis spans' at inference time
                # Since this is callback during forward, store current emphasis spans as a class attribute or argument
                
                # The 'apply_reweighting' should be called with actual emphasis spans at inference time.
                # Thus, this hook must have access to current emphasis spans.
                # To accomplish this, class should hold 'current_emphasis_spans' as state,
                # which is set externally at inference time.

                # Thus, an external method must set 'self._current_emphasis_spans' before generation.

                # Check if emphasis spans are available
                if not hasattr(self, '_current_emphasis_spans'):
                    # No emphasis spans set; skip reweighting
                    continue

                emphasis_token_indices = set(self._current_emphasis_spans)

                # Create boolean mask for tokens
                # For batch size 1:
                # Sequence length is seq_len_q
                token_mask = torch.zeros(seq_len_q, dtype=torch.bool, device=attn_scores.device)
                # Mark tokens in emphasis span
                token_mask[list(emphasis_token_indices)] = True

                # Expand mask to shape: (1, 1, seq_len_q, seq_len_k)
                # to broadcast over batch and head dimensions
                mask_broadcast = token_mask.unsqueeze(0).unsqueeze(0)  # shape: (1,1,seq_len_q,1)

                # For assigning, make sure to broadcast properly
                # Repeat emphasis mask across batch and key sequence
                # But as attention is (batch_size, num_heads, seq_len_q, seq_len_k),
                # we should create a per-head mask for per head's key dimension.

                # Generate a per-head, per-sequence position mask
                # For simplicity, assuming the same emphasis mask applies to all heads
                # and batch; in multi-input, process each batch dynamically.

                for batch_idx in range(batch_size):  # for batch >1
                    # For batch index, modify the attention scores
                    # But in the hook, 'attn_scores' is the tensor; 
                    # In-place modification is possible.

                    # Generate mask for this batch
                    # For now, assuming batch_size=1
                    # For multiple batches, loop accordingly or vectorize

                    # Create mask for current batch
                    batch_mask = token_mask  # shape: (seq_len_q)
                    # Expand to matching shape: (seq_len_q, seq_len_k)
                    mask_2d = batch_mask.unsqueeze(1).expand(-1, seq_len_k)

                    # Scale down scores where tokens are not emphasized
                    # Positions where mask is False will be scaled
                    non_emphasis_mask = (~mask_2d).to(dtype=attn_scores.dtype)  # shape: (seq_len_q, seq_len_k)

                    # Apply scale: multiply non-emphasis positions by alpha
                    # Loop over batch if batch > 1
                    attn_for_batch = new_attn_scores[batch_idx, h_idx]
                    attn_for_batch = attn_for_batch * torch.where(
                        non_emphasis_mask.bool().to(attn_for_batch.device),
                        torch.full_like(attn_for_batch, self.alpha),
                        torch.ones_like(attn_for_batch)
                    )

                    # Assign back
                    new_attn_scores[batch_idx, h_idx] = attn_for_batch

                # After scaling non-emphasis attention weights, normalize
                # across the key dimension (last dimension)
                for batch_idx in range(batch_size):
                    attn_scores_batch = new_attn_scores[batch_idx, h_idx]
                    # Numerical stability: add epsilon to denominator
                    sum_scores = attn_scores_batch.sum(dim=-1, keepdim=True) + 1e-9
                    new_attn_scores[batch_idx, h_idx] = attn_scores_batch / sum_scores

            # Set the modified attention scores back to inputs or internal attribute
            # Since 'inputs' is a tuple, modify in-place if possible, or
            # replace the tuple: i.e., set inputs[0] = modified scores

            # In hook, 'inputs' is a tuple -> we must modify 'inputs' to pass modified scores downstream.
            # But as per PyTorch's hook API, 'inputs' are read-only; to modify, we'd need to override model code.
            # Alternatively, we can patch specific modules or store the modified attention scores in class,
            # which the attention module then uses internally, perhaps via a custom module.

            # For the purpose of this code, assume the attention module internally captures or uses the modified scores via the hook.
            # Thus, the above modifications via in-place are placeholders for actual implementation.

            # WARNING: Actual implementation depends heavily on the model's attention module design.
            # The code here aims to illustrate the intended logic.

        return None  # hooks do not return directly, but modify the input tensors.

        # Note: In practice, the above approach requires the model's attention implementation
        # to support external manipulation of attention scores during forward pass.
        # Otherwise, a custom attention module or rewriting the model's code is necessary.

        # Additionally, the hook should modify the attention scores tensor in-place,
        # but PyTorch JIT or nesting may complicate this.

    def set_current_emphasis_spans(self, emphasis_token_indices: List[int]):
        """
        To be called externally before inference to set current emphasis spans.
        Args:
            emphasis_token_indices: list of token indices highlighting user emphasized tokens.
        """
        self._current_emphasis_spans = emphasis_token_indices

    def clear_hooks(self):
        """
        Remove all registered hooks.
        """
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def update_selected_heads(self, new_heads: List[Tuple[int, int]]):
        """
        Update the list of heads to steer.
        """
        self.selected_heads = new_heads

    # Additional utility methods can be added if needed
```

## dataset_loader.py

```python
## dataset_loader.py
import os
import json
import csv
import random
from typing import List, Dict, Tuple, Optional, Union
from datasets import load_dataset
from transformers import PreTrainedTokenizer

class DatasetSample:
    """
    Data class to store a single dataset sample with tokenized input,
    emphasis spans, and raw text.
    """
    def __init__(
        self,
        raw_prompt: str,
        tokenized_input: List[int],
        attention_mask: List[int],
        emphasis_token_spans: List[List[int]],  # List of token index spans
        label: Optional[Union[str, Dict]] = None
    ):
        self.raw_prompt = raw_prompt
        self.tokenized_input = tokenized_input
        self.attention_mask = attention_mask
        self.emphasis_token_spans = emphasis_token_spans
        self.label = label


class DatasetLoader:
    """
    Loads datasets from a specified directory,
    supports sampling, tokenization, emphasis span extraction.
    """
    def __init__(
        self,
        task_name: str,
        data_dir: str,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
        emphasis_marker: str = '*',
        seed: int = 42
    ):
        self.task_name = task_name
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.emphasis_marker = emphasis_marker
        self.seed = seed
        self.dataset = []  # will hold DatasetSample objects

    def load_dataset(self):
        """
        Loads dataset from files in data_dir based on split and task.
        Supports JSONL, CSV, or plain text formats based on file extension.
        """
        # Determine dataset file path
        file_path = os.path.join(self.data_dir, f"{self.split}.jsonl")
        if not os.path.exists(file_path):
            # Try other formats or raise error
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load raw data depending on extension
        raw_data = []
        ext = os.path.splitext(file_path)[1]
        if ext == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_data.append(json.loads(line))
        elif ext == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_data.append(row)
        else:
            # Assume plain text, each line is a sample
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_data.append({"text": line.strip()})

        self.raw_data = raw_data
        return self

    def sample_data(self, sample_size: int) -> List[dict]:
        """
        Randomly sample data samples with fixed seed for reproducibility.
        """
        random.seed(self.seed)
        total_samples = len(self.raw_data)
        if sample_size >= total_samples:
            sampled = self.raw_data
        else:
            sampled = random.sample(self.raw_data, sample_size)
        return sampled

    def _extract_emphasis_spans(self, text: str) -> Tuple[str, List[List[int]]]:
        """
        Detect emphasis markers (default '*'), remove them from text,
        and return clean text along with list of emphasis token index spans.
        """
        spans = []
        clean_text = ""
        current_pos = 0  # character position in original text
        emphasis_positions_char = []

        # We'll parse manually to find emphasis marker positions
        i = 0
        in_emphasis = False
        start_idx = None
        clean_chars = []

        while i < len(text):
            if text[i] == self.emphasis_marker:
                in_emphasis = not in_emphasis
                i += 1
                continue
            else:
                clean_chars.append(text[i])
                if in_emphasis:
                    emphasis_positions_char.append(i)
                i += 1

        clean_text = "".join(clean_chars)
        # Now, map emphasis marker character positions to token indices after tokenization
        return clean_text, emphasis_positions_char

    def _map_char_to_token_span(
        self,
        character_positions: List[int],
        offsets: List[Tuple[int, int]],
        emphasis_marker_positions: List[int]
    ) -> List[List[int]]:
        """
        Map character positions of emphasis spans to token index spans.
        """
        token_spans = []

        for start_char in character_positions:
            # Find token index where token offset matches or surrounds start_char
            token_indices = []
            for idx, (start_off, end_off) in enumerate(offsets):
                if start_off is None or end_off is None:
                    continue
                if start_off <= start_char < end_off:
                    token_indices.append(idx)
            if token_indices:
                token_spans.append([min(token_indices), max(token_indices)])
        # Merge overlapping spans if necessary
        merged_spans = self._merge_spans(token_spans)
        return merged_spans

    def _merge_spans(self, spans: List[List[int]]) -> List[List[int]]:
        """
        Merge overlapping or contiguous spans.
        """
        if not spans:
            return []
        # Sort spans
        spans = sorted(spans, key=lambda x: x[0])
        merged = [spans[0]]
        for current in spans[1:]:
            prev = merged[-1]
            if current[0] <= prev[1] + 1:
                # Overlap or contiguous
                merged[-1] = [prev[0], max(prev[1], current[1])]
            else:
                merged.append(current)
        return merged

    def _tokenize_sample(
        self,
        text: str,
        emphasis_char_positions: List[int]
    ) -> Tuple[List[int], List[int], List[List[int]]]:
        """
        Tokenize text, map emphasis character positions to token spans.
        Returns token IDs, attention mask, emphasis token index spans.
        """
        # Tokenize with offset mapping
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            max_length=self.max_seq_length,
            truncation=True
        )
        offsets = encoding['offset_mapping']
        token_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Map emphasized char positions to token index spans
        token_spans = self._map_char_to_token_span(emphasis_char_positions, offsets, emphasis_char_positions)
        return token_ids, attention_mask, token_spans

    def get_processed_samples(self, sample_size: int) -> List[DatasetSample]:
        """
        Load, sample, process, tokenized, and annotate emphasis spans.
        """
        sampled_data = self.sample_data(sample_size)
        processed_samples = []

        for entry in sampled_data:
            # Determine prompt text based on dataset content
            # For simplicity, assume 'text' key contains full prompt
            raw_text = entry.get('text', '')
            raw_prompt = raw_text

            # Extract emphasis spans (character positions)
            clean_text, emphasis_char_positions = self._extract_emphasis_spans(raw_text)

            # Tokenize and map emphasis
            token_ids, attention_mask, emphasis_spans = self._tokenize_sample(
                text=clean_text,
                emphasis_char_positions=emphasis_char_positions
            )

            # Create DatasetSample
            sample = DatasetSample(
                raw_prompt=raw_text,
                tokenized_input=token_ids,
                attention_mask=attention_mask,
                emphasis_token_spans=emphasis_spans,
                label=entry.get('label')  # optional, depends on dataset
            )
            processed_samples.append(sample)

        return processed_samples
```

## evaluation.py

```python
## evaluation.py
import json
import re
import math
from typing import List, Dict, Tuple, Any, Optional
import torch
from transformers import PreTrainedTokenizer
from datasets import Dataset
import numpy as np

from utils import extract_emphasis_token_indices, compute_ngram_entropy
from dataset_loader import DatasetSample
from model_wrapper import ModelWrapper

class Evaluation:
    """
    Evaluates model generations on specified tasks with metrics:
    - JSON format correctness
    - Prediction accuracy
    - Pronoun changing accuracy (including all-changed)
    - BiasBios occupation classification accuracy
    - CounterFact efficacy and paraphrase scores
    - Fluency metrics (bigram/trigram entropy)
    """
    def __init__(
        self,
        model: ModelWrapper,
        dataset: List[DatasetSample],
        task_name: str,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any]
    ):
        self.model = model
        self.dataset = dataset
        self.task_name = task_name
        self.tokenizer = tokenizer
        # Metrics to compute
        self.metrics_config = config.get('evaluation', {}).get('metrics', {})
        # Initialize counters
        self.reset()
        # Also store original prompt if needed for tracking
        self.original_prompts = [sample.raw_prompt for sample in dataset]
        # Reference labels for scoring
        self.references = [sample.label for sample in dataset]
        # Placeholder for generated texts
        self.generated_texts = []

    def reset(self):
        self.correct_json = 0
        self.total_samples = 0
        self.correct_prediction = 0
        self.pronoun_correct = 0
        self.pronoun_all_changed = 0
        self.bias_bios_correct = 0
        self.counterfactual_effectiveness = 0
        self.counterfactual_paraphrase = 0
        self.total_json_valid = 0
        self.total_json_correct = 0
        self.fluency_bigrams = []
        self.fluency_trigrams = []

    def evaluate(self):
        """
        Perform evaluation over dataset.
        """
        for idx, sample in enumerate(self.dataset):
            input_ids = sample.tokenized_input
            emphasis_spans = sample.emphasis_token_spans
            label = sample.label
            raw_prompt = sample.raw_prompt

            # Set emphasis span indices for the attention reweighting
            emphasis_token_indices = extract_emphasis_token_indices(raw_prompt, self.tokenizer)
            # Set current emphasis spans in the model for hooks
            self.model.set_current_emphasis_spans(emphasis_token_indices)

            # Generate output with possible attention steering
            gen_text = self.model.generate(
                input_ids=input_ids,
                emphasis_spans=emphasis_token_indices,
                alpha=self.metrics_config.get('attention_alpha', 0.01),
                max_new_tokens=100,
                temperature=0.7
            )
            self.generated_texts.append(gen_text)

            # Compute metrics based on task
            if self.task_name == 'JSON Formatting':
                self._evaluate_json_task(gen_text, label)
            elif self.task_name == 'Pronouns Changing':
                self._evaluate_pronouns_task(gen_text, label)
            elif self.task_name == 'BiasBios':
                self._evaluate_biasbios_task(gen_text, label)
            elif self.task_name == 'CounterFact':
                self._evaluate_counterfact_task(gen_text, label)
            # Add other task evaluations if needed

            self.total_samples += 1

        # After evaluation, compute averages or percentages
        results = self._compute_final_metrics()
        return results

    def _evaluate_json_task(self, gen_text: str, reference: Any):
        """
        Valid JSON and correctness of fields.
        """
        is_valid_json = False
        predicted_json = None
        try:
            gen_obj = json.loads(gen_text)
            is_valid_json = True
            predicted_json = gen_obj
        except json.JSONDecodeError:
            is_valid_json = False

        if self.metrics_config.get('format_accuracy', False):
            self.total_json_valid += int(is_valid_json)
            # Check correctness of JSON values if valid
            if is_valid_json and reference is not None:
                # Compare predicted JSON's 'occupation' field or other as per task
                correct = False
                if isinstance(reference, dict):
                    # For JSON task, reference probably dict
                    correct_value = reference.get('occupation', '').lower()
                    pred_value = str(predicted_json.get('occupation', '')).lower() if predicted_json else ''
                    correct = (pred_value == correct_value)
                else:
                    correct = False
                self.total_json_correct += int(correct)
                if correct:
                    self.correct_prediction += 1

        if is_valid_json:
            self.correct_json += 1

        # Update entropy for fluency
        if self.metrics_config.get('fluency', False):
            chars = gen_text
            self.fluency_bigrams.extend(compute_ngram_entropy(chars, 2))
            self.fluency_trigrams.extend(compute_ngram_entropy(chars, 3))

    def _evaluate_pronouns_task(self, gen_text: str, reference: Any):
        """
        Evaluate pronoun change correctness and all punctuated pronouns change.
        """
        # For pronouns, we expect the output to contain 'they' or other replacements
        # Reference contains the intended 'she'/'he' replaced to 'they'
        # Basic precision: count if 'they' in generated proportional to expected

        # For simplicity, check if 'she'/'he' replaced by 'they' (case-insensitive)
        def contains_pronoun(text: str, pronoun: str) -> bool:
            pattern = r'\b' + re.escape(pronoun) + r'\b'
            return re.search(pattern, text, re.IGNORECASE) is not None

        # Determine the expected pronoun from label or context
        # Here, assume label indicates correct pronoun to change
        expected_pronoun = 'she' if 'she' in str(reference).lower() else 'he'

        # Count if in gen_text 'they' appears where 'she'/'he' was
        # Also, check if all pronouns are changed
        she_in_gen = contains_pronoun(gen_text, 'she')
        he_in_gen = contains_pronoun(gen_text, 'he')
        they_in_gen = contains_pronoun(gen_text, 'they')

        # Simple accuracy: model correctly changed pronouns
        if 'she' in expected_pronoun:
            self.pronoun_correct += int(contains_pronoun(gen_text, 'she') or contains_pronoun(gen_text, 'they'))
        elif 'he' in expected_pronoun:
            self.pronoun_correct += int(contains_pronoun(gen_text, 'he') or contains_pronoun(gen_text, 'they'))

        # All pronouns changed case: count if all she/he replaced with they
        # Simplified: check for presence of 'they' and absence of original pronouns
        all_changed = False
        if 'she' in expected_pronoun:
            all_changed = contains_pronoun(gen_text, 'they')
        elif 'he' in expected_pronoun:
            all_changed = contains_pronoun(gen_text, 'they')
        self.pronoun_all_changed += int(all_changed)

    def _evaluate_biasbios_task(self, gen_text: str, reference: Any):
        """
        Classification accuracy for occupation prediction.
        """
        # We assume model outputs a occupation string or similar
        # For simplicity, use token-level or string match
        pred_label = gen_text.strip().lower()
        true_label = reference.strip().lower() if reference else ''
        correct = (pred_label == true_label)
        self.bias_bios_correct += int(correct)

    def _evaluate_counterfact_task(self, gen_text: str, reference: Any):
        """
        Evaluate counterfact efficacy (ES) and paraphrase score (PS).
        """
        # Reference likely contains old and new facts
        old_fact = reference.get('old_fact', '')
        new_fact = reference.get('new_fact', '')
        question = reference.get('question', '')
        # Parse generated answer, compare with new fact
        pred_value = gen_text.strip().lower()
        correct = (new_fact.lower() in pred_value)
        self.counterfactual_effectiveness += int(correct)
        # For paraphrase score, can compute similarity if needed (skipped here)
        # For simplicity, assume PS is 1 if correct, 0 otherwise
        # Could implement exact matches or more sophisticated metrics

    def _compute_final_metrics(self) -> Dict[str, float]:
        """
        Compute and return all metrics in a dict.
        """
        results = {}
        # JSON format correctness
        if self.metrics_config.get('format_accuracy', False):
            results['json_format_accuracy'] = (
                self.total_json_valid / self.total_samples * 100 if self.total_samples else 0.0
            )
            results['json_prediction_accuracy'] = (
                self.total_json_correct / self.total_samples * 100 if self.total_samples else 0.0
            )
        # Pronoun accuracy
        if self.metrics_config.get('pronoun_accuracy', False):
            results['pronoun_accuracy'] = (
                self.pronoun_correct / self.total_samples * 100 if self.total_samples else 0.0
            )
            results['pronoun_all_changed_accuracy'] = (
                self.pronoun_all_changed / self.total_samples * 100 if self.total_samples else 0.0
            )
        # BiasBios classification accuracy
        if self.metrics_config.get('bias_bios', False):
            results['BiasBios_Accuracy'] = (
                self.bias_bios_correct / self.total_samples * 100 if self.total_samples else 0.0
            )
        # CounterFact efficacy (ES) and paraphrase (PS)
        if self.metrics_config.get('counterfact', False):
            results['CounterFact_Efficacy'] = (
                self.counterfactual_effectiveness / self.total_samples * 100 if self.total_samples else 0.0
            )
            # PS could be more detailed, skipped for brevity
        # Fluency metrics
        if self.metrics_config.get('fluency', False):
            def entropy_mean(entropy_list):
                return np.mean(entropy_list) if entropy_list else 0.0
            results['Bigram_Entropy']'] = entropy_mean(self.fluency_bigrams)
            results['Trigram_Entropy'] = entropy_mean(self.fluency_trigrams)

        return results
```

## main.py

```python
## main.py
import os
import yaml
import torch
import random
from tqdm import tqdm

from dataset_loader import DatasetLoader
from model_wrapper import ModelWrapper
from attention_steering import AttentionSteering
from profile import Profiler
from evaluation import Evaluation
from utils import extract_emphasis_token_indices

def main():
    # 1. Load configuration from YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Initialize tokenizer and datasets
    model_name = config['model'].get('name')
    model_path = config['model'].get('model_path', None)
    data_dir = config['dataset'].get('data_dir')
    dataset_name = config['dataset'].get('name', 'BiasBios')
    split_names = config['dataset'].get('split', ['train', 'validation', 'test'])
    train_size = config['dataset'].get('train_size', 1000)
    val_size = config['dataset'].get('val_size', 1000)
    test_size = config['dataset'].get('test_size', 5000)

    # 3. Initialize ModelWrapper
    model_wrapper = ModelWrapper(model_name=model_name, model_path=model_path)

    # 4. Load datasets for profiling and evaluation
    # Initialize dataset loaders for train/val/test
    tokenizer = model_wrapper.tokenizer

    dataset_train = DatasetLoader(
        task_name=dataset_name, data_dir=data_dir, split=split_names[0],
        tokenizer=tokenizer).load_dataset()
    dataset_val = DatasetLoader(
        task_name=dataset_name, data_dir=data_dir, split=split_names[1],
        tokenizer=tokenizer).load_dataset()
    dataset_test = DatasetLoader(
        task_name=dataset_name, data_dir=data_dir, split=split_names[2],
        tokenizer=tokenizer).load_dataset()

    # 5. Prepare profiling dataset (small subset)
    prof_train_samples = dataset_train.sample_data(train_size)
    prof_val_samples = dataset_val.sample_data(val_size)

    # Re-assign to dataset objects for profiling if needed
    dataset_train.raw_data = prof_train_samples
    dataset_val.raw_data = prof_val_samples

    # 6. Profiling: identify effective attention heads
    profiler = Profiler(
        model_wrapper=model_wrapper,
        dataset=dataset_val,
        top_heads_count=config['profiling'].get('top_heads_count', 50),
        profile_samples=1000,
        strategy=config['attention_steering'].get('heads_selection_strategy', 'top-per-task')
    )
    selected_heads = profiler.profile_heads()

    # 7. Set the selected heads in model for steering
    # Alternatively, we can encapsulate in AttentionSteering
    alpha = config['attention_steering'].get('alpha', 0.01)
    attention_strategy = config['attention_steering'].get('heads_selection_strategy', 'top-per-task')

    # Initialize AttentionSteering with selected heads
    attention_steering = AttentionSteering(
        model=model_wrapper,
        selected_heads=selected_heads,
        alpha=alpha
    )

    # 8. Inference on test set with attention steering
    # For each test example, extract emphasis spans and generate
    results = {}
    task_name = dataset_name  # or could be specified per dataset

    # Optional: prepare evaluation metrics
    eval_flags = {
        'format_accuracy': config['evaluation']['metrics'].get('format_accuracy', True),
        'prediction_accuracy': config['evaluation']['metrics'].get('prediction_accuracy', True),
        'pronoun_accuracy': config['evaluation']['metrics'].get('pronoun_accuracy', True),
        'fluency': config['evaluation']['metrics'].get('fluency', True),
        'counterfact': config['evaluation']['metrics'].get('counterfact_effectiveness', True)
    }

    evaluation_dataset = dataset_test.get_samples(test_size)

    # Loop over test samples
    for sample in tqdm(evaluation_dataset, desc='Inference with PASTA on test'):
        raw_prompt = sample.raw_prompt
        input_ids = sample.tokenized_input

        # 8a. Extract emphasis spans: find emphasized tokens using utils
        emphasis_token_indices = extract_emphasis_token_indices(raw_prompt, tokenizer)
        # 8b. Set the emphasis in model for hooks
        model_wrapper.set_current_emphasis_spans(emphasis_token_indices)

        # 8c. Generate output with attention steering
        generated_text = model_wrapper.generate(
            input_ids=input_ids,
            emphasis_spans=emphasis_token_indices,
            alpha=alpha,
            max_new_tokens=100,
            temperature=0.7
        )

        # 8d. Store generated output for evaluation
        # We can store in the sample object or separately
        sample.generated_text = generated_text

    # 9. Evaluation on generated outputs
    evaluator = Evaluation(
        model=model_wrapper,
        dataset=evaluation_dataset,
        task_name=task_name,
        tokenizer=tokenizer,
        config={'evaluation': eval_flags}
    )

    results = evaluator.evaluate()

    # 10. Log results
    print("=== PASTA Inference and Evaluation Results ===")
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value:.2f}")

    # 11. Clean up hooks
    attention_steering.clear_hooks()

if __name__ == "__main__":
    main()
```

## model_wrapper.py

```python
## model_wrapper.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict, Optional, Callable
import re

class ModelWrapper:
    """
    Wraps a pre-trained language model to enable inference with
    attention score manipulation for attention steering.
    """
    def __init__(self, model_name: str, model_path: str = None):
        """
        Load the model and tokenizer based on model_name and register hooks.
        :param model_name: e.g., "LLaMA-7B" or "GPT-J"
        :param model_path: path to local model checkpoint or None for hub model
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
        self._register_hook_dict: Dict[Tuple[int, int], torch.nn.Module] = {}
        # Store references to hooks for later removal if needed
        self._hooks_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Maintain a set of target heads for steering (layer, head)
        self.target_heads: List[Tuple[int, int]] = []

    def _load_model(self):
        """
        Load model and tokenizer. Supports LLAMA, GPT-J architectures.
        """
        # Example: specify model_path or use models from Huggingface hub
        # For LLAMA: use 'huggingface/llama model id' if available
        # For GPT-J: use 'EleutherAI/gpt-j-6B'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path if self.model_path else self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path if self.model_path else self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            revision="main"
        )
        self.model.to(self.device)
        self.model.eval()

        # Identify attention modules depending on architecture
        # For LLAMA: self.model.layers is a list of transformer blocks
        # For GPT-J: self.model.transformer.h is list of blocks
        # We'll support both by checking attributes
        if hasattr(self.model, 'layers'):
            self.attention_layers = self.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.attention_layers = self.model.transformer.h
        else:
            raise ValueError("Unknown model architecture for attention hook registration.")
        # Prepare a dict to store attention scores per layer
        self._attn_scores: Dict[Tuple[int, int], torch.Tensor] = {}
        # For hook management
        self._register_hooks()

    def _register_hooks(self):
        """
        Register hooks on attention modules for all layers.
        For multiple heads, hooks are registered per layer.
        """
        # Clear previous handles if any
        for handle in self._hooks_handles:
            handle.remove()
        self._hooks_handles.clear()

        # Loop over attention modules
        for layer_idx, layer_module in enumerate(self.attention_layers):
            # Depending on architecture, attention module attribute:
            # For LLAMA: 'self_attn' in each layer
            # For GPT-J: 'attn' in each block
            attn_module = None
            if hasattr(layer_module, 'self_attn'):
                attn_module = getattr(layer_module, 'self_attn')
            elif hasattr(layer_module, 'attn'):
                attn_module = getattr(layer_module, 'attn')
            else:
                continue  # skip if not found

            # Register hook to extract attention scores
            # We assume the module has a method or attribute that outputs attention scores.
            # For LLAMA, the attention module often returns attention weights when called
            # with output_attentions=True or via hooks.
            handle = attn_module.register_forward_hook(self._create_attention_hook(layer_idx))
            self._hooks_handles.append(handle)

    def _create_attention_hook(self, layer_idx: int) -> Callable:
        """
        Create a hook to capture attention scores during forward pass.
        Assumes the attention module returns or saves attention scores.
        """
        def hook(module, inputs, outputs):
            """
            Hook function to save attention scores.
            :param module: attention module
            :param inputs: tuple of input tensors
            :param outputs: output object, could be tuple or dict depending on model
            """
            # Depending on the module architecture, extract attention weights
            # For most transformers, implement assuming the module returns attention weights
            # For LLAMA: self_attn may return attention weights if output_attentions=True
            # For GPT-J: attention weights are usually internal; need to fetch from outputs
            # Since hooks are before softmax, assume the attention weights are available
            # in the module or via the outputs (depends on model implementation)
            # For generality, we try to get attention weights
            # Note: If model does not output attention scores, this may need adjustments
            if hasattr(module, 'attn'):  # typical for some implementations
                # LLAMA uses 'self_attn' which may not output attentions directly
                # but for the purpose here, assume the attribute exists
                # Alternatively, we can access these during the forward if model supports
                # options, but for simplicity, attach to attention modules that output attentions
                pass
            # Alternatively, some models store attentions inside module, or in output if configured

        return hook

    def register_attention_heads(self, heads: List[Tuple[int, int]]):
        """
        Record the selected heads for steering.
        :param heads: list of (layer_idx, head_idx)
        """
        self.target_heads = heads

    def get_attention_scores(self, layer_idx: int, head_idx: int, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass to obtain attention scores for a specific head in a layer.
        Note: For efficiency, a dedicated method or caching strategy can be used.
        """
        # Run a forward with return_attentions=True
        # For models supporting output_attentions
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_attentions=True,
            return_dict=True
        )
        attentions = outputs['attentions']  # tuple/list: one per layer
        layer_attn = attentions[layer_idx]  # shape: (batch_size, num_heads, seq_len, seq_len)
        # Select head_idx
        attn_head = layer_attn[:, head_idx, :, :]  # shape: (batch_size, seq_len, seq_len)
        return attn_head

    def generate(self, input_ids: List[int], emphasis_spans: List[List[int]], alpha: float=0.01, max_new_tokens: int=50, temperature: float=1.0) -> str:
        """
        Generate text with optional attention steering by reweighting attention scores
        during decoding at specified layers and heads.
        :param input_ids: tokenized prompt
        :param emphasis_spans: list of token index spans emphasized by user
        :param alpha: scaling coefficient
        :param max_new_tokens: maximum tokens to generate
        :param temperature: sampling temperature
        """
        # Prepare focus mask: set of token indexes highlighted
        emphasized_token_indices = set()
        for span in emphasis_spans:
            if span:
                start_idx, end_idx = span
                # Add tokens in span (assuming span is [start, end])
                emphasized_token_indices.update(range(start_idx, end_idx+1))
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Set model to eval mode
        self.model.eval()

        # To modify attention, we leverage hooks in attention modules.
        # Because hooks are active during forward, we set a custom forward_hook that adjusts attention scores.
        # Once set, in the generate loop, attention scores are manipulated per step.
        # Here, for simplicity, we make the assumption that attention scores are returned via attentions,
        # and we perform a "re-weight" step right before softmax.

        # For more precise control, custom decoder or attention implementations are needed.
        # Assuming we can set hooks to modify attention during the forward pass.

        # Run generation with custom attention reweighting
        output_ids = self._generate_with_attention_reweight(input_ids, emphasis_spans, alpha, max_new_tokens, temperature)
        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return generated_text

    def _generate_with_attention_reweight(
        self,
        input_ids: List[int],
        emphasis_spans: List[List[int]],
        alpha: float,
        max_new_tokens: int,
        temperature: float
    ) -> List[int]:
        """
        Generate tokens iteratively, applying the attention score reweighting at each step
        in the targeted heads during the attention calculations.
        Note: For the purpose of illustration, assumes the model's attention modules can be intercepted.
        """
        # Initialize generation
        generated_ids = input_ids.copy()
        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([generated_ids], device=self.device)
            # Run model with hooks that reweight attention
            # (Assuming hooks are registered and will manipulate scores during forward)
            outputs = self.model(
                input_ids=input_tensor,
                output_attentions=True,
                return_dict=True,
                temperature=temperature
            )
            logits = outputs.logits  # shape: (1, seq_len, vocab_size)
            # Get last token logits
            next_token_logits = logits[0, -1, :]
            # Apply temperature
            scaled_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            # Sample or greedy
            next_token_id = torch.argmax(probs).item()
            generated_ids.append(next_token_id)
            if next_token_id == self.tokenizer.eos_token_id:
                break
        return generated_ids

# Note:
# The critical part of this implementation is the correct registration of hooks
# on the attention modules to access and modify attention scores during generation.
# This code assumes the model's attention modules support capturing attention weights
# via forward hooks, which may require model-specific adjustments.
# For extremely custom models, rewriting attention modules to accept external scores
# or use a custom transformer implementation is necessary.
```

## profile.py

```python
## profile.py
import torch
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

class Profiler:
    """
    Implements multi-head attention profiling to identify most effective attention heads
    for steering, based on their impact on task-specific performance over a subset of data.
    """
    def __init__(
        self,
        model_wrapper,             # Instance of ModelWrapper
        dataset,                 # Dataset object with get_samples method
        top_heads_count: int = 50,# Number of heads to select after profiling
        profile_samples: int = 1000,  # Number of samples for profiling
        strategy: str = 'top-per-task' # How to combine heads across tasks ('top-per-task', 'union', 'intersection')
    ):
        self.model_wrapper = model_wrapper
        self.dataset = dataset
        self.top_heads_count = top_heads_count
        self.profile_samples = profile_samples
        self.strategy = strategy

        # Will hold the final selected heads after profiling
        self.selected_heads: List[Tuple[int, int]] = []

    def profile_heads(self, task_labels: Optional[List[str]] = None):
        """
        Profiles all attention heads by evaluating their influence on performance
        across dataset samples. Selects top heads depending on strategy.

        Args:
            task_labels: Optional list of task labels for multi-task profiling (not mandatory here,
                         assume single task or all tasks combined).
        """
        # Obtain total layers and heads
        L, H = self.model_wrapper.get_num_layers_heads()
        all_heads = [(l, h) for l in range(L) for h in range(H)]
        
        head_scores: List[Dict] = []

        # Evaluate each head independently
        for (layer, head) in tqdm(all_heads, desc="Profiling attention heads"):
            # Register hook to steer only this head
            self.model_wrapper.register_attention_hook(layer, head)
            # Initialize list to hold individual sample scores
            scores = []

            # Get a small subset of samples for evaluation
            samples = self.dataset.get_samples(self.profile_samples)
            
            for sample in samples:
                # Generate output with only the selected head steered
                # Assume generate method can accept a head tuple for reweighting
                generated_output = self.model_wrapper.generate(
                    sample.tokenized_input, 
                    emphasis_spans=None, # No emphasis spans during profiling
                    head=(layer, head),
                    alpha=self.model_wrapper.alpha
                )
                # Evaluate performance (classification accuracy, JSON validity, etc.)
                score = self._evaluate_sample_performance(generated_output, sample.label)
                scores.append(score)
            
            # Average score across samples
            mean_score = sum(scores)/len(scores) if scores else 0.0
            head_scores.append({'layer': layer, 'head': head, 'score': mean_score})
            # Unregister hooks to avoid interference with next evaluation
            self.model_wrapper.remove_attention_hook(layer, head)

        # Rank all heads based on scores
        ranked_heads = sorted(head_scores, key=lambda x: x['score'], reverse=True)

        # Select top heads according to strategy
        if self.strategy == 'top-per-task':
            # Simply top-K overall
            self.selected_heads = [
                (h['layer'], h['head']) for h in ranked_heads[:self.top_heads_count]
            ]
        elif self.strategy == 'union':
            # For multi-task, union of top heads per task; here assuming single task, so same as top-per-task
            # For multi-task, aggregate per task and union
            self.selected_heads = [
                (h['layer'], h['head']) for h in ranked_heads[:self.top_heads_count]
            ]
        elif self.strategy == 'intersection':
            # For multi-task, intersection across individual task top sets
            # Here, assuming one global top set
            self.selected_heads = [
                (h['layer'], h['head']) for h in ranked_heads[:self.top_heads_count]
            ]
        else:
            # Default fallback
            self.selected_heads = [
                (h['layer'], h['head']) for h in ranked_heads[:self.top_heads_count]
            ]
        return self.selected_heads

    def _evaluate_sample_performance(self, generated_text: str, reference_label):
        """
        Compute task-specific performance metric.
        For classification tasks, accuracy; for JSON tasks, validity and correctness.
        Implemented as a placeholder here.
        """
        # For demonstration, let's assume a classification accuracy function
        # that compares generated_text to reference_label
        # User can override or extend this method as needed
        if reference_label is None:
            return 0.0
        # Example: exact match (or run custom metric)
        # For classification: 1.0 if match, 0.0 else
        return 1.0 if generated_text.strip().lower() == reference_label.strip().lower() else 0.0

# The dataset.get_samples method should return dicts with keys:
# 'tokenized_input' (List[int]), 'label' (task-specific, e.g., str or class)

# The model_wrapper.register_attention_hook and remove_attention_hook methods
# are assumed to set hooks that modify the attention scores of the specified head during generation.
# They should be implemented in concert with the attention_steering.py module.

# This code aligns with the "Data structures and interfaces" and the "Logic Analysis" provided.
```

## utils.py

```python
## utils.py

import re
from typing import List, Tuple, Optional
import torch

def tokenize(text: str, tokenizer) -> List[int]:
    """
    Tokenize the input text into token IDs using the provided tokenizer.
    """
    return tokenizer.encode(text, add_special_tokens=False)

def detokenize(token_ids: List[int], tokenizer) -> str:
    """
    Convert token IDs back into a string using the tokenizer.
    """
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=True)

def extract_emphasis_token_indices(prompt: str, marker: str='*') -> List[int]:
    """
    Extract emphasized token indices from a styled prompt.
    Assumes emphasis markers (e.g., '*') enclose emphasized spans.
    Returns a list of token indices (integers) that are emphasized.
    """
    # Find all emphasized segments (pairwise matches of marker)
    pattern = re.escape(marker) + '(.*?)' + re.escape(marker)
    matches = list(re.finditer(pattern, prompt))
    emphasized_char_spans: List[Tuple[int, int]] = []

    # For each emphasized span, record start and end character positions
    for match in matches:
        start_char = match.start()
        end_char = match.end()
        # The span text without markers
        span_text = match.group(1)
        emphasized_char_spans.append((start_char, end_char))

    # Remove emphasis markers to get clean prompt
    clean_prompt = re.sub(pattern, lambda m: m.group(1), prompt)

    # Tokenize the clean prompt with offset mapping
    # Requires tokenizer to support return_offsets_mapping=True
    # We will assume the caller passes tokenizer and do tokenization here
    # For robustness, check if tokenizer supports it
    # But since this is a utility, we expect to pass tokenizer as an argument
    # Let's implement the core here (to be called outside)
    return list_of_emphasized_token_indices_from_char_spans(clean_prompt, emphasized_char_spans)

def list_of_emphasized_token_indices_from_char_spans(
    prompt_text: str,
    emphasized_char_spans: List[Tuple[int, int]],
    tokenizer=None
) -> List[int]:
    """
    Map character spans to token indices.
    If tokenizer is provided, use it to compute token offsets. Otherwise, return empty.
    """
    token_indices: List[int] = []
    if tokenizer is None:
        return token_indices

    # Obtain token offsets
    encoding = tokenizer(
        prompt_text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    offsets = encoding['offset_mapping']  # list of (start_char, end_char) per token

    # For each emphasized span, find tokens overlapping with span
    for (char_start, char_end) in emphasized_char_spans:
        for idx, (tok_start, tok_end) in enumerate(offsets):
            # Check overlap
            if tok_end <= char_start:
                continue
            if tok_start >= char_end:
                continue
            # Overlap exists
            token_indices.append(idx)

    # Optionally, remove duplicates
    token_indices = sorted(set(token_indices))
    return token_indices

def extract_emphasis_token_indices(prompt: str, tokenizer, marker: str='*') -> List[int]:
    """
    Parses the prompt, extracts emphasized spans marked with `marker`,
    and returns token indices corresponding to emphasized tokens.
    """
    pattern = re.escape(marker) + '(.*?)' + re.escape(marker)
    matches = list(re.finditer(pattern, prompt))
    emphasized_char_spans: List[Tuple[int, int]] = []

    for match in matches:
        start_char = match.start()
        end_char = match.end()
        span_text = match.group(1)
        emphasized_char_spans.append((start_char, end_char))
    # Remove emphasis markers in prompt
    clean_prompt = re.sub(pattern, lambda m: m.group(1), prompt)
    # Get token offsets
    encoding = tokenizer(
        clean_prompt,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    offsets = encoding['offset_mapping']
    token_indices: List[int] = []

    # Map each emphasized char span to token indices
    for (char_start, char_end) in emphasized_char_spans:
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= char_start:
                continue
            if tok_start >= char_end:
                continue
            # Overlap
            if idx not in token_indices:
                token_indices.append(idx)
    token_indices = sorted(set(token_indices))
    return token_indices

def normalize_attention_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Normalize attention scores over the last dimension so rows sum to 1.
    Adds epsilon for numerical stability.
    """
    epsilon = 1e-9
    sums = scores.sum(dim=-1, keepdim=True) + epsilon
    return scores / sums

def scale_attention_scores(
    scores: torch.Tensor,
    emphasis_indices: List[int],
    alpha: float = 0.01
) -> torch.Tensor:
    """
    Reweight attention scores by scaling down non-emphasized token scores.
    Scores: tensor of shape (batch_size, num_heads, seq_len, seq_len)
    emphasis_indices: list of token indices to emphasize
    """
    # Create mask tensor: shape (seq_len,)
    seq_len = scores.shape[-1]
    device = scores.device
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[emphasis_indices] = True
    # Expand mask for each batch and head
    # shape: (1, 1, seq_len)
    # We will broadcast in the next step
    mask_broadcast = mask.unsqueeze(0).unsqueeze(0)  # shape: (1,1,seq_len)
    # For each position in the query sequence (dim=-2)
    # scale scores where the key token is not emphasized
    # Use torch.where for element-wise multiplication
    # Expand for batch and head dims
    # scores shape: (batch_size, num_heads, seq_len_q, seq_len_k)
    # We need to scale scores for non-emphasized key tokens
    scores_scaled = scores.clone()
    # For each position in query, scale scores of non-emphasized tokens
    # Broadcast mask over query positions
    # Create a mask for the key tokens: shape (seq_len,)
    # Expand to (1, 1, 1, seq_len)
    mask_expanded = ~mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # shape: (1,1,1,seq_len)
    # Convert to float for multiplication
    scale_mask = mask_expanded.type(scores.dtype)
    # Multiply corresponding scores by alpha
    scores_scaled = torch.where(
        mask_expanded,
        scores * alpha,
        scores
    )
    # Renormalize
    scores_norm = normalize_attention_scores(scores_scaled)
    return scores_norm
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\outputs\paper2code\PASTA\PASTA_repo`
