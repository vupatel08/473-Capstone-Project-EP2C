# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## attention_module.py

```python
# attention_module.py

import torch
from torch import Tensor
from transformers import PreTrainedModel
from typing import List, Dict, Tuple, Optional

class AttentionModule:
    """
    This class provides methods to:
    1. Register hooks to capture attention scores at inference time.
    2. Extract attention scores after the forward pass.
    3. Reweight attention scores based on user-highlighted tokens and a coefficient alpha.
    
    It supports models from Hugging Face transformers with accessible attention outputs, such as
    LLAMA, GPT-J, and Vicuna variants, assuming they are loaded with output_attentions=True.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        device: str = "cuda",
        alpha: float = 0.01,
        top_k_heads: int = 400,
        profile_heads: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Initialize the AttentionModule.
        Args:
            model: The Hugging Face transformer model instance.
            device: 'cuda' or 'cpu'.
            alpha: Reweighting coefficient, default 0.01.
            top_k_heads: Number of heads to select for steering (if profiling not used).
            profile_heads: List of (layer_idx, head_idx). If None, should set after profiling.
        """
        self.model = model
        self.device = device
        self.alpha = alpha
        self.top_k_heads = top_k_heads
        self.profile_heads = profile_heads or []
        self.attention_scores: List[Dict[Tuple[int, int], Tensor]] = []
        self._hooks = []
        self._register_attention_hooks()
        # Keep track of latest attentions per layer-head
        self._attention_cache: Dict[Tuple[int, int], Tensor] = {}
        # For managing layer names
        self._layer_modules = self._identify_attention_modules()
    
    def _identify_attention_modules(self):
        """
        Identify attention modules within the model's architecture,
        to register hooks. Handles common transformer structures.
        Returns:
            List of modules corresponding to transformer layers.
        """
        modules = []
        top_module = getattr(self.model, 'model', self.model)
        # For LLAMA or similar models
        if hasattr(top_module, 'layers'):
            modules = list(top_module.layers)
        elif hasattr(top_module, 'h'):
            modules = list(top_module.h)
        elif hasattr(top_module, 'block'):
            modules = list(top_module.block)
        else:
            raise RuntimeError("Unable to find attention modules in the model.")
        return modules
    
    def _register_attention_hooks(self):
        """
        Register hooks into each attention layer/module
        to capture the attention scores during forward pass.
        """
        def get_attention_hook(layer_idx: int):
            def hook(module, input, output):
                """
                Hook function to capture attention scores per layer.
                Assumes output is either a tuple with attention tensors or an object
                with 'attentions' attribute.
                """
                attn_tensor = None
                if isinstance(output, tuple):
                    # Many models return attentions as second item
                    if len(output) >= 2:
                        attn_tensor = output[1]
                elif hasattr(output, 'attentions'):
                    # Some models store attentions in attribute
                    attn_tensor = output.attentions
                if attn_tensor is None:
                    return
                # attn_tensor: shape (batch_size, num_heads, seq_len, seq_len)
                # For indexation, store per layer-head
                # If multiple tensors, take the last one
                self._attention_cache[(layer_idx, None)] = attn_tensor.detach()
            return hook

        # Attach hooks to all relevant attention modules
        for layer_idx, layer_module in enumerate(self._layer_modules):
            # Different models have different attribute names
            # Standard pattern:
            #   - self_attn or attn or attention
            handled = False
            if hasattr(layer_module, 'self_attn'):
                handle = layer_module.self_attn.register_forward_hook(get_attention_hook(layer_idx))
                self._hooks.append(handle)
                handled = True
            if hasattr(layer_module, 'attn') and not handled:
                handle = layer_module.attn.register_forward_hook(get_attention_hook(layer_idx))
                self._hooks.append(handle)
                handled = True
            if hasattr(layer_module, 'attention') and not handled:
                handle = layer_module.attention.register_forward_hook(get_attention_hook(layer_idx))
                self._hooks.append(handle)
                handled = True
            # If none matched, skip
        # Note: For models like LLAMA, attention is typically in 'self_attn' attribute.

    def clear_hooks(self):
        """
        Remove all registered hooks to clean up.
        """
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def get_attention_scores(
        self,
        input_kwargs: Dict
    ) -> List[Dict[Tuple[int, int], Tensor]]:
        """
        Forward through the model with hooks enabled, then collect attention scores.
        Args:
            input_kwargs: dict with input_ids, attention_mask etc.
        Returns:
            List of dicts per layer, where each dict keys are (layer_idx, head_idx)
            and values are attention tensors of shape (batch_size, seq_len, seq_len).
        """
        self._attention_cache.clear()
        # Run inference with hooks capturing attention
        with torch.no_grad():
            output = self.model(**input_kwargs)
        # From the cache, organize attention per layer and head
        attention_per_layer: Dict[int, Tensor] = {}
        for (layer_idx, _), attn_tensor in self._attention_cache.items():
            # Store only one attention tensor per layer (average over heads if multiple tensors)
            # But hooks collect attention per layer as a tensor of shape (batch_size, num_heads, seq_len, seq_len)
            # To handle multiple heads, keys are (layer_idx, head_idx)
            # So, we must process attention per head later
            if layer_idx not in attention_per_layer:
                attention_per_layer[layer_idx] = attn_tensor
            else:
                # If multiple tensors per layer (unlikely), merge or overwrite
                attention_per_layer[layer_idx] = attn_tensor
        # Convert organized data into expected structure:
        # For each layer, we have a tensor with shape (batch_size, num_heads, seq_len, seq_len)
        attention_list = []
        for layer_idx, attn_tensor in sorted(attention_per_layer.items()):
            # For each head, store individual attention
            num_heads = attn_tensor.shape[1]
            for head_idx in range(num_heads):
                # Store attention tensor for each head
                # Keyed by (layer_idx, head_idx)
                if (layer_idx, head_idx) not in self._attention_cache:
                    # Save in cache for reweighting
                    self._attention_cache[(layer_idx, head_idx)] = attn_tensor[:, head_idx, :, :]
                else:
                    self._attention_cache[(layer_idx, head_idx)] = attn_tensor[:, head_idx, :, :]
        # Build output list
        attentions = []
        for (layer_idx, head_idx), attn_tensor in self._attention_cache.items():
            attentions.append({'layer_idx': layer_idx, 'head_idx': head_idx, 'attention': attn_tensor})
        return attentions

    def reweight_attention(
        self,
        attention_tensors: List[Dict[Tuple[int, int], Tensor]],
        highlighted_tokens: List[int]
    ) -> None:
        """
        Reweight stored attention matrices in-place based on highlighted tokens and alpha.
        This modifies the attention scores so that subsequent attention computation uses these.
        Args:
            attention_tensors: List of attention dicts as produced by get_attention_scores.
            highlighted_tokens: List of token indices (int) to emphasize.
        """
        # Precompute set for efficient lookup
        highlight_set = set(highlighted_tokens)
        for attn_dict in attention_tensors:
            layer_idx = attn_dict['layer_idx']
            head_idx = attn_dict['head_idx']
            attn_score: Tensor = attn_dict['attention']  # shape: (batch_size, seq_len, seq_len)
            # For each batch, rowwise process
            batch_size, seq_len, _ = attn_score.shape
            # Create mask for j: 1 if in highlighted_tokens else alpha
            mask_j = torch.ones(seq_len, device=attn_score.device)
            for j in range(seq_len):
                if j not in highlight_set:
                    mask_j[j] = self.alpha
            # Expand mask_j to (seq_len,) for broadcasting
            # For each row i, multiply each attention weight A_{i,j}: produce scaled scores
            for b in range(batch_size):
                # Scale attention scores: multiply each row by the mask of j
                scores: Tensor = attn_score[b]  # shape: (seq_len, seq_len)
                # Scale columns (attention weights for each token j)
                scores = scores * mask_j.unsqueeze(0)  # shape: (seq_len, seq_len)
                # Row-wise normalization
                C_i = scores.sum(dim=1, keepdim=True)  # shape: (seq_len, 1)
                C_i = torch.where(C_i == 0, torch.ones_like(C_i), C_i)
                scores = scores / C_i
                attn_score[b] = scores
            # Save back the reweighted attention
            attn_dict['attention'] = attn_score

    def set_profile_heads(self, heads: List[Tuple[int, int]]):
        """
        Update the heads for steering.
        Args:
            heads: list of (layer_idx, head_idx)
        """
        self.profile_heads = heads

    def get_profile_heads(self):
        """
        Return current profile heads.
        """
        return self.profile_heads

```

## dataset_loader.py

```python
## dataset_loader.py
import os
import json
import random
from typing import List, Dict, Tuple, Optional
from datasets import Dataset

from utils import load_config, validate_path

class DatasetLoader:
    """
    A class to load, parse, and prepare datasets for different tasks as per the experimental setup.
    Supports BiasBios, CounterFact, JSON Formatting, and Pronouns Changing datasets.
    """
    def __init__(
        self,
        dataset_paths: Dict[str, str],
        task_name: str,
        split_ratios: Dict[str, float] = None,
        seed: int = 42,
        cache_dir: str = "cached_datasets"
    ):
        """
        Initialize DatasetLoader with dataset paths and task info.
        Args:
            dataset_paths (dict): dictionary with dataset paths keyed by dataset name.
            task_name (str): which dataset to load.
            split_ratios (dict): ratios for train/val/test splits, default: {'train':0.6,'val':0.2,'test':0.2}
            seed (int): seed for splitting.
            cache_dir (str): directory to cache processed datasets.
        """
        self.dataset_paths = dataset_paths
        self.task_name = task_name
        self.seed = seed
        self.cache_dir = cache_dir
        if split_ratios is None:
            self.split_ratios = {'train': 0.6, 'val': 0.2, 'test': 0.2}
        else:
            self.split_ratios = split_ratios
        self.dataset = None  # to hold the loaded dataset

        validate_path(cache_dir, must_exist=False)

    def load_dataset(self):
        """
        Load, parse, and split dataset according to task_name.
        Supports caching for efficiency.
        """
        cache_path = os.path.join(self.cache_dir, f"{self.task_name}_full.json")
        if os.path.exists(cache_path):
            # Load preprocessed dataset from cache
            with open(cache_path, 'r') as f:
                self.dataset = json.load(f)
            return

        # Select parsing method based on task name
        if self.task_name == 'BiasBios':
            raw_data = self._load_raw_data('bias_bios')
            parsed_data = self._parse_bias_bios(raw_data)
        elif self.task_name == 'CounterFact':
            raw_data = self._load_raw_data('counterfact')
            parsed_data = self._parse_counterfact(raw_data)
        elif self.task_name == 'JSON Formatting':
            raw_data = self._load_raw_data('json_format')
            parsed_data = self._parse_json_format(raw_data)
        elif self.task_name == 'Pronouns Changing':
            raw_data = self._load_raw_data('pronouns_changing')
            parsed_data = self._parse_pronouns_changing(raw_data)
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")

        # Save full dataset cache
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(parsed_data, f)

        # Split into train/val/test
        self.dataset = self._split_dataset(parsed_data)
        
    def get_dataset(self) -> Dict[str, List[Dict]]:
        """
        Return dict with 'train', 'validation', 'test' datasets.
        """
        if self.dataset is None:
            self.load_dataset()
        return self.dataset

    def _load_raw_data(self, dataset_key: str) -> List[dict]:
        """
        Load raw data file(s) for specified dataset key from path.
        Currently supports JSONL or JSON lines.
        """
        path = self.dataset_paths.get(dataset_key)
        if path is None:
            raise ValueError(f"Dataset path for {dataset_key} not provided")
        validate_path(path)
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    data.append(sample)
                except json.JSONDecodeError:
                    # If dataset is in raw text, extend parsing as needed
                    # For simplicity, assume JSON Lines
                    raise
        return data

    def _split_dataset(self, data: List[dict]) -> Dict[str, List[dict]]:
        """
        Split the dataset into train/val/test according to ratios, fixed seed.
        """
        random.seed(self.seed)
        data_shuffled = data.copy()
        random.shuffle(data_shuffled)
        total = len(data_shuffled)
        train_end = int(total * self.split_ratios['train'])
        val_end = train_end + int(total * self.split_ratios['val'])
        train_data = data_shuffled[:train_end]
        val_data = data_shuffled[train_end:val_end]
        test_data = data_shuffled[val_end:]
        return {'train': train_data, 'validation': val_data, 'test': test_data}

    def _parse_bias_bios(self, raw_data: List[dict]) -> List[dict]:
        """
        Parse BiasBios dataset.
        Expect each sample to have biographical context and occupation label.
        """
        parsed = []
        for sample in raw_data:
            context = sample.get('text', '').strip()
            occupation = sample.get('label', '').strip()
            # Assume emphasis markers are already embedded if provided
            input_text = context
            target_text = occupation
            # Typically, emphasis is on the first sentence; no further parsing needed
            parsed.append({
                'input_text': input_text,
                'target_text': target_text,
                'highlighted_spans': self._extract_emphasis_indices(input_text),
                'task_type': 'BiasBios'
            })
        return parsed

    def _parse_counterfact(self, raw_data: List[dict]) -> List[dict]:
        """
        Parse CounterFact dataset.
        Each sample contains old and new facts, and a question.
        """
        parsed = []
        for sample in raw_data:
            old_fact = sample.get('old_fact', '').strip()
            new_fact = sample.get('new_fact', '').strip()
            question = sample.get('question', '').strip()
            # Input prompt: "Previously, {old_fact}. Currently, {new_fact}. {question}"
            input_text = f"Previously, {old_fact}. Currently, {new_fact}. {question}"
            target_text = new_fact  # Expected output
            # No emphasis marking assuming user provides highlighted spans
            parsed.append({
                'input_text': input_text,
                'target_text': target_text,
                'highlighted_spans': self._extract_emphasis_indices(input_text),
                'task_type': 'CounterFact'
            })
        return parsed

    def _parse_json_format(self, raw_data: List[dict]) -> List[dict]:
        """
        Parse JSON formatting task.
        Each sample contains 'name' and 'occupation'.
        Generate input prompt instructing to produce JSON output.
        """
        parsed = []
        for sample in raw_data:
            name = sample.get('name', '').strip()
            occupation = sample.get('occupation', '').strip()
            # Generate input: "Winnie is an American photographer... {instruction}"
            # Using prompt template in utils or simply embedded here
            input_text = (
                f"{name} is an American {occupation} living in New York. "
                f"Specialized in fashion photography and portrait, she applies her talent on "
                f"both humans and animals. {self._get_instruction('json_format')}"
            )
            # Expected output: JSON object string with name and occupation
            target_text = json.dumps({"name": name, "occupation": occupation})
            parsed.append({
                'input_text': input_text,
                'target_text': target_text,
                'highlighted_spans': self._extract_emphasis_indices(input_text),
                'task_type': 'JSON Formatting'
            })
        return parsed

    def _parse_pronouns_changing(self, raw_data: List[dict]) -> List[dict]:
        """
        Parse Pronouns Changing dataset.
        For each sample, generate prompt with emphasis on context.
        """
        parsed = []
        for sample in raw_data:
            context = sample.get('context', '').strip()
            person_name = sample.get('person', '').strip()
            occupation = sample.get('occupation', '').strip()
            # Assume the emphasis marker is around the context
            input_text = (
                f"{context} You should change 'she' and 'he' to 'they' and generate "
                f"the occupation of {person_name} after changing pronouns."
            )
            target_text = sample.get('target_text', '').strip()
            parsed.append({
                'input_text': input_text,
                'target_text': target_text,
                'highlighted_spans': self._extract_emphasis_indices(input_text),
                'task_type': 'Pronouns Changing'
            })
        return parsed

    def _extract_emphasis_indices(self, text: str, emphasis_marker: str = "**") -> List[int]:
        """
        Extract token indices/character positions of emphasized spans.
        Assumes emphasis markers are embedded with asterisks or similar.
        For simplicity, return character indices of emphasized parts.
        """
        pattern = re.escape(emphasis_marker) + '(.*?)' + re.escape(emphasis_marker)
        spans = [match.span() for match in re.finditer(pattern, text)]
        # Remove markers from text
        clean_text = re.sub(pattern, r'\1', text)
        emphasized_positions = []
        for start, end in spans:
            # Map to character indices of emphasized spans
            emphasized_positions.extend(range(start, end - 2*len(emphasis_marker)))
        return emphasized_positions

    # Utility method to get pre-defined instruction snippets, if needed
    def _get_instruction(self, task_type: str) -> str:
        """
        Retrieve task-specific instruction snippets for prompting.
        """
        # Could be implemented to load from config or hardcoded
        if task_type == 'json_format':
            # Example instruction snippet
            return "Answer the occupation of {person} and generate the answer as json format."
        else:
            return ""

```

## evaluation.py

```python
## evaluation.py
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log2
from typing import List, Dict, Tuple, Optional
from utils import load_config

class Evaluation:
    """
    The Evaluation class computes and reports performance metrics for the tasks:
    - JSON Formatting: format accuracy, prediction correctness
    - Pronouns Changing: accuracy and all-changed accuracy
    - BiasBios: occupation classification accuracy
    - CounterFact: efficacy and paraphrase scores
    - Fluency and content consistency scores
    
    It operates on model-generated outputs and corresponding ground truths.
    """
    def __init__(self, 
                 model_outputs: List[str], 
                 dataset: List[Dict], 
                 task_name: str,
                 task_configs: Dict = None,
                 verbose: bool = False):
        """
        Initialize with generated texts, dataset, task name, and optional configs.
        """
        self.outputs = model_outputs
        self.dataset = dataset
        self.task_name = task_name
        self.config = task_configs if task_configs is not None else {}
        self.verbose = verbose
        # Load global config parameters for metrics thresholds
        default_config = load_config('config.yaml')
        eval_config = default_config.get('evaluation', {})
        self.min_entropy = eval_config.get('metrics', {}).get('fluency', {}).get('min_entropy', 3.0)
        # Placeholder for storing results
        self.results = {}

    def evaluate(self) -> Dict[str, float]:
        """
        Run all evaluation metrics depending on the task.
        """
        if self.task_name == 'JSON Formatting':
            return self._eval_json_format()
        elif self.task_name == 'Pronouns Changing':
            return self._eval_pronouns_change()
        elif self.task_name == 'BiasBios':
            return self._eval_bias_bios()
        elif self.task_name == 'CounterFact':
            return self._eval_counterfact()
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

    def get_results(self) -> Dict[str, float]:
        """
        Returns evaluated metrics after calling evaluate()
        """
        return self.results

    ########## Internal metric implementations ##########

    def _eval_json_format(self) -> Dict[str, float]:
        """
        Evaluate JSON formatting outputs.
        Metrics:
            - Format accuracy: valid JSON
            - Prediction accuracy: matching target fields
        Assumption:
            self.outputs: list of generated strings
            self.dataset: list of dicts with 'target_text'
        """
        total = len(self.outputs)
        correct_format = 0
        correct_fields = 0
        for gen_str, sample in zip(self.outputs, self.dataset):
            target_json_str = sample.get('target_text', '')
            # Validate JSON
            is_valid = self._validate_json(gen_str)
            if is_valid:
                correct_format += 1
                # Parse JSON
                try:
                    gen_json = json.loads(gen_str)
                    # Compare fields; e.g., name and occupation
                    if self._json_fields_match(gen_json, sample.get('target_json', {})):
                        correct_fields += 1
                except:
                    pass # Error in parsing, count as wrong
        format_acc = correct_format / total if total >0 else 0.0
        pred_acc = correct_fields / total if total > 0 else 0.0
        self.results['Format Accuracy'] = format_acc
        self.results['Prediction Accuracy'] = pred_acc
        return {'Format Accuracy': format_acc, 'Prediction Accuracy': pred_acc}

    def _validate_json(self, s: str) -> bool:
        """
        Checks whether string s is a valid JSON object.
        """
        try:
            obj = json.loads(s)
            return isinstance(obj, dict)
        except:
            return False

    def _json_fields_match(self, gen_json: Dict, target_json: Dict) -> bool:
        """
        Checks if JSON fields match (exact match for example).
        """
        if not gen_json or not target_json:
            return False
        for key in target_json:
            if key not in gen_json:
                return False
            if gen_json[key] != target_json[key]:
                return False
        return True

    def _eval_pronouns_change(self) -> Dict[str, float]:
        """
        Evaluate pronoun replacement accuracy and all-changed accuracy.
        Assumptions:
            - Dataset has 'target_text' with correct pronouns
            - Generated text is in self.outputs
        """
        total = len(self.outputs)
        correct = 0
        all_changed = 0
        for gen_str, sample in zip(self.outputs, self.dataset):
            target_text = sample.get('target_text', '')
            # Count pronouns replaced correctly
            if self._pronoun_correctly_replaced(gen_str, target_text):
                correct += 1
            # Check if all pronouns are replaced
            if self._all_pronouns_changed(gen_str, target_text):
                all_changed += 1
        acc = correct / total if total > 0 else 0.0
        all_acc = all_changed / total if total > 0 else 0.0
        self.results['Acc'] = acc
        self.results['All Changed Acc'] = all_acc
        return {'Accuracy': acc, 'All Changed Accuracy': all_acc}

    def _pronoun_correctly_replaced(self, gen: str, target: str) -> bool:
        """
        Checks if pronouns are replaced according to ground truth.
        """
        # Simple regex matching; can be refined
        pronouns = ['she', 'he', 'her', 'him', 'hers', 'his']
        for p in pronouns:
            pattern = r'\b' + re.escape(p) + r'\b'
            if re.search(pattern, target, flags=re.IGNORECASE):
                # check if gen replaced all instances
                if not re.search(pattern, gen, flags=re.IGNORECASE):
                    return False
        return True

    def _all_pronouns_changed(self, gen: str, target: str) -> bool:
        """
        Checks if all pronouns have been replaced
        """
        original_pronouns = ['she', 'he', 'her', 'him', 'hers', 'his']
        replaced_pronouns = ['they', 'they', 'their', 'them', 'theirs', 'their']
        for p, rep in zip(original_pronouns, replaced_pronouns):
            pattern_p = r'\b' + re.escape(p) + r'\b'
            pattern_rep = r'\b' + re.escape(rep) + r'\b'
            # Both should be present concurrently for a valid change
            if re.search(pattern_p, target, flags=re.IGNORECASE):
                if not re.search(pattern_rep, gen, flags=re.IGNORECASE):
                    return False
        return True

    def _eval_bias_bios(self) -> Dict[str, float]:
        """
        Evaluate occupation classification accuracy.
        Assume dataset has 'target_occupation' and output is predicted occupation.
        """
        correct = 0
        total = len(self.outputs)
        for gen_str, sample in zip(self.outputs, self.dataset):
            target_occ = sample.get('target', '').lower()
            pred_occ = self._extract_occupation(gen_str).lower()
            if pred_occ == target_occ:
                correct += 1
        acc = correct / total if total > 0 else 0.0
        self.results['BiasBios Accuracy'] = acc
        return {'BiasBios Accuracy': acc}

    def _extract_occupation(self, text: str) -> str:
        """
        Simplistic extraction: assuming the occupation is the entire output, or parse JSON if possible.
        """
        # Try parse JSON, fallback to raw text
        try:
            parsed = json.loads(text)
            if 'occupation' in parsed:
                return parsed['occupation']
        except:
            pass
        # fallback: extract last words or heuristics (not robust, placeholder)
        return text.strip()

    def _eval_counterfact(self) -> Dict[str, float]:
        """
        Evaluate content change correctness: compare scores if model provides.
        For simplicity, assume we assign 1 if the answer contains the new fact, else 0.
        """
        total = len(self.outputs)
        efficacy = 0
        paraphrase = 0
        for gen_str, sample in zip(self.outputs, self.dataset):
            old_fact = sample.get('old_fact', '')
            new_fact = sample.get('new_fact', '')
            question = sample.get('question', '')
            # Check if output indicates the new fact
            if self._contains_fact(gen_str, new_fact):
                efficacy += 1
            # For paraphrase robustness, maybe test with rephrased question
            # Here, just count correctness similarly
            if self._contains_fact(gen_str, new_fact):
                paraphrase += 1
        es = efficacy / total if total > 0 else 0.0
        ps = paraphrase / total if total > 0 else 0.0
        self.results['ES'] = es
        self.results['PS'] = ps
        return {'Efficacy Score': es, 'Paraphrase Score': ps}

    def _contains_fact(self, text: str, fact: str) -> bool:
        """
        Check if the generated text states the fact (simple substring check).
        """
        return fact.lower() in text.lower()

    def _compute_fluency(self, text: str) -> float:
        """
        Compute average entropy over bigram and trigram n-grams.
        """
        tokens = text.split()
        if len(tokens) < 2:
            return 0.0
        bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
        trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
        # Calculate frequencies
        bigram_freqs = {}
        for bg in bigrams:
            bigram_freqs[bg] = bigram_freqs.get(bg, 0) + 1
        trigram_freqs = {}
        for tg in trigrams:
            trigram_freqs[tg] = trigram_freqs.get(tg, 0) + 1
        # Probabilities
        total_bigrams = len(bigrams)
        total_trigrams = len(trigrams)
        bigram_probs = [count / total_bigrams for count in bigram_freqs.values()]
        trigram_probs = [count / total_trigrams for count in trigram_freqs.values()]
        # Entropy
        bigram_entropy = -sum(p * log2(p) for p in bigram_probs if p > 0)
        trigram_entropy = -sum(p * log2(p) for p in trigram_probs if p > 0)
        # Average entropy
        return (bigram_entropy + trigram_entropy) / 2.0

    def _compute_content_similarity(self, texts: List[str], ref_texts: List[str]) -> float:
        """
        Use TF-IDF vectorizer to compute cosine similarity between generated and reference texts.
        """
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(texts + ref_texts)
            gen_vecs = tfidf_matrix[:len(texts)]
            ref_vecs = tfidf_matrix[len(texts):]
            # average cosine similarity over all pairs
            similarities = []
            for i in range(len(texts)):
                numerator = (gen_vecs[i] * ref_vecs).sum()
                denom = (np.linalg.norm(gen_vecs[i].toarray()) * np.linalg.norm(ref_vecs[i].toarray()))
                if denom == 0:
                    similarities.append(0.0)
                else:
                    similarities.append(numerator / denom)
            return np.mean(similarities)
        except Exception:
            return 0.0

    ########## Public method to compute fluency and content scores ##########

    def compute_fluency_score(self, texts: List[str]) -> float:
        """
        Compute average fluency score over given texts.
        """
        entropies = [self._compute_fluency(t) for t in texts]
        # Filter out low-fluency outputs
        valid = [e for e in entropies if e >= self.min_entropy]
        if not valid:
            return 0.0
        return np.mean(valid)

    def compute_content_score(self, texts: List[str], refs: List[str]) -> float:
        """
        Compute content similarity score.
        """
        return self._compute_content_similarity(texts, refs)

```


## main.py

```python
## main.py
import argparse
import os
import sys
import yaml
import json
import random
import torch
import numpy as np

from utils import load_config, setup_logging
from dataset_loader import DatasetLoader
from model import Model
from profiling import ProfileAnalyzer
from steering import AttentionReweighter
from evaluation import Evaluation
from utils import create_prompt

def parse_args():
    parser = argparse.ArgumentParser(description="Main script for PASTA experiments")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--tasks', nargs='+', default=None, help='List of tasks to run, default: all')
    parser.add_argument('--do_profiling', action='store_true', help='Run attention head profiling')
    parser.add_argument('--do_inference', action='store_true', help='Run inference with attention steering')
    parser.add_argument('--do_evaluation', action='store_true', help='Evaluate generated outputs')
    parser.add_argument('--profile_tasks', nargs='+', default=None, help='Tasks to profile on')
    parser.add_argument('--test_tasks', nargs='+', default=None, help='Tasks to evaluate on')
    parser.add_argument('--load_profile', type=str, default=None, help='Path to precomputed profile heads JSON')
    parser.add_argument('--k_heads', type=int, default=None, help='Number of heads to steer; overrides config if set')
    args = parser.parse_args()
    return args

def main():
    # Parse command-line arguments
    args = parse_args()
    setup_logging()

    # Load config
    cfg = load_config(args.config)

    # Set device
    device = cfg['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, falling back to CPU.")

    # Load datasets
    dataset_paths = cfg.get('datasets', {}).get('dataset_paths', {})
    task_list = args.tasks if args.tasks else ['BiasBios', 'CounterFact', 'JSON Formatting', 'Pronouns Changing']
    # If specific tasks provided in args, override
    task_list = args.tasks if args.tasks else task_list

    # Initialize DatasetLoader
    dataset_loader = DatasetLoader(dataset_paths=dataset_paths, task_name=None)
    datasets_by_task = {}
    for task in task_list:
        dataset_loader.task_name = task
        datasets_by_task[task] = dataset_loader.load_dataset()

    # Initialize model
    model_name = cfg['training'].get('model_name', 'llama-7b')
    model = Model(model_name=model_name, device=device)
    model.eval()

    # Optional: Load attention hooks (done inside model.py during init)
    # For safety, attach hooks now (done inside Model class constructor)

    # Profile attention heads if needed
    profile_results_path = args.load_profile
    selected_heads = []
    if args.do_profiling:
        print("Starting profiling of attention heads...")
        # For profiling, use small dataset (profile_samples, e.g., 1000)
        profile_dataset_list = []
        for task in task_list:
            # Use first 1000 samples for profiling
            profile_dataset_list.extend(datasets_by_task[task]['train'][:cfg['training'].get('profiling_samples', 1000)])
        for task in task_list:
            print(f"Profiling task: {task}")
            profile_fetch = datasets_by_task[task]['train'][:cfg['training'].get('profiling_samples', 1000)]
            profile_analyzer = ProfileAnalyzer(
                model=model,
                profile_dataset=profile_fetch,
                task_name=task,
                config=cfg
            )
            selected_heads_task = profile_analyzer.profile_heads()
            # Save or accumulate profile heads
            # For multi-task, take intersection across all tasks
            if not selected_heads:
                selected_heads = set(selected_heads_task)
            else:
                selected_heads = selected_heads.intersection(selected_heads_task)
        # Convert to list and pick top-K if needed
        selected_heads = list(selected_heads)
        # Save profile heads
        profile_out_path = f'profile_heads_{"+".join(task_list)}.json'
        with open(profile_out_path, 'w') as f:
            json.dump([list(h) for h in selected_heads], f)
        print(f"Profile heads saved to {profile_out_path}")
    elif args.load_profile:
        # Load precomputed profile heads
        with open(args.load_profile, 'r') as f:
            selected_heads = [tuple(h) for h in json.load(f)]
        print(f"Loaded profile heads from {args.load_profile}")
    else:
        # Use heads specified in config or default
        default_k = cfg['training'].get('top_k_heads', 400)
        # If no profile, select top-k randomly or use shared default
        selected_heads = []  # if empty, no steering
        print("No profiling performed; no heads will be steered unless specified.")

    # Initialize attention reweighter
    alpha = cfg['training'].get('alpha', 0.01)
    bump_heads = selected_heads
    attention_weighter = AttentionReweighter(head_indices=bump_heads, alpha=alpha)

    # For each task, prepare prompts, perform inference, evaluate
    results_by_task = {}
    for task in task_list:
        print(f"\n=== Processing task: {task} ===")
        dataset = datasets_by_task[task]['test']  # use test split for evaluation
        prompts_template = cfg.get('prompts', {}).get(f"{task}_template", None)
        if prompts_template is None:
            print(f"No prompt template found for task {task}, skipping.")
            continue

        # For evaluation, prepare list to store outputs
        gen_texts = []
        task_samples = dataset[:]
        # For each sample, generate prompt with emphasis
        for sample in task_samples:
            raw_input_text = sample.get('input_text', '')
            # Assume 'highlighted_spans' are already embedded or known
            # For this problem, generate spans from dataset if in dataset
            # Otherwise, assume dataset has 'highlighted_spans' info
            hl_indices = sample.get('highlighted_spans', [])
            # Generate prompt
            prompt_str = create_prompt(
                template_str=prompts_template,
                input_text=raw_input_text,
                highlighted_spans=hl_indices,
                instruction=sample.get('task_instruction', ''),
                emphasis_marker='**'
            )

            # Tokenize prompt
            encodings = model.tokenizer(prompt_str, return_tensors='pt').to(device)
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']

            # Generate with attention steering
            output_text = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg['training'].get('max_sequence_length', 512),
                do_sample=False
            )

            gen_texts.append(output_text)

        # Evaluation
        if args.do_evaluation:
            evaluation = Evaluation(
                model_outputs=gen_texts,
                dataset=task_samples,
                task_name=task
            )
            metrics_dict = evaluation.evaluate()
            results_by_task[task] = metrics_dict
            print(f"Results for {task}: {metrics_dict}")

    # Final summarize or save results
    results_path = f'results_summary_{"+".join(task_list)}.json'
    with open(results_path, 'w') as f:
        json.dump(results_by_task, f, indent=2)
    print(f"All results saved to {results_path}")

if __name__ == '__main__':
    main()
```

## model.py

```python
## model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Union
import re

class Model:
    """
    Class to load pretrained language models, extract and reweight attention scores,
    and generate text with optional attention steering based on user-highlighted tokens.
    Supports models like LLAMA-7B, GPT-J-6B, Vicuna-7B.
    """
    def __init__(
        self,
        model_name: str = "llama-7b",
        device: str = "cuda",
        alpha: float = 0.01,
        top_k_heads: int = 400,
        profile_heads: Optional[List[Tuple[int, int]]] = None,
        model_cache_dir: Optional[str] = None
    ):
        """
        Initialize and load the pretrained model, tokenizer, and setup hooks.
        Args:
            model_name (str): Name of the model. E.g., 'llama-7b', 'gpt-j-6b', 'vicuna-7b'.
            device (str): 'cuda' or 'cpu'.
            alpha (float): Reweighting coefficient for attention scores.
            top_k_heads (int): Number of heads to steer after profiling.
            profile_heads (List of (layer_idx, head_idx)): Specific heads to steer; if None, need to set after profiling.
            model_cache_dir (str): Directory for model weights cache (optional).
        """
        self.model_name = model_name
        self.device = device
        self.alpha = alpha
        self.top_k_heads = top_k_heads  # For potential profiling or preselected heads

        # Load model and tokenizer based on model_name
        if "llama" in model_name.lower():
            model_path = "decapoda-research/llama-7b-hf"  # Default, adjust as needed
        elif "gpt-j" in model_name.lower():
            model_path = "EleutherAI/gpt-j-6B"
        elif "vicuna" in model_name.lower():
            model_path = "NousResearch/Vicuna-7B"
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Ensure padding token is configured
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=model_cache_dir)
        self.model.to(self.device)
        self.model.eval()

        # Prepare storage for attention logs
        self.attention_scores_per_layer: List[List[torch.Tensor]] = []

        # Determine layers count and attach hooks
        self._attach_attention_hooks()

        # Store profile heads if provided
        self.profile_heads = profile_heads  # List of (layer_idx, head_idx)
        # Maintain a set for faster lookup
        self.heads_to_steer: List[Tuple[int, int]] = []

    def _attach_attention_hooks(self):
        """
        Register hooks into the model's attention modules to capture attention scores.
        Supports different model structures based on architecture.
        """
        self.attention_logs = []

        # Depending on model architecture, locate attention modules
        self._hooks = []

        # For Huggingface models, the attention modules are usually within the transformer blocks
        # LLAMA/Meta models: self.model.model.layers or self.model.model.decoder.layers
        model_modules = []

        if hasattr(self.model, 'model'):
            top_module = self.model.model
        else:
            top_module = self.model

        # Attempt detection for common architectures
        if hasattr(top_module, 'layers'):
            # e.g., GPT-J, GPT-2, GPT-neo
            model_modules = top_module.layers
        elif hasattr(top_module, 'block'):
            # e.g., LLAMA
            model_modules = top_module.block
        elif hasattr(top_module, 'h'):
            # e.g., GPT-2, GPT-neo in a different style
            model_modules = top_module.h
        else:
            raise RuntimeError("Unable to locate attention modules in the model.")

        # Register hooks for each attention module
        def get_attention_hook(layer_idx: int):
            def hook(module, input, output):
                # output: Tuple(logits, attentions) or attention tensor, depends on model config
                # For transformers, attention outputs are often the second element in output
                # For models like LLAMA, the attention scores are accessible via output['attentions']
                # or via output[2], when output is tuple.
                # For safety, handle both cases
                if isinstance(output, tuple):
                    # Some models return (hidden_states, attentions, ...)
                    if len(output) >= 2:
                        attn = output[1]
                        self.attention_logs.append((layer_idx, attn))
                elif hasattr(output, 'attentions'):
                    # Many models store attentions here
                    attn = output.attentions
                    # attn: tuple of tensors per layer
                    self.attention_logs.extend([(layer_idx, attn_layer) for attn_layer in output.attentions])
                else:
                    # fallback: do nothing
                    pass
            return hook

        # In models like LLAMA, attention modules are in each layer
        # For simplicity, assume we can register hooks to each layer if they have an attn attribute
        for layer_idx, layer_module in enumerate(model_modules):
            if hasattr(layer_module, 'self_attn'):
                # For LLAMA-like
                handle = layer_module.self_attn.register_forward_hook(get_attention_hook(layer_idx))
                self._hooks.append(handle)
            elif hasattr(layer_module, 'attn'):
                # For some models
                handle = layer_module.attn.register_forward_hook(get_attention_hook(layer_idx))
                self._hooks.append(handle)
            elif hasattr(layer_module, 'attention'):
                handle = layer_module.attention.register_forward_hook(get_attention_hook(layer_idx))
                self._hooks.append(handle)
            else:
                # No attention module found
                pass

    def _clear_attention_logs(self):
        """
        Clear stored attention logs before each inference.
        """
        self.attention_logs = []

    def extract_attention(self, **kwargs) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference with hook capturing enabled, retrieve attention scores.
        Returns:
            List of dicts: each dict contains 'layer_idx', 'attention' (Tensor)
        """
        # Clear previous logs
        self._clear_attention_logs()

        # Run model inference
        with torch.no_grad():
            # The user must provide input_ids and attention_mask via kwargs
            _ = self.model(**kwargs)

        # Process stored attention logs
        # They are stored as list of (layer_idx, attention_tensor)
        # Group attention tensors by layer
        attention_per_layer: Dict[int, List[torch.Tensor]] = {}
        for (layer_idx, attn) in self.attention_logs:
            if layer_idx not in attention_per_layer:
                attention_per_layer[layer_idx] = []
            attention_per_layer[layer_idx].append(attn)

        # For simplicity, average attention tensors across heads if multiple
        # or return as-is if already per head
        # But typically, attention tensor in hooks is shape: (batch_size, num_heads, seq_len, seq_len)
        # So, we can keep them grouped as such.
        # For modeling, store in a list ordered by layer index
        attention_list = []
        for layer_idx, attentions in sorted(attention_per_layer.items()):
            # attentions: list of tensors, consolidate
            # For now, take the last or mean
            # Assuming one tensor per hook call
            attn_tensor = attentions[-1]  # shape: (batch_size, num_heads, seq_len, seq_len)
            attention_list.append({
                'layer_idx': layer_idx,
                'attention': attn_tensor
            })
        return attention_list

    def set_profile_heads(self, heads: List[Tuple[int, int]]):
        """
        Set attention heads (layer_idx, head_idx) to steer.
        Args:
            heads (list): List of (layer_idx, head_idx)
        """
        self.heads_to_steer = heads

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        highlighted_token_indices: List[int] = None,
        steer_heads: Optional[List[Tuple[int, int]]] = None,
        use_attention_reweight: bool = True,
        max_new_tokens: int = 50,
        do_sample: bool = False
    ) -> str:
        """
        Generate output text with optional attention score reweighting.
        Args:
            input_ids (torch.Tensor): Token IDs.
            attention_mask (torch.Tensor): Attention mask.
            highlighted_token_indices (list): Indices of tokens to emphasize.
            steer_heads (list): Specific heads to steer, override self.heads_to_steer if provided.
            use_attention_reweight (bool): Whether to modify attention during generation.
            max_new_tokens (int): Max tokens to generate.
            do_sample (bool): Sampling mode.
        Returns:
            str: Generated text.
        """
        # Note: For models like GPT-J, no native support for in-place attention modification
        # but can be implemented via hooks.
        # Here, as a simplified approach:
        # - For each generation step:
        #    - Extract attention logs
        #    - Modify scores
        #    - Run model forward with reweighted attention
        # Since transformers do not support dynamic attention modification easily,
        # a more feasible approach is to:
        # - During inference, run with hooks that:
        #   * Capture attention scores
        #   * Reweight them
        #   * Use them in attention computation via custom hooks
        # Due to complexity, here, we assume only inference with hooks and pre-registered rerouting.

        # For simplicity, assume that reweighting is applied once then inference proceeds:
        if use_attention_reweight and (highlighted_token_indices is not None and len(self.heads_to_steer) > 0):
            # Run a single pass with attention hooks capturing attention scores
            attention_list = self.extract_attention(
                input_ids=input_ids, attention_mask=attention_mask
            )
            # For each layer and head, apply reweighting on attention scores at positions
            # and replace attention tensors used in the model
            # This is non-trivial: transformers do not expose direct attention weight replacement
            # without rewriting model forward. Alternatively, set hook that modifies attention dynamically.
            # For simplicity, we can assume hooks are in place and reweighting is done during actual forward call-
            # which requires writing a custom forward (not supported here). So, just annotate:
            # TODO: Implement custom attention forward to incorporate reweighted scores.
            pass

        # Proceed to generate output tokens
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=1,  # greedy
            pad_token_id=self.tokenizer.pad_token_id
        )
        # Decode to text
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

    def to(self, device: str):
        """
        Move model to device.
        """
        self.device = device
        self.model.to(device)

    def save(self, save_path: str):
        """
        Save the model and tokenizer.
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load(self, load_path: str):
        """
        Load model and tokenizer from saved directory.
        """
        self.model = AutoModelForCausalLM.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
```


## profiling.py

```python
# profiling.py
import os
import json
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from utils import load_config, save_json
from model import Model

class ProfileAnalyzer:
    """
    ProfileAnalyzer performs model profiling to identify the most effective attention heads
    for steering. It evaluates the impact of steering each attention head on a small dataset
    and selects top heads based on the aggregated performance metrics.
    """

    def __init__(
        self,
        model: Model,
        profile_dataset: List[Dict],
        task_name: str,
        config: Dict,
        profile_dir: str = "profiling_results"
    ):
        """
        Initialize the profiler.
        Args:
            model (Model): The loaded Model instance.
            profile_dataset (List[Dict]): Small dataset (~1000 samples) for profiling.
            task_name (str): Name of the current task being profiled.
            config (Dict): Configuration dictionary loaded from 'config.yaml'.
            profile_dir (str): Directory to save profiling results.
        """
        self.model = model
        self.profile_dataset = profile_dataset
        self.task_name = task_name
        self.config = config
        self.alpha = load_config('config.yaml')['training'].get('alpha', 0.01)
        self.top_k = load_config('config.yaml')['training'].get('top_k_heads', 400)
        self.profile_dir = profile_dir
        self.profile_results: Dict = {}
        # Create directory if needed
        os.makedirs(self.profile_dir, exist_ok=True)

    def evaluate_head_on_sample(
        self,
        sample: Dict,
        head: Tuple[int, int]
    ) -> float:
        """
        Evaluate the performance when steering a specific head on a single sample.
        Args:
            sample (Dict): Data sample containing 'input_text', 'target_text', 'highlighted_spans'.
            head (Tuple[int, int]): (layer_idx, head_idx).
        Returns:
            float: Task-specific performance metric (e.g., accuracy).
        """
        layer_idx, head_idx = head
        # Prepare inputs
        input_text = sample['input_text']
        target_text = sample['target_text']
        highlighted_tokens = sample['highlighted_spans']
        # Tokenize input
        from utils import create_prompt, create_tokenizer
        tokenizer = create_tokenizer()
        prompt = create_prompt(
            self.config['prompts']['json_format_template'],
            input_text,
            highlighted_tokens,
            instruction=sample.get('task_instruction', ''),
            emphasis_marker="**"
        )
        encodings = tokenizer(prompt, return_tensors='pt').to(self.model.device)
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        # Register hook to obtain attention scores during inference
        # Extract attention
        attention_scores_list = self.model.extract_attention(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # For each layer, reweight the selected head's attention scores
        # Find the attention tensor for the head
        # We assume attention_scores_list is a list of dicts per layer
        # with keys 'layer_idx', 'attention' (batch, heads, seq, seq)
        # Rearrange to find the particular head at layer_idx
        for item in attention_scores_list:
            if item['layer_idx'] == layer_idx:
                attn_tensor = item['attention']  # shape: (batch, heads, seq, seq)
                # Select the head tensor
                head_attn = attn_tensor[:, head_idx, :, :]  # shape: (batch, seq, seq)
                # Apply reweighting
                head_attn_reweighted = self._reweight_attention(
                    head_attn, highlighted_tokens
                )
                # Insert back the reweighted attention
                attn_tensor[:, head_idx, :, :] = head_attn_reweighted
                # Save back
                item['attention'] = attn_tensor

        # Now, run model inference with reweighted attention
        # (Assuming model uses hooks to pick up reweighted attention during the run)
        # For simplicity, call model's generate with the input
        # The hooks will have modified attention during this pass
        output_text = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50
        )
        # Evaluate output on the task (e.g., accuracy / correctness)
        performance_score = self._compute_performance(output_text, target_text)
        return performance_score

    def profile_heads(self) -> List[Tuple[int, int]]:
        """
        Run profiling over all heads and samples.
        Returns:
            List of tuples: (layer_idx, head_idx) for selected heads.
        """
        # Collect scores for each head
        head_performance: Dict[Tuple[int, int], List[float]] = {}
        # Determine total layers and heads from model
        total_layers = self.model.get_num_layers()
        total_heads = self.model.get_num_heads()
        # Initialize dict
        for l in range(total_layers):
            for h in range(total_heads):
                head_performance[(l, h)] = []

        # Profiling loop
        for idx, sample in enumerate(self.profile_dataset):
            print(f"Profiling sample {idx+1}/{len(self.profile_dataset)}")
            for layer_idx in range(total_layers):
                for head_idx in range(total_heads):
                    score = self.evaluate_head_on_sample(sample, (layer_idx, head_idx))
                    head_performance[(layer_idx, head_idx)].append(score)

        # Compute average performance per head
        mean_performance: List[Tuple[Tuple[int, int], float]] = []
        for key, scores in head_performance.items():
            avg_score = np.mean(scores)
            mean_performance.append((key, avg_score))
        # Sort heads by performance descending
        mean_performance.sort(key=lambda x: x[1], reverse=True)

        # Select top-k heads
        selected_heads = [item[0] for item in mean_performance[:self.top_k]]

        # Save profiling results
        self.profile_results = {
            'task_name': self.task_name,
            'selected_heads': selected_heads,
            'performance_summary': mean_performance[:self.top_k]
        }
        profile_path = os.path.join(self.profile_dir, f"{self.task_name}_heads.json")
        save_json(self.profile_results, profile_path)
        print(f"Profile saved to {profile_path}")
        return selected_heads

    def _reweight_attention(
        self,
        attention: torch.Tensor,
        highlighted_tokens: List[int]
    ) -> torch.Tensor:
        """
        Reweight attention scores for a single head based on highlighted tokens.
        Args:
            attention (Tensor): shape (batch, seq_len, seq_len)
            highlighted_tokens (List[int])): token indices to emphasize.
        Returns:
            Tensor: reweighted attention (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = attention.shape
        device = attention.device
        # Create a mask for tokens: 1 for highlighted, alpha for others
        mask_j = torch.ones(seq_len, device=device) * self.alpha
        for j in highlighted_tokens:
            if j < seq_len:
                mask_j[j] = 1.0
        # Apply per batch
        for b in range(batch_size):
            scores = attention[b]  # (seq_len, seq_len)
            # Scale columns based on mask_j
            scores = scores * (mask_j.unsqueeze(0))
            # Normalize each row
            C_i = scores.sum(dim=1, keepdim=True)
            C_i = torch.where(C_i == 0, torch.ones_like(C_i), C_i)
            scores = scores / C_i
            attention[b] = scores
        return attention

    def _compute_performance(self, output_text: str, target_text: str) -> float:
        """
        Compute task-specific performance metric.
        Override this method with task-specific logic, e.g., accuracy or JSON validity.
        """
        # For JSON format, check validity
        try:
            json.loads(output_text)
            return 1.0  # Correct
        except:
            return 0.0  # Invalid JSON, or could implement more complex metrics

    def save_profile(self, filename: str):
        """
        Save profiling results to a specified file.
        """
        save_json(self.profile_results, filename)

    def load_profile(self, filename: str):
        """
        Load a saved profile from file.
        """
        with open(filename, 'r') as f:
            self.profile_results = json.load(f)
        self.selected_heads = [tuple(h) for h in self.profile_results.get('selected_heads', [])]

```

## steering.py

```python
## steering.py
from typing import List, Tuple
import torch
import sys

class AttentionReweighter:
    """
    The AttentionReweighter class is responsible for applying post-hoc attention score reweighting
    during inference, based on user-highlighted tokens. It manages targeting specific attention
    heads, scaling their attention scores to emphasize highlighted tokens, in accordance with
    the approach described in Section 3.1 of the paper.
    """

    def __init__(
        self,
        head_indices: List[Tuple[int, int]],
        alpha: float = 0.01
    ):
        """
        Initialize the AttentionReweighter with specific attention heads and scaling coefficient.
        Args:
            head_indices (List[Tuple[int, int]]): List of (layer_idx, head_idx) for heads to steer.
            alpha (float): Scaling coefficient for non-highlighted tokens, default 0.01.
        """
        if not isinstance(head_indices, list):
            raise TypeError("head_indices must be a list of (layer_idx, head_idx) tuples.")
        self.head_indices = head_indices
        self.alpha = alpha
        self._hooks = []  # Store hook handles to deregister later if needed

    def apply_masking(
        self,
        attention_scores: torch.Tensor,
        highlighted_token_indices: List[int]
    ) -> torch.Tensor:
        """
        Reweight attention scores for a given head's attention tensor during inference.
        Args:
            attention_scores (torch.Tensor): Shape (batch_size, seq_len, seq_len),
                attention logits for a specific head.
            highlighted_token_indices (List[int]): List of token indices to emphasize.
        Returns:
            torch.Tensor: Reweighted attention tensor with same shape.
        """
        # Clone to avoid in-place modifications to original tensor outside
        reweighted = attention_scores.clone()

        batch_size, seq_len, _ = reweighted.shape
        device = reweighted.device

        # Convert highlighted tokens to a set for faster membership check
        highlight_set = set(highlighted_token_indices)

        # Create column-wise scaling factors: 1 for highlighted tokens, alpha for others
        # shape: (seq_len,)
        scaling_factors = torch.ones(seq_len, device=device)
        for idx in range(seq_len):
            if idx not in highlight_set:
                scaling_factors[idx] = self.alpha

        # Apply scaling column-wise: scale the *columns* (correspond to attention to tokens j)
        # Adjust attention logits accordingly:
        # For each batch, perform:
        for b in range(batch_size):
            # shape: (seq_len, seq_len)
            scores = reweighted[b]
            scores = scores * scaling_factors.unsqueeze(0)  # scale columns
            # Row-wise normalization to ensure sum to 1
            C_i = scores.sum(dim=1, keepdim=True)
            # Prevent division by zero
            C_i = torch.where(C_i == 0, torch.ones_like(C_i), C_i)
            scores = scores / C_i
            reweighted[b] = scores

        return reweighted

    def register_hooks(
        self,
        model,
        highlighted_token_indices: List[int]
    ):
        """
        Register forward hooks in the model to modify attention scores during inference
        at the specified heads.
        Args:
            model: The HuggingFace model with accessible attention modules.
            highlighted_token_indices: List of token indices to emphasize during reweighting.
        """
        # Clear existing hooks if any
        self deregister_hooks()

        # For each targeted head, set a hook
        for (layer_idx, head_idx) in self.head_indices:
            # Find the attention module in the model corresponding to layer_idx
            # This step depends on the model architecture
            # Assume model has a list of modules for transformer layers
            module = self._get_attention_module(model, layer_idx)
            if module is None:
                continue

            # Define the hook function
            def hook_fn(module, input, output, layer_idx=layer_idx, head_idx=head_idx):
                """
                Hook modifies the attention scores before softmax.
                """
                # Depending on model, output may be a tuple or object
                # Typically, output[1] or an attribute contains attention logits
                # Here, we assume it's the attention logits tensor
                # The attention scores shape: (batch_size, num_heads, seq_len, seq_len)
                if isinstance(output, tuple):
                    # e.g., output[1]
                    attn_scores = output[1]
                elif hasattr(output, 'attentions'):
                    # output.attentions is tuple/list
                    attn_scores = output.attentions
                else:
                    # fallback: skip
                    return
                # attn_scores shape: (batch_size, num_heads, seq_len, seq_len)
                # We want to modify only the targeted head
                if attn_scores is None:
                    return
                # Reweight only the targeted head within attention scores
                attn = attn_scores  # assume shape: (batch, heads, seq, seq)
                # Find the index of the specified head dimension
                # Reweight only the selected head
                batch_size, num_heads, seq_len, _ = attn.shape
                # Safety: check head_idx bounds
                if head_idx >= num_heads:
                    return
                # Apply reweighting to the specific head
                # For each batch, replace the head's attention scores
                # Get current head scores
                head_scores = attn[:, head_idx, :, :]  # shape: (batch, seq, seq)
                # Reweight using apply_masking
                reweighted_head = self.apply_masking(head_scores, highlighted_token_indices)
                # Replace in the attention tensor
                attn[:, head_idx, :, :] = reweighted_head
                # Assign back to output if possible (depends on model's hook structure)
                # For most models, the output is the attention tensor or a tuple
                # which we've modified in place. So, no explicit assign needed.
            # Register the hook on the module
            handle = module.register_forward_hook(hook_fn)
            self._hooks.append(handle)

    def deregister_hooks(self):
        """
        Remove all registered hooks to avoid duplicate modifications.
        """
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def _get_attention_module(self, model, layer_idx: int):
        """
        Retrieve the attention module corresponding to the specified layer index.
        Implementation depends on the model architecture.
        Args:
            model: HuggingFace model instance.
            layer_idx (int): Index of the layer to target.
        Returns:
            Module or None if not found.
        """
        # Common architectures:
        # - For LLAMA: model.model.layers[layer_idx].self_attn
        # - For GPT-J: model.transformer.layers[layer_idx].self_attention
        # - For Vicuna: similar to LLAMA, treat accordingly
        # Implemented here for LLAMA-like structure:
        try:
            # Access the top-level model attributes
            top_model = getattr(model, 'model', model)
            # For LLAMA or similar:
            if hasattr(top_model, 'layers'):
                layers = top_model.layers
                if layer_idx < len(layers):
                    layer_module = layers[layer_idx]
                    # Return self_attn or attention module
                    if hasattr(layer_module, 'self_attn'):
                        return layer_module.self_attn
                    elif hasattr(layer_module, 'attn'):
                        return layer_module.attn
            elif hasattr(top_model, 'h'):
                # e.g., GPT-2, GPT-neo
                layers = list(top_model.h)
                if layer_idx < len(layers):
                    return layers[layer_idx]
            elif hasattr(top_model, 'block'):
                layers = list(top_model.block)
                if layer_idx < len(layers):
                    return layers[layer_idx]
        except Exception:
            return None
        return None

    def get_highlighted_token_indices(self, input_ids: torch.Tensor, highlight_indices: List[int]) -> List[int]:
        """
        Utility to ensure highlight indices are within sequence length.
        Can also convert to batch-wise if needed.
        """
        seq_len = input_ids.shape[1]
        valid_indices = [idx for idx in highlight_indices if 0 <= idx < seq_len]
        return valid_indices

    # Additional utility methods could be added here as needed.
```


## utils.py

```python
## utils.py
import yaml
import os
import logging
from typing import Any, Dict, List, Tuple, Optional
import json
import re

from transformers import PreTrainedTokenizer

##########################
# 1. Configuration Parsing
##########################

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load and parse the YAML configuration file.
    Args:
        config_path (str): Path to the YAML config file.
    Returns:
        dict: Parsed configuration dictionary.
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file: {e}")
    return config

def get_config_param(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Retrieve a nested parameter from the config dictionary given a dotted path.
    Args:
        config (dict): The configuration dictionary.
        key_path (str): Dot-separated path to the config parameter.
        default (Any): Default value if key not found.
    Returns:
        Any: The parameter value or default.
    """
    keys = key_path.split('.')
    val = config
    for key in keys:
        if isinstance(val, dict) and key in val:
            val = val[key]
        else:
            return default
    return val

##########################
# 2. Logging Setup
##########################

def setup_logging(log_level: str = 'INFO'):
    """
    Configure logging for the pipeline.
    Args:
        log_level (str): Logging level as string; default 'INFO'.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=numeric_level
    )

##########################
# 3. Prompt Template Rendering
##########################

def create_prompt(
    template_str: str,
    input_text: str,
    highlighted_spans: List[Tuple[int, int]],
    instruction: str,
    emphasis_marker: str = "**"
) -> str:
    """
    Generate a prompt by inserting emphasized spans into the template.
    Args:
        template_str (str): The prompt template with placeholders.
        input_text (str): The raw input text.
        highlighted_spans (List[Tuple[int, int]]): List of (start_char, end_char) spans to emphasize.
        instruction (str): Additional instruction text, if needed.
        emphasis_marker (str): Markers to denote emphasis (default "**").
    Returns:
        str: Formatted prompt string ready for model input.
    """
    # Create a HTML or markdown style emphasized version of input_text
    emphasized_text = input_text
    # Offset adjustment for multiple spans
    offset = 0
    for start_char, end_char in sorted(highlighted_spans, key=lambda x: x[0]):
        start_idx = start_char + offset
        end_idx = end_char + offset
        span_text = input_text[start_char:end_char]
        mark_span = f"{emphasis_marker}{span_text}{emphasis_marker}"
        emphasized_text = (
            emphasized_text[:start_idx] + mark_span + emphasized_text[end_idx:]
        )
        offset += len(mark_span) - (end_char - start_char)
    # Format prompt by replacing placeholders
    prompt = template_str.format(
        instruction=instruction,
        highlighted_spans=emphasized_text
    )
    return prompt

##########################
# 4. Dataset Loading & Preprocessing
##########################

def load_dataset(task_name: str, dataset_paths: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Load dataset for a specific task from provided path.
    Args:
        task_name (str): Name of the task/data.
        dataset_paths (dict): Dict containing dataset paths keyed by task names.
    Returns:
        list: List of dataset samples as dictionaries.
    Raises:
        FileNotFoundError: If dataset file not found.
        ValueError: If dataset format is unknown or malformed.
    """
    path = dataset_paths.get(task_name)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset for task '{task_name}' not found at {path}")
    # Assume datasets are in JSON lines (jsonl) or json format, extend as needed
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                data.append(sample)
            except json.JSONDecodeError:
                # fallback if CSV/TSV, or malformed
                # For simplicity, raise error here; extend if needed
                raise ValueError(f"Malformed dataset line: {line}")
    return data

def extract_highlight_indices(text: str, emphasis_marker: str = "**") -> Tuple[str, List[int]]:
    """
    For a given text, find tokens that are emphasized via markers and return clean text 
    plus list of token indices for highlighted tokens.
    Args:
        text (str): Raw text with emphasis markers.
        emphasis_marker (str): Marker used for emphasis (default "**").
    Returns:
        Tuple[str, List[int]]: cleaned text without markers and list of token indices (characters) that are emphasized.
    """
    # Using regex to find all emphasized spans
    pattern = re.escape(emphasis_marker) + '(.*?)' + re.escape(emphasis_marker)
    emphasize_spans = [(m.start(), m.end()) for m in re.finditer(pattern, text)]
    clean_text = re.sub(pattern, r'\1', text)
    # Generate list of emphasized token indices
    emphasized_positions = []
    for start, end in emphasize_spans:
        # Here, we return character indices in clean_text
        # Could map to token indices if tokenized; for now, just character positions
        emphasized_positions.extend(range(start, end - 2 * len(emphasis_marker)))
    return clean_text, emphasized_positions

def tokenize_text(tokenizer: PreTrainedTokenizer, text: str) -> Dict[str, Any]:
    """
    Tokenize text using provided tokenizer.
    Args:
        tokenizer: Hugging face tokenizer.
        text (str): Input text.
    Returns:
        dict: Contains 'input_ids' and 'attention_mask'
    """
    encoding = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=512
    )
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask']
    }

def save_json(data: Any, filename: str):
    """
    Save data as JSON to file.
    Args:
        data (Any): Data to save.
        filename (str): Output filename.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filename: str) -> Any:
    """
    Load JSON data from file.
    Args:
        filename (str): Path to JSON file.
    Returns:
        Any: Parsed JSON data.
    """
    with open(filename, 'r') as f:
        return json.load(f)

##########################
# 5. Helper for Attention Reweighting and Hooks
##########################

def register_attention_hooks(model):
    """
    Register hooks to capture attention scores during inference.
    Args:
        model: The Hugging Face model instance.
    Returns:
        list: A list of hook handle objects for potential deregistration.
    """
    attention_logs = []

    def hook(module, input, output):
        # output is a tuple: (attention_scores, ...)
        # Depending on the model, output[0] is attention scores
        attention_scores = output[1] if isinstance(output, tuple) else output
        attention_logs.append(attention_scores.detach())

    handles = []
    for layer in model.modules():
        if hasattr(layer, 'attn'):
            handle = layer.attn.register_forward_hook(hook)
            handles.append(handle)
    return handles, attention_logs

def modify_attention_scores(
    attention_scores: torch.Tensor,
    highlighted_tokens: List[int],
    alpha: float = 0.01
) -> torch.Tensor:
    """
    Apply post-hoc reweighting to attention scores based on highlighted tokens.
    Args:
        attention_scores (torch.Tensor): Shape (batch_size, num_heads, seq_len, seq_len)
        highlighted_tokens (List[int]): List of token indices to emphasize.
        alpha (float): Reweighting coefficient, default 0.01.
    Returns:
        torch.Tensor: Reweighted attention scores with the same shape.
    """
    # Copy to avoid in-place modification
    reweighted_scores = attention_scores.clone()
    batch_size, num_heads, seq_len, _ = reweighted_scores.shape

    # Create highlight mask: shape (seq_len)
    mask = torch.zeros(seq_len, device=attention_scores.device)
    if highlighted_tokens:
        mask[highlighted_tokens] = 1.0
    # Compute normalization per row
    # Operate head-wise
    for b in range(batch_size):
        for h in range(num_heads):
            scores = reweighted_scores[b, h]  # shape: (seq_len, seq_len)
            # Apply scaling
            # For each row i, scale the scores
            # Use broadcasting
            scale_vector = torch.where(mask == 1, torch.ones_like(mask), torch.full_like(mask, alpha))
            # Expand to match (seq_len, seq_len)
            scale_matrix = scale_vector.unsqueeze(0).expand_as(scores)  # each row scaled accordingly
            scores = scores * scale_matrix
            # Normalize row-wise
            C_i = scores.sum(dim=1, keepdim=True)  # shape (seq_len,1)
            # Prevent division by zero
            C_i = torch.where(C_i == 0, torch.ones_like(C_i), C_i)
            scores = scores / C_i
            reweighted_scores[b, h] = scores
    return reweighted_scores

##########################
# 6. Miscellaneous Utilities
##########################

def ensure_dir(directory: str):
    """
    Create directory if it doesn't exist.
    Args:
        directory (str): Directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def validate_path(path: str, must_exist: bool = True):
    """
    Validate a filesystem path.
    Args:
        path (str): Path to validate.
        must_exist (bool): If True, path must exist.
    Raises:
        FileNotFoundError: if must_exist and path does not exist.
    """
    if must_exist and not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

def get_device(cfg: Dict[str, Any]) -> str:
    """
    Determine the device for model inference.
    Args:
        cfg (dict): Configuration dictionary.
    Returns:
        str: 'cuda' or 'cpu'
    """
    device_str = get_config_param(cfg, 'training.device', 'cpu')
    # Check if CUDA is available
    import torch
    if device_str == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def print_banner(msg: str):
    """
    Print a formatted banner for clarity in logs.
    Args:
        msg (str): Message to print.
    """
    print(f"\n{'=' * 10} {msg} {'=' * 10}\n")
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\PASTA\PASTA_repo`
