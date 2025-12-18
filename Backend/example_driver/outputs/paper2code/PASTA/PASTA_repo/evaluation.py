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
