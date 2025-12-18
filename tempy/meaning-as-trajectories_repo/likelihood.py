## likelihood.py
import torch
from typing import List
from transformers import AutoModelForCausalLM

class LikelihoodCalculator:
    """
    A class to compute the log probability of sequences given prompts using the provided model wrapper.
    Supports caching for efficiency and batch calculations if required.
    """

    def __init__(self, model_wrapper, cache=None):
        """
        Initialize the LikelihoodCalculator.
        Args:
            model_wrapper (ModelWrapper): Instance of the model wrapper with inference methods.
            cache (dict, optional): Dictionary to cache computed log probs for sequences.
        """
        self.model = model_wrapper
        self.cache = cache if cache is not None else {}

    def compute_log_prob(self, sequence: str, prompt: str) -> float:
        """
        Compute the log probability of a sequence conditioned on a prompt.
        Checks cache before computation.
        Args:
            sequence (str): Trajectory continuation sequence.
            prompt (str): The prompt string.
        Returns:
            float: Log probability of the sequence given the prompt.
        """
        cache_key = self._create_cache_key(sequence, prompt)
        if cache_key in self.cache:
            return self.cache[cache_key]
        log_prob = self._calculate_sequence_log_prob(sequence, prompt)
        self.cache[cache_key] = log_prob
        return log_prob

    def _calculate_sequence_log_prob(self, sequence: str, prompt: str) -> float:
        """
        Calculate the sequence log likelihood conditioned on prompt using model's token probs.
        Args:
            sequence (str): Trajectory continuation.
            prompt (str): The prompt string.
        Returns:
            float: Log likelihood.
        """
        # Tokenize prompt and sequence
        prompt_ids = self.model.tokenize(prompt)
        seq_ids = self.model.tokenize(sequence)
        # Full token sequence: prompt + sequence
        full_ids = prompt_ids + seq_ids

        # Convert to tensor
        input_ids = torch.tensor([full_ids], device=self.model.device)

        with torch.no_grad():
            outputs = self.model.model(input_ids=input_ids)
            logits = outputs.logits.squeeze(0)  # shape: (seq_len, vocab_size)

        # Calculate token-wise probabilities for sequence (excluding prompt)
        log_probs = []
        total_prompt_len = len(prompt_ids)

        for i in range(total_prompt_len, len(full_ids)):
            # Obtain logits for position i-1
            logits_i = logits[i - 1]
            probs = torch.softmax(logits_i, dim=-1)
            token_id = full_ids[i]
            prob = probs[token_id].item()
            # Avoid log(0)
            if prob <= 1e-12:
                prob = 1e-12
            log_probs.append(torch.log(prob))

        total_log_prob = sum(log_probs)
        return total_log_prob.item()

    def _create_cache_key(self, sequence: str, prompt: str):
        """
        Create a cache key based on tokenized input for cache lookup.
        """
        # Use token IDs of prompt + sequence as key for uniqueness
        prompt_ids = self.model.tokenize(prompt)
        sequence_ids = self.model.tokenize(sequence)
        # To avoid excessively large keys, tuple of token IDs
        return tuple(prompt_ids + sequence_ids)
