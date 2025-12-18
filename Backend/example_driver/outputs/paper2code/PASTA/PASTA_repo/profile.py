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
