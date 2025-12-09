## evaluation.py
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer

from utils import compute_accuracy, compute_rouge, compute_retrieval_precision

class Evaluation:
    """
    The Evaluation class runs inference on a dataset using the provided model, incorporating
    the custom long sequence inference logic with external memory. It computes metrics such as
    accuracy, ROUGE, retrieval precision, and logs resource usage and cache statistics.
    """

    def __init__(self, model, dataset_metrics: list, device: str = 'cuda'):
        """
        Initialize the evaluator.

        Args:
            model: Instance of ModelWrapper.
            dataset_metrics (list): List of metrics to compute, e.g., ['accuracy', 'ROUGE', 'R.PK'].
            device (str): Computation device.
        """
        self.model = model
        self.metrics = dataset_metrics
        self.device = device

        # Initialize metric accumulators
        self.accuracies: List[float] = []
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.rouge_scores: List[dict] = []
        self.retrieval_precisions: List[float] = []

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

        # To store detailed per-sequence results if needed
        self.sequence_results = []

    def evaluate_sequence(self, prompt: str, long_input_tokens: list):
        """
        Run streaming inference on a sequence, updating memory and collecting predictions.

        Args:
            prompt (str): The prompt text.
            long_input_tokens (list): Token list for the long sequence.

        Returns:
            dict: Metrics result for the sequence.
        """
        # Initialize memory cache state
        memory = self.model.memory  # shared with model; manages cache
        # Reset cache statistics for this sequence
        self.cache_hits = 0
        self.cache_misses = 0

        total_tokens = len(long_input_tokens)
        chunk_size = self.model.config.get('inference', {}).get('chunk_size', 4096)
        offset = 0
        generated_tokens = []

        start_time = time.perf_counter()

        while offset < total_tokens:
            end_idx = min(offset + chunk_size, total_tokens)
            chunk_tokens = long_input_tokens[offset:end_idx]
            # Run inference chunk with custom attention + memory
            output_text = self.model.generate(
                prompt=prompt,
                long_input=chunk_tokens,
                max_new_tokens=chunk_size,
                do_streaming=True,
                memory=memory
            )

            # Here, one would extract generated tokens
            # For simplicity, assume output_text decoded to tokens (mocked)
            # But in actual code, you'd get token IDs or decode the output
            # For demonstration, we skip token extraction.

            # After each chunk, update external memory with important blocks
            # To do that, examine evicted tokens/blocks, compute importance scores, and update
            # For brevity, assume a function 'update_memory_for_chunk' is called here.
            # It would:
            # - Compute importance scores for evicted tokens
            # - Form blocks
            # - Insert blocks into memory manager
            # Since not implemented, we leave as placeholder.

            offset = end_idx

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Evaluate metrics for this sequence
        metrics_result = {}
        # For placeholder, assign mock metrics
        # For actual evaluation, extract predictions, compare with ground truth labels
        # e.g., accuracy = compute_accuracy(predicted, labels)
        # e.g., rouge scores = compute_rouge(predicted_text, reference_text)
        # Since labels are not available, skip actual calculation.
        for met in self.metrics:
            metrics_result[met] = None  # or compute if data available

        # Log resource usage
        cache_stats = {
            'hits': self.cache_hits,
            'misses': self.cache_misses
        }

        return {
            'metrics': metrics_result,
            'time': elapsed_time,
            'cache_stats': cache_stats
        }

    def evaluate_all(self, dataset):
        """
        Run evaluation over entire dataset.

        Args:
            dataset: Dataset loader with stream_chunks() method.

        Returns:
            dict: Summary of evaluation over all sequences.
        """
        sequence_count = 0
        total_metrics = {met: [] for met in self.metrics}
        total_times = []

        for sequence in dataset.stream_chunks():
            prompt = sequence['prompt']
            long_input_tokens = sequence['long_input']
            result = self.evaluate_sequence(prompt, long_input_tokens)
            sequence_count += 1
            total_times.append(result['time'])

            # Collect metrics
            for met in self.metrics:
                val = result['metrics'].get(met, None)
                if val is not None:
                    total_metrics[met].append(val)

            # Log per-sequence info
            print(f"Sequence {sequence['metadata']['sequence_id']} processed in {result['time']:.2f}s.")
            print(f"Cache hits/misses: {result['cache_stats']['hits']}/{result['cache_stats']['misses']}.")

        # Compute averages over sequences for each metric
        report = {}
        for met in self.metrics:
            vals = [v for v in total_metrics[met] if v is not None]
            if len(vals) > 0:
                if isinstance(vals[0], (int, float)):
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    report[met] = {'mean': mean_val, 'std': std_val}
                else:
                    # For non-numeric, aggregate as needed
                    report[met] = vals
            else:
                report[met] = None
        avg_time = np.mean(total_times) if total_times else None
        report['average_time_sec'] = avg_time

        # Cache statistics: sum over sequences
        total_hits = getattr(self, 'cache_hits', 0)
        total_misses = getattr(self, 'cache_misses', 0)
        report['cache_hits'] = total_hits
        report['cache_misses'] = total_misses
        if total_hits + total_misses > 0:
            report['cache_hit_rate'] = total_hits / (total_hits + total_misses)
        else:
            report['cache_hit_rate'] = None

        return report
