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

