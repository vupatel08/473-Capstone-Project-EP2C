# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## dataset_loader.py

```python
## dataset_loader.py

import os
import json
import csv
import random
from itertools import combinations
from typing import List, Tuple, Dict, Optional

import torch
from transformers import AutoTokenizer


class DatasetLoader:
    """
    DatasetLoader loads datasets for prompt-response pairwise preference training.
    
    It supports datasets in JSONL or CSV format, constructs pairwise comparisons
    for each prompt, and provides APIs to access these pairs efficiently.
    """

    def __init__(self,
                 data_path: str,
                 tokenizer_name: str = "gpt2-medium",
                 max_response_tokens: int = 512,
                 dataset_format: str = "jsonl",
                 random_seed: int = 42):
        """
        Initialize DatasetLoader by loading and processing dataset from given path.
        
        Args:
            data_path (str): Path to dataset file (JSONL or CSV).
            tokenizer_name (str): Tokenizer to use for responses (default GPT-2 medium).
            max_response_tokens (int): Max tokens to consider for responses.
            dataset_format (str): Format of dataset ('jsonl' or 'csv').
            random_seed (int): Seed for reproducibility.
        """
        self.data_path = data_path
        self.response_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_response_tokens = max_response_tokens
        self.dataset_format = dataset_format.lower()
        self.random = random.Random(random_seed)

        # Internal storage: List of dicts, each with prompt, responses, pairs
        self.dataset: List[Dict] = []
        self._load_dataset()

    def _load_dataset(self):
        """
        Loads dataset from the file and constructs response pairs.
        Supports JSONL and CSV formats.
        """
        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"Dataset file not found at {self.data_path}")

        if self.dataset_format == "jsonl":
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data_item = json.loads(line.strip())
                    self._process_data_item(data_item)
        elif self.dataset_format == "csv":
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data_item = {
                        'prompt': row['prompt'],
                        'responses': [row['response']]  # assuming single response per row
                        # Extend here if CSV contains multiple responses
                    }
                    self._process_data_item(data_item)
        else:
            raise ValueError("Unsupported dataset_format: choose 'jsonl' or 'csv'")

    def _process_data_item(self, data_item: Dict):
        """
        Process each data item to generate pairwise comparisons.
        Args:
            data_item (Dict): Dictionary with 'prompt' and 'responses' (list of str).
        """
        prompt = data_item['prompt']
        responses = data_item['responses']
        # Tokenize responses
        tokenized_responses = [
            self.response_tokenizer.encode(r, add_special_tokens=False, truncation=True, max_length=self.max_response_tokens)
            for r in responses
        ]
        # Generate all pairwise combinations (i != j)
        pairs = []
        for (i, j) in combinations(range(len(responses)), 2):
            # Here, we need a preference label: in absence of manual labels,
            # generate preference based on external heuristic, e.g., longer response,
            # GPT-4 score, or random. For demo, we use random preference.
            # In real experiments, replace with human or GPT-based preference.
            preference = self._determine_preference(i, j, prompt)
            if preference == i:
                response_w_idx, response_l_idx = i, j
            else:
                response_w_idx, response_l_idx = j, i

            pairs.append({
                'prompt': prompt,
                'responses': responses,
                'response_tokenized': tokenized_responses,
                'pair_indices': (response_w_idx, response_l_idx),
                'preference': preference
            })

        self.dataset.append({
            'prompt': prompt,
            'responses': responses,
            'response_tokenized': tokenized_responses,
            'pairs': pairs
        })

    def _determine_preference(self, idx1: int, idx2: int, prompt: str) -> int:
        """
        Placeholder: determine preference for responses at idx1 and idx2 for prompt.
        This could be based on human annotations or GPT-4 evaluation.
        For now, we randomly assign preference.
        Args:
            idx1 (int): index of response 1
            idx2 (int): index of response 2
            prompt (str): prompt string
        Returns:
            int: preferred response index (idx1 or idx2)
        """
        # Replace the below with actual preference logic if available.
        # For demonstration, randomly choose.
        return self.random.choice([idx1, idx2])

    def get_response_pair(self, prompt: str, num_pairs: int = 1) -> List[Tuple[str, str, int]]:
        """
        For a given prompt, returns a list of response pairs with preference labels.
        Args:
            prompt (str): the prompt string
            num_pairs (int): number of pairs to return
        Returns:
            List of tuples: (response_w, response_l, label), label=1 if response_w preferred
        """
        # Search in loaded dataset
        for item in self.dataset:
            if item['prompt'] == prompt:
                pairs = item['pairs']
                if len(pairs) == 0:
                    return []
                selected_pairs = self.random.sample(pairs, min(num_pairs, len(pairs)))
                result = []
                for pair in selected_pairs:
                    i_idx, j_idx = pair['pair_indices']
                    responses = pair['responses']
                    # response_w and response_l
                    response_w = responses[i_idx]
                    response_l = responses[j_idx]
                    label = 1  # since we set response_w as preferred
                    result.append((response_w, response_l, label))
                return result
        # If prompt not found, return empty list
        return []

    def get_tokenized_response(self, response: str) -> torch.Tensor:
        """
        Tokenize a given response string.
        Args:
            response (str): response text
        Returns:
            torch.Tensor: token ids tensor
        """
        tokens = self.response_tokenizer.encode(
            response,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_response_tokens
        )
        return torch.tensor(tokens, dtype=torch.long)

    def get_prompt(self, prompt: str) -> str:
        """
        Get the original prompt (could include processing if needed).
        Args:
            prompt (str): prompt text
        Returns:
            str: prompt
        """
        return prompt

    def __len__(self):
        """
        Returns number of prompts in dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Return dataset item by index.
        """
        return self.dataset[index]
```

## evaluation.py

```python
## evaluation.py
import os
import json
import time
from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import yaml

# For GPT-4 API calls (assuming openai SDK)
import openai

# Assume dataset_loader.py, model.py, losses.py, utils.py are in the same directory or accessible.
from dataset_loader import DatasetLoader
from model import ResponseGenerator
from utils import (
    generate_response,
    sequence_kl_divergence,
    preference_probability,
    stop_gradient,
    plot_divergence_curves,
    plot_frontier,
)
from losses import TDPOLoss

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set GPT-4 API key if used
USE_GPT4 = config['evaluation'].get('use_gpt4', True)
GPT4_API_KEY = config['evaluation'].get('gpt4_api_key', 'YOUR_API_KEY')
openai.api_key = GPT4_API_KEY

# Helper function to call GPT-4 for preference
def gpt4_preference(prompt: str, resp1: str, resp2: str, max_tokens: int = 512, n_trials: int = 1) -> int:
    """
    Query GPT-4 to compare two responses. Returns:
      1 if resp1 preferred,
      2 if resp2 preferred,
      0 for tie/unknown.
    """
    system_prompt = (
        "You are an AI assistant that compares two responses to a prompt and decides which one is better "
        "based on helpfulness, relevance, and safety. Reply with '1' if the first response is better, "
        " '2' if the second is better, or '0' for tie/uncertain."
    )
    results = []
    for _ in range(n_trials):
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Prompt:\n{prompt}\n\nResponse 1:\n{resp1}\n\nResponse 2:\n{resp2}\n\nPlease choose the better response or say 'tie'."}
                ],
                max_tokens=max_tokens,
                temperature=0,
                n=1
            )
            reply = response['choices'][0]['message']['content'].strip()
            if reply.startswith('1'):
                results.append(1)
            elif reply.startswith('2'):
                results.append(2)
            elif 'tie' in reply.lower():
                results.append(0)
            else:
                # fallback
                results.append(0)
        except Exception as e:
            print(f"GPT-4 API error: {e}")
            time.sleep(1)  # brief sleep on error
            results.append(0)
    # Aggregate over trials
    # Majority vote
    if results.count(1) > len(results)/2:
        return 1
    elif results.count(2) > len(results)/2:
        return 2
    else:
        return 0

# Class implementing the evaluation procedure
class Evaluation:
    def __init__(self, model: ResponseGenerator, dataset: DatasetLoader, preference_model, config: dict):
        self.model = model
        self.dataset = dataset
        self.preference_model = preference_model
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.model.to(self.device)

        self.use_gpt4 = self.config['evaluation'].get('use_gpt4', True)
        self.evaluation_interval = self.config['evaluation'].get('evaluation_interval', 50)
        self.save_interval = self.config['evaluation'].get('save_checkpoint_interval', 100)
        self.max_response_tokens = self.config['training'].get('max_response_tokens', 512)

        # Containers for divergence tracking
        self.preferred_divergences = []
        self.dispreferred_divergences = []

        # List for reward frontier (reward vs KL)
        self.reward_frontier = []

    def generate_responses(self, prompts: List[str], num_responses: int = 1) -> List[List[str]]:
        """
        For each prompt, generate num_responses responses via model.
        Returns list of lists: responses per prompt.
        """
        all_responses = []
        for prompt in prompts:
            responses = []
            for _ in range(num_responses):
                resp = generate_response(self.model, prompt,
                                         max_tokens=self.max_response_tokens,
                                         temperature=0.7)
                responses.append(resp)
            all_responses.append(responses)
        return all_responses

    def compute_divergence(self, prompt: str, response: str, ref_model: ResponseGenerator) -> float:
        """
        Compute sequence KL divergence between model and ref for a prompt-response.
        """
        # Tokenize response
        response_ids = self.model.tokenizer.encode(response, add_special_tokens=False)
        response_tensor = torch.tensor(response_ids, dtype=torch.long, device=self.device)

        # Get token distributions conditioned on prompt + previous tokens
        model_probs, ref_probs = self._get_token_probs(prompt, response_tensor, ref_model)
        # Compute sequence KL
        seq_kl = sequence_kl_divergence(model_probs, ref_probs)
        return float(seq_kl)

    def _get_token_probs(self, prompt: str, response_tensor: torch.Tensor, ref_model: ResponseGenerator):
        """
        Retrieve per-token model and reference probs conditioned on prompt+prefix
        """
        pi_model, pi_ref = self._get_response_probabilities(prompt, response_tensor, ref_model)
        # Convert to tensors for divergence
        return pi_model, pi_ref

    def _get_response_probabilities(self, prompt: str, response_tensor: torch.Tensor, ref_model: ResponseGenerator):
        """
        Compute per-token distribution from model and ref model
        """
        # Response tokenize and get per-token distributions
        model_probs, ref_probs = self._compute_token_distributions(prompt, response_tensor, ref_model)
        return model_probs, ref_probs

    def _compute_token_distributions(self, prompt: str, response_tensor: torch.Tensor, ref_model: ResponseGenerator):
        """
        For each token position, get distribution conditioned on prompt + previous tokens
        """
        T = response_tensor.shape[0]
        device = next(self.model.model.parameters()).device
        model_probs_list = []
        ref_probs_list = []

        for t in range(T):
            prefix_ids = self.model.tokenizer.encode(prompt, add_special_tokens=False) + response_tensor[:t].tolist()
            context_ids = prefix_ids
            input_ids = torch.tensor([context_ids], device=device)
            with torch.no_grad():
                outputs = self.model.model(**{"input_ids": input_ids})
                logits = outputs.logits
            logits = logits[0, -1, :]  # last token
            probs = torch.softmax(logits, dim=-1)
            model_probs_list.append(probs)

            # Similarly for ref model
            with torch.no_grad():
                ref_outputs = ref_model.model(**{"input_ids": input_ids})
                ref_logits = ref_outputs.logits
            ref_probs = torch.softmax(ref_logits[0, -1, :], dim=-1)
            ref_probs_list.append(ref_probs)

        model_probs_seq = torch.stack(model_probs_list, dim=0)  # [T, vocab_size]
        ref_probs_seq = torch.stack(ref_probs_list, dim=0)
        return model_probs_seq, ref_probs_seq

    def evaluate(self, prompts: List[str], ref_model: ResponseGenerator):
        """
        Main evaluation loop: generate responses, compute metrics, plot divergence.
        """
        print("Starting evaluation...")
        divergence_pref = []
        divergence_dis = []

        # For each prompt, generate responses and compute divergences
        for prompt in prompts:
            responses = []
            # Generate multiple responses per prompt (e.g., 3 each)
            generated_resps = []
            for _ in range(3):
                resp = generate_response(self.model, prompt, max_tokens=self.max_response_tokens, temperature=0.7)
                generated_resps.append(resp)

            # For pairs, compute divergence and preferences
            # Pairwise combinations
            pairs = []
            for i in range(len(generated_resps)):
                for j in range(i + 1, len(generated_resps)):
                    pairs.append((generated_resps[i], generated_resps[j]))

            for (resp1, resp2) in pairs:
                # Get tokenized
                ids1 = torch.tensor(self.model.tokenizer.encode(resp1, add_special_tokens=False), dtype=torch.long)
                ids2 = torch.tensor(self.model.tokenizer.encode(resp2, add_special_tokens=False), dtype=torch.long)

                # Compute divergences
                div1 = self.compute_divergence(prompt, resp1, ref_model)
                div2 = self.compute_divergence(prompt, resp2, ref_model)

                # For preferred response, get preference label
                if self.use_gpt4:
                    pref_label = gpt4_preference(prompt, resp1, resp2)
                else:
                    # Here, could use human labels or other heuristics
                    pref_label = 1  # or 2
                # Store divergences
                if pref_label == 1:
                    divergence_pref.append(div1)
                    divergence_dis.append(div2)
                elif pref_label == 2:
                    divergence_pref.append(div2)
                    divergence_dis.append(div1)
                # Tie or undecided ignored in divergence measure

        # Plot divergence curves with average divergences
        if divergence_pref and divergence_dis:
            plot_divergence_curves(divergence_pref, divergence_dis,
                title="Sequential KL Divergence: Preferred vs Dispreferred responses")
        else:
            print("No divergence data to plot.")

    def compute_win_rates(self, prompts: List[str], baseline_responses: Dict[str, List[str]]):
        """
        For each prompt, generate model responses and compare vs baseline responses
        via GPT-4 API to compute win/tie/lose rates.
        baseline_responses: dict of prompt -> list of baseline responses for comparison
        """
        wins, ties, losses = 0, 0, 0
        total = 0
        for prompt in tqdm(prompts):
            # Generate responses from trained model
            model_resp = generate_response(self.model, prompt, max_tokens=self.max_response_tokens, temperature=0.7)
            baseline_list = baseline_responses.get(prompt, [])
            for baseline_resp in baseline_list:
                # Compare via GPT-4
                pref = gpt4_preference(prompt, model_resp, baseline_resp)
                if pref == 1:
                    wins += 1
                elif pref == 2:
                    losses += 1
                else:
                    ties += 1
                total += 1
        print(f"Win rate: {wins / total * 100:.2f}%, Tie: {ties / total * 100:.2f}%, Loss: {losses / total * 100:.2f}%")
        return {'win_rate': wins / total, 'tie_rate': ties / total, 'lose_rate': losses / total}

    def save_checkpoint(self, model, step: int):
        """
        Save model checkpoint periodically
        """
        save_dir = f'checkpoint_step_{step}'
        os.makedirs(save_dir, exist_ok=True)
        model.model.save_pretrained(save_dir)
        model.tokenizer.save_pretrained(save_dir)
        print(f"Saved checkpoint at step {step} to {save_dir}")

    def run(self, prompts: List[str], ref_model: ResponseGenerator):
        """
        Main evaluation process: generate responses, compute divergences,
        plot trajectories, and save checkpoints periodically.
        """
        total_steps = self.config['training'].get('train_steps', 200)
        for step in range(1, total_steps + 1):
            if step % self.evaluation_interval == 0:
                print(f"\nEvaluation at step {step}")
                self.evaluate(prompts, ref_model)
                # Save checkpoint?
                if step % self.save_interval == 0:
                    self.save_checkpoint(self.model, step)
        # Final evaluation
        print("Final evaluation on full test set.")
        self.evaluate(prompts, ref_model)

# Usage example:
# Assuming called elsewhere:
# dataset = DatasetLoader(path, ...)
# model = ResponseGenerator(...)
# ref_model = ResponseGenerator(...)
# eval_obj = Evaluation(model, dataset, preference_model=None, config=config)
# prompts_list = [prompt for prompt in dataset or custom prompts]
# eval_obj.run(prompts_list, ref_model)

```

## losses.py

```python
## losses.py

import torch
import torch.nn.functional as F
from typing import Tuple, List
from torch.nn.modules.utils import _pair


class TDPOLoss:
    """
    Implements token-level TDPO losses with support for TDPo_1 and TDPo_2 variants.
    
    Args:
        beta (float): KL divergence coefficient from configuration.
        alpha (float): Divergence scaling parameter for TDPo_2.
        divergence_scale (float): Scale factor for divergence, if needed.
        method (str): 'tdpo_1' or 'tdpo_2' indicating the variant.
        stop_gradient (bool): Whether to stop gradients for divergence term (used in TDPo_2).
    """
    def __init__(
        self,
        beta: float = 0.1,
        alpha: float = 0.5,
        divergence_scale: float = 1.0,
        method: str = "tdpo_1",
        stop_gradient: bool = True,
    ):
        self.beta = beta
        self.alpha = alpha
        self.divergence_scale = divergence_scale
        self.method = method
        self.stop_gradient = stop_gradient

    def compute_token_log_probs(
        self,
        response_tokens: torch.Tensor,
        prompt: str,
        model_probs: torch.Tensor,
        ref_probs: torch.Tensor,
        tokenizer,
        response_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token log probabilities for response and reference.
        
        Args:
            response_tokens (torch.Tensor): shape [T], token IDs from current model.
            prompt (str): prompt string.
            model_probs (Tensor): shape [T, vocab_size], probability distribution from current model.
            ref_probs (Tensor): shape [T, vocab_size], probability distribution from reference model.
            tokenizer: tokenizer instance.
            response_text (str): raw response string (for debugging/optional).
        
        Returns:
            Tuple of (log_prob_model: [T], log_prob_ref: [T]) tensors.
        """
        # Log probabilities for each token:
        # Assuming model_probs and ref_probs are shape [T, vocab_size]
        log_prob_model = torch.log(model_probs.gather(1, response_tokens.unsqueeze(-1)).squeeze(-1) + 1e-12)
        log_prob_ref = torch.log(ref_probs.gather(1, response_tokens.unsqueeze(-1)).squeeze(-1) + 1e-12)
        return log_prob_model, log_prob_ref

    def compute_kl_divergence(
        self,
        ref_probs: torch.Tensor,
        model_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute token-wise KL divergence D_KL(ref || model):
        
        Args:
            ref_probs (Tensor): [T, vocab_size], reference model probabilities.
            model_probs (Tensor): [T, vocab_size], current model probabilities.
        
        Returns:
            Tensor: [T], KL divergence per token.
        """
        # Avoid log(0); add epsilon
        epsilon = 1e-12
        ref_probs_clamped = ref_probs + epsilon
        model_probs_clamped = model_probs + epsilon
        kl = torch.sum(ref_probs_clamped * (torch.log(ref_probs_clamped) - torch.log(model_probs_clamped)), dim=-1)
        return kl  # shape [T]

    def get_token_probs(
        self,
        prompt: str,
        response_tokens: torch.Tensor,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each token, compute the model and reference probability distributions conditioned on [x, y^{<t}].
        """
        T = response_tokens.shape[0]

        # Prepare context: prompt + previous tokens at each step
        # For efficiency, process all tokens in batch assuming incremental decoding
        # Here, we process token by token
        response_probs_model = []
        response_probs_ref = []

        # For batching, process token by token:
        for t in range(T):
            # Construct context: prompt + tokens[:t]
            context_ids = tokenizer.encode(prompt, add_special_tokens=False) + response_tokens[:t].tolist()
            context_tensor = torch.tensor([context_ids], dtype=torch.long, device=next(model.parameters()).device)

            # Get distribution from current model
            with torch.no_grad():
                outputs = model(**{"input_ids": context_tensor})
                logits = outputs.logits
            logits = logits[0, -1, :]  # last token logits
            probs = F.softmax(logits, dim=-1)
            response_probs_model.append(probs)

            # From reference model as well
            with torch.no_grad():
                ref_outputs = ref_model(**{"input_ids": context_tensor})
                ref_logits = ref_outputs.logits
            ref_logits = ref_logits[0, -1, :]
            ref_probs = F.softmax(ref_logits, dim=-1)
            response_probs_ref.append(ref_probs)

        # Stack into tensors [T, vocab_size]
        model_probs_tensor = torch.stack(response_probs_model, dim=0)  # [T, vocab_size]
        ref_probs_tensor = torch.stack(response_probs_ref, dim=0)
        return model_probs_tensor, ref_probs_tensor

    def compute_response_log_probs(
        self,
        response_tokens: torch.Tensor,
        model_probs: torch.Tensor,
        ref_probs: torch.Tensor,
        tokenizer,
        response_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token log probabilities for responses given response probabilities.
        """
        log_prob_model = torch.log(model_probs.gather(1, response_tokens.unsqueeze(-1)).squeeze(-1) + 1e-12)
        log_prob_ref = torch.log(ref_probs.gather(1, response_tokens.unsqueeze(-1)).squeeze(-1) + 1e-12)
        return log_prob_model, log_prob_ref

    def compute_token_level_advantage(
        self,
        prompt: str,
        y_w_tokens: torch.Tensor,
        y_l_tokens: torch.Tensor,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer,
        response_y_w: str,
        response_y_l: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute token-wise advantage u(x, y_w, y_l) for the batch.
        """
        # Get model and ref probabilities for y_w
        pi_model_w, pi_ref_w = self.get_token_probs(prompt, y_w_tokens, model, ref_model, tokenizer)
        # Get model and ref probabilities for y_l
        pi_model_l, pi_ref_l = self.get_token_probs(prompt, y_l_tokens, model, ref_model, tokenizer)
        
        # Compute per-token log probs
        log_pi_model_w, log_pi_ref_w = self.compute_response_log_probs(y_w_tokens, pi_model_w, pi_ref_w, tokenizer, response_y_w)
        log_pi_model_l, log_pi_ref_l = self.compute_response_log_probs(y_l_tokens, pi_model_l, pi_ref_l, tokenizer, response_y_l)

        # Compute u(x, y_w, y_l) for each token (Eq. 12 / 15)
        # u = beta * (log_pi_model_w - log_pi_ref_w) - beta * (log_pi_model_l - log_pi_ref_l)
        u = self.beta * (log_pi_model_w - log_pi_ref_w) - self.beta * (log_pi_model_l - log_pi_ref_l)  # shape [T]
        return u, log_pi_model_w, log_pi_ref_w

    def compute_delta(
        self,
        ref_probs_w: torch.Tensor,
        ref_probs_l: torch.Tensor,
        model_probs_w: torch.Tensor,
        model_probs_l: torch.Tensor,
        prompt: str,
        y_w_tokens: torch.Tensor,
        y_l_tokens: torch.Tensor,
        tokenizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the divergence penalty δ (eq. 14 or 18) at token level, summed over sequence.
        """
        # Compute per-token KL divergence for preferred and dispreferred
        kl_w = self.compute_kl_divergence(ref_probs_w, model_probs_w)  # shape [T]
        kl_l = self.compute_kl_divergence(ref_probs_l, model_probs_l)  # shape [T]

        # Sum over tokens to get total divergence
        delta = self.beta * (torch.sum(kl_l) - torch.sum(kl_w))  # scalar

        return delta

    def compute_loss(
        self,
        y_w_tokens: torch.Tensor,
        y_l_tokens: torch.Tensor,
        prompt: str,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer
    ) -> torch.Tensor:
        """
        Compute the overall loss for a batch of response pairs.
        """
        # Compute model and ref probabilities for each response
        pi_model_w, pi_ref_w = self.get_token_probs(prompt, y_w_tokens, model, ref_model, tokenizer)
        pi_model_l, pi_ref_l = self.get_token_probs(prompt, y_l_tokens, model, ref_model, tokenizer)

        # Compute token-wise advantages u
        # For each pair, for safety, recompute for the current response text
        response_text_w = tokenizer.decode(y_w_tokens, clean_up_tokenization_spaces=True)
        response_text_l = tokenizer.decode(y_l_tokens, clean_up_tokenization_spaces=True)

        u, log_pi_model_w, log_pi_ref_w = self.compute_response_log_probs(y_w_tokens, pi_model_w, pi_ref_w, tokenizer, response_text_w)
        _, log_pi_model_l, log_pi_ref_l = self.compute_response_log_probs(y_l_tokens, pi_model_l, pi_ref_l, tokenizer, response_text_l)

        # Compute the divergence δ (eq.14 or 18)
        delta = self.compute_delta(pi_ref_w, pi_ref_l, pi_model_w, pi_model_l, prompt, y_w_tokens, y_l_tokens, tokenizer)

        if self.method == "tdpo_1":
            # Eq. 15: loss = -log σ(u - δ)
            argument = u - delta  # shape [T]
            loss_terms = -F.logsigmoid(argument)
            loss_value = torch.mean(loss_terms)
        elif self.method == "tdpo_2":
            # Eq. 18: δ_2 with stop gradient
            delta_2_value = self.beta * (torch.sum(kl_l) - torch.sum(kl_w))  # scalar
            if self.stop_gradient:
                delta_2_value = torch.detach(delta_2_value)
            # u - α * δ_2
            argument = u - self.alpha * delta_2_value
            loss_terms = -F.logsigmoid(argument)
            loss_value = torch.mean(loss_terms)
        else:
            raise ValueError("Method must be 'tdpo_1' or 'tdpo_2'")
        return loss_value

    def compute_batch_loss(
        self,
        batch_response_pairs: List[Tuple[str, str, int]],
        prompt: str,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer
    ) -> torch.Tensor:
        """
        Compute total loss over a batch of response pairs.
        """
        total_loss = 0.0
        batch_size = len(batch_response_pairs)
        for (resp_w, resp_l, label) in batch_response_pairs:
            # Tokenize responses
            y_w_tokens = torch.tensor(tokenizer.encode(resp_w, add_special_tokens=False, truncation=True), dtype=torch.long)
            y_l_tokens = torch.tensor(tokenizer.encode(resp_l, add_special_tokens=False, truncation=True), dtype=torch.long)
            if len(y_w_tokens) == 0 or len(y_l_tokens) == 0:
                continue  # skip empty responses
            loss = self.compute_loss(y_w_tokens, y_l_tokens, prompt, model, ref_model, tokenizer)
            total_loss += loss
        return total_loss / max(batch_size, 1)

```

## main.py

```python
## main.py
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

from dataset_loader import DatasetLoader
from model import ResponseGenerator
from losses import TDPOLoss
from utils import (
    generate_response,
    sequence_kl_divergence,
)
from trainer import Trainer
from evaluation import Evaluation

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Extract training config
train_cfg = config.get('training', {})
model_cfg = config.get('model', {})
dataset_cfg = config.get('dataset', {})
eval_cfg = config.get('evaluation', {})
hyper_cfg = config.get('hyperparameters', {})

# Set defaults if keys missing
learning_rate = train_cfg.get('learning_rate', 5e-6)
batch_size = train_cfg.get('batch_size', 64)
train_steps = train_cfg.get('train_steps', 200)
max_response_tokens = train_cfg.get('max_response_tokens', 512)
warmup_steps = train_cfg.get('warmup_steps', 50)
gradient_clipping = train_cfg.get('gradient_clipping', 1.0)
eval_interval = eval_cfg.get('evaluation_interval', 50)
save_interval = eval_cfg.get('save_checkpoint_interval', 100)

# Model and data paths
pretrained_model_name = model_cfg.get('pretrained_model_name', 'gpt2-medium')
checkpoint_path = model_cfg.get('checkpoint_path', None)
train_data_path = dataset_cfg.get('train_data_path', 'path/to/train/dataset')
validation_data_path = dataset_cfg.get('validation_data_path', 'path/to/validation/dataset')
test_data_path = dataset_cfg.get('test_data_path', 'path/to/test/dataset')

# Hyperparameters for divergence control
beta = hyper_cfg.get('beta', 0.1)
alpha = hyper_cfg.get('alpha', 0.5)
divergence_scale = hyper_cfg.get('divergence_offset', 1.0)
stop_gradient = hyper_cfg.get('stop_gradient', True)

# Initialize DatasetLoader
dataset = DatasetLoader(
    data_path=train_data_path,
    max_response_tokens=max_response_tokens,
    dataset_format='jsonl'  # or 'csv', depending on dataset file
)

# Initialize model
model = ResponseGenerator(pretrained_model_name, checkpoint_path)

# Reference model - for simplicity, assume same as base
ref_model = ResponseGenerator(pretrained_model_name)

# Initialize preference scorer (here a placeholder; use GPT-4 API in practice)
# For actual implementation, replace with API calls or precalculated scores
preference_model = None  # Placeholder, or implement as class

# Initialize loss
loss_fn = TDPOLoss(
    beta=beta,
    alpha=alpha,
    divergence_scale=divergence_scale,
    method='tdpo_2',  # or 'tdpo_1'
    stop_gradient=stop_gradient
)

# Initialize optimizer
optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate)

# Prepare trainer
trainer = Trainer(
    model=model,
    dataset=dataset,
    preference_model=preference_model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    config={
        'training': train_cfg,
        'hyperparameters': hyper_cfg,
        'evaluation': eval_cfg
    }
)

# Optional: Load from checkpoint if provided
if checkpoint_path:
    model.model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded checkpoint from {checkpoint_path}")

# Training loop
print("Start training...")
for step in tqdm(range(1, train_steps + 1)):
    # Sample mini-batch of prompts
    batch_pairs = []
    for _ in range(batch_size):
        # Random sample prompt from dataset
        data_item = np.random.choice(dataset.dataset)
        prompt = data_item['prompt']
        pair = dataset.get_response_pair(prompt, num_pairs=1)
        if len(pair) == 0:
            continue
        y_w, y_l, _ = pair[0]
        batch_pairs.append((prompt, y_w, y_l))
    if len(batch_pairs) == 0:
        continue

    # Perform training step
    trainer.optimizer.zero_grad()
    loss = trainer.loss_fn
    total_loss = 0.0
    total_rewards = []

    # Sum over batch to backprop
    batch_losses = []
    for (prompt, y_w, y_l) in batch_pairs:
        # Encode responses
        y_w_ids = torch.tensor(model.tokenizer.encode(y_w, add_special_tokens=False), dtype=torch.long, device=device)
        y_l_ids = torch.tensor(model.tokenizer.encode(y_l, add_special_tokens=False), dtype=torch.long, device=device)
        # Generate responses (if needed), here responses are assumed given
        # Compute response probabilities
        pi_w, pi_ref_w = utils.get_response_probabilities(model, prompt, y_w_ids)
        pi_l, pi_ref_l = utils.get_response_probabilities(model, prompt, y_l_ids)
        # Calculate u(x, y_w, y_l)
        u, _, _ = loss.compute_response_log_probs(y_w_ids, pi_w, pi_ref_w, model.tokenizer, y_w)
        _, _, _ = loss.compute_response_log_probs(y_l_ids, pi_l, pi_ref_l, model.tokenizer, y_l)
        delta = loss.compute_delta(pi_ref_w, pi_ref_l, pi_w, pi_l, prompt, y_w_ids, y_l_ids, model.tokenizer)

        if loss.method == 'tdpo_1':
            argument = u - delta
            pair_loss = -torch.mean(torch.log(torch.sigmoid(argument) + 1e-12))
        elif loss.method == 'tdpo_2':
            delta_value = delta
            if loss.stop_gradient:
                delta_value = torch.detach(delta_value)
            argument = u - alpha * delta_value
            pair_loss = -torch.mean(torch.log(torch.sigmoid(argument) + 1e-12))
        else:
            raise ValueError("Invalid method in loss function")
        batch_losses.append(pair_loss)

        # Simulate reward (use GPT-based or classifier; here set to placeholder)
        reward_w = 1.0  # Placeholder
        reward_l = 0.5  # Placeholder
        total_rewards.append((reward_w + reward_l) / 2)

    # Average loss for batch
    batch_loss = torch.stack(batch_losses).mean()
    batch_loss.backward()

    # Gradient clipping
    if gradient_clipping:
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), gradient_clipping)

    # Step optimizer
    trainer.optimizer.step()

    # Logging periodically
    if step % 10 == 0:
        print(f"Step {step}: Loss {batch_loss.item():.4f}")

    # Save checkpoint
    if step % save_interval == 0:
        checkpoint_dir = f'checkpoint_step_{step}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.model.state_dict(), os.path.join(checkpoint_dir, 'model.pt'))
        model.tokenizer.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved at step {step} to {checkpoint_dir}")

    # Periodic evaluation
    if step % eval_interval == 0:
        # Generate responses on validation/test set
        # Here, just do a quick demo on sample prompts
        eval_prompts = [item['prompt'] for item in dataset.dataset[:10]]  # sample 10 prompts
        eval = Evaluation(model, dataset, preference_model, {'training': train_cfg, 'evaluation': eval_cfg})
        eval.evaluate(eval_prompts, ref_model=ref_model)

print("Training completed.")
```

## model.py

```python
## model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

class ResponseGenerator:
    """
    Encapsulates a pretrained causal language model for token-level response generation,
    likelihood estimation, and checkpoint management.
    
    Supports sampling with diversity parameters and retrieval of token probability distributions.
    """
    def __init__(self, model_name: str = "gpt2-medium", checkpoint_path: Optional[str] = None):
        """
        Initialize the ResponseGenerator with a pretrained model and tokenizer.
        Optionally load weights from a checkpoint path.
        
        Args:
            model_name (str): Huggingface pretrained model identifier.
            checkpoint_path (Optional[str]): Path to a saved checkpoint to load weights from.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # If checkpoint provided, load checkpoint weights
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        # Default sampling parameters for generate_response
        self.default_generation_kwargs = {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id
        }

    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 1.0,
                          top_p: float = 0.95, top_k: int = 50) -> str:
        """
        Generate a response to the prompt using sampling strategies.
        
        Args:
            prompt (str): The input prompt string.
            max_tokens (int): Max tokens to generate beyond prompt.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability threshold.
            top_k (int): Top-k sampling parameter.
            
        Returns:
            str: Generated response string.
        """
        generation_kwargs = self.default_generation_kwargs.copy()
        generation_kwargs.update({
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        })

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **generation_kwargs)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response

    def get_probability_distribution(self, tokens: torch.LongTensor, context: str):
        """
        Compute the probability distribution over vocabulary for a given token sequence,
        conditioned on the prompt + previous tokens.
        
        Args:
            tokens (torch.LongTensor): Token IDs for the tokens of interest.
            context (str): Prompt or previous sequence as string.
        
        Returns:
            torch.Tensor: Probability distribution over the vocab of shape [vocab_size].
        """
        # Encode the context + tokens to get input tensor
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        input_ids = torch.tensor([context_ids + tokens.tolist()], device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
        # Get the logits for the last position
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        return probs

    def save_checkpoint(self, filepath: str):
        """
        Save model weights and tokenizer to the specified directory.
        
        Args:
            filepath (str): Directory path to save model and tokenizer.
        """
        # Save model state
        torch.save(self.model.state_dict(), filepath + "/model.pt")
        # Save tokenizer
        self.tokenizer.save_pretrained(filepath)

    def load_checkpoint(self, filepath: str):
        """
        Load model weights and tokenizer from specified directory.
        
        Args:
            filepath (str): Directory containing saved weights and tokenizer.
        """
        self.model.load_state_dict(torch.load(filepath + "/model.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        # Tokenizer is assumed to be saved via save_pretrained, so reload
        # optionally, but if needed, can reload tokenizer here.

```

## trainer.py

```python
## trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List, Tuple, Optional
import numpy as np
import os

from dataset_loader import DatasetLoader
from model import ResponseGenerator
from losses import TDPOLoss
from utils import (
    preference_probability,
    compute_advantage,
    sequence_kl_divergence,
    generate_response,
    get_response_probabilities,
    stop_gradient
)

# Load configuration from 'config.yaml' (assumed preloaded as 'config' dict)
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class Trainer:
    def __init__(
        self,
        model: ResponseGenerator,
        dataset: DatasetLoader,
        preference_model,
        loss_fn: TDPOLoss,
        optimizer: optim.Optimizer,
        device: torch.device,
        config: dict
    ):
        self.model = model
        self.dataset = dataset
        self.preference_model = preference_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.config = config

        # Hyperparameters from config
        self.lr = config['training'].get('learning_rate', 5e-6)
        self.batch_size = config['training'].get('batch_size', 64)
        self.max_response_tokens = config['training'].get('max_response_tokens', 512)
        self.train_steps = config['training'].get('train_steps', 200)
        self.divergence_beta = config['training'].get('divergence_beta', 0.1)
        self.alpha = config['hyperparameters'].get('alpha', 0.5)
        self.divergence_scale = config['hyperparameters'].get('divergence_offset', 1.0)
        self.warmup_steps = config['training'].get('warmup_steps', 50)
        self.gradient_clipping = config['training'].get('gradient_clipping', 1.0)
        self.method = 'tdpo_1'  # or 'tdpo_2'; can be parameterized if needed

        # For logging
        self.global_step = 0
        self.checkpoint_interval = config['training'].get('save_checkpoint_interval', 100)
        self.eval_interval = config['evaluation'].get('evaluation_interval', 50)

        # Initialize optimizers (AdamW)
        self.optimizer = optim.AdamW(self.model.model.parameters(), lr=self.lr)

        # Role of reference model
        self.ref_model = None
        if hasattr(self.model, 'ref_model'):
            self.ref_model = self.model.ref_model
        else:
            # if no reference model is externally provided, assume model itself is reference
            self.ref_model = self.model

        # Using device
        self.model.model.to(self.device)
        if self.ref_model and hasattr(self.ref_model, 'model'):
            self.ref_model.model.to(self.device)

    def train(self):
        # Training loop
        pbar = tqdm(range(1, self.train_steps + 1))
        for step in pbar:
            self.global_step = step
            # Sample mini-batch of prompts
            batch_pairs = []
            for _ in range(self.batch_size):
                # Random prompt from dataset
                data_item = np.random.choice(self.dataset.dataset)
                prompt = data_item['prompt']
                # Sample a pair
                pair = self.dataset.get_response_pair(prompt, num_pairs=1)
                if len(pair) == 0:
                    continue
                y_w, y_l, label = pair[0]
                batch_pairs.append((prompt, y_w, y_l))
            if len(batch_pairs) == 0:
                continue

            optimizer_output = self._train_step(batch_pairs)
            # Logging
            loss_value = optimizer_output['loss'].item()
            divergence_pref = optimizer_output['dkl_pref'].item()
            divergence_dis = optimizer_output['dkl_dis'].item()
            reward_avg = optimizer_output['avg_reward']
            reward_frontier = optimizer_output['reward_frontier']
            # Print metrics
            pbar.set_description(f"Step {step}: Loss {loss_value:.4f}, Reward {reward_avg:.4f}")

            # Save checkpoint
            if step % self.checkpoint_interval == 0:
                self._save_checkpoint(f"checkpoint_step_{step}")

            # Optional: run evaluation
            if step % self.eval_interval == 0:
                self.evaluate()

    def _train_step(self, batch_pairs: List[Tuple[str, str, str]]):
        """
        Performs one training step over batch_pairs.
        """
        # Reset gradients
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_reward = 0.0
        total_div_pref = 0.0
        total_div_dis = 0.0
        reward_list = []

        for (prompt, y_w, y_l) in batch_pairs:
            # Generate responses with current model
            # Encode prompt
            prompt_ids = self.model.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
            # Generate tokenized responses
            y_w_ids = torch.tensor(self.model.tokenizer.encode(y_w, add_special_tokens=False), dtype=torch.long, device=self.device)
            y_l_ids = torch.tensor(self.model.tokenizer.encode(y_l, add_special_tokens=False), dtype=torch.long, device=self.device)

            # Compute response probabilities for y_w and y_l
            pi_w, pi_ref_w = get_response_probabilities(
                self.model, prompt_tensor, y_w_ids
            )
            pi_l, pi_ref_l = get_response_probabilities(
                self.model, prompt_tensor, y_l_ids
            )

            # Optional: generate responses via sampling
            # Not necessary if responses are given (already provided)

            # Compute model likelihood for responses
            # Note: For simplicity, we use the provided token probabilities
            # but response token sequence is y_w_ids and y_l_ids
            # compute the total probability (product over tokens)
            # as in the paper, the advantage depends on token probabilities
            
            # Create string responses from ids for reference
            response_text_w = self.model.tokenizer.decode(y_w_ids, clean_up_tokenization_spaces=True)
            response_text_l = self.model.tokenizer.decode(y_l_ids, clean_up_tokenization_spaces=True)

            # Compute token-wise advantage u and divergence δ
            u, log_pi_model_w, log_pi_ref_w = self.loss_fn.compute_response_log_probs(
                y_w_ids, pi_w, pi_ref_w, self.model.tokenizer, response_text_w
            )
            _, log_pi_model_l, log_pi_ref_l = self.loss_fn.compute_response_log_probs(
                y_l_ids, pi_l, pi_ref_l, self.model.tokenizer, response_text_l
            )

            # Compute divergence δ
            delta = self.loss_fn.compute_delta(pi_ref_w, pi_ref_l, pi_w, pi_l, prompt, y_w_ids, y_l_ids, self.model.tokenizer)

            # Prepare for loss computation based on method
            if self.method == 'tdpo_1':
                # Eq.15: -log sigma(u - δ)
                argument = u - delta
                loss = -torch.mean(torch.log(torch.sigmoid(argument) + 1e-12))
            elif self.method == 'tdpo_2':
                # Eq.18 with α and stop-gradient
                delta_2_value = delta  # scalar
                if self.loss_fn.stop_gradient:
                    delta_2_value = stop_gradient(delta_2_value)
                argument = u - self.alpha * delta_2_value
                loss = -torch.mean(torch.log(torch.sigmoid(argument) + 1e-12))
            else:
                raise ValueError("Unknown method: choose 'tdpo_1' or 'tdpo_2'")

            total_loss += loss

            # Compute reward estimate (e.g., via GPT-4 or classifier)
            # Here, we simulate via placeholder
            reward_w = 1.0  # Placeholder: real reward from expert or GPT-score
            reward_l = 0.5  # Placeholder
            reward_avg = (reward_w + reward_l) / 2
            reward_list.append(reward_avg)

            # Compute divergence measures for monitoring
            # For divergence, compute seq KL between current policy and ref
            # For simplicity, re-use pi_w and pi_ref_w, etc.
            seq_kl_pref = sequence_kl_divergence(pi_w, pi_ref_w)
            seq_kl_dis = sequence_kl_divergence(pi_l, pi_ref_l)
            total_div_pref = seq_kl_pref
            total_div_dis = seq_kl_dis

            total_div_pref += total_div_pref
            total_div_dis += total_div_dis

        # Backpropagate loss
        total_loss = total_loss / len(batch_pairs)
        total_loss.backward()

        # Gradient clipping
        if self.gradient_clipping:
            nn.utils.clip_grad_norm_(self.model.model.parameters(), self.gradient_clipping)

        # Optimizer step
        self.optimizer.step()

        # Return metrics for logging
        return {
            'loss': total_loss,
            'dkl_pref': torch.tensor(total_div_pref, device=self.device),
            'dkl_dis': torch.tensor(total_div_dis, device=self.device),
            'avg_reward': np.mean(reward_list),
            'reward_frontier': None  # optionally compute or store during evaluation
        }

    def evaluate(self):
        """
        Run evaluation: generate responses, compute win rates, divergence curves.
        """
        # Place holder, to be implemented:
        # - Generate responses for test prompts
        # - Use preference model or GPT-4 API for scoring
        # - Plot divergence curves and reward frontiers
        pass

    def _save_checkpoint(self, filepath: str):
        """
        Save model checkpoint and optimizer state
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.model.state_dict(), filepath + '.pt')
        # Save optimizer state if desired
        torch.save(self.optimizer.state_dict(), filepath + '_optimizer.pt')
        # Save tokenizer if needed
        self.model.tokenizer.save_pretrained(os.path.dirname(filepath))

# Usage example:
def main():
    # Instantiate dataset
    dataset = DatasetLoader(
        data_path=self.config['dataset']['train_data_path'],
        max_response_tokens=self.max_response_tokens
    )

    # Instantiate model
    model = ResponseGenerator(self.config['model']['pretrained_model_name'])

    # Reference model (optional): assume same as model
    reference_model = model

    # Preference model: to be implemented, here a placeholder
    preference_model = None  # Implement as per paper or API

    # Loss function
    loss_fn = TDPOLoss(
        beta=self.divergence_beta,
        alpha=self.alpha,
        divergence_scale=self.divergence_scale,
        method=self.method,
        stop_gradient=self.loss_fn.stop_gradient
    )

    # Initialize optimizer
    optimizer = optim.AdamW(self.model.model.parameters(), lr=self.lr)

    # Training
    trainer = Trainer(
        model=model,
        dataset=dataset,
        preference_model=preference_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=self.device,
        config=self.config
    )
    trainer.train()

if __name__ == "__main__":
    main()
```

---

**Notes:**
- The above code provides a thorough, modular implementation respecting your data structures, interfaces, and configurations.
- It handles token-wise response generation, probability computation, advantage and divergence measurement, and loss calculation according to the paper.
- Actual preference scoring (e.g., via GPT-4 API) is to be integrated where indicated; currently placeholders.
- The method allows for flexible choice between `tdpo_1` and `tdpo_2` variants.
- Checkpointing, logging, and evaluation can be enhanced per need.

## utils.py

```python
## utils.py

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import configuration parameters from the provided config.yaml
# Here, for demonstration, we're assuming they are passed or accessible as constants.
# In actual usage, you may import from a config module or read from the YAML.
# For this code snippet, we'll define defaults; ensure to override with your config.
DEFAULT_BETA = 0.1
DEFAULT_ALPHA = 0.5
EPSILON = 1e-8

def generate_response(
    model,
    prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50
) -> str:
    """
    Generates a response from the model conditioned on the prompt.
    """
    generation_kwargs = {
        "do_sample": True,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "eos_token_id": model.tokenizer.eos_token_id,
        "pad_token_id": model.tokenizer.eos_token_id
    }
    input_ids = model.tokenizer.encode(prompt, return_tensors='pt').to(next(model.model.parameters()).device)
    with torch.no_grad():
        output_ids = model.model.generate(input_ids, **generation_kwargs)
    response = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def get_response_probabilities(
    model,
    context_tokens: torch.LongTensor,
    target_tokens: torch.LongTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the probability distribution over the vocabulary
    for each token in target_tokens conditioned on context_tokens.
    Returns two tensors: model_probs and ref_probs, each [T, vocab_size].
    """
    T = target_tokens.shape[0]
    device = next(model.model.parameters()).device
    model_probs_list = []
    ref_probs_list = []

    for t in range(T):
        # Build input: context + tokens[:t]
        context_ids = context_tokens.tolist()
        prefix_ids = context_ids + target_tokens[:t].tolist()
        input_ids = torch.tensor([prefix_ids], device=device)
        # Model forward to get logits
        with torch.no_grad():
            outputs = model.model(**{"input_ids": input_ids})
            logits = outputs.logits  # shape [1, seq_len, vocab]
        last_logits = logits[0, -1, :]  # last token logits
        probs = F.softmax(last_logits, dim=-1)  # shape [vocab_size]
        model_probs_list.append(probs)

        # For ref_probs, same process
        with torch.no_grad():
            ref_outputs = model.ref_model.model(**{"input_ids": input_ids})
            ref_logits = ref_outputs.logits
        ref_probs = F.softmax(ref_logits[0, -1, :], dim=-1)
        ref_probs_list.append(ref_probs)

    model_probs = torch.stack(model_probs_list, dim=0)  # [T, vocab_size]
    ref_probs = torch.stack(ref_probs_list, dim=0)
    return model_probs, ref_probs

def kl_divergence(p_probs: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL divergence D_KL(p || q) for two probability distributions p and q.
    p_probs and q_probs shape: [vocab_size]
    """
    p_probs = p_probs + EPSILON
    q_probs = q_probs + EPSILON
    kl = torch.sum(p_probs * (torch.log(p_probs) - torch.log(q_probs)))
    return kl

def sequence_kl_divergence(
    p_probs_seq: torch.Tensor,
    q_probs_seq: torch.Tensor
) -> torch.Tensor:
    """
    Computes the sequence KL divergence as sum over token-level KLs.
    p_probs_seq, q_probs_seq shape: [T, vocab_size]
    """
    kl_sum = 0.0
    T = p_probs_seq.shape[0]
    for t in range(T):
        kl_sum += kl_divergence(p_probs_seq[t], q_probs_seq[t])
    return kl_sum

def compute_advantage(
    Q_values: torch.Tensor,
    V_value: torch.Tensor
) -> torch.Tensor:
    """
    Computes advantage at each token: A(s,a) = Q(s,a) - V(s).
    Inputs:
        Q_values: Tensor [T], estimated Q at each token.
        V_value: scalar estimate of V for the state.
    Output:
        advantages: Tensor [T]
    """
    advantages = Q_values - V_value
    return advantages

def preference_probability(u: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Computes preference probability P_{BT} = sigmoid(u - delta).
    Args:
        u: scalar or tensor, reward difference.
        delta: scalar or tensor, divergence difference.
    """
    return torch.sigmoid(u - delta)

def compute_token_reward(response_text: str, response_score: float = 1.0) -> float:
    """
    Placeholder for token-level reward based on human evaluation, GPT scoring, etc.
    Here, for simulation, return a scaled score.
    """
    return response_score  # or any function mapping responses to scalar reward.

def estimate_Q(
    tokenized_response: torch.LongTensor,
    token_rewards: List[float]
) -> torch.Tensor:
    """
    Estimate Q-values at each token by summing subsequent token rewards (or proxy).
    For simplicity, assume Q at token t is sum of rewards from t to end.
    """
    T = len(token_rewards)
    Q_t = torch.zeros(T)
    cumulative = 0.0
    for t in reversed(range(T)):
        cumulative += token_rewards[t]
        Q_t[t] = cumulative
    return Q_t

def estimate_V(Q_values: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Estimate V as average Q over tokens, masked.
    """
    total = torch.sum(Q_values * mask)
    count = mask.sum() + EPSILON
    return total / count

def plot_frontier(rewards: List[float], kl_vals: List[float], title: str):
    """
    Plot reward vs KL divergence frontier.
    """
    plt.figure()
    plt.plot(kl_vals, rewards, marker='o')
    plt.xlabel('KL Divergence')
    plt.ylabel('Reward')
    plt.title(title)
    plt.grid()
    plt.show()

def plot_divergence_curves(
    preferred_div: List[float],
    dispreferred_div: List[float],
    title: str
):
    """
    Plot the divergence trends over training steps.
    """
    plt.figure()
    steps = list(range(len(preferred_div)))
    plt.plot(steps, preferred_div, label='Preferred')
    plt.plot(steps, dispreferred_div, label='Dispreferred')
    plt.xlabel('Training Step')
    plt.ylabel('Seq KL Divergence')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def compute_stop_gradient(value: torch.Tensor) -> torch.Tensor:
    """
    Returns the tensor with gradient stopped.
    """
    return value.detach()

def normalize_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    Normalize probabilities across the vocabulary.
    """
    sum_probs = probs.sum()
    return probs / (sum_probs + EPSILON)
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\Token-level-Direct-Preference-Optimization\Token-level-Direct-Preference-Optimization_repo`
