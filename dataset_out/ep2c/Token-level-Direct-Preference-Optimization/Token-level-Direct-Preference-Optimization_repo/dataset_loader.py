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
