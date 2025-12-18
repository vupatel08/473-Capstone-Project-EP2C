## dataset_loader.py
import os
import random
from typing import List, Tuple, Dict, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

class DatasetLoader:
    """
    Responsible for loading datasets, tokenizing, sampling sequences, and extracting signals
    suitable for PCA spectrum analysis, consistent with experimental setup.
    """
    def __init__(
        self,
        dataset_name: str = 'WikiText-2',
        dataset_path: Optional[str] = None,
        model_name_or_path: str = 'gpt2',  # default, can be overridden
        sample_size: int = 1024,
        sequence_length: int = 2048,
        batch_size: int = 32,
        seed: int = 42
    ):
        """
        Initialize dataset loader.
        Args:
            dataset_name (str): Name of the dataset ('WikiText-2' or 'Alpaca')
            dataset_path (str): Path to dataset if local, else None
            model_name_or_path (str): Hugging Face model identifier for tokenizer
            sample_size (int): Number of sequences to sample for PCA
            sequence_length (int): Length of each sequence in tokens
            batch_size (int): Batch size for processing
            seed (int): Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.model_name_or_path = model_name_or_path
        self.sample_size = sample_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.seed = seed

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        # Ensure the tokenizer does not add special tokens if not desired
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset
        self.dataset = self._load_dataset()

        # Tokenize entire dataset into token id list
        self.tokenized_data = self._tokenize_dataset()

        # Collect total size
        self.total_tokens = len(self.tokenized_data)

    def _load_dataset(self):
        """
        Load dataset from HuggingFace datasets or local files.
        Returns:
            dict or Dataset object with columns 'text' (or similar)
        """
        if self.dataset_name.lower() == 'wikitext-2':
            # load from Hugging Face dataset
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        elif self.dataset_name.lower() == 'alpaca':
            # For Alpaca with local files, replace with your dataset path
            if self.dataset_path is None:
                raise ValueError("Path must be provided for custom datasets like Alpaca.")
            # Assuming a simple text file with one data point per line
            # implement as needed; here, a placeholder
            # For the example, we skip actual implementation as data may not be available
            raise NotImplementedError("Custom dataset loading for Alpaca not implemented.")
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")
        return dataset

    def _tokenize_dataset(self) -> List[int]:
        """
        Tokenize the entire dataset's text to a flat list of token ids.
        Returns:
            token_list (List[int]) - concatenated tokens from dataset
        """
        token_list: List[int] = []

        # If dataset is a HuggingFace dataset
        if hasattr(self.dataset, 'column_names'):
            # For datasets as BatchDicts
            texts = []
            for example in self.dataset:
                if isinstance(example, dict):
                    # pick a text field, commonly 'text' for Wikitext-2
                    text = example.get('text', None)
                    if text is None:
                        # fallback if structure differs
                        text = list(example.values())[0]
                else:
                    text = example
                texts.append(text)
            # Concatenate all texts
            full_text = '\n'.join(texts)
        else:
            # fallback: assume dataset is a list of strings
            full_text = '\n'.join(self.dataset['text'])

        # Tokenize entire dataset text
        tokens = self.tokenizer(full_text, add_special_tokens=False, truncation=False)['input_ids']
        return tokens

    def sample_sequences(self) -> torch.Tensor:
        """
        Sample `sample_size` sequences of length `sequence_length` from tokenized data.
        Returns:
            signals tensor of shape (N, D), where N = total tokens sampled, D = model embedding dim
        """
        # Set seed for reproducibility
        random.seed(self.seed)

        max_start_index = self.total_tokens - self.sequence_length
        if max_start_index <= 0:
            raise ValueError("Dataset too small for the specified sequence length.")

        # Sample start indices without replacement
        start_indices = random.sample(range(max_start_index), self.sample_size)

        # Collect sequences
        sequences: List[List[int]] = []
        for start_idx in start_indices:
            seq = self.tokenized_data[start_idx:start_idx + self.sequence_length]
            sequences.append(seq)

        # Convert list of sequences to tensor
        token_ids = torch.tensor(sequences, dtype=torch.long)  # shape: (sample_size, seq_len)

        # Flatten to (N, D) - N = total tokens, D=1 initially
        # but per our context, signals are the inputs to subsequent layers.
        # For simplicity, get embedding-like signals: here, use token embeddings as signals.
        # Alternatively, for more accurate signals (e.g., layer inputs), integrate with model.
        # But for dataset loader, use embeddings as signals placeholder:
        return token_ids

    def get_signals(self, model_wrapper) -> torch.Tensor:
        """
        Given a model wrapper with accessible embedding layer, generate signals
        for each sequence by passing token ids and extracting layer inputs.
        Args:
            model_wrapper: instance of ModelWrapper with method to get activations
        Returns:
            signals: torch.Tensor of shape (N, D)
        """
        signals_list: List[torch.Tensor] = []

        # Batch processing
        num_batches = (self.sample_size + self.batch_size - 1) // self.batch_size
        current_idx = 0

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, self.sample_size)
            batch_size_actual = batch_end - batch_start

            # Get batch token ids
            batch_token_ids = self._get_batch_token_ids(batch_start, batch_end)

            # Run through model to extract activations
            # Assume model_wrapper has method `extract_layer_inputs`
            layer_inputs = model_wrapper.extract_layer_inputs(batch_token_ids)

            # layer_inputs shape: (batch_size_actual, seq_len, D)
            # Reshape to (batch_size_actual * seq_len, D)
            batch_signals = layer_inputs.reshape(-1, layer_inputs.shape[-1])

            signals_list.append(batch_signals.cpu())

        # Concatenate all signals
        signals = torch.cat(signals_list, dim=0)  # shape: (N, D)
        return signals

    def _get_batch_token_ids(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Retrieve token id sequences for a batch between start_idx and end_idx
        by index from tokenized_data.
        """
        batch_token_ids = self.tokenized_data[start_idx:self.sample_size].unsqueeze(0)
        # But since tokenized_data is a 1D tensor, slice accordingly
        # Make a batch of sequences:
        sequences = []
        for i in range(start_idx, end_idx):
            seq = self.tokenized_data[i:i + self.sequence_length]
            if len(seq) < self.sequence_length:
                # pad with eos_token or truncate, here truncate
                seq = torch.cat([seq, torch.full((self.sequence_length - len(seq),), self.tokenizer.eos_token_id)])
            sequences.append(seq.unsqueeze(0))
        batch = torch.cat(sequences, dim=0)
        return batch

    def load_dataset(self):
        return self._load_dataset()
