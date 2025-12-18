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
