## dataset_loader.py

import os
import json
import csv
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
import re

@dataclass
class Sample:
    query: str
    responses: List[str]  # For pairwise, responses are [Response1, Response2]; for single response, list with one element
    scenario: str
    annotation: Optional[Union[int, str]] = None  # Preference label or rating
    critique: Optional[str] = None

@dataclass
class Dataset:
    samples: List[Sample]

class DatasetLoader:
    def __init__(self, config: dict):
        """
        Initialize DatasetLoader with configuration parameters.
        """
        # Paths to raw data files; expects keys: 'train_files', 'val_files', 'test_files'
        self.train_files: List[str] = config.get('data', {}).get('train_files', [])
        self.val_files: List[str] = config.get('data', {}).get('val_files', [])
        self.test_files: List[str] = config.get('data', {}).get('test_files', [])
        # Scenario list and mapping
        self.scenario_list: List[str] = config.get('dataset', {}).get('scenario_list', [])
        self.scenario_map: Dict[str, str] = self._generate_scenario_mapping()
        # Heuristics parameters, e.g., min/max response length, confidence thresholds
        self.min_response_length: int = 5
        self.max_response_length: int = 2000  # prevent excessively long
        # Language filtering: Remove non-English samples based on heuristic
        self.english_char_ratio_threshold: float = 0.85

    def _generate_scenario_mapping(self) -> Dict[str, str]:
        """
        Generate mapping from raw scenario labels to standardized scenario list.
        Placeholder implementation: assume raw labels are same as scenario_list.
        Can be extended when loading data.
        """
        # For simplicity, assume raw labels match scenario names
        return {scenario: scenario for scenario in self.scenario_list}

    def _is_english(self, text: str) -> bool:
        """
        Simple heuristic: check the proportion of ASCII letters.
        """
        if not text:
            return False
        ascii_chars = sum(c.isascii() and c.isprintable() for c in text)
        ratio = ascii_chars / max(1, len(text))
        return ratio >= self.english_char_ratio_threshold

    def _truncate_middle(self, text: str, max_length: int) -> str:
        """
        Truncate text from the middle and insert ellipsis if it exceeds max_length.
        """
        if len(text) <= max_length:
            return text
        half = (max_length - 3) // 2
        return text[:half] + "..." + text[-half:]

    def load_data(self, split: str = 'train') -> Dataset:
        """
        Load and process dataset based on the split.
        """
        if split == 'train':
            files = self.train_files
        elif split == 'val':
            files = self.val_files
        elif split == 'test':
            files = self.test_files
        else:
            raise ValueError(f"Invalid dataset split: {split}")

        samples: List[Sample] = []

        for file_path in files:
            if not os.path.exists(file_path):
                print(f"Warning: file {file_path} does not exist.")
                continue
            # Support different formats: JSONL, CSV, or raw JSON
            ext = os.path.splitext(file_path)[1].lower()
            with open(file_path, 'r', encoding='utf-8') as f:
                if ext in ['.jsonl', '.json']:
                    for line in f:
                        data = json.loads(line)
                        self._parse_sample(data, samples)
                elif ext == '.csv':
                    reader = csv.DictReader(f)
                    for row in reader:
                        self._parse_sample(row, samples)
                else:
                    # For unrecognized format, assume raw JSON list
                    try:
                        data_list = json.load(f)
                        for data in data_list:
                            self._parse_sample(data, samples)
                    except json.JSONDecodeError:
                        print(f"Unrecognized file format for {file_path}. Skipping.")
        # Filtering process
        filtered_samples = self._filter_samples(samples)

        return Dataset(filtered_samples)

    def _parse_sample(self, raw_data, samples: List[Sample]):
        """
        Parse raw sample data into Sample class. Supports multiple formats.
        """
        # Initialize variables
        query = ""
        responses: List[str] = []
        scenario = ""
        annotation = None

        # Support different raw data schemas:
        # 1. Dict with keys: 'query', 'response', 'scenario', 'annotation'
        # 2. Dict with 'responses' as list, 'scenario', 'annotation'
        # 3. Different keys based on source
        if isinstance(raw_data, dict):
            query = raw_data.get('query') or raw_data.get('question') or raw_data.get('input')
            scenario_raw = raw_data.get('scenario') or raw_data.get('scenario_name') or raw_data.get('scenario_label')
            # Map scenario to standard
            if scenario_raw in self.scenario_map:
                scenario = self.scenario_map[scenario_raw]
            elif scenario_raw:
                # fallback: check mapping or assign 'others' if unknown
                scenario = scenario_raw if scenario_raw in self.scenario_list else 'others'
            else:
                scenario = 'others'

            # Responses: handle pairwise or single
            if 'responses' in raw_data:
                responses = list(raw_data['responses'])
            elif 'response' in raw_data:
                responses = [raw_data['response']]
            elif 'responses_single' in raw_data:
                responses = list(raw_data['responses_single'])
            elif 'response1' in raw_data and 'response2' in raw_data:
                responses = [raw_data['response1'], raw_data['response2']]
            else:
                responses = [raw_data.get('response', '')]

            # Annotation (preference or rating)
            annotation = raw_data.get('annotation')
            if annotation is None:
                annotation = raw_data.get('preference')
            if annotation is None:
                annotation = raw_data.get('rating')

            # Critique if available
            critique = raw_data.get('critique')

        else:
            # For other data formats, skip
            return

        # Ensure responses are non-empty
        responses = [resp.strip() for resp in responses if resp and len(resp.strip()) >= self.min_response_length]

        # Add sample if valid
        if not query or not responses:
            return

        # Limit response length
        responses = [self._truncate_middle(r, self.max_response_length) for r in responses]

        sample = Sample(
            query=query,
            responses=responses,
            scenario=scenario,
            annotation=annotation,
            critique=critique
        )
        samples.append(sample)

    def _filter_samples(self, samples: List[Sample]) -> List[Sample]:
        """
        Apply heuristics to filter noisy or low-quality samples.
        """
        filtered: List[Sample] = []
        for sample in samples:
            # Filter based on response length
            if not all(self.min_response_length <= len(r) <= self.max_response_length for r in sample.responses):
                continue
            # Filter non-English
            if not self._is_english(sample.query):
                continue
            for resp in sample.responses:
                if not self._is_english(resp):
                    break  # skip sample if any response isn't English
            else:
                # All responses are English
                # Filter annotations if available
                if sample.annotation is not None:
                    # For ratings, ensure within scale
                    if isinstance(sample.annotation, (int, float)):
                        if not (1 <= sample.annotation <= 10):
                            continue
                    # For preference labels, accept 'win', 'tie', 'lose'
                    elif isinstance(sample.annotation, str):
                        if sample.annotation.lower() not in ['win', 'tie', 'lose', '1', '2']:
                            continue
                    # For other cases, keep for now
                # Additional heuristics can be added here
                filtered.append(sample)
        return filtered
